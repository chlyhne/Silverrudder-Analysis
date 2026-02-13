"""
Silverrudder figure generation pipeline.

This script builds the PDF figures consumed by the LaTeX report. It intentionally
keeps all logic in one place so the data-preparation method and the plotting
method use the same assumptions.

High-level idea:
1. Load race metadata + raw GPS samples for each boat.
2. Convert each sample onto a shared 1D "progress along route" axis.
3. Build local fleet baselines (speed/pace) at each progress window.
4. Compute per-sample deltas (boat value - fleet baseline value).
5. Aggregate deltas by leg and by boat.
6. Export two figure sets (desktop and phone) with profile-specific layout.

Important terms used throughout:
- routeIdx: Each sample's nearest index on the average route polyline.
- progress: routeIdx converted to [0, 1] for leg slicing and interpolation.
- window stats: Fleet summary (min/max/mean) computed over local progress windows.
- pace: Minutes per nautical mile, computed as 60 / speed(knots).
- delta: Boat metric minus local fleet baseline at the same progress.

Output behavior:
- Figures are always exported to documentation/figures/{desktop,phone}/...
- This file does not compile LaTeX. It only generates figure PDFs.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}"
matplotlib.rcParams["axes.unicode_minus"] = False


import box_plot_utils as box_plots
import silver_helpers as sh
import silver_plot_helpers as sph

# Reader guide for this file:
# - main() orchestrates the full pipeline and is the best place to start.
# - Plotting/tick/map boilerplate is in silver_plot_helpers.py.
# - "Waypoint / leg helper functions" here convert progress into leg-aware slices.
# - "Speed/Pace delta sampling..." computes reusable data used by plot exporters.
# - "plot_*_box_plot" functions are the report-facing exporters.
# - All plotting functions accept export_dir/export_and_close to support both
#   batch mode and interactive debugging mode without code duplication.

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------
# Keep all style knobs centralized so plot appearance remains consistent.
# Speed and pace share most styling, but axis tick behavior differs slightly.
# Physical y-scaling is controlled elsewhere through units-per-inch settings.
BOX_PLOT_STYLE_COMMON = box_plots.BoxPlotStyle()
BOX_PLOT_AXIS_STYLE_COMMON = box_plots.BoxPlotAxisStyle()
BOX_PLOT_STYLE_SPEED = BOX_PLOT_STYLE_COMMON
BOX_PLOT_AXIS_STYLE_SPEED = BOX_PLOT_AXIS_STYLE_COMMON
BOX_PLOT_STYLE_PACE = BOX_PLOT_STYLE_COMMON
BOX_PLOT_AXIS_STYLE_PACE = replace(BOX_PLOT_AXIS_STYLE_COMMON, major_tick=2.0)
BOX_PLOT_WHISKER_SCALE = 1.5


@dataclass(frozen=True)
class FigureProfile:
    """
    Rendering profile for one output family (desktop or phone).

    Why this exists:
    - The same data must be readable in two contexts with very different widths.
    - We want textual sizing and physical axis spacing to stay intentional.
    - By grouping settings in one dataclass, the rest of the plotting code can
      stay profile-agnostic.
    """

    name: str
    output_subdir: str
    rc_params: Dict[str, object]
    waypoint_label_fontsize: float
    scalar_marker_size: float
    pace_units_per_inch_factor: float
    speed_units_per_inch_factor: float
    boxplot_top_margin_in: float
    boxplot_bottom_margin_in: float
    boxplot_left_margin_in: float
    boxplot_right_margin_in: float


FIGURE_PROFILES = {
    "desktop": FigureProfile(
        name="desktop",
        output_subdir="desktop",
        rc_params={
            # Match LaTeX text width on A4 with 25 mm margins: 160 mm.
            "figure.figsize": (160.0 / 25.4, 5.2),
            "font.size": 11.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 11.0,
        },
        waypoint_label_fontsize=9.0,
        scalar_marker_size=6.0,
        pace_units_per_inch_factor=1.10,
        speed_units_per_inch_factor=0.95,
        boxplot_top_margin_in=0.30,
        boxplot_bottom_margin_in=0.95,
        boxplot_left_margin_in=0.75,
        boxplot_right_margin_in=0.20,
    ),
    "phone": FigureProfile(
        name="phone",
        output_subdir="phone",
        rc_params={
            # Match LaTeX text width on phone layout: 108 mm paper with 4 mm margins => 100 mm.
            "figure.figsize": (100.0 / 25.4, 5.2),
            "font.size": 11.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 11.0,
        },
        waypoint_label_fontsize=11.0,
        scalar_marker_size=8.0,
        pace_units_per_inch_factor=1.00,
        speed_units_per_inch_factor=1.00,
        boxplot_top_margin_in=0.30,
        boxplot_bottom_margin_in=0.95,
        boxplot_left_margin_in=0.60,
        boxplot_right_margin_in=0.15,
    ),
}
ACTIVE_FIGURE_PROFILE = FIGURE_PROFILES["desktop"]


# ---------------------------------------------------------------------------
# Small utility helpers used by many plotting functions
# ---------------------------------------------------------------------------
def set_active_figure_profile(profile):
    """
    Set the global active profile.

    Plot helpers read ACTIVE_FIGURE_PROFILE at render time for marker size,
    waypoint label font size, margin policy, and physical axis scaling factors.
    """
    global ACTIVE_FIGURE_PROFILE
    ACTIVE_FIGURE_PROFILE = profile


def prepare_figure(fig, export_and_close=False):
    """
    Apply interactive-only figure prep.

    During batch export we deliberately avoid maximizing windows because exported
    size should come from explicit figsize/layout settings, not screen geometry.
    """
    if not export_and_close:
        sh.maximize_figure(fig)


def save_figure(fig, output_path):
    """
    Save figure exactly as configured.

    We do not pass bbox/scale overrides here; canvas size and subplot margins
    are treated as the single source of truth for text and geometry.
    """
    fig.savefig(output_path)


def latex_display_name(name: str) -> str:
    """
    Map selected boat names to explicit LaTeX spellings for report text.

    Keep source metadata unchanged (for IDs/filenames) and only alter rendered
    labels/titles in plots.
    """
    if name == "Aegir 2.0":
        return "Ægir 2.0"
    if name == "Nordri":
        return r"Nor\dh ri"
    return name


def apply_boxplot_physical_layout(fig, local_range, units_per_inch, extra_bottom_in=0.0):
    """
    Resize a box-plot figure so y-units map to consistent physical height.

    Parameters:
    - local_range: Data range currently shown on y-axis.
    - units_per_inch: Desired conversion from y-units -> inches.
    - extra_bottom_in: Additional bottom margin (inches) for long rotated labels.

    This is the key function behind "comparable plots":
    if two plots have the same units_per_inch, one extra unit on the y-axis
    occupies the same physical height in both exported PDFs.
    """
    fig_width = fig.get_size_inches()[0]
    data_height_in = local_range / units_per_inch if units_per_inch > 0 else fig.get_size_inches()[1]
    top_in = ACTIVE_FIGURE_PROFILE.boxplot_top_margin_in
    bottom_in = ACTIVE_FIGURE_PROFILE.boxplot_bottom_margin_in + max(0.0, extra_bottom_in)
    left_in = ACTIVE_FIGURE_PROFILE.boxplot_left_margin_in
    right_in = ACTIVE_FIGURE_PROFILE.boxplot_right_margin_in

    # Final figure height = data band height + fixed top/bottom UI margins.
    fig_height = data_height_in + top_in + bottom_in
    fig.set_size_inches(fig_width, fig_height, forward=True)

    left = left_in / fig_width
    right = 1.0 - (right_in / fig_width)
    bottom = bottom_in / fig_height
    top = 1.0 - (top_in / fig_height)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def main():
    # -----------------------------------------------------------------------
    # Stage 1: Load raw inputs
    # -----------------------------------------------------------------------
    # End-to-end pipeline:
    # - load metadata + tracks
    # - align samples to a reference route
    # - compute baseline statistics
    # - build/export figure families
    data_root = Path("data") / "silverrudder_2025"
    filename = str(data_root / "Silverrudder 2025_Keelboats Small_gps_data.csv")
    metadata_path = data_root / "race_metadata.json"

    # Metadata contains stable IDs/names/colors and race-time information.
    # All later labeling/order/color behavior depends on this mapping.
    race_meta = sh.load_race_metadata(metadata_path)
    track_id_keys = race_meta["track_id_keys"]
    boat_names = race_meta["boat_names"]

    # Build per-boat track structs from the raw CSV.
    # We attach names early so every downstream warning/plot already has human labels.
    tracking_data = sh.read_tracking_csv_as_struct(filename)
    tracks = sh.build_tracks(
        tracking_data,
        "tracked_object_id",
        "Latitude",
        "Longitude",
        "SampleTime",
        "Speed",
    )
    tracks = sh.convert_speed_to_knots(tracks)
    track_name_map = dict(zip(track_id_keys, boat_names))
    tracks = sh.apply_track_names(tracks, track_name_map)
    # First-leg scalar metrics are gate-time based (start -> first gate),
    # so we need a reliable absolute start timestamp.
    start_time_utc = parse_start_time_utc(race_meta)

    # -----------------------------------------------------------------------
    # Stage 2: Route/gate normalization
    # -----------------------------------------------------------------------
    # Detect Start/Finish gate positions and crop each track to that race segment.
    # This avoids contamination from pre-start and post-finish samples.
    geo_data_path = metadata_path.parent / "waypoints" / "geo_data.json"
    waypoint_gates, start_gate_pos, finish_gate_pos, first_leg_gate_pos = load_waypoint_gates(geo_data_path)
    gate_times_by_track = sh.compute_gate_crossings(tracks, waypoint_gates)
    tracks = sh.trim_tracks_by_gate_times(tracks, gate_times_by_track, start_gate_pos, finish_gate_pos)
    # Apply fixed boat colors so all plots stay visually coherent.
    boat_colors = race_meta.get("boat_colors", {})
    track_color_map = sh.build_boat_color_map(boat_names, boat_colors)
    tracks = sh.apply_track_colors(tracks, track_color_map)

    route_sample_count = 20000
    # Search window limits route-index jumps when mapping GPS points to route points.
    # This reduces accidental back-and-forth index spikes.
    route_search_window_half_width = int(np.ceil(route_sample_count / 100))
    # The average route is the shared geometry used to define "same location on course".
    # Every boat sample is projected onto this route to get a comparable progress value.
    average_route = sh.compute_average_route(route_sample_count, geo_data_path)
    tracks = sh.map_track_points_to_route(tracks, average_route, route_search_window_half_width)
    tracks = sh.remove_route_index_spikes(tracks, route_sample_count)
    # Waypoint progress values define leg boundaries for all leg-based plots.
    way_point_progress, way_point_names = sh.compute_waypoint_progress_from_gates(average_route, waypoint_gates)

    window_sample_count = 50
    window_step_samples = 1
    filter_alpha = 0.01
    # -----------------------------------------------------------------------
    # Stage 3: Fleet baseline construction
    # -----------------------------------------------------------------------
    # Compute fleet window stats along progress. This is the baseline for:
    # - map alpha coloring
    # - speed delta calculations
    # - pace delta calculations
    tracks, speed_window_stats = sh.compute_sample_alpha_by_route_windows(
        tracks, route_sample_count, window_sample_count, window_step_samples, filter_alpha
    )

    # -----------------------------------------------------------------------
    # Stage 4: Export policy and plot selection
    # -----------------------------------------------------------------------
    # We run in batch-export mode by default: no interactive windows, write PDFs, close.
    export_and_close_figures = True
    figure_output_root = Path("documentation") / "figures"
    # Turn off interactive backends for predictable batch operation.
    if export_and_close_figures:
        plt.ioff()

    # Toggle matrix for each output group.
    # Keeping toggles centralized makes experimentation easier without touching logic.
    show_map_plot = False
    show_pace_box_plots = True
    show_pace_box_plots_by_boat = True
    show_speed_box_plots = True
    show_speed_box_plots_by_boat = True
    show_pace_range_plot = False

    # -----------------------------------------------------------------------
    # Stage 5: Shared data prep for speed/pace variants
    # -----------------------------------------------------------------------
    # Precompute pace inputs once and reuse for both "by leg" + "by boat" pace plots.
    pace_box_plot_data = None
    if show_pace_box_plots or show_pace_box_plots_by_boat:
        pace_box_plot_data = prepare_pace_delta_box_plot_data(
            tracks,
            speed_window_stats,
            way_point_progress,
            way_point_names,
            average_route,
            start_time_utc,
            first_leg_gate_pos,
        )

    # Precompute speed inputs once and reuse for both speed plot families.
    speed_box_plot_data = None
    if show_speed_box_plots or show_speed_box_plots_by_boat:
        speed_box_plot_data = prepare_speed_delta_box_plot_data(
            tracks,
            speed_window_stats,
            way_point_progress,
            way_point_names,
            average_route,
            start_time_utc,
            first_leg_gate_pos,
        )

    # -----------------------------------------------------------------------
    # Stage 6: Render profile families
    # -----------------------------------------------------------------------
    # Always export both families so LaTeX wrappers can choose inclusion profile later.
    figure_profiles = [FIGURE_PROFILES["desktop"], FIGURE_PROFILES["phone"]]
    for figure_profile in figure_profiles:
        set_active_figure_profile(figure_profile)
        # Profile subfolder keeps outputs isolated:
        # documentation/figures/desktop/*
        # documentation/figures/phone/*
        export_dir = (
            figure_output_root / figure_profile.output_subdir
            if export_and_close_figures
            else None
        )
        # rc_context ensures profile typography/size settings do not leak globally.
        with matplotlib.rc_context(figure_profile.rc_params):
            if show_map_plot:
                plot_colored_tracks(
                    tracks,
                    average_route,
                    speed_window_stats,
                    way_point_progress,
                    way_point_names,
                    export_path=(export_dir / "map.pdf") if export_and_close_figures else None,
                    export_and_close=export_and_close_figures,
                )

            if show_pace_box_plots:
                plot_pace_delta_box_plot(
                    tracks,
                    pace_box_plot_data,
                    export_dir=export_dir,
                    export_and_close=export_and_close_figures,
                )

            if show_pace_box_plots_by_boat:
                plot_pace_delta_box_plot_by_boat(
                    tracks,
                    pace_box_plot_data,
                    export_dir=export_dir,
                    export_and_close=export_and_close_figures,
                )

            if show_speed_box_plots:
                plot_speed_delta_box_plot(
                    tracks,
                    speed_box_plot_data,
                    export_dir=export_dir,
                    export_and_close=export_and_close_figures,
                )

            if show_speed_box_plots_by_boat:
                plot_speed_delta_box_plot_by_boat(
                    tracks,
                    speed_box_plot_data,
                    export_dir=export_dir,
                    export_and_close=export_and_close_figures,
                )

            if show_pace_range_plot:
                plot_speed_range_along_route(
                    tracks,
                    speed_window_stats,
                    way_point_progress,
                    way_point_names,
                    export_path=(export_dir / "pace-range-along-route.pdf") if export_and_close_figures else None,
                    export_and_close=export_and_close_figures,
                )

    # In interactive mode only, keep figures open for manual inspection.
    if not export_and_close_figures:
        plt.show()


# ---------------------------------------------------------------------------
# Waypoint / leg helper functions
# ---------------------------------------------------------------------------
def parse_start_time_utc(race_meta):
    """
    Parse race start time from metadata into UTC epoch seconds.

    The metadata stores a local timestamp plus an offset; we convert once here
    so every first-leg calculation uses the same time basis.
    """
    # Metadata stores local time plus offset; convert once so downstream metrics stay consistent.
    local_offset_hours = race_meta.get("local_offset_hours", 0)
    start_time_local = race_meta.get("start_time_local", "")
    start_time_utc = sh.parse_local_time_to_utc(start_time_local, local_offset_hours)
    if start_time_utc is None:
        raise ValueError("start_time_local missing or invalid in race_metadata.json")
    return start_time_utc


def load_waypoint_gates(geo_data_path):
    """
    Load gate geometry and discover canonical Start/Finish gate indices.

    Returns:
    - waypoint_gates: Full gate list from geo data.
    - start_gate_pos: Index of Start gate in waypoint_gates.
    - finish_gate_pos: Index of Finish gate in waypoint_gates.
    - first_leg_gate_pos: Index of first mark after Start.
    """
    _, waypoint_gates = sh.load_geo_data(geo_data_path)
    start_gate_pos, finish_gate_pos = sh.find_start_finish_gate_positions(waypoint_gates)
    # The first leg is defined as Start -> next gate, used for scalar delta metrics.
    first_leg_gate_pos = start_gate_pos + 1
    if first_leg_gate_pos >= len(waypoint_gates):
        raise ValueError("No waypoint available after Start for first leg.")
    return waypoint_gates, start_gate_pos, finish_gate_pos, first_leg_gate_pos


def build_leg_labels(progress_values, waypoint_labels):
    """
    Build per-leg labels from consecutive waypoint labels.

    Example:
    [Start, Troense, Thuro] -> ["Start-Troense", "Troense-Thuro"]
    """
    leg_labels = []
    # Keep labels usable even if some waypoint names are missing in metadata.
    for leg_index in range(len(progress_values) - 1):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")
    return leg_labels


def get_leg_bounds(progress_values, leg_index):
    """Return monotonically ordered [start, end] progress bounds for one leg."""
    leg_start = progress_values[leg_index]
    leg_end = progress_values[leg_index + 1]
    if leg_start > leg_end:
        leg_start, leg_end = leg_end, leg_start
    return leg_start, leg_end


def compute_first_leg_distance_m(average_route, progress_values):
    """
    Compute physical first-leg distance (meters) from route geometry + progress.

    First-leg scalar pace/speed metrics use gate timing and therefore need a
    physical distance estimate for Start -> first gate.
    """
    if len(progress_values) < 2:
        return np.nan

    # Use the full rhumb-line length to convert fractional progress into meters.
    route_distance_m = sh.cumulative_distance_meters(
        np.asarray(average_route["lat"], dtype=float),
        np.asarray(average_route["lon"], dtype=float),
    )[-1]
    if not np.isfinite(route_distance_m) or route_distance_m <= 0:
        return np.nan
    return abs(progress_values[1] - progress_values[0]) * route_distance_m


def compute_first_leg_delta(values_by_track):
    """
    Convert raw first-leg scalar values into deltas against fleet mean.

    The output semantics match later per-sample deltas:
    negative pace delta = faster; positive speed delta = faster.
    """
    # Center first-leg values on the fleet mean so deltas are comparable to later legs.
    if values_by_track.size:
        mean_value = float(np.nanmean(values_by_track))
    else:
        mean_value = np.nan
    return values_by_track - mean_value


def get_leg_samples_for_track(progress, delta_values, progress_values, leg_index, first_leg_value):
    """
    Return a track's samples for one leg.

    Behavior:
    - leg_index == 0: returns a synthetic 1-value array from first_leg_value.
    - later legs: slices delta_values by progress interval.

    This is the key abstraction that lets first leg and later legs share the
    same plotting pipeline even though they originate from different data.
    """
    if leg_index == 0:
        # The first leg uses the gate-based scalar delta to avoid sparse route-window sampling.
        if not np.isfinite(first_leg_value):
            return None
        return np.asarray([first_leg_value], dtype=float)

    if progress.size == 0:
        return None

    # Later legs use route-progress windows to select all samples within the leg bounds.
    leg_start, leg_end = get_leg_bounds(progress_values, leg_index)
    leg_mask = (progress >= leg_start) & (progress <= leg_end)
    leg_samples = delta_values[leg_mask]
    if leg_samples.size == 0:
        return None
    return leg_samples


def build_leg_ordered(track_progress_by_track, delta_by_track, progress_values, leg_index, first_leg_delta_by_track):
    """
    Collect + rank tracks by leg mean delta.

    Returns list entries as:
    (track_index, leg_mean, leg_samples)
    """
    leg_ordered = []
    # Sorting by mean delta makes the box-plot ordering reflect relative performance.
    for track_index, (progress, delta_values) in enumerate(
        zip(track_progress_by_track, delta_by_track)
    ):
        first_leg_value = first_leg_delta_by_track[track_index]
        leg_samples = get_leg_samples_for_track(
            progress, delta_values, progress_values, leg_index, first_leg_value
        )
        if leg_samples is None:
            continue
        leg_ordered.append((track_index, float(np.mean(leg_samples)), leg_samples))
    return leg_ordered


def compute_track_robust_delta_range(progress, delta_values, progress_values, first_leg_value):
    """
    Compute a single track's robust min/max delta range across all legs.

    "Robust" here means Tukey-style whisker bounds via `update_whisker_range`,
    which are less sensitive to extreme outliers than raw min/max.
    """
    local_lower = np.inf
    local_upper = -np.inf
    # Aggregate whisker bounds across all legs to keep per-boat y-scales consistent.
    for leg_index in range(len(progress_values) - 1):
        leg_samples = get_leg_samples_for_track(
            progress, delta_values, progress_values, leg_index, first_leg_value
        )
        if leg_samples is None:
            continue
        local_lower, local_upper = box_plots.update_whisker_range(
            local_lower,
            local_upper,
            leg_samples,
            whisker_scale=BOX_PLOT_WHISKER_SCALE,
        )
    if not np.isfinite(local_lower) or not np.isfinite(local_upper):
        return None
    return local_lower, local_upper


def compute_leg_robust_delta_range_across_tracks(
    track_progress_by_track,
    delta_by_track,
    progress_values,
    leg_index,
    first_leg_delta_by_track,
):
    """
    Compute robust min/max delta range for one leg across all tracks.

    Uses the same Tukey-style whisker bound method as
    `compute_track_robust_delta_range` for consistency.
    """
    local_lower = np.inf
    local_upper = -np.inf
    for track_index, (progress, delta_values) in enumerate(
        zip(track_progress_by_track, delta_by_track)
    ):
        first_leg_value = first_leg_delta_by_track[track_index]
        leg_samples = get_leg_samples_for_track(
            progress, delta_values, progress_values, leg_index, first_leg_value
        )
        if leg_samples is None:
            continue
        local_lower, local_upper = box_plots.update_whisker_range(
            local_lower,
            local_upper,
            leg_samples,
            whisker_scale=BOX_PLOT_WHISKER_SCALE,
        )
    if not np.isfinite(local_lower) or not np.isfinite(local_upper):
        return None
    return local_lower, local_upper


def compute_metric_scale_context(
    track_progress_by_track,
    delta_by_track,
    progress_values,
    first_leg_delta_by_track,
    units_per_inch_factor=1.0,
):
    """
    Build metric-wide scaling context for physically comparable plots.

    Returns:
    - global_lower/global_upper/global_range: robust global y-range
    - units_per_inch: y-unit -> inch conversion used in figure sizing

    This context is shared by:
    - per-leg plots for that metric
    - per-boat plots for that metric
    """
    global_lower = float("inf")
    global_upper = float("-inf")
    for track_index, (progress, delta_values) in enumerate(
        zip(track_progress_by_track, delta_by_track)
    ):
        if progress.size == 0:
            continue
        leg_range = compute_track_robust_delta_range(
            progress, delta_values, progress_values, first_leg_delta_by_track[track_index]
        )
        if leg_range is None:
            continue
        global_lower = min(global_lower, leg_range[0])
        global_upper = max(global_upper, leg_range[1])

    global_range_result = box_plots.compute_global_plot_range(global_lower, global_upper)
    if global_range_result is None:
        return None
    global_lower, global_upper, global_range = global_range_result
    units_per_inch = box_plots.compute_units_per_inch(global_range)
    units_per_inch *= units_per_inch_factor
    return global_lower, global_upper, global_range, units_per_inch


def plot_colored_tracks(
    tracks,
    average_route,
    speed_window_stats,
    waypoint_progress,
    waypoint_names,
    export_path=None,
    export_and_close=False,
):
    """
    Plot route map with all tracks, colored by local relative speed.

    This is mainly a diagnostic/inspection figure:
    - validates route mapping
    - validates alpha baseline behavior
    - supports interactive datatips for manual checks
    """
    # Disable TeX text rendering for the map to keep annotations lightweight.
    previous_usetex = matplotlib.rcParams.get("text.usetex", False)
    matplotlib.rcParams["text.usetex"] = False
    fig, ax = plt.subplots()
    prepare_figure(fig, export_and_close=export_and_close)
    ax.grid(True)
    # Preserve metric proportions so the route geometry is not visually distorted.
    sh.apply_local_meter_aspect(ax, average_route)

    # Coastline is purely contextual; crop to track bounds to keep it lightweight.
    sph.plot_coast_geojson_cropped(ax, tracks, "ne_10m_coastline.geojson", 0.15)

    rhumb_line, = ax.plot(
        average_route["lon"],
        average_route["lat"],
        color=(0, 0.6, 1),
        linewidth=5.0,
        label="Rhumb line",
    )
    rhumb_line.set_picker(True)
    rhumb_line.set_pickradius(6)

    sph.plot_waypoints_on_route(
        ax,
        average_route,
        waypoint_progress,
        waypoint_names,
        ACTIVE_FIGURE_PROFILE.waypoint_label_fontsize,
    )

    cmap, norm = sph.build_alpha_colormap()
    sph.add_alpha_tracks(ax, tracks, cmap, norm)

    # Colorbar uses the same normalization as the segment coloring.
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label("alpha (rel. speed)")

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Tracks colored by alpha")

    # Enable interactive data tips that combine route position and speed stats.
    sh.enable_manual_datatips(ax, tracks, average_route, speed_window_stats)
    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, export_path)
    if export_and_close:
        plt.close(fig)
    matplotlib.rcParams["text.usetex"] = previous_usetex


@dataclass(frozen=True)
class SpeedDeltaBoxPlotData:
    """
    Shared speed-delta payload used by both speed plot families.

    Storing this in a dataclass avoids recomputing expensive sampling logic and
    makes function signatures easier to reason about.
    """

    track_progress_by_track: List[np.ndarray]
    speed_delta_by_track: List[np.ndarray]
    progress_values: np.ndarray
    leg_labels: List[str]
    first_leg_delta_by_track: np.ndarray


def prepare_speed_delta_box_plot_data(
    tracks,
    speed_window_stats,
    way_point_progress,
    way_point_names,
    average_route,
    start_time_utc,
    first_leg_gate_pos,
) -> Optional[SpeedDeltaBoxPlotData]:
    """
    Prepare all speed-delta structures required for plotting.

    Returns None when required prerequisites are missing, allowing callers to
    skip speed plotting gracefully.
    """
    # Reuse the same delta samples for both "per leg" and "per boat" plots.
    track_progress_by_track, speed_delta_by_track = compute_speed_delta_samples(
        tracks, speed_window_stats
    )
    if track_progress_by_track is None:
        return None

    # Progress values define leg boundaries and also drive label generation.
    progress_values, waypoint_labels = sh.normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return None

    leg_labels = build_leg_labels(progress_values, waypoint_labels)

    # First-leg averages are gate-based, so compute them separately from route-window deltas.
    first_leg_distance_m = compute_first_leg_distance_m(average_route, progress_values)
    first_leg_speed_by_track = compute_first_leg_speed_by_track(
        tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m
    )
    first_leg_delta_by_track = compute_first_leg_delta(first_leg_speed_by_track)

    return SpeedDeltaBoxPlotData(
        track_progress_by_track=track_progress_by_track,
        speed_delta_by_track=speed_delta_by_track,
        progress_values=progress_values,
        leg_labels=leg_labels,
        first_leg_delta_by_track=first_leg_delta_by_track,
    )


def plot_speed_delta_box_plot(
    tracks,
    speed_plot_data: Optional[SpeedDeltaBoxPlotData],
    export_dir=None,
    export_and_close=False,
):
    """
    Export one speed-delta figure per leg (boats on x-axis).

    Sorting:
    - Higher speed delta is better, so ordering is descending by leg mean.
    """
    if speed_plot_data is None:
        return

    track_progress_by_track = speed_plot_data.track_progress_by_track
    speed_delta_by_track = speed_plot_data.speed_delta_by_track
    progress_values = speed_plot_data.progress_values
    leg_labels = speed_plot_data.leg_labels
    first_leg_delta_by_track = speed_plot_data.first_leg_delta_by_track

    # Iterate each leg separately to produce a dedicated figure per leg.
    leg_count = len(progress_values) - 1
    scale_context = compute_metric_scale_context(
        track_progress_by_track,
        speed_delta_by_track,
        progress_values,
        first_leg_delta_by_track,
        units_per_inch_factor=ACTIVE_FIGURE_PROFILE.speed_units_per_inch_factor,
    )
    if scale_context is None:
        return
    global_lower, global_upper, global_range, units_per_inch = scale_context

    for leg_index in range(leg_count - 1, -1, -1):
        leg_range = compute_leg_robust_delta_range_across_tracks(
            track_progress_by_track,
            speed_delta_by_track,
            progress_values,
            leg_index,
            first_leg_delta_by_track,
        )
        if leg_range is None:
            leg_range = (global_lower, global_upper)
        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            leg_range[0], leg_range[1], global_range
        )

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        apply_boxplot_physical_layout(fig, local_range, units_per_inch, extra_bottom_in=0.15)
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_SPEED)
        ax.set_ylim(local_lower, local_upper)

        # Order boats by mean delta so the ranking is visually consistent.
        leg_ordered = build_leg_ordered(
            track_progress_by_track,
            speed_delta_by_track,
            progress_values,
            leg_index,
            first_leg_delta_by_track,
        )
        if not leg_ordered:
            ax.set_title(leg_labels[leg_index])
            if export_dir is not None:
                export_dir.mkdir(parents=True, exist_ok=True)
                safe_label = sh.sanitize_filename_label(leg_labels[leg_index])
                output_path = export_dir / f"speed-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
                save_figure(fig, output_path)
            if export_and_close:
                plt.close(fig)
            continue

        # Reverse ordering so faster (higher) speed deltas appear first on speed plots.
        leg_ordered.sort(key=lambda item: item[1], reverse=True)
        for plot_index, (track_index, _, speed_leg) in enumerate(leg_ordered, start=1):
            line_color = tracks[track_index]["color"] or (0.3, 0.3, 0.3)
            if leg_index == 0:
                # First leg is a single scalar delta, plotted as a point.
                ax.plot(
                    plot_index,
                    speed_leg,
                    marker="o",
                    markersize=ACTIVE_FIGURE_PROFILE.scalar_marker_size,
                    color=line_color,
                    linestyle="None",
                )
            else:
                # Later legs have enough samples for a box plot.
                box_plots.draw_manual_box_plot(
                    ax,
                    plot_index,
                    speed_leg,
                    line_color,
                    style=BOX_PLOT_STYLE_SPEED,
                    whisker_scale=BOX_PLOT_WHISKER_SCALE,
                )

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [latex_display_name(tracks[track_index]["name"]) for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylabel(r"$\Delta \mathrm{Speed}\,[\mathrm{kn}]$")
        ax.set_ylim(local_lower, local_upper)
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            # Export per-leg figures with a stable name for reports.
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sh.sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"speed-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


def plot_speed_delta_box_plot_by_boat(
    tracks,
    speed_plot_data: Optional[SpeedDeltaBoxPlotData],
    export_dir=None,
    export_and_close=False,
):
    """
    Export one speed-delta figure per boat (legs on x-axis).

    These figures are intended for skipper-specific diagnosis across legs.
    """
    if speed_plot_data is None:
        return

    track_progress_by_track = speed_plot_data.track_progress_by_track
    speed_delta_by_track = speed_plot_data.speed_delta_by_track
    progress_values = speed_plot_data.progress_values
    leg_labels = speed_plot_data.leg_labels
    first_leg_delta_by_track = speed_plot_data.first_leg_delta_by_track

    # Compute one metric-wide scale context so per-boat and per-leg plots are physically comparable.
    leg_count = len(progress_values) - 1
    scale_context = compute_metric_scale_context(
        track_progress_by_track,
        speed_delta_by_track,
        progress_values,
        first_leg_delta_by_track,
        units_per_inch_factor=ACTIVE_FIGURE_PROFILE.speed_units_per_inch_factor,
    )
    if scale_context is None:
        return
    _, _, global_range, units_per_inch = scale_context

    # Each boat gets its own figure with a locally adjusted y-range.
    for track_index, track in enumerate(tracks):
        progress = track_progress_by_track[track_index]
        speed_delta = speed_delta_by_track[track_index]
        if progress.size == 0 or speed_delta.size == 0:
            continue

        leg_range = compute_track_robust_delta_range(
            progress, speed_delta, progress_values, first_leg_delta_by_track[track_index]
        )
        if leg_range is None:
            continue
        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            leg_range[0], leg_range[1], global_range
        )

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        # Keep a modest guard band for long rotated leg labels.
        boat_tick_extra_bottom_in = 0.40
        boat_tick_edge_padding = 0.70
        if ACTIVE_FIGURE_PROFILE.name == "phone":
            boat_tick_extra_bottom_in = 0.75
            boat_tick_edge_padding = 0.85
        # Resize vertically so value ranges occupy a comparable physical height.
        apply_boxplot_physical_layout(
            fig,
            local_range,
            units_per_inch,
            extra_bottom_in=boat_tick_extra_bottom_in,
        )
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_SPEED)

        line_color = track.get("color") or (0.3, 0.3, 0.3)
        for leg_index in range(leg_count):
            speed_leg = get_leg_samples_for_track(
                progress,
                speed_delta,
                progress_values,
                leg_index,
                first_leg_delta_by_track[track_index],
            )
            if speed_leg is None:
                continue
            if leg_index == 0:
                # First leg is a scalar delta, so plot a single point.
                ax.plot(
                    leg_index + 1,
                    speed_leg,
                    marker="o",
                    markersize=ACTIVE_FIGURE_PROFILE.scalar_marker_size,
                    color=line_color,
                    linestyle="None",
                )
            else:
                # Later legs use the full box plot to show variability.
                box_plots.draw_manual_box_plot(
                    ax,
                    leg_index + 1,
                    speed_leg,
                    line_color,
                    style=BOX_PLOT_STYLE_SPEED,
                    whisker_scale=BOX_PLOT_WHISKER_SCALE,
                )

        ax.set_xticks(range(1, leg_count + 1))
        ax.set_xticklabels(
            leg_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=ACTIVE_FIGURE_PROFILE.waypoint_label_fontsize,
        )
        ax.set_xlim(1.0 - boat_tick_edge_padding, leg_count + boat_tick_edge_padding)
        ax.set_ylabel(r"$\Delta \mathrm{Speed}\,[\mathrm{kn}]$")
        ax.set_ylim(local_lower, local_upper)
        ax.set_title(latex_display_name(track.get("name", "Boat")))

        if export_dir is not None:
            # Persist per-boat figures for inclusion in reports.
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sh.sanitize_filename_label(track.get("name", f"boat-{track_index + 1}"))
            output_path = export_dir / f"speed-delta-boat-{safe_name}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Speed/Pace delta sampling and first-leg scalar derivation
# ---------------------------------------------------------------------------
def compute_speed_delta_samples(tracks, speed_window_stats):
    """
    Build per-track speed delta samples against fleet mean-by-progress baseline.

    Output:
    - track_progress_by_track[i]: progress vector for boat i
    - speed_delta_by_track[i]: speed - baseline_speed at matching progress
    """
    if "meanSpeedByWindow" not in speed_window_stats:
        return None, None

    mean_speed_by_window = speed_window_stats["meanSpeedByWindow"]
    window_step_samples = speed_window_stats["windowStepSamples"]
    route_sample_count = speed_window_stats["routeSampleCount"]
    window_count = len(mean_speed_by_window)
    window_progress = np.asarray(speed_window_stats.get("windowProgress", []), dtype=float)

    track_progress_by_track = []
    speed_delta_by_track = []

    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        speed = track["speed"].astype(float)
        # Keep only samples that map cleanly to a valid route index.
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(speed)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
        )
        if not valid_mask.any():
            track_progress_by_track.append(np.array([]))
            speed_delta_by_track.append(np.array([]))
            continue

        route_index_valid = route_index[valid_mask]
        # Convert route index to fractional progress for leg selection.
        progress = (route_index_valid - 1) / (route_sample_count - 1)
        if window_progress.size >= 2:
            # Interpolate baseline mean speed onto continuous progress values.
            mean_mask = np.isfinite(window_progress) & np.isfinite(mean_speed_by_window)
            if np.count_nonzero(mean_mask) >= 2:
                mean_speed_for_sample = np.interp(
                    progress,
                    window_progress[mean_mask],
                    mean_speed_by_window[mean_mask],
                    left=np.nan,
                    right=np.nan,
                )
            else:
                mean_speed_for_sample = np.full_like(progress, np.nan, dtype=float)
        else:
            # Fall back to discrete window mapping when window progress is unavailable.
            home_window = np.floor((route_index_valid - 1) / window_step_samples) + 1
            home_window = np.clip(home_window, 1, window_count).astype(int)
            mean_speed_for_sample = mean_speed_by_window[home_window - 1]

        # Reject invalid baselines so they do not pollute the delta samples.
        mean_speed_for_sample = np.where(
            np.isfinite(mean_speed_for_sample) & (mean_speed_for_sample > 0),
            mean_speed_for_sample,
            np.nan,
        )

        # Delta is boat speed relative to the fleet mean at the same progress point.
        speed_delta = speed[valid_mask] - mean_speed_for_sample
        delta_mask = np.isfinite(speed_delta)
        speed_delta = speed_delta[delta_mask]
        progress = progress[delta_mask]
        if speed_delta.size == 0:
            track_progress_by_track.append(np.array([]))
            speed_delta_by_track.append(np.array([]))
            continue

        track_progress_by_track.append(progress)
        speed_delta_by_track.append(speed_delta)

    return track_progress_by_track, speed_delta_by_track


def compute_first_leg_speed_by_track(tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m):
    """
    Compute first-leg average speed (kn) from Start -> first gate crossing.

    This first leg is treated specially because route-window samples can be
    sparse/noisy during starts; gate-time based scalar is more stable here.
    """
    speed_values = []
    # First-leg speed depends on start time and gate crossing times; bail out if missing.
    if not np.isfinite(start_time_utc) or first_leg_distance_m <= 0:
        return np.full(len(tracks), np.nan, dtype=float)

    # Convert meters to nautical miles to keep speed in knots.
    leg_distance_nm = first_leg_distance_m / 1852.0
    if leg_distance_nm <= 0:
        return np.full(len(tracks), np.nan, dtype=float)

    for track in tracks:
        gate_times = track.get("gateTimes", np.array([]))
        if gate_times is None or len(gate_times) <= first_leg_gate_pos:
            speed_values.append(np.nan)
            continue

        finish_time = gate_times[first_leg_gate_pos]
        if not np.isfinite(finish_time) or finish_time <= start_time_utc:
            speed_values.append(np.nan)
            continue

        leg_time_hours = (finish_time - start_time_utc) / 3600.0
        if leg_time_hours <= 0:
            speed_values.append(np.nan)
            continue

        speed_knots = leg_distance_nm / leg_time_hours
        speed_values.append(speed_knots)

    return np.asarray(speed_values, dtype=float)


def compute_pace_delta_samples(tracks, speed_window_stats):
    """
    Build per-track pace delta samples against fleet baseline pace.

    Pace baseline preference:
    1) use meanPaceByWindow if available
    2) otherwise derive as 60 / meanSpeedByWindow
    """
    if "meanSpeedByWindow" not in speed_window_stats:
        return None, None

    mean_speed_by_window = speed_window_stats["meanSpeedByWindow"]
    window_step_samples = speed_window_stats["windowStepSamples"]
    route_sample_count = speed_window_stats["routeSampleCount"]
    window_count = len(mean_speed_by_window)
    window_progress = np.asarray(speed_window_stats.get("windowProgress", []), dtype=float)

    mean_pace_by_window = speed_window_stats.get("meanPaceByWindow")
    if mean_pace_by_window is None or len(mean_pace_by_window) == 0:
        # Derive mean pace from mean speed when the pace baseline is missing.
        mean_pace_by_window = np.where(
            np.isfinite(mean_speed_by_window) & (mean_speed_by_window > 0),
            60.0 / mean_speed_by_window,
            np.nan,
        )

    track_progress_by_track = []
    pace_delta_by_track = []

    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        speed = track["speed"].astype(float)
        # Require positive speed because pace is undefined for zero or negative values.
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(speed)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
            & (speed > 0)
        )
        if not valid_mask.any():
            track_progress_by_track.append(np.array([]))
            pace_delta_by_track.append(np.array([]))
            continue

        route_index_valid = route_index[valid_mask]
        # Progress lets us align pace samples to legs.
        progress = (route_index_valid - 1) / (route_sample_count - 1)
        if window_progress.size >= 2:
            # Interpolate the baseline pace onto continuous progress.
            mean_mask = np.isfinite(window_progress) & np.isfinite(mean_pace_by_window)
            if np.count_nonzero(mean_mask) >= 2:
                mean_pace_for_sample = np.interp(
                    progress,
                    window_progress[mean_mask],
                    mean_pace_by_window[mean_mask],
                    left=np.nan,
                    right=np.nan,
                )
            else:
                mean_pace_for_sample = np.full_like(progress, np.nan, dtype=float)
        else:
            # Fall back to discrete window mapping when progress is not available.
            home_window = np.floor((route_index_valid - 1) / window_step_samples) + 1
            home_window = np.clip(home_window, 1, window_count).astype(int)
            mean_pace_for_sample = mean_pace_by_window[home_window - 1]
        # Filter invalid baselines before computing deltas.
        mean_pace_for_sample = np.where(
            np.isfinite(mean_pace_for_sample) & (mean_pace_for_sample > 0),
            mean_pace_for_sample,
            np.nan,
        )

        # Pace is minutes per NM; delta is relative to the fleet baseline at that progress.
        pace_samples = 60.0 / speed[valid_mask]
        pace_delta = pace_samples - mean_pace_for_sample
        delta_mask = np.isfinite(pace_delta)
        pace_delta = pace_delta[delta_mask]
        progress = progress[delta_mask]
        if pace_delta.size == 0:
            track_progress_by_track.append(np.array([]))
            pace_delta_by_track.append(np.array([]))
            continue

        track_progress_by_track.append(progress)
        pace_delta_by_track.append(pace_delta)

    return track_progress_by_track, pace_delta_by_track


def compute_first_leg_pace_by_track(tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m):
    """
    Compute first-leg pace (min/NM) from Start -> first gate crossing.

    Same philosophy as first-leg speed helper: use gate timing to avoid
    start-area sampling artifacts.
    """
    pace_values = []
    # First-leg pace depends on the race start time and gate crossing time.
    if not np.isfinite(start_time_utc) or first_leg_distance_m <= 0:
        return np.full(len(tracks), np.nan, dtype=float)

    # Use nautical miles to keep pace in minutes per NM.
    leg_distance_nm = first_leg_distance_m / 1852.0
    if leg_distance_nm <= 0:
        return np.full(len(tracks), np.nan, dtype=float)

    for track in tracks:
        gate_times = track.get("gateTimes", np.array([]))
        if gate_times is None or len(gate_times) <= first_leg_gate_pos:
            pace_values.append(np.nan)
            continue

        finish_time = gate_times[first_leg_gate_pos]
        if not np.isfinite(finish_time) or finish_time <= start_time_utc:
            pace_values.append(np.nan)
            continue

        leg_time_sec = finish_time - start_time_utc
        pace_min_per_nm = (leg_time_sec / 60.0) / leg_distance_nm
        pace_values.append(pace_min_per_nm)

    return np.asarray(pace_values, dtype=float)


@dataclass(frozen=True)
class PaceDeltaBoxPlotData:
    """
    Shared pace-delta payload used by both pace plot families.

    Mirrors SpeedDeltaBoxPlotData so the two metric pipelines stay symmetric.
    """

    track_progress_by_track: List[np.ndarray]
    pace_delta_by_track: List[np.ndarray]
    progress_values: np.ndarray
    leg_labels: List[str]
    first_leg_delta_by_track: np.ndarray


def prepare_pace_delta_box_plot_data(
    tracks,
    speed_window_stats,
    way_point_progress,
    way_point_names,
    average_route,
    start_time_utc,
    first_leg_gate_pos,
) -> Optional[PaceDeltaBoxPlotData]:
    """
    Prepare all pace-delta structures required for plotting.

    Returns None when required prerequisites are missing.
    """
    # Reuse the same pace delta samples for both pace plot variants.
    track_progress_by_track, pace_delta_by_track = compute_pace_delta_samples(
        tracks, speed_window_stats
    )
    if track_progress_by_track is None:
        return None

    # Progress values define leg boundaries and label ordering.
    progress_values, waypoint_labels = sh.normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return None

    leg_labels = build_leg_labels(progress_values, waypoint_labels)

    # First-leg pace comes from start-to-gate timing, not route-window samples.
    first_leg_distance_m = compute_first_leg_distance_m(average_route, progress_values)
    first_leg_pace_by_track = compute_first_leg_pace_by_track(
        tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m
    )
    first_leg_delta_by_track = compute_first_leg_delta(first_leg_pace_by_track)

    return PaceDeltaBoxPlotData(
        track_progress_by_track=track_progress_by_track,
        pace_delta_by_track=pace_delta_by_track,
        progress_values=progress_values,
        leg_labels=leg_labels,
        first_leg_delta_by_track=first_leg_delta_by_track,
    )


def plot_pace_delta_box_plot(
    tracks,
    pace_plot_data: Optional[PaceDeltaBoxPlotData],
    export_dir=None,
    export_and_close=False,
):
    """
    Export one pace-delta figure per leg (boats on x-axis).

    Sorting:
    - Lower pace delta is better, so ordering is ascending by leg mean.
    """
    if pace_plot_data is None:
        return

    track_progress_by_track = pace_plot_data.track_progress_by_track
    pace_delta_by_track = pace_plot_data.pace_delta_by_track
    progress_values = pace_plot_data.progress_values
    leg_labels = pace_plot_data.leg_labels
    first_leg_delta_by_track = pace_plot_data.first_leg_delta_by_track

    # Generate one figure per leg for easier inspection.
    leg_count = len(progress_values) - 1
    scale_context = compute_metric_scale_context(
        track_progress_by_track,
        pace_delta_by_track,
        progress_values,
        first_leg_delta_by_track,
        units_per_inch_factor=ACTIVE_FIGURE_PROFILE.pace_units_per_inch_factor,
    )
    if scale_context is None:
        return
    global_lower, global_upper, global_range, units_per_inch = scale_context

    for leg_index in range(leg_count - 1, -1, -1):
        leg_range = compute_leg_robust_delta_range_across_tracks(
            track_progress_by_track,
            pace_delta_by_track,
            progress_values,
            leg_index,
            first_leg_delta_by_track,
        )
        if leg_range is None:
            leg_range = (global_lower, global_upper)
        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            leg_range[0], leg_range[1], global_range
        )

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        apply_boxplot_physical_layout(fig, local_range, units_per_inch, extra_bottom_in=0.15)
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_PACE)
        ax.set_ylim(local_lower, local_upper)

        # Order boats by mean pace delta to emphasize relative ranking.
        leg_ordered = build_leg_ordered(
            track_progress_by_track,
            pace_delta_by_track,
            progress_values,
            leg_index,
            first_leg_delta_by_track,
        )
        if not leg_ordered:
            ax.set_title(leg_labels[leg_index])
            if export_dir is not None:
                export_dir.mkdir(parents=True, exist_ok=True)
                safe_label = sh.sanitize_filename_label(leg_labels[leg_index])
                output_path = export_dir / f"pace-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
                save_figure(fig, output_path)
            if export_and_close:
                plt.close(fig)
            continue

        leg_ordered.sort(key=lambda item: item[1])
        for plot_index, (track_index, _, pace_leg) in enumerate(leg_ordered, start=1):
            line_color = tracks[track_index]["color"] or (0.3, 0.3, 0.3)
            if leg_index == 0:
                # First leg is a single scalar delta, plotted as a point.
                ax.plot(
                    plot_index,
                    pace_leg,
                    marker="o",
                    markersize=ACTIVE_FIGURE_PROFILE.scalar_marker_size,
                    color=line_color,
                    linestyle="None",
                )
            else:
                # Later legs use full box plots to show spread.
                box_plots.draw_manual_box_plot(
                    ax,
                    plot_index,
                    pace_leg,
                    line_color,
                    style=BOX_PLOT_STYLE_PACE,
                    whisker_scale=BOX_PLOT_WHISKER_SCALE,
                )

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [latex_display_name(tracks[track_index]["name"]) for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylabel(r"$\Delta \mathrm{Pace}\,[\mathrm{min}\,\mathrm{NM}^{-1}]$")
        ax.set_ylim(local_lower, local_upper)
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            # Export per-leg figures for report inclusion.
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sh.sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"pace-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


def plot_pace_delta_box_plot_by_boat(
    tracks,
    pace_plot_data: Optional[PaceDeltaBoxPlotData],
    export_dir=None,
    export_and_close=False,
):
    """
    Export one pace-delta figure per boat (legs on x-axis).

    These plots are typically used in the "Boat summaries" section of the report.
    """
    if pace_plot_data is None:
        return

    track_progress_by_track = pace_plot_data.track_progress_by_track
    pace_delta_by_track = pace_plot_data.pace_delta_by_track
    progress_values = pace_plot_data.progress_values
    leg_labels = pace_plot_data.leg_labels
    first_leg_delta_by_track = pace_plot_data.first_leg_delta_by_track

    # Compute one metric-wide scale context so per-boat and per-leg plots are physically comparable.
    leg_count = len(progress_values) - 1
    scale_context = compute_metric_scale_context(
        track_progress_by_track,
        pace_delta_by_track,
        progress_values,
        first_leg_delta_by_track,
        units_per_inch_factor=ACTIVE_FIGURE_PROFILE.pace_units_per_inch_factor,
    )
    if scale_context is None:
        return
    _, _, global_range, units_per_inch = scale_context

    for track_index, track in enumerate(tracks):
        progress = track_progress_by_track[track_index]
        pace_delta = pace_delta_by_track[track_index]
        if progress.size == 0 or pace_delta.size == 0:
            continue

        leg_range = compute_track_robust_delta_range(
            progress, pace_delta, progress_values, first_leg_delta_by_track[track_index]
        )
        if leg_range is None:
            continue
        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            leg_range[0], leg_range[1], global_range
        )

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        # Keep a modest guard band for long rotated leg labels.
        boat_tick_extra_bottom_in = 0.40
        boat_tick_edge_padding = 0.70
        if ACTIVE_FIGURE_PROFILE.name == "phone":
            boat_tick_extra_bottom_in = 0.75
            boat_tick_edge_padding = 0.85
        # Maintain a consistent data-to-physical-height ratio across boats.
        apply_boxplot_physical_layout(
            fig,
            local_range,
            units_per_inch,
            extra_bottom_in=boat_tick_extra_bottom_in,
        )
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_PACE)

        line_color = track.get("color") or (0.3, 0.3, 0.3)
        for leg_index in range(leg_count):
            pace_leg = get_leg_samples_for_track(
                progress,
                pace_delta,
                progress_values,
                leg_index,
                first_leg_delta_by_track[track_index],
            )
            if pace_leg is None:
                continue
            if leg_index == 0:
                # First leg uses a single scalar delta.
                ax.plot(
                    leg_index + 1,
                    pace_leg,
                    marker="o",
                    markersize=ACTIVE_FIGURE_PROFILE.scalar_marker_size,
                    color=line_color,
                    linestyle="None",
                )
            else:
                # Later legs use a box plot to show distribution.
                box_plots.draw_manual_box_plot(
                    ax,
                    leg_index + 1,
                    pace_leg,
                    line_color,
                    style=BOX_PLOT_STYLE_PACE,
                    whisker_scale=BOX_PLOT_WHISKER_SCALE,
                )

        ax.set_xticks(range(1, leg_count + 1))
        ax.set_xticklabels(
            leg_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=ACTIVE_FIGURE_PROFILE.waypoint_label_fontsize,
        )
        ax.set_xlim(1.0 - boat_tick_edge_padding, leg_count + boat_tick_edge_padding)
        ax.set_ylabel(r"$\Delta \mathrm{Pace}\,[\mathrm{min}\,\mathrm{NM}^{-1}]$")
        ax.set_ylim(local_lower, local_upper)
        ax.set_title(latex_display_name(track.get("name", "Boat")))

        if export_dir is not None:
            # Export per-boat figures for documentation.
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sh.sanitize_filename_label(track.get("name", f"boat-{track_index + 1}"))
            output_path = export_dir / f"pace-delta-boat-{safe_name}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------
def plot_speed_range_along_route(
    tracks,
    speed_window_stats,
    waypoint_progress=None,
    waypoint_names=None,
    export_path=None,
    export_and_close=False,
):
    """
    Diagnostic plot: all boats' smoothed speed vs progress + fleet envelope.

    This is not part of the default report flow; it's a sanity/analysis aid to
    inspect baseline construction quality.
    """
    if not speed_window_stats or speed_window_stats.get("routeSampleCount", 0) < 2:
        return

    route_sample_count = speed_window_stats["routeSampleCount"]
    min_speed = speed_window_stats["minSpeedByWindow"]
    max_speed = speed_window_stats["maxSpeedByWindow"]
    mean_speed = speed_window_stats.get("meanSpeedByWindow", np.array([]))
    filter_alpha = speed_window_stats.get("filterAlpha", 0)

    window_progress = np.asarray(speed_window_stats.get("windowProgress", []), dtype=float)
    if window_progress.size == 0:
        # Fall back to window midpoints when explicit progress is not stored.
        window_start_index = speed_window_stats["windowStartIndex"]
        window_end_index = speed_window_stats["windowEndIndex"]
        window_center = (window_start_index + window_end_index) / 2
        window_progress = (window_center - 1) / (route_sample_count - 1)

    fig, ax = plt.subplots()
    prepare_figure(fig, export_and_close=export_and_close)
    ax.grid(True)

    default_color = plt.cm.hsv(np.linspace(0, 1, max(len(tracks), 1)))
    for idx, track in enumerate(tracks):
        # Resample on the same grid as the window stats to compare with the envelope.
        speed_on_grid = sph.resample_track_speed(
            track, window_progress, route_sample_count, filter_alpha
        )
        if speed_on_grid is None:
            continue
        line_color = track["color"] or default_color[idx]
        ax.plot(
            window_progress,
            speed_on_grid,
            color=line_color,
            linewidth=0.8,
            label=latex_display_name(track["name"]),
        )

    # Envelope lines show fleet-wide min/mean/max at each progress window.
    ax.plot(window_progress, min_speed, "b-", linewidth=1.5, label="Slowest speed (min)")
    ax.plot(window_progress, max_speed, "r-", linewidth=1.5, label="Fastest speed (max)")
    if mean_speed.size:
        ax.plot(window_progress, mean_speed, "k-", linewidth=1.5, label="Mean speed")

    sph.apply_waypoint_ticks(ax, waypoint_progress, waypoint_names)

    ax.set_xlabel(r"$\mathrm{Progress}\,[-]$")
    ax.set_ylabel(r"$\mathrm{Speed}\,[\mathrm{kn}]$")
    ax.set_title(r"$\mathrm{Speed}$ with min/mean/max envelope")
    ax.legend(loc="best")

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, export_path)
    if export_and_close:
        plt.close(fig)


if __name__ == "__main__":
    main()


