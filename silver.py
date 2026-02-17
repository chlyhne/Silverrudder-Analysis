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
import concurrent.futures
import multiprocessing as mp
import os
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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
# Distribution bounds policy used for both y-scaling and violin clipping.
DELTA_RANGE_LOWER_PERCENTILE = 0.0
DELTA_RANGE_UPPER_PERCENTILE = 100.0
# Hard upper cap for pace delta plots [min/NM].
PACE_DELTA_UPPER_CAP = 10.0


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
        pace_units_per_inch_factor=0.8,
        speed_units_per_inch_factor=1,
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
        pace_units_per_inch_factor=1.00,
        speed_units_per_inch_factor=1.00,
        boxplot_top_margin_in=0.30,
        boxplot_bottom_margin_in=0.95,
        boxplot_left_margin_in=0.60,
        boxplot_right_margin_in=0.15,
    ),
}
ACTIVE_FIGURE_PROFILE = FIGURE_PROFILES["desktop"]
PARALLEL_RENDER_CONTEXT = None


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


def resolve_plot_worker_count(default_max_workers=4):
    """
    Resolve the process count for parallel plot rendering.

    Priority:
    1) SILVER_PLOT_WORKERS env var (if valid integer >= 1)
    2) default min(default_max_workers, CPU count)
    """
    cpu_count = max(1, int(os.cpu_count() or 1))
    default_workers = max(1, min(default_max_workers, cpu_count))
    env_value = os.getenv("SILVER_PLOT_WORKERS", "").strip()
    if not env_value:
        return default_workers
    try:
        parsed_value = int(env_value)
    except ValueError:
        return default_workers
    return max(1, parsed_value)


def initialize_parallel_render_context(context):
    """
    Store immutable render context for process workers.
    """
    global PARALLEL_RENDER_CONTEXT
    PARALLEL_RENDER_CONTEXT = context


def render_plot_family_task(profile_name, family_name, export_dir_path, export_and_close):
    """
    Render one plot family for one figure profile.

    This function is safe to execute in worker processes.
    """
    context = PARALLEL_RENDER_CONTEXT
    if context is None:
        raise RuntimeError("Parallel render context has not been initialized.")

    profile = FIGURE_PROFILES[profile_name]
    set_active_figure_profile(profile)

    export_dir = None if export_dir_path is None else Path(export_dir_path)
    tracks = context["tracks"]
    average_route = context["average_route"]
    speed_window_stats = context["speed_window_stats"]
    way_point_progress = context["way_point_progress"]
    way_point_names = context["way_point_names"]
    start_gate_pos = context.get("start_gate_pos")
    pace_box_plot_data = context["pace_box_plot_data"]
    speed_box_plot_data = context["speed_box_plot_data"]
    comparison_pairs = context["comparison_pairs"]

    with matplotlib.rc_context(profile.rc_params):
        if family_name == "map":
            plot_colored_tracks(
                tracks,
                average_route,
                speed_window_stats,
                way_point_progress,
                way_point_names,
                export_path=(export_dir / "map.pdf") if export_and_close and export_dir is not None else None,
                export_and_close=export_and_close,
            )
            return

        if family_name == "pace_leg":
            plot_pace_delta_box_plot(
                tracks,
                pace_box_plot_data,
                export_dir=export_dir,
                export_and_close=export_and_close,
            )
            return

        if family_name == "pace_boat":
            plot_pace_delta_box_plot_by_boat(
                tracks,
                pace_box_plot_data,
                export_dir=export_dir,
                export_and_close=export_and_close,
            )
            return

        if family_name == "pace_pair":
            plot_pace_delta_split_violin_by_pair(
                tracks,
                pace_box_plot_data,
                comparison_pairs,
                export_dir=export_dir,
                export_and_close=export_and_close,
            )
            return

        if family_name == "speed_leg":
            plot_speed_delta_box_plot(
                tracks,
                speed_box_plot_data,
                export_dir=export_dir,
                export_and_close=export_and_close,
            )
            return

        if family_name == "speed_boat":
            plot_speed_delta_box_plot_by_boat(
                tracks,
                speed_box_plot_data,
                export_dir=export_dir,
                export_and_close=export_and_close,
            )
            return

        if family_name == "speed_pair":
            plot_speed_delta_split_violin_by_pair(
                tracks,
                speed_box_plot_data,
                comparison_pairs,
                export_dir=export_dir,
                export_and_close=export_and_close,
            )
            return

        if family_name == "pace_range":
            plot_speed_range_along_route(
                tracks,
                speed_window_stats,
                way_point_progress,
                way_point_names,
                export_path=(
                    export_dir / "pace-range-along-route.pdf"
                    if export_and_close and export_dir is not None
                    else None
                ),
                export_and_close=export_and_close,
            )
            return

        if family_name == "time_delta":
            plot_time_delta_along_route(
                tracks,
                speed_window_stats,
                way_point_progress,
                way_point_names,
                start_gate_pos=start_gate_pos,
                export_path=(
                    export_dir / "time-delta-along-route.pdf"
                    if export_and_close and export_dir is not None
                    else None
                ),
                export_and_close=export_and_close,
            )
            return

    raise ValueError(f"Unknown render family: {family_name}")


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
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)


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


def draw_manual_violin_distribution(
    ax,
    x_position,
    samples,
    line_color,
    style=None,
    clip_lower=None,
    clip_upper=None,
):
    """
    Draw one distribution as a violin with median line and mean dot.

    For a single-value sample, fall back to a point marker because density is undefined.
    """
    style = style or BOX_PLOT_STYLE_COMMON
    clean_samples = prepare_violin_samples(samples, clip_lower=clip_lower, clip_upper=clip_upper)
    if clean_samples.size == 0:
        return

    mean_value = float(np.mean(clean_samples))

    if clean_samples.size == 1:
        ax.plot(
            x_position,
            mean_value,
            marker=style.mean_marker,
            markersize=style.mean_marker_size,
            color=line_color,
            linestyle="None",
        )
        return

    violin = ax.violinplot(
        [clean_samples],
        positions=[x_position],
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    violin_body = violin["bodies"][0]
    violin_body.set_facecolor(line_color)
    violin_body.set_edgecolor(line_color)
    violin_body.set_alpha(style.box_alpha)
    violin_body.set_linewidth(style.whisker_linewidth)

    median_value = float(np.median(clean_samples))
    median_half_width = style.box_half_width
    ax.plot(
        [x_position - median_half_width, x_position + median_half_width],
        [median_value, median_value],
        color=line_color,
        linewidth=style.median_linewidth,
    )
    ax.plot(
        x_position,
        mean_value,
        marker=style.mean_marker,
        markersize=style.mean_marker_size,
        color=line_color,
        linestyle="None",
    )


def prepare_violin_samples(samples, clip_lower=None, clip_upper=None):
    """
    Clean and bound samples before violin rendering.
    """
    clean_samples = np.asarray(samples, dtype=float)
    clean_samples = clean_samples[np.isfinite(clean_samples)]
    if clean_samples.size == 0:
        return clean_samples

    percentile_bounds = compute_percentile_bounds(clean_samples)
    if percentile_bounds is not None:
        clean_samples = np.clip(clean_samples, percentile_bounds[0], percentile_bounds[1])
    if clip_lower is not None or clip_upper is not None:
        lower_bound = -np.inf if clip_lower is None else float(clip_lower)
        upper_bound = np.inf if clip_upper is None else float(clip_upper)
        clean_samples = np.clip(clean_samples, lower_bound, upper_bound)
    return clean_samples


def draw_split_violin_distribution(
    ax,
    x_position,
    left_samples,
    right_samples,
    left_color,
    right_color,
    style=None,
    clip_lower=None,
    clip_upper=None,
):
    """
    Draw one split violin (left/right halves) for comparing two datasets.
    """
    style = style or BOX_PLOT_STYLE_COMMON
    left_clean = prepare_violin_samples(left_samples, clip_lower=clip_lower, clip_upper=clip_upper)
    right_clean = prepare_violin_samples(right_samples, clip_lower=clip_lower, clip_upper=clip_upper)
    if left_clean.size == 0 and right_clean.size == 0:
        return

    violin_width = 0.8
    half_width = 0.5 * violin_width
    mean_x_offset = 0.5 * half_width

    if left_clean.size > 1:
        left_parts = ax.violinplot(
            [left_clean],
            positions=[x_position],
            widths=violin_width,
            side="low",
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        left_body = left_parts["bodies"][0]
        left_body.set_facecolor(left_color)
        left_body.set_edgecolor(left_color)
        left_body.set_alpha(style.box_alpha)
        left_body.set_linewidth(style.whisker_linewidth)

    if right_clean.size > 1:
        right_parts = ax.violinplot(
            [right_clean],
            positions=[x_position],
            widths=violin_width,
            side="high",
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        right_body = right_parts["bodies"][0]
        right_body.set_facecolor(right_color)
        right_body.set_edgecolor(right_color)
        right_body.set_alpha(style.box_alpha)
        right_body.set_linewidth(style.whisker_linewidth)

    if left_clean.size:
        left_median = float(np.median(left_clean))
        left_mean = float(np.mean(left_clean))
        ax.plot(
            [x_position - half_width, x_position],
            [left_median, left_median],
            color=left_color,
            linewidth=style.median_linewidth,
        )
        ax.plot(
            x_position - mean_x_offset,
            left_mean,
            marker=style.mean_marker,
            markersize=style.mean_marker_size,
            color=left_color,
            linestyle="None",
        )

    if right_clean.size:
        right_median = float(np.median(right_clean))
        right_mean = float(np.mean(right_clean))
        ax.plot(
            [x_position, x_position + half_width],
            [right_median, right_median],
            color=right_color,
            linewidth=style.median_linewidth,
        )
        ax.plot(
            x_position + mean_x_offset,
            right_mean,
            marker=style.mean_marker,
            markersize=style.mean_marker_size,
            color=right_color,
            linestyle="None",
        )


def main():
    figure_generation_lock = None
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
        figure_output_root.parent.mkdir(parents=True, exist_ok=True)
        figure_generation_lock = (
            figure_output_root.parent / f".figures-generating.{os.getpid()}.lock"
        )
        figure_generation_lock.write_text(
            f"Figure generation in progress (pid={os.getpid()}).\n"
            "This file is created by silver.py and removed when rendering finishes.\n",
            encoding="utf-8",
        )

    # Toggle matrix for each output group.
    # Keeping toggles centralized makes experimentation easier without touching logic.
    show_map_plot = False
    show_pace_box_plots = True
    show_pace_box_plots_by_boat = True
    show_pace_pair_plots = True
    show_speed_pair_plots = True
    comparison_pairs = build_all_ordered_comparison_pairs(tracks)
    show_speed_box_plots = True
    show_speed_box_plots_by_boat = True
    show_pace_range_plot = False
    show_time_delta_plot = True

    # -----------------------------------------------------------------------
    # Stage 5: Shared data prep for speed/pace variants
    # -----------------------------------------------------------------------
    # Precompute pace inputs once and reuse for both "by leg" + "by boat" pace plots.
    pace_box_plot_data = None
    if show_pace_box_plots or show_pace_box_plots_by_boat or show_pace_pair_plots:
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
    if show_speed_box_plots or show_speed_box_plots_by_boat or show_speed_pair_plots:
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
    try:
        figure_profiles = [FIGURE_PROFILES["desktop"], FIGURE_PROFILES["phone"]]
        render_tasks = []
        for figure_profile in figure_profiles:
            # Profile subfolder keeps outputs isolated:
            # documentation/figures/desktop/*
            # documentation/figures/phone/*
            export_dir = (
                figure_output_root / figure_profile.output_subdir
                if export_and_close_figures
                else None
            )
            export_dir_path = None if export_dir is None else str(export_dir)
            if show_map_plot:
                render_tasks.append(
                    (figure_profile.name, "map", export_dir_path, export_and_close_figures)
                )
            if show_pace_box_plots:
                render_tasks.append(
                    (figure_profile.name, "pace_leg", export_dir_path, export_and_close_figures)
                )
            if show_pace_box_plots_by_boat:
                render_tasks.append(
                    (figure_profile.name, "pace_boat", export_dir_path, export_and_close_figures)
                )
            if show_pace_pair_plots:
                render_tasks.append(
                    (figure_profile.name, "pace_pair", export_dir_path, export_and_close_figures)
                )
            if show_speed_box_plots:
                render_tasks.append(
                    (figure_profile.name, "speed_leg", export_dir_path, export_and_close_figures)
                )
            if show_speed_box_plots_by_boat:
                render_tasks.append(
                    (figure_profile.name, "speed_boat", export_dir_path, export_and_close_figures)
                )
            if show_speed_pair_plots:
                render_tasks.append(
                    (figure_profile.name, "speed_pair", export_dir_path, export_and_close_figures)
                )
            if show_pace_range_plot:
                render_tasks.append(
                    (figure_profile.name, "pace_range", export_dir_path, export_and_close_figures)
                )
            if show_time_delta_plot:
                render_tasks.append(
                    (figure_profile.name, "time_delta", export_dir_path, export_and_close_figures)
                )

        render_context = {
            "tracks": tracks,
            "average_route": average_route,
            "speed_window_stats": speed_window_stats,
            "way_point_progress": way_point_progress,
            "way_point_names": way_point_names,
            "start_gate_pos": start_gate_pos,
            "pace_box_plot_data": pace_box_plot_data,
            "speed_box_plot_data": speed_box_plot_data,
            "comparison_pairs": comparison_pairs,
        }

        render_worker_count = resolve_plot_worker_count()
        use_parallel_rendering = (
            export_and_close_figures
            and render_worker_count > 1
            and len(render_tasks) > 1
        )
        if use_parallel_rendering:
            print(
                f"Rendering {len(render_tasks)} plot families with {render_worker_count} workers."
            )
            multiprocessing_context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=render_worker_count,
                mp_context=multiprocessing_context,
                initializer=initialize_parallel_render_context,
                initargs=(render_context,),
            ) as executor:
                futures = [
                    executor.submit(render_plot_family_task, *task) for task in render_tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            initialize_parallel_render_context(render_context)
            for task in render_tasks:
                render_plot_family_task(*task)
    finally:
        if figure_generation_lock is not None and figure_generation_lock.exists():
            figure_generation_lock.unlink()

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


def compute_route_distance_m(average_route):
    """
    Compute total route length (meters) from rhumb-line geometry.
    """
    route_distance_m = sh.cumulative_distance_meters(
        np.asarray(average_route["lat"], dtype=float),
        np.asarray(average_route["lon"], dtype=float),
    )[-1]
    if not np.isfinite(route_distance_m) or route_distance_m <= 0:
        return np.nan
    return float(route_distance_m)


def compute_gate_leg_metric_by_track(
    tracks,
    start_time_utc,
    first_leg_gate_pos,
    progress_values,
    average_route,
    metric_name,
):
    """
    Compute raw per-track per-leg metric from gate-crossing times.

    metric_name:
    - "speed": leg_length / leg_time  [kn]
    - "pace": leg_time / leg_length   [min/NM]
    """
    leg_count = len(progress_values) - 1
    metric_by_track = np.full((len(tracks), max(0, leg_count)), np.nan, dtype=float)
    if leg_count < 1:
        return metric_by_track
    if not np.isfinite(start_time_utc) or first_leg_gate_pos < 0:
        return metric_by_track
    if metric_name not in {"speed", "pace"}:
        raise ValueError(f"Unsupported metric_name '{metric_name}'.")

    route_distance_m = compute_route_distance_m(average_route)
    if not np.isfinite(route_distance_m):
        return metric_by_track
    leg_distance_nm = np.abs(np.diff(np.asarray(progress_values, dtype=float))) * route_distance_m / 1852.0

    for track_index, track in enumerate(tracks):
        gate_times = np.asarray(track.get("gateTimes", np.array([])), dtype=float)
        for leg_index in range(leg_count):
            distance_nm = leg_distance_nm[leg_index]
            if not np.isfinite(distance_nm) or distance_nm <= 0:
                continue

            if leg_index == 0:
                if gate_times.size <= first_leg_gate_pos:
                    continue
                leg_start_time = start_time_utc
                leg_end_time = gate_times[first_leg_gate_pos]
            else:
                gate_start_index = first_leg_gate_pos + leg_index - 1
                gate_end_index = first_leg_gate_pos + leg_index
                if gate_times.size <= gate_end_index:
                    continue
                leg_start_time = gate_times[gate_start_index]
                leg_end_time = gate_times[gate_end_index]

            if not np.isfinite(leg_start_time) or not np.isfinite(leg_end_time):
                continue
            if leg_end_time <= leg_start_time:
                continue

            leg_time_hours = (leg_end_time - leg_start_time) / 3600.0
            if leg_time_hours <= 0:
                continue

            if metric_name == "speed":
                metric_by_track[track_index, leg_index] = distance_nm / leg_time_hours
            else:
                leg_time_min = leg_time_hours * 60.0
                metric_by_track[track_index, leg_index] = leg_time_min / distance_nm

    return metric_by_track


def compute_leg_delta_by_track(raw_leg_metric_by_track):
    """
    Center raw per-leg metric values on fleet mean per leg.
    """
    delta_by_track = np.full_like(raw_leg_metric_by_track, np.nan, dtype=float)
    if raw_leg_metric_by_track.ndim != 2:
        return delta_by_track

    for leg_index in range(raw_leg_metric_by_track.shape[1]):
        leg_values = raw_leg_metric_by_track[:, leg_index]
        finite_values = leg_values[np.isfinite(leg_values)]
        if finite_values.size == 0:
            continue
        fleet_mean = float(np.mean(finite_values))
        delta_by_track[:, leg_index] = leg_values - fleet_mean

    return delta_by_track


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


def build_leg_ordered(
    track_progress_by_track,
    delta_by_track,
    progress_values,
    leg_index,
    first_leg_delta_by_track,
    clip_lower=None,
    clip_upper=None,
):
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
        display_samples = prepare_violin_samples(
            leg_samples, clip_lower=clip_lower, clip_upper=clip_upper
        )
        if display_samples.size == 0:
            continue
        leg_mean = float(np.mean(display_samples))
        if clip_lower is not None or clip_upper is not None:
            lower_bound = -np.inf if clip_lower is None else float(clip_lower)
            upper_bound = np.inf if clip_upper is None else float(clip_upper)
            leg_mean = float(np.clip(leg_mean, lower_bound, upper_bound))
        leg_ordered.append((track_index, leg_mean, leg_samples))
    return leg_ordered


def compute_percentile_bounds(samples, lower_percentile=DELTA_RANGE_LOWER_PERCENTILE, upper_percentile=DELTA_RANGE_UPPER_PERCENTILE):
    """
    Compute inclusive percentile bounds for one sample vector.

    Returns None when samples contain no finite values.
    """
    clean_samples = np.asarray(samples, dtype=float)
    clean_samples = clean_samples[np.isfinite(clean_samples)]
    if clean_samples.size == 0:
        return None
    lower_value = float(np.percentile(clean_samples, lower_percentile))
    upper_value = float(np.percentile(clean_samples, upper_percentile))
    if not np.isfinite(lower_value) or not np.isfinite(upper_value):
        return None
    if lower_value > upper_value:
        lower_value, upper_value = upper_value, lower_value
    return lower_value, upper_value


def update_percentile_range(current_lower, current_upper, samples):
    """
    Expand a (lower, upper) range using configured percentile bounds.
    """
    bounds = compute_percentile_bounds(samples)
    if bounds is None:
        return current_lower, current_upper
    return min(current_lower, bounds[0]), max(current_upper, bounds[1])


def clamp_range_bounds(lower_value, upper_value, clip_lower=None, clip_upper=None):
    """
    Clamp range endpoints to optional hard bounds and keep ordering valid.
    """
    lower = float(lower_value)
    upper = float(upper_value)
    if clip_lower is not None or clip_upper is not None:
        lower_bound = -np.inf if clip_lower is None else float(clip_lower)
        upper_bound = np.inf if clip_upper is None else float(clip_upper)
        lower = float(np.clip(lower, lower_bound, upper_bound))
        upper = float(np.clip(upper, lower_bound, upper_bound))
    if upper < lower:
        upper = lower
    return lower, upper


def compute_track_robust_delta_range(progress, delta_values, progress_values, first_leg_value):
    """
    Compute a single track's robust min/max delta range across all legs.

    "Robust" here means percentile bounds (1st/99th) per leg, aggregated
    across legs for the track.
    """
    local_lower = np.inf
    local_upper = -np.inf
    # Aggregate percentile bounds across all legs to keep per-boat y-scales consistent.
    for leg_index in range(len(progress_values) - 1):
        leg_samples = get_leg_samples_for_track(
            progress, delta_values, progress_values, leg_index, first_leg_value
        )
        if leg_samples is None:
            continue
        local_lower, local_upper = update_percentile_range(local_lower, local_upper, leg_samples)
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

    Uses the same percentile-bound method as
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
        local_lower, local_upper = update_percentile_range(local_lower, local_upper, leg_samples)
    if not np.isfinite(local_lower) or not np.isfinite(local_upper):
        return None
    return local_lower, local_upper


def compute_metric_scale_context(
    track_progress_by_track,
    delta_by_track,
    progress_values,
    first_leg_delta_by_track,
    units_per_inch_factor=1.0,
    clip_lower=None,
    clip_upper=None,
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
        clipped_lower, clipped_upper = clamp_range_bounds(
            leg_range[0], leg_range[1], clip_lower=clip_lower, clip_upper=clip_upper
        )
        global_lower = min(global_lower, clipped_lower)
        global_upper = max(global_upper, clipped_upper)

    global_range_result = box_plots.compute_global_plot_range(global_lower, global_upper)
    if global_range_result is None:
        return None
    global_lower, global_upper, global_range = global_range_result
    units_per_inch = box_plots.compute_units_per_inch(global_range)
    units_per_inch *= units_per_inch_factor
    return global_lower, global_upper, global_range, units_per_inch


def expand_y_limits_by_subticks(local_lower, local_upper, axis_style, subticks_each_side=2):
    """
    Expand y-limits by a fixed number of minor y-tick intervals on each side.

    This adds consistent visual headroom independent of data spread.
    """
    if not np.isfinite(local_lower) or not np.isfinite(local_upper):
        return local_lower, local_upper, local_upper - local_lower

    major_tick = float(getattr(axis_style, "major_tick", 0.0))
    minor_ticks_per_major = int(getattr(axis_style, "minor_ticks_per_major", 0))
    if major_tick <= 0:
        return local_lower, local_upper, local_upper - local_lower

    # Minor ticks are placed at major_tick / (minor_ticks_per_major + 1).
    minor_step = major_tick / max(1, minor_ticks_per_major + 1)
    padding = float(subticks_each_side) * minor_step
    local_lower -= padding
    local_upper += padding
    return local_lower, local_upper, local_upper - local_lower


def format_minutes_as_hhmm(value, _tick_pos=None):
    """Format minute values as signed h:mm."""
    if not np.isfinite(value):
        return ""
    sign = "-" if value < 0 else ""
    total_minutes = int(np.round(abs(value)))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{sign}{hours}:{minutes:02d}"


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

    # First-leg scalar comes from gate-time leg speed.
    raw_leg_speed_by_track = compute_gate_leg_metric_by_track(
        tracks,
        start_time_utc,
        first_leg_gate_pos,
        progress_values,
        average_route,
        metric_name="speed",
    )
    first_leg_delta_by_track = compute_leg_delta_by_track(raw_leg_speed_by_track)[:, 0]
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
        local_lower, local_upper, local_range = expand_y_limits_by_subticks(
            local_lower, local_upper, BOX_PLOT_AXIS_STYLE_SPEED, subticks_each_side=2
        )

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        apply_boxplot_physical_layout(fig, local_range, units_per_inch, extra_bottom_in=0)
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
            draw_manual_violin_distribution(
                ax,
                plot_index,
                speed_leg,
                line_color,
                style=BOX_PLOT_STYLE_SPEED,
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
        local_lower, local_upper, local_range = expand_y_limits_by_subticks(
            local_lower, local_upper, BOX_PLOT_AXIS_STYLE_SPEED, subticks_each_side=2
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
            draw_manual_violin_distribution(
                ax,
                leg_index + 1,
                speed_leg,
                line_color,
                style=BOX_PLOT_STYLE_SPEED,
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

    # First-leg scalar comes from gate-time leg pace.
    raw_leg_pace_by_track = compute_gate_leg_metric_by_track(
        tracks,
        start_time_utc,
        first_leg_gate_pos,
        progress_values,
        average_route,
        metric_name="pace",
    )
    first_leg_delta_by_track = compute_leg_delta_by_track(raw_leg_pace_by_track)[:, 0]
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
        clip_upper=PACE_DELTA_UPPER_CAP,
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
        leg_lower, leg_upper = clamp_range_bounds(
            leg_range[0], leg_range[1], clip_upper=PACE_DELTA_UPPER_CAP
        )
        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            leg_lower, leg_upper, global_range
        )
        local_lower, local_upper, local_range = expand_y_limits_by_subticks(
            local_lower, local_upper, BOX_PLOT_AXIS_STYLE_PACE, subticks_each_side=2
        )
        local_lower, local_upper = clamp_range_bounds(
            local_lower, local_upper, clip_upper=PACE_DELTA_UPPER_CAP
        )
        local_range = local_upper - local_lower
        if local_range <= 0:
            local_range = 1.0
            local_lower = local_upper - local_range

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        apply_boxplot_physical_layout(fig, local_range, units_per_inch, extra_bottom_in=0.0)
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_PACE)
        ax.set_ylim(local_lower, local_upper)

        # Order boats by mean pace delta to emphasize relative ranking.
        leg_ordered = build_leg_ordered(
            track_progress_by_track,
            pace_delta_by_track,
            progress_values,
            leg_index,
            first_leg_delta_by_track,
            clip_upper=PACE_DELTA_UPPER_CAP,
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
            draw_manual_violin_distribution(
                ax,
                plot_index,
                pace_leg,
                line_color,
                style=BOX_PLOT_STYLE_PACE,
                clip_upper=PACE_DELTA_UPPER_CAP,
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
        clip_upper=PACE_DELTA_UPPER_CAP,
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
        leg_lower, leg_upper = clamp_range_bounds(
            leg_range[0], leg_range[1], clip_upper=PACE_DELTA_UPPER_CAP
        )
        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            leg_lower, leg_upper, global_range
        )
        local_lower, local_upper, local_range = expand_y_limits_by_subticks(
            local_lower, local_upper, BOX_PLOT_AXIS_STYLE_PACE, subticks_each_side=2
        )
        local_lower, local_upper = clamp_range_bounds(
            local_lower, local_upper, clip_upper=PACE_DELTA_UPPER_CAP
        )
        local_range = local_upper - local_lower
        if local_range <= 0:
            local_range = 1.0
            local_lower = local_upper - local_range

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
            draw_manual_violin_distribution(
                ax,
                leg_index + 1,
                pace_leg,
                line_color,
                style=BOX_PLOT_STYLE_PACE,
                clip_upper=PACE_DELTA_UPPER_CAP,
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


def find_track_index_by_name(tracks, target_name):
    """
    Resolve a boat index by case-insensitive display name.
    """
    target_normalized = str(target_name).strip().lower()
    for track_index, track in enumerate(tracks):
        track_name = str(track.get("name", "")).strip().lower()
        if track_name == target_normalized:
            return track_index
    return None


def build_all_ordered_comparison_pairs(tracks):
    """
    Build all ordered boat pairs (A vs B) with A != B.

    Order matters for split violins:
    - left half: first boat
    - right half: second boat
    """
    track_names = []
    for track in tracks:
        track_name = str(track.get("name", "")).strip()
        if not track_name:
            continue
        track_names.append(track_name)

    ordered_pairs = []
    for left_name in track_names:
        for right_name in track_names:
            if left_name == right_name:
                continue
            ordered_pairs.append((left_name, right_name))
    return ordered_pairs


def plot_speed_delta_split_violin_by_pair(
    tracks,
    speed_plot_data: Optional[SpeedDeltaBoxPlotData],
    pair_names: Optional[List[tuple[str, str]]] = None,
    export_dir=None,
    export_and_close=False,
):
    """
    Export one speed-only split-violin figure per selected boat pair.

    Each figure compares two boats across all legs:
    - left half: first boat in pair
    - right half: second boat in pair
    """
    if speed_plot_data is None:
        return
    if not pair_names:
        return

    track_progress_by_track = speed_plot_data.track_progress_by_track
    speed_delta_by_track = speed_plot_data.speed_delta_by_track
    progress_values = speed_plot_data.progress_values
    leg_labels = speed_plot_data.leg_labels
    first_leg_delta_by_track = speed_plot_data.first_leg_delta_by_track
    leg_count = len(progress_values) - 1
    if leg_count < 1:
        return

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

    for left_name, right_name in pair_names:
        left_index = find_track_index_by_name(tracks, left_name)
        right_index = find_track_index_by_name(tracks, right_name)
        if left_index is None or right_index is None:
            continue

        pair_lower = np.inf
        pair_upper = -np.inf
        for track_index in (left_index, right_index):
            track_range = compute_track_robust_delta_range(
                track_progress_by_track[track_index],
                speed_delta_by_track[track_index],
                progress_values,
                first_leg_delta_by_track[track_index],
            )
            if track_range is None:
                continue
            pair_lower = min(pair_lower, track_range[0])
            pair_upper = max(pair_upper, track_range[1])
        if not np.isfinite(pair_lower) or not np.isfinite(pair_upper):
            continue

        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            pair_lower, pair_upper, global_range
        )
        local_lower, local_upper, local_range = expand_y_limits_by_subticks(
            local_lower, local_upper, BOX_PLOT_AXIS_STYLE_SPEED, subticks_each_side=2
        )
        if local_range <= 0:
            local_range = 1.0
            local_lower = local_upper - local_range

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        pair_tick_extra_bottom_in = 0.40
        pair_tick_edge_padding = 0.70
        if ACTIVE_FIGURE_PROFILE.name == "phone":
            pair_tick_extra_bottom_in = 0.75
            pair_tick_edge_padding = 0.85
        apply_boxplot_physical_layout(
            fig,
            local_range,
            units_per_inch,
            extra_bottom_in=pair_tick_extra_bottom_in,
        )
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_SPEED)
        ax.set_ylim(local_lower, local_upper)

        left_color = tracks[left_index].get("color") or (0.3, 0.3, 0.3)
        right_color = tracks[right_index].get("color") or (0.3, 0.3, 0.3)
        left_progress = track_progress_by_track[left_index]
        left_delta = speed_delta_by_track[left_index]
        right_progress = track_progress_by_track[right_index]
        right_delta = speed_delta_by_track[right_index]
        left_first_leg = first_leg_delta_by_track[left_index]
        right_first_leg = first_leg_delta_by_track[right_index]

        for leg_index in range(leg_count):
            left_leg = get_leg_samples_for_track(
                left_progress,
                left_delta,
                progress_values,
                leg_index,
                left_first_leg,
            )
            right_leg = get_leg_samples_for_track(
                right_progress,
                right_delta,
                progress_values,
                leg_index,
                right_first_leg,
            )
            if left_leg is None and right_leg is None:
                continue

            draw_split_violin_distribution(
                ax,
                leg_index + 1,
                [] if left_leg is None else left_leg,
                [] if right_leg is None else right_leg,
                left_color,
                right_color,
                style=BOX_PLOT_STYLE_SPEED,
            )

        ax.set_xticks(range(1, leg_count + 1))
        ax.set_xticklabels(
            leg_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=ACTIVE_FIGURE_PROFILE.waypoint_label_fontsize,
        )
        ax.set_xlim(1.0 - pair_tick_edge_padding, leg_count + pair_tick_edge_padding)
        ax.set_ylabel(r"$\Delta \mathrm{Speed}\,[\mathrm{kn}]$")
        ax.set_ylim(local_lower, local_upper)
        left_title = latex_display_name(tracks[left_index].get("name", left_name))
        right_title = latex_display_name(tracks[right_index].get("name", right_name))
        ax.set_title(f"{left_title} vs {right_title}")
        legend_handles = [
            mpatches.Patch(
                facecolor=left_color,
                edgecolor=left_color,
                alpha=BOX_PLOT_STYLE_SPEED.box_alpha,
                label=left_title,
            ),
            mpatches.Patch(
                facecolor=right_color,
                edgecolor=right_color,
                alpha=BOX_PLOT_STYLE_SPEED.box_alpha,
                label=right_title,
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_left = sh.sanitize_filename_label(tracks[left_index].get("name", left_name))
            safe_right = sh.sanitize_filename_label(tracks[right_index].get("name", right_name))
            output_path = export_dir / f"speed-delta-pair-{safe_left}-vs-{safe_right}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


def plot_pace_delta_split_violin_by_pair(
    tracks,
    pace_plot_data: Optional[PaceDeltaBoxPlotData],
    pair_names: Optional[List[tuple[str, str]]] = None,
    export_dir=None,
    export_and_close=False,
):
    """
    Export one pace-only split-violin figure per selected boat pair.

    Each figure compares two boats across all legs:
    - left half: first boat in pair
    - right half: second boat in pair
    """
    if pace_plot_data is None:
        return
    if not pair_names:
        return

    track_progress_by_track = pace_plot_data.track_progress_by_track
    pace_delta_by_track = pace_plot_data.pace_delta_by_track
    progress_values = pace_plot_data.progress_values
    leg_labels = pace_plot_data.leg_labels
    first_leg_delta_by_track = pace_plot_data.first_leg_delta_by_track
    leg_count = len(progress_values) - 1
    if leg_count < 1:
        return

    scale_context = compute_metric_scale_context(
        track_progress_by_track,
        pace_delta_by_track,
        progress_values,
        first_leg_delta_by_track,
        units_per_inch_factor=ACTIVE_FIGURE_PROFILE.pace_units_per_inch_factor,
        clip_upper=PACE_DELTA_UPPER_CAP,
    )
    if scale_context is None:
        return
    _, _, global_range, units_per_inch = scale_context

    for left_name, right_name in pair_names:
        left_index = find_track_index_by_name(tracks, left_name)
        right_index = find_track_index_by_name(tracks, right_name)
        if left_index is None or right_index is None:
            continue

        pair_lower = np.inf
        pair_upper = -np.inf
        for track_index in (left_index, right_index):
            track_range = compute_track_robust_delta_range(
                track_progress_by_track[track_index],
                pace_delta_by_track[track_index],
                progress_values,
                first_leg_delta_by_track[track_index],
            )
            if track_range is None:
                continue
            range_lower, range_upper = clamp_range_bounds(
                track_range[0], track_range[1], clip_upper=PACE_DELTA_UPPER_CAP
            )
            pair_lower = min(pair_lower, range_lower)
            pair_upper = max(pair_upper, range_upper)
        if not np.isfinite(pair_lower) or not np.isfinite(pair_upper):
            continue

        local_lower, local_upper, local_range = box_plots.compute_local_plot_range(
            pair_lower, pair_upper, global_range
        )
        local_lower, local_upper, local_range = expand_y_limits_by_subticks(
            local_lower, local_upper, BOX_PLOT_AXIS_STYLE_PACE, subticks_each_side=2
        )
        local_lower, local_upper = clamp_range_bounds(
            local_lower, local_upper, clip_upper=PACE_DELTA_UPPER_CAP
        )
        local_range = local_upper - local_lower
        if local_range <= 0:
            local_range = 1.0
            local_lower = local_upper - local_range

        fig, ax = plt.subplots()
        prepare_figure(fig, export_and_close=export_and_close)
        pair_tick_extra_bottom_in = 0.40
        pair_tick_edge_padding = 0.70
        if ACTIVE_FIGURE_PROFILE.name == "phone":
            pair_tick_extra_bottom_in = 0.75
            pair_tick_edge_padding = 0.85
        apply_boxplot_physical_layout(
            fig,
            local_range,
            units_per_inch,
            extra_bottom_in=pair_tick_extra_bottom_in,
        )
        box_plots.apply_boxplot_axis_style(ax, BOX_PLOT_AXIS_STYLE_PACE)
        ax.set_ylim(local_lower, local_upper)

        left_color = tracks[left_index].get("color") or (0.3, 0.3, 0.3)
        right_color = tracks[right_index].get("color") or (0.3, 0.3, 0.3)
        left_progress = track_progress_by_track[left_index]
        left_delta = pace_delta_by_track[left_index]
        right_progress = track_progress_by_track[right_index]
        right_delta = pace_delta_by_track[right_index]
        left_first_leg = first_leg_delta_by_track[left_index]
        right_first_leg = first_leg_delta_by_track[right_index]

        for leg_index in range(leg_count):
            left_leg = get_leg_samples_for_track(
                left_progress,
                left_delta,
                progress_values,
                leg_index,
                left_first_leg,
            )
            right_leg = get_leg_samples_for_track(
                right_progress,
                right_delta,
                progress_values,
                leg_index,
                right_first_leg,
            )
            if left_leg is None and right_leg is None:
                continue

            draw_split_violin_distribution(
                ax,
                leg_index + 1,
                [] if left_leg is None else left_leg,
                [] if right_leg is None else right_leg,
                left_color,
                right_color,
                style=BOX_PLOT_STYLE_PACE,
                clip_upper=PACE_DELTA_UPPER_CAP,
            )

        ax.set_xticks(range(1, leg_count + 1))
        ax.set_xticklabels(
            leg_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=ACTIVE_FIGURE_PROFILE.waypoint_label_fontsize,
        )
        ax.set_xlim(1.0 - pair_tick_edge_padding, leg_count + pair_tick_edge_padding)
        ax.set_ylabel(r"$\Delta \mathrm{Pace}\,[\mathrm{min}\,\mathrm{NM}^{-1}]$")
        ax.set_ylim(local_lower, local_upper)
        left_title = latex_display_name(tracks[left_index].get("name", left_name))
        right_title = latex_display_name(tracks[right_index].get("name", right_name))
        ax.set_title(f"{left_title} vs {right_title}")
        legend_handles = [
            mpatches.Patch(
                facecolor=left_color,
                edgecolor=left_color,
                alpha=BOX_PLOT_STYLE_PACE.box_alpha,
                label=left_title,
            ),
            mpatches.Patch(
                facecolor=right_color,
                edgecolor=right_color,
                alpha=BOX_PLOT_STYLE_PACE.box_alpha,
                label=right_title,
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_left = sh.sanitize_filename_label(tracks[left_index].get("name", left_name))
            safe_right = sh.sanitize_filename_label(tracks[right_index].get("name", right_name))
            output_path = export_dir / f"pace-delta-pair-{safe_left}-vs-{safe_right}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------
def compute_time_delta_samples_along_route(
    tracks,
    route_sample_count,
    start_gate_pos=None,
    progress_sample_count=1200,
):
    """
    Build cumulative time-delta samples versus progress for all boats.

    For each boat:
    1) Convert route index -> progress in [0, 1]
    2) Build elapsed time [min] from Start gate crossing when available
    3) Interpolate elapsed time onto a shared progress grid

    The returned delta is:
        boat elapsed time - fleet mean elapsed time
    at the same progress.
    """
    if route_sample_count < 2:
        return None, None

    progress_grid = np.linspace(0.0, 1.0, int(max(2, progress_sample_count)))
    elapsed_on_grid_by_track = []

    for track in tracks:
        route_index = np.asarray(track.get("routeIdx", []), dtype=float)
        time_values = np.asarray(track.get("t", []), dtype=float)
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(time_values)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
        )
        if np.count_nonzero(valid_mask) < 2:
            elapsed_on_grid_by_track.append(np.full(progress_grid.size, np.nan, dtype=float))
            continue

        progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
        start_time = time_values[valid_mask][0]
        gate_times = np.asarray(track.get("gateTimes", []), dtype=float)
        if (
            start_gate_pos is not None
            and gate_times.size > int(start_gate_pos)
            and np.isfinite(gate_times[int(start_gate_pos)])
        ):
            start_time = float(gate_times[int(start_gate_pos)])
        elif gate_times.size:
            finite_gate_times = gate_times[np.isfinite(gate_times)]
            if finite_gate_times.size:
                start_time = float(np.min(finite_gate_times))
        elapsed_minutes = (time_values[valid_mask] - start_time) / 60.0

        sort_index = np.argsort(progress)
        progress = progress[sort_index]
        elapsed_minutes = elapsed_minutes[sort_index]

        progress_unique, unique_index = np.unique(progress, return_index=True)
        elapsed_unique = elapsed_minutes[unique_index]
        if progress_unique.size < 2:
            elapsed_on_grid_by_track.append(np.full(progress_grid.size, np.nan, dtype=float))
            continue

        # Guard against tiny non-monotonic artifacts introduced by de-duplication.
        elapsed_unique = np.maximum.accumulate(elapsed_unique)
        elapsed_on_grid = np.interp(
            progress_grid,
            progress_unique,
            elapsed_unique,
            left=np.nan,
            right=np.nan,
        )
        elapsed_on_grid_by_track.append(elapsed_on_grid)

    if not elapsed_on_grid_by_track:
        return progress_grid, []

    elapsed_matrix = np.vstack(elapsed_on_grid_by_track)
    fleet_mean_elapsed = np.full(progress_grid.size, np.nan, dtype=float)
    finite_counts = np.sum(np.isfinite(elapsed_matrix), axis=0)
    valid_baseline = finite_counts > 0
    if np.any(valid_baseline):
        fleet_mean_elapsed[valid_baseline] = np.nanmean(
            elapsed_matrix[:, valid_baseline], axis=0
        )
    time_delta_matrix = np.full_like(elapsed_matrix, np.nan)
    time_delta_matrix[:, valid_baseline] = (
        elapsed_matrix[:, valid_baseline] - fleet_mean_elapsed[None, valid_baseline]
    )
    time_delta_by_track = [time_delta_matrix[idx] for idx in range(time_delta_matrix.shape[0])]
    return progress_grid, time_delta_by_track


def plot_time_delta_along_route(
    tracks,
    speed_window_stats,
    waypoint_progress=None,
    waypoint_names=None,
    start_gate_pos=None,
    export_path=None,
    export_and_close=False,
):
    """
    Diagnostic plot: cumulative time lost/gained versus progress for each boat.

    Positive values mean a boat is behind the fleet mean at that progress.
    Negative values mean the boat is ahead.
    """
    if not speed_window_stats or speed_window_stats.get("routeSampleCount", 0) < 2:
        return

    route_sample_count = int(speed_window_stats["routeSampleCount"])
    progress_grid, time_delta_by_track = compute_time_delta_samples_along_route(
        tracks, route_sample_count, start_gate_pos=start_gate_pos
    )
    if progress_grid is None or not time_delta_by_track:
        return

    fig, ax = plt.subplots()
    prepare_figure(fig, export_and_close=export_and_close)
    ax.grid(True)

    default_colors = plt.cm.hsv(np.linspace(0, 1, max(len(tracks), 1)))
    has_any_series = False
    for track_index, (track, time_delta) in enumerate(zip(tracks, time_delta_by_track)):
        finite_mask = np.isfinite(time_delta)
        if np.count_nonzero(finite_mask) < 2:
            continue
        has_any_series = True
        line_color = track.get("color") or default_colors[track_index]
        ax.plot(
            progress_grid[finite_mask],
            time_delta[finite_mask],
            color=line_color,
            linewidth=1.2,
            label=latex_display_name(track.get("name", f"Boat {track_index + 1}")),
        )

    if not has_any_series:
        if export_and_close:
            plt.close(fig)
        return

    ax.axhline(0.0, color=(0.2, 0.2, 0.2), linewidth=1.0, linestyle="--")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(60.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_minutes_as_hhmm))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(15.0))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="y", which="minor", labelleft=False)
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="y", alpha=0.35)
    sph.apply_waypoint_ticks(ax, waypoint_progress, waypoint_names)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\mathrm{Progress}\,[-]$")
    ax.set_ylabel(r"$\Delta t\,[\mathrm{min}]$")
    ax.set_title("Time lost/gained vs progress")
    legend_columns = 1 if ACTIVE_FIGURE_PROFILE.name == "phone" else 2
    ax.legend(loc="best", ncol=legend_columns)

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, export_path)
    if export_and_close:
        plt.close(fig)


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
