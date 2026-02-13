import argparse
import json
from dataclasses import dataclass, replace
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}"
matplotlib.rcParams["axes.unicode_minus"] = False


import box_plot_utils as box_plots
import silver_helpers as sh

# Centralize box-plot styling so every figure uses shared defaults with per-metric overrides.
BOX_PLOT_STYLE_COMMON = box_plots.BoxPlotStyle()
BOX_PLOT_AXIS_STYLE_COMMON = box_plots.BoxPlotAxisStyle()
BOX_PLOT_STYLE_SPEED = BOX_PLOT_STYLE_COMMON
BOX_PLOT_AXIS_STYLE_SPEED = BOX_PLOT_AXIS_STYLE_COMMON
BOX_PLOT_STYLE_PACE = BOX_PLOT_STYLE_COMMON
BOX_PLOT_AXIS_STYLE_PACE = replace(BOX_PLOT_AXIS_STYLE_COMMON, major_tick=2.0)
BOX_PLOT_WHISKER_SCALE = 1.5


@dataclass(frozen=True)
class FigureProfile:
    """Profile-specific export settings for figure layout and typography."""

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


def parse_cli_args():
    """Parse command-line options for figure export profiles."""
    parser = argparse.ArgumentParser(
        description="Generate Silverrudder analysis figures."
    )
    parser.add_argument(
        "--figure-profile",
        choices=("desktop", "phone", "both"),
        default="both",
        help="Figure style/export target to generate.",
    )
    parser.add_argument(
        "--figure-root",
        default=str(Path("documentation") / "figures"),
        help="Root output folder for exported figures.",
    )
    return parser.parse_args()


def resolve_figure_profiles(profile_name):
    """Return ordered figure profiles to export."""
    if profile_name == "both":
        return [FIGURE_PROFILES["desktop"], FIGURE_PROFILES["phone"]]
    return [FIGURE_PROFILES[profile_name]]


def set_active_figure_profile(profile):
    """Set active profile used by plot helpers for marker/text sizing."""
    global ACTIVE_FIGURE_PROFILE
    ACTIVE_FIGURE_PROFILE = profile


def prepare_figure(fig, export_and_close=False):
    """Keep interactive maximize behavior without overriding export sizing."""
    if not export_and_close:
        sh.maximize_figure(fig)


def save_figure(fig, output_path):
    """Save figure without external scaling; keep canvas size as the source of truth."""
    fig.savefig(output_path)


def apply_boxplot_physical_layout(fig, local_range, units_per_inch, extra_bottom_in=0.0):
    """Set figure size so y-axis data spacing is consistent in physical units."""
    fig_width = fig.get_size_inches()[0]
    data_height_in = local_range / units_per_inch if units_per_inch > 0 else fig.get_size_inches()[1]
    top_in = ACTIVE_FIGURE_PROFILE.boxplot_top_margin_in
    bottom_in = ACTIVE_FIGURE_PROFILE.boxplot_bottom_margin_in + max(0.0, extra_bottom_in)
    left_in = ACTIVE_FIGURE_PROFILE.boxplot_left_margin_in
    right_in = ACTIVE_FIGURE_PROFILE.boxplot_right_margin_in

    # Figure height is data-height plus fixed non-data margins.
    fig_height = data_height_in + top_in + bottom_in
    fig.set_size_inches(fig_width, fig_height, forward=True)

    left = left_in / fig_width
    right = 1.0 - (right_in / fig_width)
    bottom = bottom_in / fig_height
    top = 1.0 - (top_in / fig_height)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def main():
    args = parse_cli_args()

    # End-to-end pipeline: load metadata and tracks, align samples to a reference route,
    # compute per-window baseline stats, then prepare plot-ready data for the chosen outputs.
    data_root = Path("data") / "silverrudder_2025"
    filename = str(data_root / "Silverrudder 2025_Keelboats Small_gps_data.csv")
    metadata_path = data_root / "race_metadata.json"

    # Metadata provides stable IDs, display names, and optional colors used across plots.
    race_meta = sh.load_race_metadata(metadata_path)
    track_id_keys = race_meta["track_id_keys"]
    boat_names = race_meta["boat_names"]

    # Build per-boat tracks from raw samples; names are attached early for consistent labeling.
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
    # Start time anchors the first-leg pace/speed metrics to an absolute clock.
    start_time_utc = parse_start_time_utc(race_meta)

    # Gate crossings let us trim each track to the common race segment (start -> finish).
    geo_data_path = metadata_path.parent / "waypoints" / "geo_data.json"
    waypoint_gates, start_gate_pos, finish_gate_pos, first_leg_gate_pos = load_waypoint_gates(geo_data_path)
    gate_times_by_track = sh.compute_gate_crossings(tracks, waypoint_gates)
    tracks = sh.trim_tracks_by_gate_times(tracks, gate_times_by_track, start_gate_pos, finish_gate_pos)
    # Colors are mapped by boat name to keep figures visually consistent across plots.
    boat_colors = race_meta.get("boat_colors", {})
    track_color_map = sh.build_boat_color_map(boat_names, boat_colors)
    tracks = sh.apply_track_colors(tracks, track_color_map)

    route_sample_count = 20000
    # Search window limits route-index jumps when mapping samples onto the reference route.
    route_search_window_half_width = int(np.ceil(route_sample_count / 100))
    # The average route provides a 1D progress axis for comparing boats at the same location.
    average_route = sh.compute_average_route(route_sample_count, geo_data_path)
    tracks = sh.map_track_points_to_route(tracks, average_route, route_search_window_half_width)
    tracks = sh.remove_route_index_spikes(tracks, route_sample_count)
    # Waypoint progress is used to define leg boundaries and label x-axes.
    way_point_progress, way_point_names = sh.compute_waypoint_progress_from_gates(average_route, waypoint_gates)

    window_sample_count = 50
    window_step_samples = 1
    filter_alpha = 0.01
    # Per-window stats establish the fleet baseline used for alpha coloring and delta plots.
    tracks, speed_window_stats = sh.compute_sample_alpha_by_route_windows(
        tracks, route_sample_count, window_sample_count, window_step_samples, filter_alpha
    )

    # When enabled, figures are saved to PDF and immediately closed to limit memory use.
    export_and_close_figures = True
    figure_output_root = Path(args.figure_root)
    # Disable interactive display when exporting so figures do not pop up during batch runs.
    if export_and_close_figures:
        plt.ioff()

    # Plot toggles let us selectively generate the heavier figures when needed.
    show_map_plot = False
    show_pace_box_plots = True
    show_pace_box_plots_by_boat = True
    show_speed_box_plots = True
    show_speed_box_plots_by_boat = True
    show_pace_range_plot = False

    # Precompute pace deltas once when either pace plot is enabled.
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

    # Precompute speed deltas once when either speed plot is enabled.
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

    figure_profiles = resolve_figure_profiles(args.figure_profile)
    for figure_profile in figure_profiles:
        set_active_figure_profile(figure_profile)
        export_dir = (
            figure_output_root / figure_profile.output_subdir
            if export_and_close_figures
            else None
        )
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

    # Only display figures interactively when we are not in export-and-close mode.
    if not export_and_close_figures:
        plt.show()


def parse_start_time_utc(race_meta):
    """Parse the race start time from metadata into UTC seconds."""
    # Metadata stores local time plus offset; convert once so downstream metrics stay consistent.
    local_offset_hours = race_meta.get("local_offset_hours", 0)
    start_time_local = race_meta.get("start_time_local", "")
    start_time_utc = sh.parse_local_time_to_utc(start_time_local, local_offset_hours)
    if start_time_utc is None:
        raise ValueError("start_time_local missing or invalid in race_metadata.json")
    return start_time_utc


def load_waypoint_gates(geo_data_path):
    """Load waypoint gates and locate start/finish indices."""
    _, waypoint_gates = sh.load_geo_data(geo_data_path)
    start_gate_pos, finish_gate_pos = sh.find_start_finish_gate_positions(waypoint_gates)
    # The first leg is defined as Start -> next gate, used for scalar delta metrics.
    first_leg_gate_pos = start_gate_pos + 1
    if first_leg_gate_pos >= len(waypoint_gates):
        raise ValueError("No waypoint available after Start for first leg.")
    return waypoint_gates, start_gate_pos, finish_gate_pos, first_leg_gate_pos


def normalize_waypoint_series(waypoint_progress, waypoint_names):
    """Return finite waypoint progress values with a padded label list."""
    progress_values = np.asarray(waypoint_progress, dtype=float).flatten()
    label_values = list(waypoint_names) if waypoint_names is not None else []

    # Keep labels aligned with progress values even when metadata is short/long.
    if label_values and progress_values.size:
        if len(label_values) < progress_values.size:
            label_values += [""] * (progress_values.size - len(label_values))
        if len(label_values) > progress_values.size:
            label_values = label_values[: progress_values.size]

    # Drop NaN progress entries and the matching labels to avoid misaligned ticks.
    finite_mask = np.isfinite(progress_values)
    progress_values = progress_values[finite_mask]
    label_values = [label for label, keep in zip(label_values, finite_mask) if keep]
    return progress_values, label_values


def filter_waypoints_to_unit_interval(progress_values, label_values):
    """Clamp waypoint progress to [0, 1] and keep labels aligned."""
    # Progress outside [0, 1] does not lie on the plotted route, so ignore it.
    in_range_mask = (progress_values >= 0) & (progress_values <= 1)
    return progress_values[in_range_mask], [
        label for label, keep in zip(label_values, in_range_mask) if keep
    ]


def plot_waypoints_on_route(ax, average_route, waypoint_progress, waypoint_names):
    """Draw waypoint markers and labels along the rhumb line."""
    progress_values, label_values = normalize_waypoint_series(waypoint_progress, waypoint_names)
    progress_values, label_values = filter_waypoints_to_unit_interval(
        progress_values, label_values
    )
    if progress_values.size == 0:
        return

    route_count = len(average_route["lat"])
    if route_count == 0:
        return

    # Convert fractional progress into route indices so labels land on the plotted line.
    waypoint_indices = np.unique(
        np.clip(np.rint(progress_values * (route_count - 1)).astype(int), 0, route_count - 1)
    )
    waypoint_lon = np.asarray(average_route["lon"])[waypoint_indices]
    waypoint_lat = np.asarray(average_route["lat"])[waypoint_indices]
    ax.scatter(waypoint_lon, waypoint_lat, s=90, color="black", zorder=5)

    # Offset labels slightly so they remain readable on top of the route line.
    for progress_value, name in zip(progress_values, label_values):
        if not name:
            continue
        point_index = int(round(progress_value * (route_count - 1)))
        point_index = max(0, min(route_count - 1, point_index))
        ax.annotate(
            name,
            (average_route["lon"][point_index], average_route["lat"][point_index]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=ACTIVE_FIGURE_PROFILE.waypoint_label_fontsize,
            color="black",
            zorder=6,
        )


def build_leg_labels(progress_values, waypoint_labels):
    """Create leg labels from consecutive waypoint names."""
    leg_labels = []
    # Keep labels usable even if some waypoint names are missing in metadata.
    for leg_index in range(len(progress_values) - 1):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")
    return leg_labels


def get_leg_bounds(progress_values, leg_index):
    """Return ordered start/end bounds for a leg index."""
    leg_start = progress_values[leg_index]
    leg_end = progress_values[leg_index + 1]
    if leg_start > leg_end:
        leg_start, leg_end = leg_end, leg_start
    return leg_start, leg_end


def compute_first_leg_distance_m(average_route, progress_values):
    """Compute first-leg distance in meters from the rhumb line."""
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
    """Return per-track delta from the mean for the first-leg metric."""
    # Center first-leg values on the fleet mean so deltas are comparable to later legs.
    if values_by_track.size:
        mean_value = float(np.nanmean(values_by_track))
    else:
        mean_value = np.nan
    return values_by_track - mean_value


def get_leg_samples_for_track(progress, delta_values, progress_values, leg_index, first_leg_value):
    """Extract samples for a specific leg, or the first-leg scalar value."""
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
    """Order tracks by mean value within a leg."""
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


def compute_leg_whisker_range(progress, delta_values, progress_values, first_leg_value):
    """Return whisker bounds for all legs for a single track."""
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


def compute_leg_whisker_range_across_tracks(
    track_progress_by_track,
    delta_by_track,
    progress_values,
    leg_index,
    first_leg_delta_by_track,
):
    """Return whisker bounds for one leg aggregated across all tracks."""
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
    """Return global y-range and units-per-inch for one metric (speed or pace)."""
    global_lower = float("inf")
    global_upper = float("-inf")
    for track_index, (progress, delta_values) in enumerate(
        zip(track_progress_by_track, delta_by_track)
    ):
        if progress.size == 0:
            continue
        leg_range = compute_leg_whisker_range(
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


def apply_waypoint_ticks(ax, waypoint_progress, waypoint_names):
    """Replace progress ticks with waypoint labels plus Start/Finish."""
    progress_values, label_values = normalize_waypoint_series(waypoint_progress, waypoint_names)
    if progress_values.size == 0:
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Start", "Finish"], rotation=45, ha="right")
        return

    # Normalize progress and remove any labels that already encode Start/Finish.
    progress_values = np.clip(progress_values, 0.0, 1.0)
    tolerance = 1e-6

    def is_start_finish_label(label_text):
        label_text = str(label_text).strip().lower()
        return label_text.startswith("start") or label_text.startswith("finish")

    filtered_progress = []
    filtered_labels = []
    for value, label in zip(progress_values.tolist(), label_values):
        if is_start_finish_label(label):
            continue
        filtered_progress.append(value)
        filtered_labels.append(label)

    # Always include Start and Finish even if the input list omitted them.
    filtered_progress.extend([0.0, 1.0])
    filtered_labels.extend(["Start", "Finish"])

    progress_values = np.asarray(filtered_progress, dtype=float)
    label_values = [filtered_labels[idx] for idx in np.argsort(progress_values)]
    progress_values = np.sort(progress_values)

    # Merge labels that land on the same progress to avoid overlapping ticks.
    merged_progress = []
    merged_labels = []
    for progress_value, label in zip(progress_values, label_values):
        if abs(progress_value - 0.0) <= tolerance:
            label = "Start"
        if abs(progress_value - 1.0) <= tolerance:
            label = "Finish"
        if not merged_progress:
            merged_progress.append(progress_value)
            merged_labels.append(label)
            continue
        if abs(progress_value - merged_progress[-1]) <= tolerance:
            if label:
                if abs(progress_value - 0.0) <= tolerance:
                    merged_labels[-1] = "Start"
                elif abs(progress_value - 1.0) <= tolerance:
                    merged_labels[-1] = "Finish"
                elif merged_labels[-1]:
                    merged_labels[-1] = f"{merged_labels[-1]} / {label}"
                else:
                    merged_labels[-1] = label
            continue
        merged_progress.append(progress_value)
        merged_labels.append(label)

    # Escape labels when TeX rendering is enabled to avoid LaTeX compile errors.
    if matplotlib.rcParams.get("text.usetex", False):
        merged_labels = [sh.escape_latex_text(lbl) for lbl in merged_labels]

    ax.set_xticks(merged_progress)
    ax.set_xticklabels(merged_labels, rotation=45, ha="right")


def resample_track_speed(track, window_progress, route_sample_count, filter_alpha):
    """Resample and low-pass a track's speed onto the window progress grid."""
    route_index = track["routeIdx"]
    speed = track["speed"]
    valid_mask = (
        np.isfinite(route_index)
        & np.isfinite(speed)
        & (speed > 0)
        & (route_index >= 1)
        & (route_index <= route_sample_count)
    )
    if not valid_mask.any():
        return None

    # Map each speed sample to fractional progress so we can compare across boats.
    progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
    speed_valid = speed[valid_mask]
    progress_unique, unique_index = np.unique(progress, return_index=True)
    speed_unique = speed_valid[unique_index]
    if progress_unique.size < 2:
        return None

    # Interpolate onto the window grid then smooth to match the alpha baseline.
    sort_index = np.argsort(progress_unique)
    progress_unique = progress_unique[sort_index]
    speed_unique = speed_unique[sort_index]

    speed_on_grid = np.interp(window_progress, progress_unique, speed_unique)
    outside_mask = (window_progress < progress_unique[0]) | (window_progress > progress_unique[-1])
    speed_on_grid[outside_mask] = np.nan
    return sh.lowpass_forward_backward(speed_on_grid, filter_alpha)


def build_alpha_colormap():
    """Return the trimmed 'hot' colormap and normalization for alpha."""
    # Trim the brightest tail to keep the palette from washing out at high alpha.
    base_cmap = plt.get_cmap("hot", 256)
    cmap_colors = base_cmap(np.linspace(0, 229 / 255, 230))
    cmap_colors[0, :] = np.array([0, 0, 0, 1])
    cmap = LinearSegmentedColormap.from_list("hot_trim", cmap_colors)
    norm = Normalize(0, 1)
    return cmap, norm


def add_alpha_tracks(ax, tracks, cmap, norm):
    """Add alpha-colored line collections to the map."""
    line_collections = []
    for track in tracks:
        longitude = track["lon"]
        latitude = track["lat"]
        alpha_values = track["alpha"].astype(float)
        alpha_values[~np.isfinite(alpha_values)] = 0.5

        if longitude.size < 2:
            continue

        # Build per-segment line collections so each segment can be colored by alpha.
        points = np.column_stack([longitude, latitude])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        segment_values = alpha_values[:-1]

        line_collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5)
        line_collection.set_array(segment_values)
        line_collection.set_picker(True)
        line_collection.set_pickradius(6)
        # Attach metadata for the custom data-tip handler.
        line_collection.track_name = track["name"]
        line_collection.track_speed = track["speed"]
        ax.add_collection(line_collection)
        line_collections.append(line_collection)

    return line_collections


def plot_colored_tracks(
    tracks,
    average_route,
    speed_window_stats,
    waypoint_progress,
    waypoint_names,
    export_path=None,
    export_and_close=False,
):
    """Plot each competitor track colored by local relative speed."""
    # Disable TeX text rendering for the map to keep annotations lightweight.
    previous_usetex = matplotlib.rcParams.get("text.usetex", False)
    matplotlib.rcParams["text.usetex"] = False
    fig, ax = plt.subplots()
    prepare_figure(fig, export_and_close=export_and_close)
    ax.grid(True)
    # Preserve metric proportions so the route geometry is not visually distorted.
    sh.apply_local_meter_aspect(ax, average_route)

    # Coastline is purely contextual; crop to track bounds to keep it lightweight.
    plot_coast_geojson_cropped(ax, tracks, "ne_10m_coastline.geojson", 0.15)

    rhumb_line, = ax.plot(
        average_route["lon"],
        average_route["lat"],
        color=(0, 0.6, 1),
        linewidth=5.0,
        label="Rhumb line",
    )
    rhumb_line.set_picker(True)
    rhumb_line.set_pickradius(6)

    plot_waypoints_on_route(ax, average_route, waypoint_progress, waypoint_names)

    cmap, norm = build_alpha_colormap()
    add_alpha_tracks(ax, tracks, cmap, norm)

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
    """Shared inputs for speed delta box plots."""

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
    """Compute shared inputs for speed delta box plots, or return None when unavailable."""
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
    """Plot speed-delta box plots per leg using precomputed inputs."""
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
        leg_range = compute_leg_whisker_range_across_tracks(
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
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
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
    """Plot speed-delta box plots per boat using precomputed inputs."""
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

        leg_range = compute_leg_whisker_range(
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
        ax.set_title(track.get("name", "Boat"))

        if export_dir is not None:
            # Persist per-boat figures for inclusion in reports.
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sh.sanitize_filename_label(track.get("name", f"boat-{track_index + 1}"))
            output_path = export_dir / f"speed-delta-boat-{safe_name}.pdf"
            save_figure(fig, output_path)
        if export_and_close:
            plt.close(fig)


def compute_speed_delta_samples(tracks, speed_window_stats):
    """Prepare per-track progress and speed-delta samples for reuse in multiple plots."""
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
    """Compute per-boat average speed for the first leg from start to the first gate."""
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
    """Prepare per-track progress and pace-delta samples for reuse in multiple plots."""
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
    """Compute per-boat pace for the first leg from start time to first gate."""
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
    """Shared inputs for pace delta box plots."""

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
    """Compute shared inputs for pace delta box plots, or return None when unavailable."""
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
    """Plot pace-delta box plots per leg using precomputed inputs."""
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
        leg_range = compute_leg_whisker_range_across_tracks(
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
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
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
    """Plot pace-delta box plots per boat using precomputed inputs."""
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

        leg_range = compute_leg_whisker_range(
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
        ax.set_title(track.get("name", "Boat"))

        if export_dir is not None:
            # Export per-boat figures for documentation.
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sh.sanitize_filename_label(track.get("name", f"boat-{track_index + 1}"))
            output_path = export_dir / f"pace-delta-boat-{safe_name}.pdf"
            save_figure(fig, output_path)
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
    """Debug plot of speed vs progress with min/max/mean envelope."""
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
        speed_on_grid = resample_track_speed(track, window_progress, route_sample_count, filter_alpha)
        if speed_on_grid is None:
            continue
        line_color = track["color"] or default_color[idx]
        ax.plot(window_progress, speed_on_grid, color=line_color, linewidth=0.8, label=track["name"])

    # Envelope lines show fleet-wide min/mean/max at each progress window.
    ax.plot(window_progress, min_speed, "b-", linewidth=1.5, label="Slowest speed (min)")
    ax.plot(window_progress, max_speed, "r-", linewidth=1.5, label="Fastest speed (max)")
    if mean_speed.size:
        ax.plot(window_progress, mean_speed, "k-", linewidth=1.5, label="Mean speed")

    apply_waypoint_ticks(ax, waypoint_progress, waypoint_names)

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


def plot_coast_geojson_cropped(ax, tracks, geojson_file, margin_deg):
    """Plot coastline lines from a GeoJSON, cropped to tracks bbox."""
    if not Path(geojson_file).is_file():
        return

    # Use the track bounds to crop the coastline and avoid drawing irrelevant geometry.
    all_lat = np.concatenate([track["lat"] for track in tracks if track["lat"].size])
    all_lon = np.concatenate([track["lon"] for track in tracks if track["lon"].size])
    finite_mask = np.isfinite(all_lat) & np.isfinite(all_lon)
    if not finite_mask.any():
        return

    all_lat = all_lat[finite_mask]
    all_lon = all_lon[finite_mask]

    min_lat = float(np.min(all_lat) - margin_deg)
    max_lat = float(np.max(all_lat) + margin_deg)
    min_lon = float(np.min(all_lon) - margin_deg)
    max_lon = float(np.max(all_lon) + margin_deg)

    with open(geojson_file, encoding="utf-8") as handle:
        geojson_data = json.load(handle)

    features = geojson_data.get("features", [])
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        line_strings = sh.geojson_geometry_to_lines(geometry)
        for line in line_strings:
            if line.size == 0:
                continue
            # Mask points outside the bounding box instead of clipping geometry in-place.
            lon = line[:, 0]
            lat = line[:, 1]
            inside = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            lon = lon.astype(float)
            lat = lat.astype(float)
            lon[~inside] = np.nan
            lat[~inside] = np.nan
            ax.plot(lon, lat, color=(0.35, 0.35, 0.35), linewidth=1.0)


if __name__ == "__main__":
    main()


