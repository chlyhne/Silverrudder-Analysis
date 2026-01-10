import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}"
matplotlib.rcParams["axes.unicode_minus"] = False


import silver_helpers as sh

def main():
    # --- Input data ---
    filename = str(Path("data") / "silverrudder_2025" / "Silverrudder 2025_Keelboats Small_gps_data.csv")

    # --- Metadata ---
    metadata_path = Path("data") / "silverrudder_2025" / "race_metadata.json"
    race_meta = sh.load_race_metadata(metadata_path)

    trackIdKeys = race_meta["track_id_keys"]
    boatNames = race_meta["boat_names"]

    # Read CSV into a dict-of-columns, then build per-boat tracks
    trackingData = sh.read_tracking_csv_as_struct(filename)
    tracks = sh.build_tracks(
        trackingData,
        "tracked_object_id",
        "Latitude",
        "Longitude",
        "SampleTime",
        "Speed",
    )

    # Convert speed from km/h to knots (apply once before any calculations)
    tracks = sh.convert_speed_to_knots(tracks)

    # --- Assign human-readable names to tracks (keyed by tracked_object_id) ---
    trackNameMap = dict(zip(trackIdKeys, boatNames))
    tracks = sh.apply_track_names(tracks, trackNameMap)

    localOffsetHours = race_meta.get("local_offset_hours", 0)
    startTimeLocal = race_meta.get("start_time_local", "")
    startTimeUtc = sh.parse_local_time_to_utc(startTimeLocal, localOffsetHours)
    if startTimeUtc is None:
        raise ValueError("start_time_local missing or invalid in race_metadata.json")

    geo_data_path = metadata_path.parent / "waypoints" / "geo_data.json"
    _, waypoint_gates = sh.load_geo_data(geo_data_path)
    start_gate_pos, finish_gate_pos = sh.find_start_finish_gate_positions(waypoint_gates)
    first_leg_gate_pos = start_gate_pos + 1
    if first_leg_gate_pos >= len(waypoint_gates):
        raise ValueError("No waypoint available after Start for first leg.")
    gate_times_by_track = sh.compute_gate_crossings(tracks, waypoint_gates)
    tracks = sh.trim_tracks_by_gate_times(tracks, gate_times_by_track, start_gate_pos, finish_gate_pos)

    boatColors = race_meta.get("boat_colors", {})
    trackColorMap = sh.build_boat_color_map(boatNames, boatColors)
    tracks = sh.apply_track_colors(tracks, trackColorMap)

    # --- Average route and mapping ---
    routeSampleCount = 20000
    averageRoute = sh.compute_average_route(routeSampleCount, geo_data_path)

    routeSearchWindowHalfWidth = 200

    tracks = sh.map_track_points_to_route(tracks, averageRoute, routeSearchWindowHalfWidth)
    tracks = sh.remove_route_index_spikes(tracks, routeSampleCount)

    wayPointProgress, wayPointNames = sh.compute_waypoint_progress_from_gates(
        averageRoute, waypoint_gates
    )

    # --- Route-sample window normalization (per-sample alpha) ---
    windowSampleCount = 50
    windowStepSamples = 1
    filterAlpha = 0.01
    tracks, speedWindowStats = sh.compute_sample_alpha_by_route_windows(
        tracks, routeSampleCount, windowSampleCount, windowStepSamples, filterAlpha
    )

    # --- Plots ---
    export_pdf = True
    export_pdf_dir = Path("documentation") / "figures"

    show_map_plot = True
    show_pace_box_plots = False
    show_pace_box_plots_by_boat = False
    show_speed_box_plots = True
    show_speed_box_plots_by_boat = True
    show_pace_range_plot = True

    if show_map_plot:
        plot_colored_tracks(tracks, averageRoute, speedWindowStats, wayPointProgress, wayPointNames)

    if show_pace_box_plots:
        plot_pace_delta_box_plot(
            tracks,
            speedWindowStats,
            wayPointProgress,
            wayPointNames,
            averageRoute,
            startTimeUtc,
            first_leg_gate_pos,
            export_dir=export_pdf_dir if export_pdf else None,
        )

    if show_pace_box_plots_by_boat:
        plot_pace_delta_box_plot_by_boat(
            tracks,
            speedWindowStats,
            wayPointProgress,
            wayPointNames,
            averageRoute,
            startTimeUtc,
            first_leg_gate_pos,
            export_dir=export_pdf_dir if export_pdf else None,
            latex_include_path=(
                Path("documentation") / "boat-pace-delta-figures.tex" if export_pdf else None
            ),
        )

    if show_speed_box_plots:
        plot_speed_delta_box_plot(
            tracks,
            speedWindowStats,
            wayPointProgress,
            wayPointNames,
            averageRoute,
            startTimeUtc,
            first_leg_gate_pos,
            export_dir=export_pdf_dir if export_pdf else None,
        )

    if show_speed_box_plots_by_boat:
        plot_speed_delta_box_plot_by_boat(
            tracks,
            speedWindowStats,
            wayPointProgress,
            wayPointNames,
            averageRoute,
            startTimeUtc,
            first_leg_gate_pos,
            export_dir=export_pdf_dir if export_pdf else None,
        )

    if show_pace_range_plot:
        plot_speed_range_along_route(
            tracks,
            speedWindowStats,
            wayPointProgress,
            wayPointNames,
            export_path=(export_pdf_dir / "pace-range-along-route.pdf") if export_pdf else None,
        )

    plt.show()


def plot_colored_tracks(tracks, average_route, speed_window_stats, waypoint_progress, waypoint_names):
    """Plot each competitor track colored by local relative speed."""
    previous_usetex = matplotlib.rcParams.get("text.usetex", False)
    matplotlib.rcParams["text.usetex"] = False
    fig, ax = plt.subplots()
    sh.maximize_figure(fig)
    ax.grid(True)
    sh.apply_local_meter_aspect(ax, average_route)

    plot_coast_geojson_cropped(ax, tracks, "ne_10m_coastline.geojson", 0.15)

    waypoint_progress = np.asarray(waypoint_progress, dtype=float)
    waypoint_names = list(waypoint_names) if waypoint_names is not None else []
    if len(waypoint_names) < waypoint_progress.size:
        waypoint_names += [""] * (waypoint_progress.size - len(waypoint_names))
    if len(waypoint_names) > waypoint_progress.size:
        waypoint_names = waypoint_names[: waypoint_progress.size]

    waypoint_mask = np.isfinite(waypoint_progress)
    waypoint_progress = waypoint_progress[waypoint_mask]
    waypoint_names = [name for name, keep in zip(waypoint_names, waypoint_mask) if keep]
    waypoint_progress = waypoint_progress[(waypoint_progress >= 0) & (waypoint_progress <= 1)]

    rhumb_line, = ax.plot(
        average_route["lon"],
        average_route["lat"],
        color=(0, 0.6, 1),
        linewidth=5.0,
        label="Rhumb line",
    )
    rhumb_line.set_picker(True)
    rhumb_line.set_pickradius(6)

    if waypoint_progress.size:
        route_count = len(average_route["lat"])
        waypoint_indices = np.unique(
            np.clip(np.rint(waypoint_progress * (route_count - 1)).astype(int), 0, route_count - 1)
        )
        waypoint_lon = np.asarray(average_route["lon"])[waypoint_indices]
        waypoint_lat = np.asarray(average_route["lat"])[waypoint_indices]
        ax.scatter(waypoint_lon, waypoint_lat, s=90, color="black", zorder=5)

        for progress_value, name in zip(waypoint_progress, waypoint_names):
            if not name:
                continue
            point_index = int(round(progress_value * (route_count - 1)))
            point_index = max(0, min(route_count - 1, point_index))
            ax.annotate(
                name,
                (average_route["lon"][point_index], average_route["lat"][point_index]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color="black",
                zorder=6,
            )

    base_cmap = plt.get_cmap("hot", 256)
    cmap_colors = base_cmap(np.linspace(0, 229 / 255, 230))
    cmap_colors[0, :] = np.array([0, 0, 0, 1])
    cmap = LinearSegmentedColormap.from_list("hot_trim", cmap_colors)
    norm = Normalize(0, 1)

    line_collections = []
    for track in tracks:
        longitude = track["lon"]
        latitude = track["lat"]
        alpha_values = track["alpha"].astype(float)
        alpha_values[~np.isfinite(alpha_values)] = 0.5

        if longitude.size < 2:
            continue

        points = np.column_stack([longitude, latitude])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        segment_values = alpha_values[:-1]

        line_collection = LineCollection(
            segments, cmap=cmap, norm=norm, linewidths=1.5
        )
        line_collection.set_array(segment_values)
        line_collection.set_picker(True)
        line_collection.set_pickradius(6)
        line_collection.track_name = track["name"]
        line_collection.track_speed = track["speed"]
        ax.add_collection(line_collection)
        line_collections.append(line_collection)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label("alpha (rel. speed)")

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Tracks colored by alpha")

    sh.enable_manual_datatips(ax, tracks, average_route, speed_window_stats)
    matplotlib.rcParams["text.usetex"] = previous_usetex


def plot_speed_delta_box_plot(
    tracks,
    speed_window_stats,
    way_point_progress,
    way_point_names,
    average_route,
    start_time_utc,
    first_leg_gate_pos,
    export_dir=None,
):
    """Box plots of speed difference per leg, one figure per leg."""
    track_progress_by_track, speed_delta_by_track = compute_speed_delta_samples(
        tracks, speed_window_stats
    )
    if track_progress_by_track is None:
        return

    progress_values, waypoint_labels = sh.normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

    route_distance_m = sh.cumulative_distance_meters(
        np.asarray(average_route["lat"], dtype=float),
        np.asarray(average_route["lon"], dtype=float),
    )[-1]
    first_leg_distance_m = np.nan
    if leg_count >= 1 and np.isfinite(route_distance_m) and route_distance_m > 0:
        first_leg_distance_m = abs(progress_values[1] - progress_values[0]) * route_distance_m
    first_leg_speed_by_track = compute_first_leg_speed_by_track(
        tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m
    )
    if first_leg_speed_by_track.size:
        first_leg_mean = float(np.nanmean(first_leg_speed_by_track))
    else:
        first_leg_mean = np.nan
    first_leg_delta_by_track = first_leg_speed_by_track - first_leg_mean

    track_count = len(tracks)
    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    for leg_index in range(leg_count - 1, -1, -1):
        fig, ax = plt.subplots()
        sh.maximize_figure(fig)
        apply_pace_boxplot_axis_style(ax)

        leg_start = progress_values[leg_index]
        leg_end = progress_values[leg_index + 1]
        if leg_start > leg_end:
            leg_start, leg_end = leg_end, leg_start

        leg_ordered = []
        if leg_index == 0:
            for track_index in range(track_count):
                speed_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(speed_delta_value):
                    continue
                leg_ordered.append((track_index, float(speed_delta_value), speed_delta_value))
        else:
            for track_index, (progress, speed_delta) in enumerate(
                zip(track_progress_by_track, speed_delta_by_track)
            ):
                if progress.size == 0:
                    continue
                leg_mask = (progress >= leg_start) & (progress <= leg_end)
                speed_leg = speed_delta[leg_mask]
                if speed_leg.size == 0:
                    continue

                leg_ordered.append((track_index, float(np.mean(speed_leg)), speed_leg))

        if not leg_ordered:
            ax.set_title(leg_labels[leg_index])
            continue

        leg_ordered.sort(key=lambda item: item[1])
        for plot_index, (track_index, _, speed_leg) in enumerate(leg_ordered, start=1):
            line_color = tracks[track_index]["color"] or (0.3, 0.3, 0.3)
            if leg_index == 0:
                ax.plot(
                    plot_index,
                    speed_leg,
                    marker="o",
                    markersize=6,
                    color=line_color,
                    linestyle="None",
                )
            else:
                sh.draw_manual_box_plot(ax, plot_index, speed_leg, line_color)

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylabel(r"$\Delta v\,[\mathrm{kn}]$")
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sh.sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"speed-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            fig.savefig(output_path, bbox_inches="tight")


def plot_speed_delta_box_plot_by_boat(
    tracks,
    speed_window_stats,
    way_point_progress,
    way_point_names,
    average_route,
    start_time_utc,
    first_leg_gate_pos,
    export_dir=None,
):
    """Box plot per boat with each leg on the x-axis (speed delta)."""
    track_progress_by_track, speed_delta_by_track = compute_speed_delta_samples(
        tracks, speed_window_stats
    )
    if track_progress_by_track is None:
        return

    progress_values, waypoint_labels = sh.normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

    route_distance_m = sh.cumulative_distance_meters(
        np.asarray(average_route["lat"], dtype=float),
        np.asarray(average_route["lon"], dtype=float),
    )[-1]
    first_leg_distance_m = np.nan
    if leg_count >= 1 and np.isfinite(route_distance_m) and route_distance_m > 0:
        first_leg_distance_m = abs(progress_values[1] - progress_values[0]) * route_distance_m
    first_leg_speed_by_track = compute_first_leg_speed_by_track(
        tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m
    )
    if first_leg_speed_by_track.size:
        first_leg_mean = float(np.nanmean(first_leg_speed_by_track))
    else:
        first_leg_mean = np.nan
    first_leg_delta_by_track = first_leg_speed_by_track - first_leg_mean

    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    def compute_whisker_bounds(samples):
        clean_samples = samples[np.isfinite(samples)]
        if clean_samples.size == 0:
            return None
        quartiles, _ = sh.quantiles_no_toolbox(clean_samples, [0.25, 0.5, 0.75])
        q1 = float(quartiles[0])
        q3 = float(quartiles[2])
        if not np.isfinite(q1) or not np.isfinite(q3):
            return None
        iqr_value = q3 - q1
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        lower_candidates = clean_samples[clean_samples >= lower_bound]
        upper_candidates = clean_samples[clean_samples <= upper_bound]
        lower_whisker = float(np.min(lower_candidates)) if lower_candidates.size else float(np.min(clean_samples))
        upper_whisker = float(np.max(upper_candidates)) if upper_candidates.size else float(np.max(clean_samples))
        return lower_whisker, upper_whisker

    global_lower = np.inf
    global_upper = -np.inf
    for track_index, (progress, speed_delta) in enumerate(
        zip(track_progress_by_track, speed_delta_by_track)
    ):
        if progress.size == 0:
            continue
        for leg_index in range(leg_count):
            if leg_index == 0:
                speed_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(speed_delta_value):
                    continue
                speed_leg = np.array([speed_delta_value], dtype=float)
            else:
                leg_start = progress_values[leg_index]
                leg_end = progress_values[leg_index + 1]
                if leg_start > leg_end:
                    leg_start, leg_end = leg_end, leg_start
                leg_mask = (progress >= leg_start) & (progress <= leg_end)
                speed_leg = speed_delta[leg_mask]
                if speed_leg.size == 0:
                    continue
            whisker_bounds = compute_whisker_bounds(speed_leg)
            if whisker_bounds is None:
                continue
            global_lower = min(global_lower, whisker_bounds[0])
            global_upper = max(global_upper, whisker_bounds[1])

    if not np.isfinite(global_lower) or not np.isfinite(global_upper):
        return
    global_range = global_upper - global_lower
    if global_range <= 0:
        global_lower, global_upper = -1.0, 1.0
        global_range = global_upper - global_lower
    else:
        padding = 0.05 * global_range
        global_lower -= padding
        global_upper += padding
        global_range = global_upper - global_lower

    base_figsize = plt.rcParams.get("figure.figsize", (6.4, 4.8))
    base_height = float(base_figsize[1]) if len(base_figsize) > 1 else 4.8
    units_per_inch = global_range / base_height if global_range > 0 else 1.0

    for track_index, track in enumerate(tracks):
        progress = track_progress_by_track[track_index]
        speed_delta = speed_delta_by_track[track_index]
        if progress.size == 0 or speed_delta.size == 0:
            continue

        local_lower = np.inf
        local_upper = -np.inf
        for leg_index in range(leg_count):
            if leg_index == 0:
                speed_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(speed_delta_value):
                    continue
                speed_leg = np.array([speed_delta_value], dtype=float)
            else:
                leg_start = progress_values[leg_index]
                leg_end = progress_values[leg_index + 1]
                if leg_start > leg_end:
                    leg_start, leg_end = leg_end, leg_start
                leg_mask = (progress >= leg_start) & (progress <= leg_end)
                speed_leg = speed_delta[leg_mask]
                if speed_leg.size == 0:
                    continue
            whisker_bounds = compute_whisker_bounds(speed_leg)
            if whisker_bounds is None:
                continue
            local_lower = min(local_lower, whisker_bounds[0])
            local_upper = max(local_upper, whisker_bounds[1])

        if not np.isfinite(local_lower) or not np.isfinite(local_upper):
            continue

        local_range = local_upper - local_lower
        if local_range <= 0:
            local_range = max(global_range * 0.02, 1.0)
            mid_point = (local_lower + local_upper) / 2.0
            local_lower = mid_point - 0.5 * local_range
            local_upper = mid_point + 0.5 * local_range
        else:
            padding = 0.02 * local_range
            local_lower -= padding
            local_upper += padding
            local_range = local_upper - local_lower

        fig, ax = plt.subplots()
        sh.maximize_figure(fig)
        fig_width = fig.get_size_inches()[0]
        fig_height = local_range / units_per_inch if units_per_inch > 0 else fig.get_size_inches()[1]
        fig.set_size_inches(fig_width, fig_height, forward=True)
        apply_pace_boxplot_axis_style(ax)

        line_color = track.get("color") or (0.3, 0.3, 0.3)
        for leg_index in range(leg_count):
            if leg_index == 0:
                speed_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(speed_delta_value):
                    continue
                ax.plot(
                    leg_index + 1,
                    speed_delta_value,
                    marker="o",
                    markersize=6,
                    color=line_color,
                    linestyle="None",
                )
                continue

            leg_start = progress_values[leg_index]
            leg_end = progress_values[leg_index + 1]
            if leg_start > leg_end:
                leg_start, leg_end = leg_end, leg_start

            leg_mask = (progress >= leg_start) & (progress <= leg_end)
            speed_leg = speed_delta[leg_mask]
            if speed_leg.size == 0:
                continue

            sh.draw_manual_box_plot(ax, leg_index + 1, speed_leg, line_color)

        ax.set_xticks(range(1, leg_count + 1))
        ax.set_xticklabels(leg_labels, rotation=45, ha="right")
        ax.set_xlim(0.5, leg_count + 0.5)
        ax.set_ylabel(r"$\Delta v\,[\mathrm{kn}]$")
        ax.set_ylim(local_lower, local_upper)
        ax.set_title(track.get("name", "Boat"))

        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sh.sanitize_filename_label(track.get("name", f"boat-{track_index + 1}"))
            output_path = export_dir / f"speed-delta-boat-{safe_name}.pdf"
            fig.savefig(output_path, bbox_inches="tight")


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
        progress = (route_index_valid - 1) / (route_sample_count - 1)
        if window_progress.size >= 2:
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
            home_window = np.floor((route_index_valid - 1) / window_step_samples) + 1
            home_window = np.clip(home_window, 1, window_count).astype(int)
            mean_speed_for_sample = mean_speed_by_window[home_window - 1]

        mean_speed_for_sample = np.where(
            np.isfinite(mean_speed_for_sample) & (mean_speed_for_sample > 0),
            mean_speed_for_sample,
            np.nan,
        )

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
    if not np.isfinite(start_time_utc) or first_leg_distance_m <= 0:
        return np.full(len(tracks), np.nan, dtype=float)

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
        progress = (route_index_valid - 1) / (route_sample_count - 1)
        if window_progress.size >= 2:
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
            home_window = np.floor((route_index_valid - 1) / window_step_samples) + 1
            home_window = np.clip(home_window, 1, window_count).astype(int)
            mean_pace_for_sample = mean_pace_by_window[home_window - 1]
        mean_pace_for_sample = np.where(
            np.isfinite(mean_pace_for_sample) & (mean_pace_for_sample > 0),
            mean_pace_for_sample,
            np.nan,
        )

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
    if not np.isfinite(start_time_utc) or first_leg_distance_m <= 0:
        return np.full(len(tracks), np.nan, dtype=float)

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


def apply_pace_boxplot_axis_style(ax):
    """Apply y-axis grid/tick styling for pace box plots."""
    ax.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(True, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", linewidth=0.6, alpha=0.5)
    ax.grid(False, axis="x", which="both")
    ax.tick_params(axis="y", which="minor", length=2)


def plot_pace_delta_box_plot(
    tracks,
    speed_window_stats,
    way_point_progress,
    way_point_names,
    average_route,
    start_time_utc,
    first_leg_gate_pos,
    export_dir=None,
):
    """Box plots of pace difference per leg, one figure per leg."""
    track_progress_by_track, pace_delta_by_track = compute_pace_delta_samples(
        tracks, speed_window_stats
    )
    if track_progress_by_track is None:
        return

    progress_values, waypoint_labels = sh.normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

    route_distance_m = sh.cumulative_distance_meters(
        np.asarray(average_route["lat"], dtype=float),
        np.asarray(average_route["lon"], dtype=float),
    )[-1]
    first_leg_distance_m = np.nan
    if leg_count >= 1 and np.isfinite(route_distance_m) and route_distance_m > 0:
        first_leg_distance_m = abs(progress_values[1] - progress_values[0]) * route_distance_m
    first_leg_pace_by_track = compute_first_leg_pace_by_track(
        tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m
    )
    if first_leg_pace_by_track.size:
        first_leg_mean = float(np.nanmean(first_leg_pace_by_track))
    else:
        first_leg_mean = np.nan
    first_leg_delta_by_track = first_leg_pace_by_track - first_leg_mean

    track_count = len(tracks)
    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    for leg_index in range(leg_count - 1, -1, -1):
        fig, ax = plt.subplots()
        sh.maximize_figure(fig)
        apply_pace_boxplot_axis_style(ax)

        leg_start = progress_values[leg_index]
        leg_end = progress_values[leg_index + 1]
        if leg_start > leg_end:
            leg_start, leg_end = leg_end, leg_start

        leg_ordered = []
        if leg_index == 0:
            for track_index in range(track_count):
                pace_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(pace_delta_value):
                    continue
                leg_ordered.append((track_index, float(pace_delta_value), pace_delta_value))
        else:
            for track_index, (progress, pace_delta) in enumerate(
                zip(track_progress_by_track, pace_delta_by_track)
            ):
                if progress.size == 0:
                    continue
                leg_mask = (progress >= leg_start) & (progress <= leg_end)
                pace_leg = pace_delta[leg_mask]
                if pace_leg.size == 0:
                    continue

                leg_ordered.append((track_index, float(np.mean(pace_leg)), pace_leg))

        if not leg_ordered:
            ax.set_title(leg_labels[leg_index])
            continue

        leg_ordered.sort(key=lambda item: item[1])
        for plot_index, (track_index, _, pace_leg) in enumerate(leg_ordered, start=1):
            line_color = tracks[track_index]["color"] or (0.3, 0.3, 0.3)
            if leg_index == 0:
                ax.plot(
                    plot_index,
                    pace_leg,
                    marker="o",
                    markersize=6,
                    color=line_color,
                    linestyle="None",
                )
            else:
                sh.draw_manual_box_plot(ax, plot_index, pace_leg, line_color)

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylabel(r"$\Delta p\,[\mathrm{min}\,\mathrm{NM}^{-1}]$")
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sh.sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"pace-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            fig.savefig(output_path, bbox_inches="tight")

def plot_pace_delta_box_plot_by_boat(
    tracks,
    speed_window_stats,
    way_point_progress,
    way_point_names,
    average_route,
    start_time_utc,
    first_leg_gate_pos,
    export_dir=None,
    latex_include_path=None,
):
    """Box plot per boat with each leg on the x-axis."""
    track_progress_by_track, pace_delta_by_track = compute_pace_delta_samples(
        tracks, speed_window_stats
    )
    if track_progress_by_track is None:
        return

    progress_values, waypoint_labels = sh.normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

    route_distance_m = sh.cumulative_distance_meters(
        np.asarray(average_route["lat"], dtype=float),
        np.asarray(average_route["lon"], dtype=float),
    )[-1]
    first_leg_distance_m = np.nan
    if leg_count >= 1 and np.isfinite(route_distance_m) and route_distance_m > 0:
        first_leg_distance_m = abs(progress_values[1] - progress_values[0]) * route_distance_m
    first_leg_pace_by_track = compute_first_leg_pace_by_track(
        tracks, start_time_utc, first_leg_gate_pos, first_leg_distance_m
    )
    if first_leg_pace_by_track.size:
        first_leg_mean = float(np.nanmean(first_leg_pace_by_track))
    else:
        first_leg_mean = np.nan
    first_leg_delta_by_track = first_leg_pace_by_track - first_leg_mean

    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    def compute_whisker_bounds(samples):
        clean_samples = samples[np.isfinite(samples)]
        if clean_samples.size == 0:
            return None
        quartiles, _ = sh.quantiles_no_toolbox(clean_samples, [0.25, 0.5, 0.75])
        q1 = float(quartiles[0])
        q3 = float(quartiles[2])
        if not np.isfinite(q1) or not np.isfinite(q3):
            return None
        iqr_value = q3 - q1
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        lower_candidates = clean_samples[clean_samples >= lower_bound]
        upper_candidates = clean_samples[clean_samples <= upper_bound]
        lower_whisker = float(np.min(lower_candidates)) if lower_candidates.size else float(np.min(clean_samples))
        upper_whisker = float(np.max(upper_candidates)) if upper_candidates.size else float(np.max(clean_samples))
        return lower_whisker, upper_whisker

    global_lower = np.inf
    global_upper = -np.inf
    for track_index, (progress, pace_delta) in enumerate(
        zip(track_progress_by_track, pace_delta_by_track)
    ):
        if progress.size == 0:
            continue
        for leg_index in range(leg_count):
            if leg_index == 0:
                pace_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(pace_delta_value):
                    continue
                pace_leg = np.array([pace_delta_value], dtype=float)
            else:
                leg_start = progress_values[leg_index]
                leg_end = progress_values[leg_index + 1]
                if leg_start > leg_end:
                    leg_start, leg_end = leg_end, leg_start
                leg_mask = (progress >= leg_start) & (progress <= leg_end)
                pace_leg = pace_delta[leg_mask]
                if pace_leg.size == 0:
                    continue
            whisker_bounds = compute_whisker_bounds(pace_leg)
            if whisker_bounds is None:
                continue
            global_lower = min(global_lower, whisker_bounds[0])
            global_upper = max(global_upper, whisker_bounds[1])

    if not np.isfinite(global_lower) or not np.isfinite(global_upper):
        return
    global_range = global_upper - global_lower
    if global_range <= 0:
        global_lower, global_upper = -1.0, 1.0
        global_range = global_upper - global_lower
    else:
        padding = 0.05 * global_range
        global_lower -= padding
        global_upper += padding
        global_range = global_upper - global_lower

    base_figsize = plt.rcParams.get("figure.figsize", (6.4, 4.8))
    base_height = float(base_figsize[1]) if len(base_figsize) > 1 else 4.8
    units_per_inch = global_range / base_height if global_range > 0 else 1.0

    latex_lines = []
    if latex_include_path is not None:
        latex_lines.append("% Auto-generated by silver.py. Do not edit by hand.\n")

    for track_index, track in enumerate(tracks):
        progress = track_progress_by_track[track_index]
        pace_delta = pace_delta_by_track[track_index]
        if progress.size == 0 or pace_delta.size == 0:
            continue

        local_lower = np.inf
        local_upper = -np.inf
        for leg_index in range(leg_count):
            if leg_index == 0:
                pace_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(pace_delta_value):
                    continue
                pace_leg = np.array([pace_delta_value], dtype=float)
            else:
                leg_start = progress_values[leg_index]
                leg_end = progress_values[leg_index + 1]
                if leg_start > leg_end:
                    leg_start, leg_end = leg_end, leg_start
                leg_mask = (progress >= leg_start) & (progress <= leg_end)
                pace_leg = pace_delta[leg_mask]
                if pace_leg.size == 0:
                    continue
            whisker_bounds = compute_whisker_bounds(pace_leg)
            if whisker_bounds is None:
                continue
            local_lower = min(local_lower, whisker_bounds[0])
            local_upper = max(local_upper, whisker_bounds[1])

        if not np.isfinite(local_lower) or not np.isfinite(local_upper):
            continue

        local_range = local_upper - local_lower
        if local_range <= 0:
            local_range = max(global_range * 0.02, 1.0)
            mid_point = (local_lower + local_upper) / 2.0
            local_lower = mid_point - 0.5 * local_range
            local_upper = mid_point + 0.5 * local_range
        else:
            padding = 0.02 * local_range
            local_lower -= padding
            local_upper += padding
            local_range = local_upper - local_lower

        fig, ax = plt.subplots()
        sh.maximize_figure(fig)
        fig_width = fig.get_size_inches()[0]
        fig_height = local_range / units_per_inch if units_per_inch > 0 else fig.get_size_inches()[1]
        fig.set_size_inches(fig_width, fig_height, forward=True)
        apply_pace_boxplot_axis_style(ax)

        line_color = track.get("color") or (0.3, 0.3, 0.3)
        for leg_index in range(leg_count):
            if leg_index == 0:
                pace_delta_value = first_leg_delta_by_track[track_index]
                if not np.isfinite(pace_delta_value):
                    continue
                ax.plot(
                    leg_index + 1,
                    pace_delta_value,
                    marker="o",
                    markersize=6,
                    color=line_color,
                    linestyle="None",
                )
                continue

            leg_start = progress_values[leg_index]
            leg_end = progress_values[leg_index + 1]
            if leg_start > leg_end:
                leg_start, leg_end = leg_end, leg_start

            leg_mask = (progress >= leg_start) & (progress <= leg_end)
            pace_leg = pace_delta[leg_mask]
            if pace_leg.size == 0:
                continue

            sh.draw_manual_box_plot(ax, leg_index + 1, pace_leg, line_color)

        ax.set_xticks(range(1, leg_count + 1))
        ax.set_xticklabels(leg_labels, rotation=45, ha="right")
        ax.set_xlim(0.5, leg_count + 0.5)
        ax.set_ylabel(r"$\Delta p\,[\mathrm{min}\,\mathrm{NM}^{-1}]$")
        ax.set_ylim(local_lower, local_upper)
        ax.set_title(track.get("name", "Boat"))

        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sh.sanitize_filename_label(track.get("name", f"boat-{track_index + 1}"))
            output_path = export_dir / f"pace-delta-boat-{safe_name}.pdf"
            fig.savefig(output_path, bbox_inches="tight")

            if latex_include_path is not None:
                escaped_name = sh.escape_latex_text(track.get("name", "Boat"))
                latex_lines.extend(
                    [
                        "\\clearpage\n",
                        f"\\subsection*{{Boat: {escaped_name}}}\n",
                        "\\begin{figure}[H]\n",
                        "\\centering\n",
                        f"\\includegraphics[width=\\linewidth]{{figures/{output_path.name}}}\n",
                        f"\\caption{{Pace delta by leg for {escaped_name}.}}\n",
                        "\\end{figure}\n",
                    ]
                )

    if latex_include_path is not None:
        latex_include_path.write_text("".join(latex_lines), encoding="utf-8")

def plot_speed_range_along_route(
    tracks, speed_window_stats, waypoint_progress=None, waypoint_names=None, export_path=None
):
    """Debug plot of speed vs progress with min/max/mean envelope."""
    if not speed_window_stats or speed_window_stats.get("routeSampleCount", 0) < 2:
        return

    route_sample_count = speed_window_stats["routeSampleCount"]
    window_start_index = speed_window_stats["windowStartIndex"]
    window_end_index = speed_window_stats["windowEndIndex"]
    min_speed = speed_window_stats["minSpeedByWindow"]
    max_speed = speed_window_stats["maxSpeedByWindow"]
    mean_speed = speed_window_stats.get("meanSpeedByWindow", np.array([]))
    filter_alpha = speed_window_stats.get("filterAlpha", 0)

    window_center = (window_start_index + window_end_index) / 2
    window_progress = (window_center - 1) / (route_sample_count - 1)

    fig, ax = plt.subplots()
    sh.maximize_figure(fig)
    ax.grid(True)

    default_color = plt.cm.hsv(np.linspace(0, 1, max(len(tracks), 1)))
    for idx, track in enumerate(tracks):
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
            continue

        progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
        speed_valid = speed[valid_mask]
        progress_unique, unique_index = np.unique(progress, return_index=True)
        speed_unique = speed_valid[unique_index]
        if progress_unique.size < 2:
            continue

        sort_index = np.argsort(progress_unique)
        progress_unique = progress_unique[sort_index]
        speed_unique = speed_unique[sort_index]

        speed_on_grid = np.interp(window_progress, progress_unique, speed_unique)
        outside_mask = (window_progress < progress_unique[0]) | (window_progress > progress_unique[-1])
        speed_on_grid[outside_mask] = np.nan
        speed_on_grid = sh.lowpass_forward_backward(speed_on_grid, filter_alpha)

        line_color = track["color"] or default_color[idx]
        ax.plot(window_progress, speed_on_grid, color=line_color, linewidth=0.8, label=track["name"])

    ax.plot(window_progress, min_speed, "b-", linewidth=1.5, label="Slowest speed (min)")
    ax.plot(window_progress, max_speed, "r-", linewidth=1.5, label="Fastest speed (max)")
    if mean_speed.size:
        ax.plot(window_progress, mean_speed, "k-", linewidth=1.5, label="Mean speed")

    if waypoint_progress is not None and waypoint_names is not None:
        progress_values = np.asarray(waypoint_progress, dtype=float)
        label_values = list(waypoint_names)
        if label_values and progress_values.size:
            if len(label_values) < progress_values.size:
                label_values += [""] * (progress_values.size - len(label_values))
            if len(label_values) > progress_values.size:
                label_values = label_values[: progress_values.size]

            valid_mask = np.isfinite(progress_values)
            progress_values = progress_values[valid_mask]
            label_values = [lbl for lbl, keep in zip(label_values, valid_mask) if keep]
            if progress_values.size:
                progress_values = np.clip(progress_values, 0.0, 1.0)
                tolerance = 1e-6

                progress_list = progress_values.tolist()
                label_list = list(label_values)

                def is_start_finish_label(label_text):
                    label_text = str(label_text).strip().lower()
                    return label_text.startswith("start") or label_text.startswith("finish")

                filtered_progress = []
                filtered_labels = []
                for value, label in zip(progress_list, label_list):
                    if is_start_finish_label(label):
                        continue
                    filtered_progress.append(value)
                    filtered_labels.append(label)

                progress_list = filtered_progress
                label_list = filtered_labels

                progress_list.append(0.0)
                label_list.append("Start")
                progress_list.append(1.0)
                label_list.append("Finish")

                progress_values = np.asarray(progress_list, dtype=float)
                sort_index = np.argsort(progress_values)
                progress_values = progress_values[sort_index]
                label_values = [label_list[idx] for idx in sort_index]

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

                if matplotlib.rcParams.get("text.usetex", False):
                    merged_labels = [sh.escape_latex_text(lbl) for lbl in merged_labels]

                ax.set_xticks(merged_progress)
                ax.set_xticklabels(merged_labels, rotation=45, ha="right")
            else:
                ax.set_xticks([0.0, 1.0])
                ax.set_xticklabels(["Start", "Finish"], rotation=45, ha="right")

    ax.set_xlabel(r"$s$ (progress)")
    ax.set_ylabel(r"$v\,[\mathrm{kn}]$")
    ax.set_title(r"$v(s)$ with min/mean/max envelope")
    ax.legend(loc="best")

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, bbox_inches="tight")


def plot_coast_geojson_cropped(ax, tracks, geojson_file, margin_deg):
    """Plot coastline lines from a GeoJSON, cropped to tracks bbox."""
    if not Path(geojson_file).is_file():
        return

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


