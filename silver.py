import csv
import json
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


from silver_helpers import (
    read_tracking_csv_as_struct,
    build_tracks,
    convert_speed_to_knots,
    apply_track_names,
    apply_track_colors,
    load_race_metadata,
    parse_local_time_to_utc,
    build_boat_color_map,
    load_waypoint_csvs,
    enforce_monotonic_waypoint_times,
    get_waypoint_index_bounds,
    extract_waypoint_time_map,
    compute_reference_start_time,
    compute_waypoint_positions,
    interpolate_track_position,
    unwrap_waypoint_indices,
    map_waypoints_to_progress,
    compute_start_finish_point,
    trim_tracks_by_start_time_and_finish_proximity,
    map_track_points_to_route,
    remove_route_index_spikes,
    compute_sample_alpha_by_route_windows,
    lowpass_forward_backward,
    maximize_figure,
    apply_local_meter_aspect,
    enable_rhumbline_datatip,
    enable_track_datatips,
    enable_manual_datatips,
    draw_manual_box_plot,
    smooth_moving_average,
    normalize_waypoint_inputs,
    sanitize_filename_label,
    quantiles_no_toolbox,
    haversine_meters,
    to_string_column,
    normalize_boat_name,
    to_double_column,
    to_time_column,
    parse_utc_time,
    compute_average_route,
    cumulative_distance_meters,
    ll2xy_meters,
    xy2ll_meters,
    fill_missing_linear,
    geojson_geometry_to_lines,
    coords_to_numeric_2d,
    coords_to_cell_of_2d,
)

def main():
    # --- Input data ---
    filename = str(Path("data") / "silverrudder_2025" / "Silverrudder 2025_Keelboats Small_gps_data.csv")

    # --- Metadata ---
    metadata_path = Path("data") / "silverrudder_2025" / "race_metadata.json"
    race_meta = load_race_metadata(metadata_path)

    # Waypoints are inferred from split CSVs under data/<race>/waypoints.

    trackIdKeys = race_meta["track_id_keys"]
    boatNames = race_meta["boat_names"]

    # Read CSV into a dict-of-columns, then build per-boat tracks
    trackingData = read_tracking_csv_as_struct(filename)
    tracks = build_tracks(
        trackingData,
        "tracked_object_id",
        "Latitude",
        "Longitude",
        "SampleTime",
        "Speed",
    )

    # Convert speed from km/h to knots (apply once before any calculations)
    tracks = convert_speed_to_knots(tracks)

    # --- Assign human-readable names to tracks (keyed by tracked_object_id) ---
    trackNameMap = dict(zip(trackIdKeys, boatNames))
    tracks = apply_track_names(tracks, trackNameMap)

    localOffsetHours = race_meta.get("local_offset_hours", 0)
    startFinishRadiusMeters = float(race_meta.get("start_finish_radius_m", 200.0))
    waypoint_dir = metadata_path.parent / "waypoints"
    waypoint_data = load_waypoint_csvs(waypoint_dir, localOffsetHours)
    if not waypoint_data:
        raise FileNotFoundError(f"No waypoint CSVs found in {waypoint_dir}")

    enforce_monotonic_waypoint_times(waypoint_data)

    start_index, finish_index = get_waypoint_index_bounds(waypoint_data)
    start_times_utc = extract_waypoint_time_map(waypoint_data, start_index)
    finish_times_utc = extract_waypoint_time_map(waypoint_data, finish_index)
    if not finish_times_utc:
        raise ValueError("No finish waypoint times found to infer the start/finish point.")

    startTimeUtc = compute_reference_start_time(start_times_utc)
    startFinishPoint = compute_start_finish_point(tracks, finish_times_utc)
    tracks = trim_tracks_by_start_time_and_finish_proximity(
        tracks,
        startTimeUtc,
        startFinishPoint,
        startFinishRadiusMeters,
    )

    boatColors = race_meta.get("boat_colors", {})
    trackColorMap = build_boat_color_map(boatNames, boatColors)
    tracks = apply_track_colors(tracks, trackColorMap)

    # --- Average route and mapping ---
    routeSampleCount = 20000
    averageRoute = compute_average_route(tracks, routeSampleCount)

    routeSearchWindowHalfWidth = 200

    tracks = map_track_points_to_route(tracks, averageRoute, routeSearchWindowHalfWidth)
    tracks = remove_route_index_spikes(tracks, routeSampleCount)

    waypoint_positions = compute_waypoint_positions(waypoint_data, tracks)
    wayPointProgress, wayPointNames = map_waypoints_to_progress(waypoint_positions, averageRoute)

    # --- Route-sample window normalization (per-sample alpha) ---
    windowSampleCount = 50
    windowStepSamples = 1
    filterAlpha = 0.005
    tracks, speedWindowStats = compute_sample_alpha_by_route_windows(
        tracks, routeSampleCount, windowSampleCount, windowStepSamples, filterAlpha
    )

    # --- Plots ---
    export_pdf = True
    export_pdf_dir = Path("documentation") / "figures"

    show_map_plot = True
    show_pace_box_plots = True
    show_pace_range_plot = True

    if show_map_plot:
        plot_colored_tracks(tracks, averageRoute, speedWindowStats, wayPointProgress, wayPointNames)

    if show_pace_box_plots:
        plot_pace_delta_box_plot(
            tracks,
            speedWindowStats,
            wayPointProgress,
            wayPointNames,
            export_dir=export_pdf_dir if export_pdf else None,
        )

    if show_pace_range_plot:
        plot_speed_range_along_route(
            tracks,
            speedWindowStats,
            export_path=(export_pdf_dir / "pace-range-along-route.pdf") if export_pdf else None,
        )

    plt.show()


def plot_colored_tracks(tracks, average_route, speed_window_stats, waypoint_progress, waypoint_names):
    """Plot each competitor track colored by local relative speed."""
    fig, ax = plt.subplots()
    maximize_figure(fig)
    ax.grid(True)
    apply_local_meter_aspect(ax, average_route)

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
    colorbar.set_label("Relative speed at route position (0=slowest, 1=fastest)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Competitor tracks colored by local (point-by-point) relative speed")

    enable_rhumbline_datatip(ax, rhumb_line, speed_window_stats, average_route)
    enable_track_datatips(line_collections)
    enable_manual_datatips(ax, tracks, average_route, speed_window_stats)


def plot_alpha_box_plot(tracks, route_sample_count, way_point_progress, way_point_names, export_dir=None):
    """Box plots of alpha per leg, one figure per leg."""
    if route_sample_count < 2:
        return

    progress_values, waypoint_labels = normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

    track_progress_by_track = []
    alpha_by_track = []
    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        alpha_values = track["alpha"].astype(float)
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(alpha_values)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
        )
        if not valid_mask.any():
            track_progress_by_track.append(np.array([]))
            alpha_by_track.append(np.array([]))
            continue

        progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
        track_progress_by_track.append(progress)
        alpha_by_track.append(alpha_values[valid_mask])

    track_count = len(tracks)
    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    for leg_index in range(leg_count - 1, -1, -1):
        fig, ax = plt.subplots()
        maximize_figure(fig)
        ax.grid(True)

        leg_start = progress_values[leg_index]
        leg_end = progress_values[leg_index + 1]
        if leg_start > leg_end:
            leg_start, leg_end = leg_end, leg_start

        leg_ordered = []
        for track_index, (progress, alpha_values) in enumerate(
            zip(track_progress_by_track, alpha_by_track)
        ):
            if progress.size == 0:
                continue
            leg_mask = (progress >= leg_start) & (progress <= leg_end)
            alpha_leg = alpha_values[leg_mask]
            if alpha_leg.size == 0:
                continue

            leg_ordered.append((track_index, float(np.mean(alpha_leg)), alpha_leg))

        if not leg_ordered:
            ax.set_title(leg_labels[leg_index])
            continue

        leg_ordered.sort(key=lambda item: item[1])
        for plot_index, (track_index, _, alpha_leg) in enumerate(leg_ordered, start=1):
            line_color = tracks[track_index]["color"] or (0.3, 0.3, 0.3)
            draw_manual_box_plot(ax, plot_index, alpha_leg, line_color)

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Alpha (relative speed)")
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"alpha-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            fig.savefig(output_path, bbox_inches="tight")


def plot_speed_delta_box_plot(tracks, speed_window_stats, way_point_progress, way_point_names, export_dir=None):
    """Box plots of speed delta from mean per leg, one figure per leg."""
    if "meanSpeedByWindow" not in speed_window_stats:
        return

    mean_speed_by_window = speed_window_stats["meanSpeedByWindow"]
    window_step_samples = speed_window_stats["windowStepSamples"]
    route_sample_count = speed_window_stats["routeSampleCount"]
    window_count = len(mean_speed_by_window)

    progress_values, waypoint_labels = normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

    track_progress_by_track = []
    delta_percent_by_track = []

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
            delta_percent_by_track.append(np.array([]))
            continue

        route_index_valid = route_index[valid_mask]
        progress = (route_index_valid - 1) / (route_sample_count - 1)
        home_window = np.floor((route_index_valid - 1) / window_step_samples) + 1
        home_window = np.clip(home_window, 1, window_count).astype(int)
        mean_speed_for_sample = mean_speed_by_window[home_window - 1]
        mean_speed_for_sample = np.where(
            np.isfinite(mean_speed_for_sample) & (mean_speed_for_sample > 0),
            mean_speed_for_sample,
            np.nan,
        )

        delta_percent = 100 * (speed[valid_mask] - mean_speed_for_sample) / mean_speed_for_sample
        delta_mask = np.isfinite(delta_percent)
        delta_percent = delta_percent[delta_mask]
        progress = progress[delta_mask]
        if delta_percent.size == 0:
            track_progress_by_track.append(np.array([]))
            delta_percent_by_track.append(np.array([]))
            continue

        track_progress_by_track.append(progress)
        delta_percent_by_track.append(delta_percent)

    track_count = len(tracks)
    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    for leg_index in range(leg_count - 1, -1, -1):
        fig, ax = plt.subplots()
        maximize_figure(fig)
        ax.grid(True)

        leg_start = progress_values[leg_index]
        leg_end = progress_values[leg_index + 1]
        if leg_start > leg_end:
            leg_start, leg_end = leg_end, leg_start

        leg_ordered = []
        for track_index, (progress, delta_percent) in enumerate(
            zip(track_progress_by_track, delta_percent_by_track)
        ):
            if progress.size == 0:
                continue
            leg_mask = (progress >= leg_start) & (progress <= leg_end)
            delta_leg = delta_percent[leg_mask]
            if delta_leg.size == 0:
                continue

            leg_ordered.append((track_index, float(np.mean(delta_leg)), delta_leg))

        if not leg_ordered:
            ax.set_title(leg_labels[leg_index])
            continue

        leg_ordered.sort(key=lambda item: item[1])
        for plot_index, (track_index, _, delta_leg) in enumerate(leg_ordered, start=1):
            line_color = tracks[track_index]["color"] or (0.3, 0.3, 0.3)
            draw_manual_box_plot(ax, plot_index, delta_leg, line_color)

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylabel("Speed delta from mean (%)")
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"speed-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            fig.savefig(output_path, bbox_inches="tight")


def plot_pace_delta_box_plot(tracks, speed_window_stats, way_point_progress, way_point_names, export_dir=None):
    """Box plots of pace difference per leg, one figure per leg."""
    if "meanSpeedByWindow" not in speed_window_stats:
        return

    mean_speed_by_window = speed_window_stats["meanSpeedByWindow"]
    window_step_samples = speed_window_stats["windowStepSamples"]
    route_sample_count = speed_window_stats["routeSampleCount"]
    window_count = len(mean_speed_by_window)

    mean_pace_by_window = speed_window_stats.get("meanPaceByWindow")
    if mean_pace_by_window is None or len(mean_pace_by_window) == 0:
        mean_pace_by_window = np.where(
            np.isfinite(mean_speed_by_window) & (mean_speed_by_window > 0),
            60.0 / mean_speed_by_window,
            np.nan,
        )

    progress_values, waypoint_labels = normalize_waypoint_inputs(way_point_progress, way_point_names)
    leg_count = len(progress_values) - 1
    if leg_count < 1 or not tracks:
        return

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

    track_count = len(tracks)
    leg_labels = []
    for leg_index in range(leg_count):
        if leg_index + 1 < len(waypoint_labels):
            leg_labels.append(f"{waypoint_labels[leg_index]}-{waypoint_labels[leg_index + 1]}")
        else:
            leg_labels.append(f"Leg {leg_index + 1}")

    for leg_index in range(leg_count - 1, -1, -1):
        fig, ax = plt.subplots()
        maximize_figure(fig)
        ax.grid(True)

        leg_start = progress_values[leg_index]
        leg_end = progress_values[leg_index + 1]
        if leg_start > leg_end:
            leg_start, leg_end = leg_end, leg_start

        leg_ordered = []
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
            draw_manual_box_plot(ax, plot_index, pace_leg, line_color)

        ax.set_xticks(range(1, len(leg_ordered) + 1))
        ax.set_xticklabels(
            [tracks[track_index]["name"] for track_index, _, _ in leg_ordered],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(0.5, len(leg_ordered) + 0.5)
        ax.set_ylabel("Pace delta from mean (min per NM)")
        ax.set_title(leg_labels[leg_index])
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            safe_label = sanitize_filename_label(leg_labels[leg_index])
            output_path = export_dir / f"pace-delta-leg-{leg_index + 1:02d}-{safe_label}.pdf"
            fig.savefig(output_path, bbox_inches="tight")

def plot_speed_range_along_route(tracks, speed_window_stats, export_path=None):
    """Debug plot of pace vs progress with min/max/mean/std envelope."""
    if not speed_window_stats or speed_window_stats.get("routeSampleCount", 0) < 2:
        return

    route_sample_count = speed_window_stats["routeSampleCount"]
    window_start_index = speed_window_stats["windowStartIndex"]
    window_end_index = speed_window_stats["windowEndIndex"]
    min_speed = speed_window_stats["minSpeedByWindow"]
    max_speed = speed_window_stats["maxSpeedByWindow"]
    mean_speed = speed_window_stats.get("meanSpeedByWindow", np.array([]))
    mean_pace = speed_window_stats.get("meanPaceByWindow", np.array([]))
    if mean_pace.size == 0:
        mean_pace = np.where(
            np.isfinite(mean_speed) & (mean_speed > 0),
            60.0 / mean_speed,
            np.nan,
        )
    min_pace = np.where(
        np.isfinite(max_speed) & (max_speed > 0),
        60.0 / max_speed,
        np.nan,
    )
    max_pace = np.where(
        np.isfinite(min_speed) & (min_speed > 0),
        60.0 / min_speed,
        np.nan,
    )

    window_center = (window_start_index + window_end_index) / 2
    window_progress = (window_center - 1) / (route_sample_count - 1)

    fig, ax = plt.subplots()
    maximize_figure(fig)
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
        pace = 60.0 / speed[valid_mask]
        line_color = track["color"] or default_color[idx]
        ax.plot(progress, pace, color=line_color, linewidth=0.5, label=track["name"])

    ax.plot(window_progress, min_pace, "b-", linewidth=1.5, label="Fastest pace (min)")
    ax.plot(window_progress, max_pace, "r-", linewidth=1.5, label="Slowest pace (max)")
    if mean_pace.size:
        ax.plot(window_progress, mean_pace, "k-", linewidth=1.5, label="Mean pace")

    ax.set_xlabel("Progress along average route")
    ax.set_ylabel("Pace (min per NM)")
    ax.set_title("Pace vs progress with min/max/mean envelope")
    ax.legend(loc="best")

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, bbox_inches="tight")


def plot_alpha_pdf_by_boat(tracks, bin_count, smoothing_window):
    """Plot a smooth PDF-style histogram of alpha per boat."""
    all_alpha = []
    for track in tracks:
        alpha_values = track["alpha"]
        alpha_values = alpha_values[np.isfinite(alpha_values)]
        all_alpha.extend(alpha_values.tolist())
    if not all_alpha:
        return

    bin_edges = np.linspace(0, 1, bin_count + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots()
    maximize_figure(fig)
    ax.grid(True)

    for track in tracks:
        alpha_values = track["alpha"]
        alpha_values = alpha_values[np.isfinite(alpha_values)]
        if alpha_values.size == 0:
            continue
        counts, _ = np.histogram(alpha_values, bins=bin_edges)
        if counts.sum() == 0:
            continue
        density = counts / (counts.sum() * bin_width)
        density = smooth_moving_average(density, smoothing_window)

        line_color = track["color"] or (0.5, 0.5, 0.5)
        ax.plot(bin_centers, density, color=line_color, linewidth=1.2, label=track["name"])

    ax.set_xlabel("Alpha (relative speed)")
    ax.set_ylabel("Density")
    ax.set_title("Alpha distribution per boat (PDF style)")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")


def plot_speed_histogram_by_boat(tracks, route_sample_count, bin_count, smoothing_window, progress_sample_count):
    """Plot PDF-style speed histograms by boat, resampled by progress."""
    if route_sample_count < 2:
        return

    if progress_sample_count < 2:
        progress_sample_count = min(2000, route_sample_count)

    progress_grid = np.linspace(0, 1, progress_sample_count)
    speed_samples_by_track = []
    all_speed_samples = []

    for track in tracks:
        route_index = track["routeIdx"]
        speed = track["speed"]
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(speed)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
        )
        if not valid_mask.any():
            speed_samples_by_track.append(np.array([]))
            continue

        progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
        speed_valid = speed[valid_mask]

        progress_unique, unique_index = np.unique(progress, return_index=True)
        speed_unique = speed_valid[unique_index]
        if progress_unique.size < 2:
            speed_samples_by_track.append(np.array([]))
            continue

        sort_index = np.argsort(progress_unique)
        progress_unique = progress_unique[sort_index]
        speed_unique = speed_unique[sort_index]

        speed_on_grid = np.interp(progress_grid, progress_unique, speed_unique)
        outside_mask = (progress_grid < progress_unique[0]) | (progress_grid > progress_unique[-1])
        speed_on_grid[outside_mask] = np.nan

        speed_samples_by_track.append(speed_on_grid)
        valid_samples = speed_on_grid[np.isfinite(speed_on_grid)]
        all_speed_samples.extend(valid_samples.tolist())

    if not all_speed_samples:
        return

    min_speed = min(all_speed_samples)
    max_speed = max(all_speed_samples)
    if min_speed == max_speed:
        min_speed -= 0.5
        max_speed += 0.5

    bin_edges = np.linspace(min_speed, max_speed, bin_count + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots()
    maximize_figure(fig)
    ax.grid(True)

    for track, speed_samples in zip(tracks, speed_samples_by_track):
        if speed_samples.size == 0:
            continue
        valid_mask = np.isfinite(speed_samples)
        if not valid_mask.any():
            continue
        counts, _ = np.histogram(speed_samples[valid_mask], bins=bin_edges)
        if counts.sum() == 0:
            continue
        density = counts / (counts.sum() * bin_width)
        density = smooth_moving_average(density, smoothing_window)

        line_color = track["color"] or (0.5, 0.5, 0.5)
        ax.plot(bin_centers, density, color=line_color, linewidth=1.2, label=track["name"])

    ax.set_xlabel("Speed (knots)")
    ax.set_ylabel("Density")
    ax.set_title("Speed distribution by boat (progress-sampled)")
    ax.legend(loc="best")


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
        line_strings = geojson_geometry_to_lines(geometry)
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


