"""
Plotting-focused helpers used by silver.py.

This module intentionally contains rendering/tick/colormap/map boilerplate so
the main pipeline file can focus on race-analysis logic and figure orchestration.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

import silver_helpers as sh


def normalize_waypoint_series(waypoint_progress, waypoint_names):
    """
    Normalize waypoint progress + label arrays so they stay index-aligned.

    Handles metadata shape mismatches by padding/truncating labels and removes
    non-finite progress entries together with their corresponding labels.
    """
    progress_values = np.asarray(waypoint_progress, dtype=float).flatten()
    label_values = list(waypoint_names) if waypoint_names is not None else []

    if label_values and progress_values.size:
        if len(label_values) < progress_values.size:
            label_values += [""] * (progress_values.size - len(label_values))
        if len(label_values) > progress_values.size:
            label_values = label_values[: progress_values.size]

    finite_mask = np.isfinite(progress_values)
    progress_values = progress_values[finite_mask]
    label_values = [label for label, keep in zip(label_values, finite_mask) if keep]
    return progress_values, label_values


def filter_waypoints_to_unit_interval(progress_values, label_values):
    """Filter waypoints to the visible progress domain [0, 1]."""
    in_range_mask = (progress_values >= 0) & (progress_values <= 1)
    return progress_values[in_range_mask], [
        label for label, keep in zip(label_values, in_range_mask) if keep
    ]


def plot_waypoints_on_route(
    ax,
    average_route,
    waypoint_progress,
    waypoint_names,
    waypoint_label_fontsize,
):
    """Draw waypoint markers + labels on the route map."""
    progress_values, label_values = normalize_waypoint_series(
        waypoint_progress, waypoint_names
    )
    progress_values, label_values = filter_waypoints_to_unit_interval(
        progress_values, label_values
    )
    if progress_values.size == 0:
        return

    route_count = len(average_route["lat"])
    if route_count == 0:
        return

    waypoint_indices = np.unique(
        np.clip(
            np.rint(progress_values * (route_count - 1)).astype(int),
            0,
            route_count - 1,
        )
    )
    waypoint_lon = np.asarray(average_route["lon"])[waypoint_indices]
    waypoint_lat = np.asarray(average_route["lat"])[waypoint_indices]
    ax.scatter(waypoint_lon, waypoint_lat, s=90, color="black", zorder=5)

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
            fontsize=waypoint_label_fontsize,
            color="black",
            zorder=6,
        )


def apply_waypoint_ticks(ax, waypoint_progress, waypoint_names):
    """
    Configure progress-axis ticks using waypoint labels plus Start/Finish.

    Guarantees:
    - Start and Finish always present.
    - Duplicate progress positions merged.
    - Labels TeX-escaped when text.usetex is enabled.
    """
    progress_values, label_values = normalize_waypoint_series(
        waypoint_progress, waypoint_names
    )
    if progress_values.size == 0:
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Start", "Finish"], rotation=45, ha="right")
        return

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

    filtered_progress.extend([0.0, 1.0])
    filtered_labels.extend(["Start", "Finish"])

    progress_values = np.asarray(filtered_progress, dtype=float)
    label_values = [filtered_labels[idx] for idx in np.argsort(progress_values)]
    progress_values = np.sort(progress_values)

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


def resample_track_speed(track, window_progress, route_sample_count, filter_alpha):
    """Resample one boat speed series onto the window progress grid."""
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

    progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
    speed_valid = speed[valid_mask]
    progress_unique, unique_index = np.unique(progress, return_index=True)
    speed_unique = speed_valid[unique_index]
    if progress_unique.size < 2:
        return None

    sort_index = np.argsort(progress_unique)
    progress_unique = progress_unique[sort_index]
    speed_unique = speed_unique[sort_index]

    speed_on_grid = np.interp(window_progress, progress_unique, speed_unique)
    outside_mask = (window_progress < progress_unique[0]) | (
        window_progress > progress_unique[-1]
    )
    speed_on_grid[outside_mask] = np.nan
    return sh.lowpass_forward_backward(speed_on_grid, filter_alpha)


def build_alpha_colormap():
    """Build the colormap used for alpha-colored map line segments."""
    base_cmap = plt.get_cmap("hot", 256)
    cmap_colors = base_cmap(np.linspace(0, 229 / 255, 230))
    cmap_colors[0, :] = np.array([0, 0, 0, 1])
    cmap = LinearSegmentedColormap.from_list("hot_trim", cmap_colors)
    norm = Normalize(0, 1)
    return cmap, norm


def add_alpha_tracks(ax, tracks, cmap, norm):
    """Draw one per-segment-colored LineCollection per track."""
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

    return line_collections


def plot_coast_geojson_cropped(ax, tracks, geojson_file, margin_deg):
    """Draw coastline geometry cropped to track bounding box."""
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
            inside = (
                (lat >= min_lat)
                & (lat <= max_lat)
                & (lon >= min_lon)
                & (lon <= max_lon)
            )
            lon = lon.astype(float)
            lat = lat.astype(float)
            lon[~inside] = np.nan
            lat[~inside] = np.nan
            ax.plot(lon, lat, color=(0.35, 0.35, 0.35), linewidth=1.0)
