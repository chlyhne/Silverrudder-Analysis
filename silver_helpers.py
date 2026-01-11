import csv
import json
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


def read_tracking_csv_as_struct(filename):
    """Read tracking CSV into a dict-of-columns."""
    data = {}
    # Read into column-oriented storage so later filtering is vectorized and fast.
    with open(filename, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return data
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            # Preserve raw strings here; parsing happens in the typed converters.
            for name in data:
                data[name].append(row.get(name, ""))
    return data


def build_tracks(tracking_data, id_field, lat_field, lon_field, time_field, speed_field):
    """Group rows by id into per-track dicts, time-sorted."""
    track_id = to_string_column(tracking_data.get(id_field, []))
    latitude = to_double_column(tracking_data.get(lat_field, []))
    longitude = to_double_column(tracking_data.get(lon_field, []))
    sample_time = to_time_column(tracking_data.get(time_field, []))
    speed_value = to_double_column(tracking_data.get(speed_field, []))

    # Filter out rows with missing identifiers or invalid coordinates/timestamps.
    valid_mask = (
        (track_id != "")
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & np.isfinite(sample_time)
        & np.isfinite(speed_value)
    )

    track_id = track_id[valid_mask]
    latitude = latitude[valid_mask]
    longitude = longitude[valid_mask]
    sample_time = sample_time[valid_mask]
    speed_value = speed_value[valid_mask]

    # Preserve the input ordering of track IDs to keep plots stable across runs.
    unique_ids = []
    seen = set()
    for value in track_id:
        if value not in seen:
            unique_ids.append(value)
            seen.add(value)

    tracks = []
    for track_value in unique_ids:
        # Each track dict becomes the shared schema for downstream analysis.
        track_mask = track_id == track_value
        track_time = sample_time[track_mask]
        sort_index = np.argsort(track_time)
        track = {
            "id": track_value,
            "name": track_value,
            "color": None,
            "lat": latitude[track_mask][sort_index],
            "lon": longitude[track_mask][sort_index],
            "t": track_time[sort_index],
            "speed": speed_value[track_mask][sort_index],
            "routeIdx": np.array([], dtype=int),
            "routeDist": np.array([], dtype=float),
            "alpha": np.array([], dtype=float),
        }
        tracks.append(track)
    return tracks


def convert_speed_to_knots(tracks):
    """Convert speed fields from km/h to knots."""
    # Normalize units once so all downstream stats use consistent nautical units.
    kmh_to_knots = 0.539956803
    for track in tracks:
        track["speed"] = track["speed"] * kmh_to_knots
    return tracks


def apply_track_names(tracks, track_name_map):
    """Set track name using mapping keyed by tracked_object_id."""
    # Human-readable names improve legends and exports without changing IDs.
    for track in tracks:
        if track["id"] in track_name_map:
            track["name"] = track_name_map[track["id"]]
    return tracks


def apply_track_colors(tracks, track_color_map):
    """Set track color using mapping keyed by boat name."""
    # Colors are attached once so every plot can reuse them consistently.
    for track in tracks:
        if track["name"] in track_color_map:
            track["color"] = track_color_map[track["name"]]
    return tracks


def load_race_metadata(metadata_path):
    """Load race metadata from JSON."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    # Metadata drives naming, colors, and timing offsets in one place.
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_local_time_to_utc(time_text, local_offset_hours):
    """Convert a local time string (dd/mm/YYYY HH:MM:SS) to UTC timestamp."""
    if not time_text:
        return None
    try:
        # Metadata stores local time; convert once so elapsed times are accurate.
        local_tz = timezone(timedelta(hours=local_offset_hours))
        local_time = datetime.strptime(time_text, "%d/%m/%Y %H:%M:%S").replace(tzinfo=local_tz)
    except ValueError:
        return None
    return local_time.astimezone(timezone.utc).timestamp()


def build_boat_color_map(boat_names, boat_colors):
    """Normalize boat color mapping from metadata."""
    # Accept either explicit name->color dictionaries or ordered color lists.
    if isinstance(boat_colors, dict):
        return {name: tuple(color) for name, color in boat_colors.items()}
    if isinstance(boat_colors, list):
        return {name: tuple(color) for name, color in zip(boat_names, boat_colors)}
    return {}


def map_track_points_to_route(tracks, average_route, route_search_window_half_width):
    """Assign each track point to nearest route index."""
    route_lat = np.asarray(average_route["lat"])
    route_lon = np.asarray(average_route["lon"])
    route_sample_count = len(route_lat)

    # Windowed search limits large jumps when the nearest neighbor is ambiguous.
    use_search_window = route_search_window_half_width > 0
    # Detect looped routes so progress can wrap without being forced to monotonic.
    route_endpoint_distance = haversine_meters(
        route_lat[0], route_lon[0], route_lat[-1], route_lon[-1]
    )
    route_is_loop = route_endpoint_distance < 500

    # Time-derived progress keeps early samples from snapping to distant route points.
    time_alignment_threshold = max(10, round(0.35 * route_sample_count))

    for track in tracks:
        latitude = track["lat"]
        longitude = track["lon"]
        sample_time = track["t"]

        expected_route_index_by_time = np.full_like(sample_time, np.nan, dtype=float)
        if sample_time.size > 0 and np.isfinite(sample_time).all():
            track_start_time = sample_time[0]
            track_end_time = sample_time[-1]
            if track_end_time > track_start_time:
                # Map elapsed time onto route index as a gentle prior.
                time_progress = (sample_time - track_start_time) / (
                    track_end_time - track_start_time
                )
                expected_route_index_by_time = np.round(
                    time_progress * (route_sample_count - 1)
                ) + 1

        route_index_by_sample = np.zeros(latitude.size, dtype=int)
        route_distance_by_sample = np.full(latitude.size, np.nan, dtype=float)

        if latitude.size > 0:
            # Initialize with the closest route point to the first sample.
            distance_to_route = haversine_meters(
                latitude[0], longitude[0], route_lat, route_lon
            )
            best_index_zero = int(np.nanargmin(distance_to_route))
            route_index_by_sample[0] = best_index_zero + 1
            route_distance_by_sample[0] = distance_to_route[best_index_zero]
            previous_route_index = route_index_by_sample[0]
            start_index = 1
        else:
            previous_route_index = 1
            start_index = 0

        for sample_index in range(start_index, latitude.size):
            if use_search_window:
                search_start = max(1, previous_route_index - route_search_window_half_width)
                search_end = min(route_sample_count, previous_route_index + route_search_window_half_width)
            else:
                search_start = 1
                search_end = route_sample_count

            start_zero = search_start - 1
            end_zero = search_end

            distance_to_route = haversine_meters(
                latitude[sample_index],
                longitude[sample_index],
                route_lat[start_zero:end_zero],
                route_lon[start_zero:end_zero],
            )
            local_index = int(np.nanargmin(distance_to_route))
            best_route_index = search_start + local_index
            min_distance = distance_to_route[local_index]

            expected_index = expected_route_index_by_time[sample_index]
            if np.isfinite(expected_index) and abs(best_route_index - expected_index) > time_alignment_threshold:
                # If the closest point is far from the time-based estimate, prefer the estimate.
                best_route_index = int(expected_index)
            else:
                # Enforce non-decreasing progress unless we are wrapping around a loop.
                if best_route_index < previous_route_index:
                    if route_is_loop and previous_route_index > 0.9 * route_sample_count and \
                            best_route_index < 0.1 * route_sample_count:
                        pass
                    else:
                        best_route_index = previous_route_index

            best_route_index = max(1, min(route_sample_count, best_route_index))
            route_index_by_sample[sample_index] = best_route_index
            route_distance_by_sample[sample_index] = min_distance
            previous_route_index = best_route_index

        track["routeIdx"] = route_index_by_sample
        track["routeDist"] = route_distance_by_sample

    return tracks


def remove_route_index_spikes(tracks, route_sample_count):
    """Fix isolated route-index spikes that jump to the wrong side."""
    if route_sample_count < 3 or not tracks:
        return tracks

    # Spikes usually come from a single bad nearest-neighbor match in noisy data.
    spike_threshold = max(50, round(0.02 * route_sample_count))

    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        if route_index.size < 3:
            continue

        step_sizes = np.abs(np.diff(route_index))
        typical_step = np.nanmedian(step_sizes) if np.isfinite(step_sizes).any() else 1
        if not np.isfinite(typical_step) or typical_step < 1:
            typical_step = 1
        # Use a track-specific threshold so slow and fast boats are treated fairly.
        spike_threshold_for_track = max(spike_threshold, round(10 * typical_step))

        # Two passes catch single-point spikes without over-smoothing real jumps.
        for _ in range(2):
            previous_index = route_index[:-2]
            current_index = route_index[1:-1]
            next_index = route_index[2:]

            spike_mask = (
                (np.abs(current_index - previous_index) > spike_threshold_for_track)
                & (np.abs(current_index - next_index) > spike_threshold_for_track)
                & (np.abs(previous_index - next_index) <= spike_threshold_for_track)
            )

            if spike_mask.any():
                # Replace spikes with the midpoint of neighboring indices.
                corrected_index = np.round(
                    (previous_index[spike_mask] + next_index[spike_mask]) / 2
                )
                route_index[1:-1] = current_index
                spike_positions = np.where(spike_mask)[0] + 1
                route_index[spike_positions] = corrected_index
            else:
                break

        track["routeIdx"] = route_index.astype(int)
    return tracks


def compute_sample_alpha_by_route_windows(
    tracks, route_sample_count, window_sample_count, window_step_samples, filter_alpha
):
    """Compute per-sample alpha using min/max windows along the route."""
    if route_sample_count < 2:
        return tracks, {}

    # Windows define a fleet-wide baseline along the route for normalization.
    window_count = int(math.floor((route_sample_count - window_sample_count) / window_step_samples) + 1)
    if window_count < 1:
        window_count = 1

    window_start_index = np.arange(window_count) * window_step_samples + 1
    window_end_index = window_start_index + window_sample_count - 1
    window_end_index = np.minimum(window_end_index, route_sample_count)
    window_center_index = (window_start_index + window_end_index) / 2
    window_progress = (window_center_index - 1) / (route_sample_count - 1)

    # Collect each boat's speed on the same progress grid to compute fleet stats.
    speed_on_grid_by_track = []

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
            continue

        progress = (route_index[valid_mask] - 1) / (route_sample_count - 1)
        speed_valid = speed[valid_mask]

        # De-duplicate progress values to keep interpolation well-behaved.
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
        speed_on_grid_by_track.append(speed_on_grid)

    min_speed_by_window = np.full(window_count, np.nan)
    max_speed_by_window = np.full(window_count, np.nan)
    mean_speed_by_window = np.full(window_count, np.nan)
    mean_pace_by_window = np.full(window_count, np.nan)

    if speed_on_grid_by_track:
        # Fleet stats are computed per window over all available boats.
        speed_grid = np.vstack(speed_on_grid_by_track)
        finite_counts = np.sum(np.isfinite(speed_grid), axis=0)
        valid_windows = finite_counts > 0
        if valid_windows.any():
            min_speed_by_window[valid_windows] = np.nanmin(speed_grid[:, valid_windows], axis=0)
            max_speed_by_window[valid_windows] = np.nanmax(speed_grid[:, valid_windows], axis=0)
            mean_speed_by_window[valid_windows] = np.nanmean(speed_grid[:, valid_windows], axis=0)

        for window_index in range(window_count):
            speeds_here = speed_grid[:, window_index]
            speeds_here = speeds_here[np.isfinite(speeds_here) & (speeds_here > 0)]
            if speeds_here.size:
                # Mean pace is derived from speed to avoid bias from low speeds.
                mean_pace_by_window[window_index] = np.mean(60.0 / speeds_here)

    # Smooth the windowed stats to reduce jaggedness from sparse data.
    min_speed_by_window = lowpass_forward_backward(min_speed_by_window, filter_alpha)
    max_speed_by_window = lowpass_forward_backward(max_speed_by_window, filter_alpha)
    mean_speed_by_window = lowpass_forward_backward(mean_speed_by_window, filter_alpha)
    mean_pace_by_window = lowpass_forward_backward(mean_pace_by_window, filter_alpha)

    sum_squared_error = np.zeros(window_count)
    count_squared_error = np.zeros(window_count)

    for speed_on_grid in speed_on_grid_by_track:
        # Use smoothed mean speed as the baseline for per-window variability.
        squared_error = (speed_on_grid - mean_speed_by_window) ** 2
        filtered_squared_error = lowpass_forward_backward(squared_error, filter_alpha)
        finite_mask = np.isfinite(filtered_squared_error)
        sum_squared_error[finite_mask] += filtered_squared_error[finite_mask]
        count_squared_error[finite_mask] += 1

    std_speed_by_window = np.full(window_count, np.nan)
    valid_std_mask = count_squared_error > 0
    std_speed_by_window[valid_std_mask] = np.sqrt(
        sum_squared_error[valid_std_mask] / count_squared_error[valid_std_mask]
    )

    speed_window_stats = {
        "minSpeedByWindow": min_speed_by_window,
        "maxSpeedByWindow": max_speed_by_window,
        "meanSpeedByWindow": mean_speed_by_window,
        "meanPaceByWindow": mean_pace_by_window,
        "stdSpeedByWindow": std_speed_by_window,
        "windowStartIndex": window_start_index,
        "windowEndIndex": window_end_index,
        "windowSampleCount": window_sample_count,
        "windowStepSamples": window_step_samples,
        "routeSampleCount": route_sample_count,
        "windowProgress": window_progress,
        "filterAlpha": filter_alpha,
    }

    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        speed = track["speed"].astype(float)
        alpha = np.full_like(speed, np.nan, dtype=float)
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(speed)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
        )
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            route_index_valid = route_index[valid_indices]
            progress = (route_index_valid - 1) / (route_sample_count - 1)

            min_mask = np.isfinite(window_progress) & np.isfinite(min_speed_by_window)
            max_mask = np.isfinite(window_progress) & np.isfinite(max_speed_by_window)

            if np.count_nonzero(min_mask) >= 2 and np.count_nonzero(max_mask) >= 2:
                # Interpolate min/max envelopes onto per-sample progress positions.
                min_speed = np.interp(
                    progress,
                    window_progress[min_mask],
                    min_speed_by_window[min_mask],
                    left=np.nan,
                    right=np.nan,
                )
                max_speed = np.interp(
                    progress,
                    window_progress[max_mask],
                    max_speed_by_window[max_mask],
                    left=np.nan,
                    right=np.nan,
                )
            else:
                min_speed = np.full_like(progress, np.nan, dtype=float)
                max_speed = np.full_like(progress, np.nan, dtype=float)

            denom = max_speed - min_speed
            alpha_values = (speed[valid_indices] - min_speed) / denom
            invalid_alpha = ~np.isfinite(alpha_values) | (denom <= 0)
            # Neutral gray when the normalization is not meaningful.
            alpha_values[invalid_alpha] = 0.5
            alpha[valid_indices] = alpha_values

        track["alpha"] = alpha

    return tracks, speed_window_stats


def lowpass_forward_backward(signal, filter_alpha):
    """First-order low-pass forward and backward."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal

    if not np.isfinite(filter_alpha) or filter_alpha <= 0:
        return signal
    if filter_alpha > 1:
        filter_alpha = 1

    # Fill gaps so the filter does not propagate NaNs through the series.
    finite_mask = np.isfinite(signal)
    if not finite_mask.any():
        return signal

    filled = signal.copy()
    if (~finite_mask).any():
        indices = np.arange(signal.size)
        finite_indices = indices[finite_mask]
        filled[~finite_mask] = np.interp(
            indices[~finite_mask],
            finite_indices,
            signal[finite_mask],
            left=signal[finite_mask][0],
            right=signal[finite_mask][-1],
        )

    forward = filled.copy()
    for idx in range(1, forward.size):
        forward[idx] = filter_alpha * filled[idx] + (1 - filter_alpha) * forward[idx - 1]

    backward = forward.copy()
    for idx in range(backward.size - 2, -1, -1):
        backward[idx] = filter_alpha * forward[idx] + (1 - filter_alpha) * backward[idx + 1]

    # Restore NaNs so downstream logic can respect missing data.
    backward[~finite_mask] = np.nan
    return backward


def maximize_figure(fig):
    """Best-effort maximize across backends."""
    # Matplotlib backends expose different window APIs; try the common ones quietly.
    try:
        manager = fig.canvas.manager
        if manager is None:
            return
        if hasattr(manager, "window"):
            window = manager.window
            if hasattr(window, "state"):
                try:
                    window.state("zoomed")
                    return
                except Exception:
                    pass
            if hasattr(window, "showMaximized"):
                try:
                    window.showMaximized()
                    return
                except Exception:
                    pass
        if hasattr(manager, "full_screen_toggle"):
            try:
                manager.full_screen_toggle()
            except Exception:
                pass
    except Exception:
        pass


def apply_local_meter_aspect(ax, average_route):
    """Force square meters on the map using mean latitude of the rhumb line."""
    route_lat = np.asarray(average_route.get("lat", []), dtype=float)
    if route_lat.size == 0:
        ax.set_aspect("equal", adjustable="box")
        return

    # Longitude degrees shrink with latitude; scale so meters look square.
    mean_lat = float(np.nanmean(route_lat))
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(mean_lat))
    if meters_per_degree_lon <= 0 or not np.isfinite(meters_per_degree_lon):
        ax.set_aspect("equal", adjustable="box")
        return

    data_ratio = meters_per_degree_lat / meters_per_degree_lon
    ax.set_aspect(data_ratio, adjustable="box")


def enable_manual_datatips(ax, tracks, average_route, speed_window_stats, pick_radius_m=250.0):
    """Fallback click-tooltips when mplcursors is unreliable."""
    # Use a lightweight picker so interactive inspection works without extra dependencies.
    route_lon = np.asarray(average_route.get("lon", []), dtype=float)
    route_lat = np.asarray(average_route.get("lat", []), dtype=float)
    if route_lon.size == 0 or route_lat.size == 0:
        return

    earth_radius = 6371000.0
    meters_per_degree = math.pi / 180.0 * earth_radius
    reference_lat = float(np.nanmedian(route_lat))
    cos_lat = math.cos(math.radians(reference_lat))

    min_speed = speed_window_stats.get("minSpeedByWindow", np.array([]))
    max_speed = speed_window_stats.get("maxSpeedByWindow", np.array([]))
    window_progress = speed_window_stats.get("windowProgress", np.array([]))

    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(8, 8),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    annotation.set_usetex(False)
    annotation.set_visible(False)

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        click_lon = float(event.xdata)
        click_lat = float(event.ydata)

        best_track_dist2 = np.inf
        best_track_name = None
        best_track_speed = None
        best_track_point = None

        for track in tracks:
            longitude = track["lon"]
            latitude = track["lat"]
            speed = track["speed"]
            valid_mask = np.isfinite(longitude) & np.isfinite(latitude) & np.isfinite(speed)
            if not valid_mask.any():
                continue
            longitude = longitude[valid_mask]
            latitude = latitude[valid_mask]
            speed = speed[valid_mask]

            # Compute distance in meters to find the nearest sample in this track.
            dx = (longitude - click_lon) * cos_lat * meters_per_degree
            dy = (latitude - click_lat) * meters_per_degree
            dist2 = dx * dx + dy * dy
            idx = int(np.nanargmin(dist2))
            if dist2[idx] < best_track_dist2:
                best_track_dist2 = dist2[idx]
                best_track_name = track["name"]
                best_track_speed = speed[idx]
                best_track_point = (longitude[idx], latitude[idx])

        dx_route = (route_lon - click_lon) * cos_lat * meters_per_degree
        dy_route = (route_lat - click_lat) * meters_per_degree
        route_dist2 = dx_route * dx_route + dy_route * dy_route
        route_index = int(np.nanargmin(route_dist2)) if route_dist2.size else None
        best_route_dist2 = route_dist2[route_index] if route_index is not None else np.inf

        radius2 = pick_radius_m * pick_radius_m
        if best_track_dist2 <= best_route_dist2 and best_track_dist2 <= radius2:
            # Prefer boat info when the click is closer to a track than the route.
            speed_text = "N/A"
            if best_track_speed is not None and np.isfinite(best_track_speed):
                speed_text = f"{best_track_speed:.2f}"
            annotation.xy = best_track_point
            annotation.set_text(f"Boat: {best_track_name}\nSpeed: {speed_text}")
            annotation.set_visible(True)
        elif best_route_dist2 <= radius2 and route_index is not None:
            # Otherwise, display baseline min/max around the nearest route point.
            if route_lon.size > 1:
                progress = route_index / (route_lon.size - 1)
            else:
                progress = 0.0
            if window_progress.size >= 2:
                min_mask = np.isfinite(window_progress) & np.isfinite(min_speed)
                max_mask = np.isfinite(window_progress) & np.isfinite(max_speed)
                if np.count_nonzero(min_mask) >= 2:
                    min_value = np.interp(
                        progress,
                        window_progress[min_mask],
                        min_speed[min_mask],
                        left=np.nan,
                        right=np.nan,
                    )
                else:
                    min_value = np.nan
                if np.count_nonzero(max_mask) >= 2:
                    max_value = np.interp(
                        progress,
                        window_progress[max_mask],
                        max_speed[max_mask],
                        left=np.nan,
                        right=np.nan,
                    )
                else:
                    max_value = np.nan
                min_text = f"{min_value:.2f}" if np.isfinite(min_value) else "N/A"
                max_text = f"{max_value:.2f}" if np.isfinite(max_value) else "N/A"
            else:
                min_text = "N/A"
                max_text = "N/A"
            annotation.xy = (route_lon[route_index], route_lat[route_index])
            annotation.set_text(
                f"Boat: Rhumb line\nProgress: {progress:.4f}\n"
                f"Min speed: {min_text}\nMax speed: {max_text}"
            )
            annotation.set_visible(True)
        else:
            annotation.set_visible(False)

        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("button_press_event", on_click)


def normalize_waypoint_inputs(way_point_progress, way_point_names):
    """Normalize waypoint progress/labels for plotting."""
    progress_values = np.asarray(way_point_progress, dtype=float).flatten()
    if progress_values.size == 0:
        return progress_values, []

    # Provide fallback labels so plots remain readable without metadata names.
    waypoint_labels = [f"WP{i+1}" for i in range(len(progress_values))]
    if way_point_names:
        waypoint_labels = list(way_point_names)
        if len(waypoint_labels) < len(progress_values):
            waypoint_labels += [
                f"WP{i+1}" for i in range(len(waypoint_labels), len(progress_values))
            ]
    return progress_values, waypoint_labels


def sanitize_filename_label(label):
    """Make a label safe for filenames (ASCII, LaTeX-friendly)."""
    # Transliterate common non-ASCII characters to keep filenames portable.
    text = str(label).strip().lower()
    text = text.replace("æ", "ae").replace("ø", "o").replace("å", "aa")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "leg"


def escape_latex_text(text):
    """Escape characters that are special in LaTeX text."""
    if text is None:
        return ""
    # Escape at the string level to keep LaTeX includes robust.
    escaped = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)
    return escaped


def haversine_meters(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters). Supports scalar/vector mixing."""
    # Haversine is stable for short distances and vectorizes well with numpy.
    earth_radius = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


def to_string_column(values):
    """Normalize ids to a string numpy array."""
    # Preserve empty strings instead of NaN to keep id comparisons simple.
    return np.array([str(value).strip() if value is not None else "" for value in values], dtype=object)


def to_double_column(values):
    """Convert to float numpy array with NaNs for invalid entries."""
    output = []
    # Treat empty strings and invalid values as NaN so filters can drop them.
    for value in values:
        if value is None:
            output.append(np.nan)
            continue
        text = str(value).strip()
        if text == "":
            output.append(np.nan)
            continue
        try:
            output.append(float(text))
        except ValueError:
            output.append(np.nan)
    return np.array(output, dtype=float)


def to_time_column(values):
    """Convert time strings to POSIX seconds (UTC)."""
    # Keep conversion in one place so downstream code stays time-zone agnostic.
    output = []
    for value in values:
        output.append(parse_utc_time(value))
    return np.array(output, dtype=float)


def parse_utc_time(value):
    """Parse time string like 'YYYY-MM-DD HH:MM:SS UTC' into seconds."""
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        # Fast path for the known telemetry format.
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        pass
    try:
        # Fall back to ISO-like strings with optional timezone suffixes.
        cleaned = text.replace(" UTC", "").replace("Z", "")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return np.nan


def load_geo_data(geo_data_path):
    """Load rhumb line coordinates and waypoint gates from GeoJSON."""
    geo_data_path = Path(geo_data_path)
    if not geo_data_path.exists():
        raise FileNotFoundError(f"Geo data file not found: {geo_data_path}")

    # GeoJSON is expected to contain a rhumb line plus gate line features.
    geo_data = json.loads(geo_data_path.read_text(encoding="utf-8"))
    features = geo_data.get("features", []) if isinstance(geo_data, dict) else []

    route_coords = None
    waypoint_gates = []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        name = str(properties.get("name", "")).strip()
        feature_type = str(properties.get("type", "")).strip().lower()

        if feature_type == "waypoint":
            # Gates are stored as small line segments that intersect the rhumb line.
            gate_coords = _normalize_gate_coords(coordinates)
            gate_index = _parse_gate_index(properties)
            waypoint_gates.append(
                {"name": name or f"Waypoint {gate_index}", "index": gate_index, "coords": gate_coords}
            )
            continue

        if geometry_type == "LineString":
            # Prefer the explicitly named rhumb line, fall back to the first line string.
            if route_coords is None or name.lower().startswith("rhumb"):
                route_coords = [list(point) for point in coordinates]

    if route_coords is None or len(route_coords) < 2:
        raise ValueError("Rhumb line coordinates not found in geo data.")

    waypoint_gates.sort(key=lambda gate: gate["index"])
    return route_coords, waypoint_gates


def compute_average_route(route_sample_count, geo_data_path):
    """Resample the supplied rhumb-line GeoJSON into a high-resolution route."""
    route_coords, _ = load_geo_data(geo_data_path)
    longitude = np.array([point[0] for point in route_coords], dtype=float)
    latitude = np.array([point[1] for point in route_coords], dtype=float)
    valid_mask = np.isfinite(latitude) & np.isfinite(longitude)
    latitude = latitude[valid_mask]
    longitude = longitude[valid_mask]
    if latitude.size < 2:
        raise ValueError("Rhumb line has insufficient valid coordinates.")

    # Collapse duplicate distances to keep interpolation monotonic.
    distance_along = cumulative_distance_meters(latitude, longitude)
    keep_mask = np.concatenate([[True], np.diff(distance_along) > 0])
    latitude = latitude[keep_mask]
    longitude = longitude[keep_mask]
    distance_along = distance_along[keep_mask]
    if distance_along.size < 2 or distance_along[-1] <= 0:
        raise ValueError("Rhumb line distance is not sufficient for resampling.")

    # Resample to a uniform distance grid so progress is linear in meters.
    total_distance = distance_along[-1]
    sample_distance = np.linspace(0, total_distance, route_sample_count)
    resampled_lat = np.interp(sample_distance, distance_along, latitude)
    resampled_lon = np.interp(sample_distance, distance_along, longitude)
    route_fraction = sample_distance / total_distance

    return {"lat": resampled_lat, "lon": resampled_lon, "s": route_fraction}


def _normalize_gate_coords(coordinates):
    """Normalize gate coordinates into a list of two [lon, lat] points."""
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError("Waypoint gate must have two coordinate points.")
    gate_coords = [list(point) for point in coordinates]
    if len(gate_coords) < 2:
        raise ValueError("Waypoint gate must have two coordinate points.")
    # Only the first two points define the gate line segment.
    return gate_coords[:2]


def _parse_gate_index(properties):
    """Extract integer index from waypoint properties."""
    if "index" in properties:
        try:
            return int(properties["index"])
        except (TypeError, ValueError):
            pass
    # Fall back to keys like "index 3" when metadata is inconsistent.
    for key in properties.keys():
        match = re.match(r"^index\\s*(\\d+)$", str(key).strip())
        if match:
            return int(match.group(1))
    raise ValueError(f"Waypoint is missing a valid index: {properties}")


def _segment_intersection_fraction(p_start, p_end, q_start, q_end):
    """Return fractional position along p segment where it intersects q segment."""
    # Segment intersection is used for gate crossings and waypoint projection.
    px, py = p_start
    rx, ry = (p_end[0] - px, p_end[1] - py)
    qx, qy = q_start
    sx, sy = (q_end[0] - qx, q_end[1] - qy)

    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return None

    qmpx = qx - px
    qmpy = qy - py
    t = (qmpx * sy - qmpy * sx) / denom
    u = (qmpx * ry - qmpy * rx) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return None


def compute_gate_crossings(tracks, waypoint_gates):
    """Compute first crossing time for each gate per track, in gate order."""
    gate_times_by_track = []
    for track in tracks:
        longitude = np.asarray(track["lon"], dtype=float)
        latitude = np.asarray(track["lat"], dtype=float)
        time_values = np.asarray(track["t"], dtype=float)
        # Ignore samples without valid coordinates or timestamps.
        valid_mask = np.isfinite(longitude) & np.isfinite(latitude) & np.isfinite(time_values)
        longitude = longitude[valid_mask]
        latitude = latitude[valid_mask]
        time_values = time_values[valid_mask]

        gate_times = np.full(len(waypoint_gates), np.nan, dtype=float)
        if longitude.size < 2:
            gate_times_by_track.append(gate_times)
            continue

        # Track crossings in order so each gate uses the earliest valid segment.
        start_index = 0
        for gate_idx, gate in enumerate(waypoint_gates):
            gate_coords = gate["coords"]
            gate_start = gate_coords[0]
            gate_end = gate_coords[1]

            crossing_time = np.nan
            for sample_index in range(start_index, longitude.size - 1):
                # Check segment-by-segment intersection with the gate line.
                p_start = (longitude[sample_index], latitude[sample_index])
                p_end = (longitude[sample_index + 1], latitude[sample_index + 1])
                fraction = _segment_intersection_fraction(p_start, p_end, gate_start, gate_end)
                if fraction is None:
                    continue
                crossing_time = time_values[sample_index] + fraction * (
                    time_values[sample_index + 1] - time_values[sample_index]
                )
                # Move forward so later gates cannot match earlier segments.
                start_index = sample_index + 1
                break

            gate_times[gate_idx] = crossing_time

        gate_times_by_track.append(gate_times)
        track["gateTimes"] = gate_times

    return gate_times_by_track


def trim_tracks_by_gate_times(tracks, gate_times_by_track, start_gate_pos, finish_gate_pos):
    """Trim tracks to the interval between start/finish gate crossings."""
    trimmed_tracks = []
    for track, gate_times in zip(tracks, gate_times_by_track):
        if gate_times.size == 0:
            continue
        start_time = gate_times[start_gate_pos]
        finish_time = gate_times[finish_gate_pos]
        if not (np.isfinite(start_time) and np.isfinite(finish_time)):
            raise ValueError(f"Missing start/finish crossing for boat {track.get('name', 'Unknown')}.")
        if finish_time < start_time:
            raise ValueError(f"Finish precedes start for boat {track.get('name', 'Unknown')}.")

        # Trim all per-sample arrays to the shared race window.
        time_values = track["t"]
        latitude = track["lat"]
        longitude = track["lon"]
        speed_values = track["speed"]
        keep_mask = (time_values >= start_time) & (time_values <= finish_time)
        track["t"] = time_values[keep_mask]
        track["lat"] = latitude[keep_mask]
        track["lon"] = longitude[keep_mask]
        track["speed"] = speed_values[keep_mask]
        track["routeIdx"] = np.array([], dtype=int)
        track["routeDist"] = np.array([], dtype=float)
        track["alpha"] = np.array([], dtype=float)
        trimmed_tracks.append(track)
    return trimmed_tracks


def compute_waypoint_progress_from_gates(average_route, waypoint_gates):
    """Project waypoint gate lines onto the rhumb line to get progress values."""
    route_lat = np.asarray(average_route["lat"], dtype=float)
    route_lon = np.asarray(average_route["lon"], dtype=float)
    if route_lat.size < 2:
        raise ValueError("Rhumb line has insufficient samples for waypoint projection.")

    cumulative_distance = cumulative_distance_meters(route_lat, route_lon)
    total_distance = cumulative_distance[-1]
    if not np.isfinite(total_distance) or total_distance <= 0:
        raise ValueError("Rhumb line distance is invalid for waypoint projection.")

    # Gate projection uses line intersections with the rhumb line segments.
    progress_values = []
    waypoint_names = []
    for gate in waypoint_gates:
        gate_coords = gate["coords"]
        gate_start = gate_coords[0]
        gate_end = gate_coords[1]
        intersection_progress = None

        for idx in range(route_lat.size - 1):
            p_start = (route_lon[idx], route_lat[idx])
            p_end = (route_lon[idx + 1], route_lat[idx + 1])
            fraction = _segment_intersection_fraction(p_start, p_end, gate_start, gate_end)
            if fraction is None:
                continue
            segment_length = cumulative_distance[idx + 1] - cumulative_distance[idx]
            distance_at_intersection = cumulative_distance[idx] + fraction * segment_length
            intersection_progress = distance_at_intersection / total_distance
            break

        if intersection_progress is None:
            raise ValueError(f"Waypoint gate '{gate.get('name', '')}' does not intersect the rhumb line.")

        progress_values.append(float(intersection_progress))
        waypoint_names.append(gate.get("name", ""))

    return progress_values, waypoint_names


def find_start_finish_gate_positions(waypoint_gates):
    """Return indices for start and finish gates in the gate list."""
    start_pos = None
    finish_pos = None
    # Gate names are treated as canonical; they must be labeled explicitly.
    for idx, gate in enumerate(waypoint_gates):
        name = str(gate.get("name", "")).strip().lower()
        if name == "start":
            start_pos = idx
        elif name == "finish":
            finish_pos = idx
    if start_pos is None or finish_pos is None:
        raise ValueError("Start or finish gate not found in geo data.")
    return start_pos, finish_pos


def cumulative_distance_meters(latitude, longitude):
    """Cumulative great-circle distance along a polyline."""
    if len(latitude) < 2:
        return np.array([0.0])
    # Convert per-segment distances into a running total.
    segment = haversine_meters(latitude[:-1], longitude[:-1], latitude[1:], longitude[1:])
    segment = np.where(np.isfinite(segment), segment, 0.0)
    return np.concatenate([[0.0], np.cumsum(segment)])


def geojson_geometry_to_lines(geometry):
    """Return list of Nx2 [lon lat] arrays from GeoJSON geometry."""
    line_strings = []
    geometry_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geometry_type == "LineString":
        line = coords_to_numeric_2d(coords)
        if line.size:
            line_strings.append(line)
    elif geometry_type == "MultiLineString":
        # Multi-line geometries are expanded into separate line segments.
        line_strings.extend(coords_to_cell_of_2d(coords))
    elif geometry_type == "GeometryCollection":
        # Recursively gather all line-like geometries.
        for geom in geometry.get("geometries", []):
            line_strings.extend(geojson_geometry_to_lines(geom))
    return line_strings


def coords_to_numeric_2d(coords):
    """Convert GeoJSON coordinates to Nx2 array."""
    if coords is None:
        return np.array([])
    if isinstance(coords, list):
        # Ignore any trailing altitude dimension.
        arr = np.array(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
    return np.array([])


def coords_to_cell_of_2d(coords):
    """Convert MultiLineString coordinates to list of Nx2 arrays."""
    lines = []
    if coords is None:
        return lines
    if isinstance(coords, list):
        for part in coords:
            # Each part becomes its own line array.
            line = coords_to_numeric_2d(part)
            if line.size:
                lines.append(line)
    return lines


if __name__ == "__main__":
    main()
