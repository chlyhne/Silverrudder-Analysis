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


def read_tracking_csv_as_struct(filename):
    """Read tracking CSV into a dict-of-columns."""
    data = {}
    with open(filename, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return data
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
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

    unique_ids = []
    seen = set()
    for value in track_id:
        if value not in seen:
            unique_ids.append(value)
            seen.add(value)

    tracks = []
    for track_value in unique_ids:
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
    kmh_to_knots = 0.539956803
    for track in tracks:
        track["speed"] = track["speed"] * kmh_to_knots
    return tracks


def apply_track_names(tracks, track_name_map):
    """Set track name using mapping keyed by tracked_object_id."""
    for track in tracks:
        if track["id"] in track_name_map:
            track["name"] = track_name_map[track["id"]]
    return tracks


def apply_track_colors(tracks, track_color_map):
    """Set track color using mapping keyed by boat name."""
    for track in tracks:
        if track["name"] in track_color_map:
            track["color"] = track_color_map[track["name"]]
    return tracks


def load_race_metadata(metadata_path):
    """Load race metadata from JSON."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_local_time_to_utc(time_text, local_offset_hours):
    """Convert a local time string (dd/mm/YYYY HH:MM:SS) to UTC timestamp."""
    if not time_text:
        return None
    try:
        local_tz = timezone(timedelta(hours=local_offset_hours))
        local_time = datetime.strptime(time_text, "%d/%m/%Y %H:%M:%S").replace(tzinfo=local_tz)
    except ValueError:
        return None
    return local_time.astimezone(timezone.utc).timestamp()


def build_boat_color_map(boat_names, boat_colors):
    """Normalize boat color mapping from metadata."""
    if isinstance(boat_colors, dict):
        return {name: tuple(color) for name, color in boat_colors.items()}
    if isinstance(boat_colors, list):
        return {name: tuple(color) for name, color in zip(boat_names, boat_colors)}
    return {}


def load_waypoint_csvs(waypoint_dir, local_offset_hours):
    """Load waypoint timing data from split CSV files."""
    waypoint_dir = Path(waypoint_dir)
    if not waypoint_dir.is_dir():
        return []

    waypoint_entries = []
    for path in waypoint_dir.glob("*.csv"):
        if path.name.startswith("OMIT_"):
            continue
        match = re.search(r"\((\d+)_\d+\)", path.name)
        if not match:
            continue
        waypoint_index = int(match.group(1))

        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            map_item_line = handle.readline().strip()
            handle.readline()  # class line
            reader = csv.DictReader(handle)

            name_match = re.search(r"MapItem:\s*(.*?)\s*\(", map_item_line)
            waypoint_name = name_match.group(1).strip() if name_match else path.stem

            times_by_boat = {}
            for row in reader:
                short_name = (row.get("Short") or "").strip()
                if "-" in short_name:
                    boat_name = short_name.split("-", 1)[1].strip()
                else:
                    boat_name = short_name
                if not boat_name:
                    continue

                time_text = (row.get("At time") or "").strip()
                if not time_text:
                    continue

                time_utc = parse_local_time_to_utc(time_text, local_offset_hours)
                if time_utc is None:
                    continue

                boat_key = normalize_boat_name(boat_name)
                times_by_boat[boat_key] = time_utc

        waypoint_entries.append(
            {
                "index": waypoint_index,
                "name": waypoint_name,
                "times_by_boat": times_by_boat,
            }
        )

    waypoint_entries.sort(key=lambda entry: entry["index"])
    return waypoint_entries


def enforce_monotonic_waypoint_times(waypoint_entries):
    """Ensure waypoint times per boat increase with waypoint order."""
    last_time_by_boat = {}
    for entry in waypoint_entries:
        filtered = {}
        for boat_key, time_value in entry["times_by_boat"].items():
            last_time = last_time_by_boat.get(boat_key)
            if last_time is None or time_value >= last_time:
                filtered[boat_key] = time_value
                last_time_by_boat[boat_key] = time_value
        entry["times_by_boat"] = filtered


def get_waypoint_index_bounds(waypoint_entries):
    """Return the smallest and largest waypoint indices."""
    indices = [entry["index"] for entry in waypoint_entries]
    if not indices:
        raise ValueError("Waypoint entries are empty.")
    return min(indices), max(indices)


def extract_waypoint_time_map(waypoint_entries, waypoint_index):
    """Return the boat->time map for a specific waypoint index."""
    for entry in waypoint_entries:
        if entry["index"] == waypoint_index:
            return dict(entry["times_by_boat"])
    return {}


def compute_reference_start_time(start_times_by_boat):
    """Compute a robust start time from per-boat waypoint times."""
    if not start_times_by_boat:
        return None
    times = np.asarray(list(start_times_by_boat.values()), dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return None
    return float(np.median(times))


def compute_waypoint_positions(waypoint_entries, tracks):
    """Estimate waypoint positions from track samples at waypoint times."""
    track_lookup = {normalize_boat_name(track["name"]): track for track in tracks}
    waypoint_positions = []

    for entry in waypoint_entries:
        positions_lat = []
        positions_lon = []
        for boat_key, time_value in entry["times_by_boat"].items():
            track = track_lookup.get(boat_key)
            if track is None:
                continue
            interpolated = interpolate_track_position(track, time_value)
            if interpolated is None:
                continue
            lat_value, lon_value = interpolated
            if np.isfinite(lat_value) and np.isfinite(lon_value):
                positions_lat.append(lat_value)
                positions_lon.append(lon_value)

        if positions_lat and positions_lon:
            waypoint_lat = float(np.median(positions_lat))
            waypoint_lon = float(np.median(positions_lon))
        else:
            waypoint_lat = np.nan
            waypoint_lon = np.nan

        waypoint_positions.append(
            {
                "index": entry["index"],
                "name": entry["name"],
                "lat": waypoint_lat,
                "lon": waypoint_lon,
            }
        )

    return waypoint_positions


def interpolate_track_position(track, time_value):
    """Interpolate track latitude/longitude at a specific UTC timestamp."""
    time_values = track["t"]
    latitude = track["lat"]
    longitude = track["lon"]
    valid_mask = np.isfinite(time_values) & np.isfinite(latitude) & np.isfinite(longitude)
    if not valid_mask.any():
        return None

    time_values = time_values[valid_mask]
    latitude = latitude[valid_mask]
    longitude = longitude[valid_mask]

    unique_time, unique_indices = np.unique(time_values, return_index=True)
    if unique_time.size < 2:
        return None

    if time_value < unique_time[0] or time_value > unique_time[-1]:
        return None

    lat_value = np.interp(time_value, unique_time, latitude[unique_indices])
    lon_value = np.interp(time_value, unique_time, longitude[unique_indices])
    return lat_value, lon_value


def unwrap_waypoint_indices(waypoint_indices, route_count):
    """Unwrap waypoint indices to a monotone sequence along a looped route."""
    waypoint_indices = np.asarray(waypoint_indices, dtype=float)
    unwrapped = waypoint_indices.copy()
    last_value = None

    for index, raw_value in enumerate(waypoint_indices):
        if not np.isfinite(raw_value):
            continue
        if last_value is None:
            unwrapped[index] = raw_value
            last_value = unwrapped[index]
            continue

        candidate = raw_value
        while candidate < last_value:
            candidate += route_count
        unwrapped[index] = candidate
        last_value = candidate

    return unwrapped


def map_waypoints_to_progress(waypoint_positions, average_route):
    """Map waypoint positions to progress along the average route."""
    route_lat = np.asarray(average_route["lat"], dtype=float)
    route_lon = np.asarray(average_route["lon"], dtype=float)
    route_count = len(route_lat)
    if route_count < 2:
        return [], []

    waypoint_indices = []
    waypoint_names = []
    for entry in sorted(waypoint_positions, key=lambda item: item["index"]):
        waypoint_names.append(entry["name"])
        if not np.isfinite(entry["lat"]) or not np.isfinite(entry["lon"]):
            waypoint_indices.append(np.nan)
            continue
        distances = haversine_meters(entry["lat"], entry["lon"], route_lat, route_lon)
        if not np.isfinite(distances).any():
            waypoint_indices.append(np.nan)
            continue
        best_index = int(np.nanargmin(distances))
        waypoint_indices.append(best_index)

    waypoint_indices = np.asarray(waypoint_indices, dtype=float)
    unwrapped = unwrap_waypoint_indices(waypoint_indices, route_count)
    unwrapped = fill_missing_linear(unwrapped)

    if unwrapped.size == 0 or not np.isfinite(unwrapped).any():
        return unwrapped.tolist(), waypoint_names

    start_value = unwrapped[0]
    end_value = unwrapped[-1]
    if not np.isfinite(start_value):
        first_finite = np.where(np.isfinite(unwrapped))[0]
        start_value = unwrapped[first_finite[0]] if first_finite.size else 0.0
    if not np.isfinite(end_value):
        last_finite = np.where(np.isfinite(unwrapped))[0]
        end_value = unwrapped[last_finite[-1]] if last_finite.size else start_value

    if abs(end_value - start_value) < 1e-6:
        progress_values = (unwrapped - start_value) / max(route_count - 1, 1)
    else:
        progress_values = (unwrapped - start_value) / (end_value - start_value)

    progress_values = np.clip(progress_values, 0, 1)
    if progress_values.size:
        progress_values[0] = 0.0
        progress_values[-1] = 1.0

    return progress_values.tolist(), waypoint_names


def compute_start_finish_point(tracks, finish_time_map):
    """Estimate the shared start/finish point using finish-time positions."""
    finish_latitudes = []
    finish_longitudes = []

    for track in tracks:
        boat_key = normalize_boat_name(track["name"])
        finish_utc = finish_time_map.get(boat_key)
        if finish_utc is None:
            continue

        time_values = track["t"]
        latitude = track["lat"]
        longitude = track["lon"]
        valid_mask = np.isfinite(time_values) & np.isfinite(latitude) & np.isfinite(longitude)
        if not valid_mask.any():
            continue

        time_values = time_values[valid_mask]
        latitude = latitude[valid_mask]
        longitude = longitude[valid_mask]

        unique_time, unique_indices = np.unique(time_values, return_index=True)
        if unique_time.size < 2:
            continue

        if finish_utc < unique_time[0] or finish_utc > unique_time[-1]:
            continue

        finish_lat = np.interp(finish_utc, unique_time, latitude[unique_indices])
        finish_lon = np.interp(finish_utc, unique_time, longitude[unique_indices])

        if np.isfinite(finish_lat) and np.isfinite(finish_lon):
            finish_latitudes.append(finish_lat)
            finish_longitudes.append(finish_lon)

    if not finish_latitudes or not finish_longitudes:
        raise ValueError("No finish positions available to compute start/finish point.")

    return {
        "lat": float(np.median(finish_latitudes)),
        "lon": float(np.median(finish_longitudes)),
    }


def trim_tracks_by_start_time_and_finish_proximity(
    tracks,
    start_time_utc,
    finish_point,
    radius_meters,
):
    """Trim tracks by start time and finish proximity."""
    if not tracks:
        return tracks

    finish_lat = finish_point["lat"]
    finish_lon = finish_point["lon"]
    radius_meters = float(radius_meters)

    trimmed_tracks = []
    for track in tracks:
        time_values = track["t"]
        latitude = track["lat"]
        longitude = track["lon"]
        valid_mask = np.isfinite(time_values) & np.isfinite(latitude) & np.isfinite(longitude)
        if not valid_mask.any():
            continue

        time_values = time_values[valid_mask]
        latitude = latitude[valid_mask]
        longitude = longitude[valid_mask]
        speed_values = track["speed"][valid_mask]

        if start_time_utc is not None and np.isfinite(start_time_utc):
            start_mask = time_values >= start_time_utc
        else:
            start_mask = np.ones(time_values.size, dtype=bool)

        if not np.any(start_mask):
            trimmed_tracks.append(track)
            continue

        time_values = time_values[start_mask]
        latitude = latitude[start_mask]
        longitude = longitude[start_mask]
        speed_values = speed_values[start_mask]

        distance_to_finish = haversine_meters(latitude, longitude, finish_lat, finish_lon)
        within_finish_radius = np.where(distance_to_finish <= radius_meters)[0]
        if within_finish_radius.size > 0:
            finish_index = int(within_finish_radius[-1])
            keep_mask = np.zeros(time_values.size, dtype=bool)
            keep_mask[: finish_index + 1] = True
        else:
            keep_mask = np.ones(time_values.size, dtype=bool)

        track["t"] = time_values[keep_mask]
        track["lat"] = latitude[keep_mask]
        track["lon"] = longitude[keep_mask]
        track["speed"] = speed_values[keep_mask]
        track["routeIdx"] = np.array([], dtype=int)
        track["routeDist"] = np.array([], dtype=float)
        track["alpha"] = np.array([], dtype=float)
        trimmed_tracks.append(track)

    return trimmed_tracks



def map_track_points_to_route(tracks, average_route, route_search_window_half_width):
    """Assign each track point to nearest route index."""
    route_lat = np.asarray(average_route["lat"])
    route_lon = np.asarray(average_route["lon"])
    route_sample_count = len(route_lat)

    use_search_window = route_search_window_half_width > 0
    route_endpoint_distance = haversine_meters(
        route_lat[0], route_lon[0], route_lat[-1], route_lon[-1]
    )
    route_is_loop = route_endpoint_distance < 500

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
                time_progress = (sample_time - track_start_time) / (
                    track_end_time - track_start_time
                )
                expected_route_index_by_time = np.round(
                    time_progress * (route_sample_count - 1)
                ) + 1

        route_index_by_sample = np.zeros(latitude.size, dtype=int)
        route_distance_by_sample = np.full(latitude.size, np.nan, dtype=float)

        if latitude.size > 0:
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
                best_route_index = int(expected_index)
            else:
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

    spike_threshold = max(50, round(0.02 * route_sample_count))

    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        if route_index.size < 3:
            continue

        step_sizes = np.abs(np.diff(route_index))
        typical_step = np.nanmedian(step_sizes) if np.isfinite(step_sizes).any() else 1
        if not np.isfinite(typical_step) or typical_step < 1:
            typical_step = 1
        spike_threshold_for_track = max(spike_threshold, round(10 * typical_step))

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

    window_count = int(math.floor((route_sample_count - window_sample_count) / window_step_samples) + 1)
    if window_count < 1:
        window_count = 1

    window_start_index = np.arange(window_count) * window_step_samples + 1
    window_end_index = window_start_index + window_sample_count - 1
    window_end_index = np.minimum(window_end_index, route_sample_count)

    speed_samples_by_window = [[] for _ in range(window_count)]

    for track in tracks:
        route_index = track["routeIdx"].astype(float)
        speed = track["speed"].astype(float)
        valid_mask = (
            np.isfinite(route_index)
            & np.isfinite(speed)
            & (route_index >= 1)
            & (route_index <= route_sample_count)
        )
        route_index = route_index[valid_mask]
        speed = speed[valid_mask]
        if route_index.size == 0:
            continue

        window_min_for_sample = np.floor((route_index - window_sample_count) / window_step_samples) + 1
        window_max_for_sample = np.floor((route_index - 1) / window_step_samples) + 1
        window_min_for_sample = np.clip(window_min_for_sample, 1, window_count).astype(int)
        window_max_for_sample = np.clip(window_max_for_sample, 1, window_count).astype(int)

        for sample_index in range(route_index.size):
            for window_index in range(window_min_for_sample[sample_index], window_max_for_sample[sample_index] + 1):
                if route_index[sample_index] < window_start_index[window_index - 1] or \
                        route_index[sample_index] > window_end_index[window_index - 1]:
                    continue
                speed_samples_by_window[window_index - 1].append(speed[sample_index])

    min_speed_by_window = np.full(window_count, np.nan)
    max_speed_by_window = np.full(window_count, np.nan)
    mean_speed_by_window = np.full(window_count, np.nan)
    mean_pace_by_window = np.full(window_count, np.nan)
    for idx, samples in enumerate(speed_samples_by_window):
        samples = np.asarray(samples, dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        min_speed_by_window[idx] = np.min(samples)
        max_speed_by_window[idx] = np.max(samples)
        mean_speed_by_window[idx] = np.mean(samples)
        positive_samples = samples[samples > 0]
        if positive_samples.size:
            mean_pace_by_window[idx] = np.mean(60.0 / positive_samples)

    min_speed_by_window = lowpass_forward_backward(min_speed_by_window, filter_alpha)
    max_speed_by_window = lowpass_forward_backward(max_speed_by_window, filter_alpha)
    mean_speed_by_window = lowpass_forward_backward(mean_speed_by_window, filter_alpha)
    mean_pace_by_window = lowpass_forward_backward(mean_pace_by_window, filter_alpha)

    # Window progress grid (center of each window) for synchronized sampling
    window_center_index = (window_start_index + window_end_index) / 2
    window_progress = (window_center_index - 1) / (route_sample_count - 1)

    sum_squared_error = np.zeros(window_count)
    count_squared_error = np.zeros(window_count)

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
            home_window_index = np.floor((route_index_valid - 1) / window_step_samples) + 1
            home_window_index = np.clip(home_window_index, 1, window_count).astype(int)

            for idx, window_index in zip(valid_indices, home_window_index):
                min_speed = min_speed_by_window[window_index - 1]
                max_speed = max_speed_by_window[window_index - 1]
                if np.isfinite(min_speed) and np.isfinite(max_speed) and max_speed > min_speed:
                    alpha[idx] = (speed[idx] - min_speed) / (max_speed - min_speed)
                else:
                    alpha[idx] = 0.5

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

    backward[~finite_mask] = np.nan
    return backward


def maximize_figure(fig):
    """Best-effort maximize across backends."""
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


def apply_local_meter_aspect(ax, average_route):
    """Force square meters on the map using mean latitude of the rhumb line."""
    route_lat = np.asarray(average_route.get("lat", []), dtype=float)
    if route_lat.size == 0:
        ax.set_aspect("equal", adjustable="box")
        return

    mean_lat = float(np.nanmean(route_lat))
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(mean_lat))
    if meters_per_degree_lon <= 0 or not np.isfinite(meters_per_degree_lon):
        ax.set_aspect("equal", adjustable="box")
        return

    data_ratio = meters_per_degree_lat / meters_per_degree_lon
    ax.set_aspect(data_ratio, adjustable="box")


def enable_rhumbline_datatip(ax, rhumb_line, speed_window_stats, average_route):
    """Optional data tip for the rhumb line showing progress and min/max."""
    try:
        import mplcursors
    except Exception:
        return

    cursor = mplcursors.cursor(rhumb_line, hover=True)
    ax._rhumbline_cursor = cursor
    route_sample_count = len(average_route["lat"])
    window_step = speed_window_stats.get("windowStepSamples", 1)
    min_speed = speed_window_stats.get("minSpeedByWindow", np.array([]))
    max_speed = speed_window_stats.get("maxSpeedByWindow", np.array([]))

    @cursor.connect("add")
    def on_add(selection):
        index = selection.index
        if route_sample_count > 1:
            progress = index / (route_sample_count - 1)
        else:
            progress = 0.0

        window_count = len(min_speed)
        if window_count > 0:
            home_window = int(math.floor(index / window_step))
            home_window = max(0, min(window_count - 1, home_window))
            min_value = min_speed[home_window]
            max_value = max_speed[home_window]
            min_text = f"{min_value:.2f}" if np.isfinite(min_value) else "N/A"
            max_text = f"{max_value:.2f}" if np.isfinite(max_value) else "N/A"
        else:
            min_text = "N/A"
            max_text = "N/A"

        selection.annotation.set_text(
            f"Boat: Rhumb line\nProgress: {progress:.4f}\n"
            f"Min speed: {min_text}\nMax speed: {max_text}"
        )


def enable_track_datatips(line_collections):
    """Optional data tip for track lines showing boat name and speed."""
    if not line_collections:
        return

    try:
        import mplcursors
    except Exception:
        return

    cursor = mplcursors.cursor(line_collections, hover=True)
    if line_collections:
        line_collections[0].axes._track_cursor = cursor

    @cursor.connect("add")
    def on_add(selection):
        artist = selection.artist
        track_name = getattr(artist, "track_name", "Boat")
        speed_series = getattr(artist, "track_speed", None)

        index = selection.index
        if isinstance(index, tuple):
            index = index[0]
        if index is None:
            index = 0

        speed_text = "N/A"
        if speed_series is not None and len(speed_series) > 0:
            safe_index = min(int(index), len(speed_series) - 1)
            speed_value = speed_series[safe_index]
            if np.isfinite(speed_value):
                speed_text = f"{speed_value:.2f}"

        selection.annotation.set_text(f"Boat: {track_name}\nSpeed: {speed_text}")


def enable_manual_datatips(ax, tracks, average_route, speed_window_stats, pick_radius_m=250.0):
    """Fallback click-tooltips when mplcursors is unreliable."""
    route_lon = np.asarray(average_route.get("lon", []), dtype=float)
    route_lat = np.asarray(average_route.get("lat", []), dtype=float)
    if route_lon.size == 0 or route_lat.size == 0:
        return

    earth_radius = 6371000.0
    meters_per_degree = math.pi / 180.0 * earth_radius
    reference_lat = float(np.nanmedian(route_lat))
    cos_lat = math.cos(math.radians(reference_lat))

    window_step = speed_window_stats.get("windowStepSamples", 1)
    min_speed = speed_window_stats.get("minSpeedByWindow", np.array([]))
    max_speed = speed_window_stats.get("maxSpeedByWindow", np.array([]))
    window_count = len(min_speed)

    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(8, 8),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
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
            speed_text = "N/A"
            if best_track_speed is not None and np.isfinite(best_track_speed):
                speed_text = f"{best_track_speed:.2f}"
            annotation.xy = best_track_point
            annotation.set_text(f"Boat: {best_track_name}\nSpeed: {speed_text}")
            annotation.set_visible(True)
        elif best_route_dist2 <= radius2 and route_index is not None:
            if route_lon.size > 1:
                progress = route_index / (route_lon.size - 1)
            else:
                progress = 0.0
            if window_count > 0:
                home_window = int(math.floor(route_index / window_step))
                home_window = max(0, min(window_count - 1, home_window))
                min_value = min_speed[home_window]
                max_value = max_speed[home_window]
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


def draw_manual_box_plot(ax, x_position, samples, line_color):
    """Draw a box plot at a fixed x position (median line + mean dot)."""
    if samples.size == 0:
        return

    quartiles, median_value = quantiles_no_toolbox(samples, [0.25, 0.5, 0.75])
    q1, q3 = quartiles[0], quartiles[2]
    iqr_value = q3 - q1
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value

    lower_whisker = np.min(samples[samples >= lower_bound]) if np.any(samples >= lower_bound) else np.min(samples)
    upper_whisker = np.max(samples[samples <= upper_bound]) if np.any(samples <= upper_bound) else np.max(samples)

    box_half_width = 0.3
    ax.add_patch(
        plt.Rectangle(
            (x_position - box_half_width, q1),
            2 * box_half_width,
            q3 - q1,
            facecolor=line_color,
            edgecolor=line_color,
            alpha=0.2,
        )
    )
    ax.plot(
        [x_position - box_half_width, x_position + box_half_width],
        [median_value, median_value],
        color=line_color,
        linewidth=1.5,
    )
    mean_value = float(np.mean(samples))
    ax.plot(
        x_position,
        mean_value,
        marker="o",
        markersize=4,
        markerfacecolor=line_color,
        markeredgecolor=line_color,
    )
    ax.plot(
        [x_position, x_position],
        [lower_whisker, upper_whisker],
        color=line_color,
        linewidth=1.0,
    )
    cap_width = box_half_width * 0.8
    ax.plot(
        [x_position - cap_width, x_position + cap_width],
        [lower_whisker, lower_whisker],
        color=line_color,
        linewidth=1.0,
    )
    ax.plot(
        [x_position - cap_width, x_position + cap_width],
        [upper_whisker, upper_whisker],
        color=line_color,
        linewidth=1.0,
    )



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


def smooth_moving_average(values, window_size):
    """Simple moving average for PDF-style smoothing."""
    if window_size <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


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


def normalize_waypoint_inputs(way_point_progress, way_point_names):
    """Normalize waypoint progress/labels for plotting."""
    progress_values = np.asarray(way_point_progress, dtype=float).flatten()
    if progress_values.size == 0:
        return progress_values, []

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
    text = str(label).strip().lower()
    text = text.replace("æ", "ae").replace("ø", "o").replace("å", "aa")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "leg"


def quantiles_no_toolbox(values, quantile_points):
    """Linear-interpolated quantiles without toolboxes."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    sample_count = sorted_values.size
    if sample_count == 0:
        return np.full(len(quantile_points), np.nan), np.nan

    quantiles = []
    for p in quantile_points:
        position = 1 + (sample_count - 1) * p
        lower_index = int(math.floor(position)) - 1
        upper_index = int(math.ceil(position)) - 1
        lower_index = max(0, min(sample_count - 1, lower_index))
        upper_index = max(0, min(sample_count - 1, upper_index))
        if lower_index == upper_index:
            quantiles.append(sorted_values[lower_index])
        else:
            weight = position - math.floor(position)
            value = sorted_values[lower_index] + weight * (
                sorted_values[upper_index] - sorted_values[lower_index]
            )
            quantiles.append(value)

    quantiles = np.asarray(quantiles, dtype=float)
    median_value = quantiles[quantile_points.index(0.5)] if 0.5 in quantile_points else np.nan
    return quantiles, median_value


def haversine_meters(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters). Supports scalar/vector mixing."""
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
    return np.array([str(value).strip() if value is not None else "" for value in values], dtype=object)


def normalize_boat_name(name):
    """Normalize boat names for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def to_double_column(values):
    """Convert to float numpy array with NaNs for invalid entries."""
    output = []
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
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        pass
    try:
        cleaned = text.replace(" UTC", "").replace("Z", "")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return np.nan


def compute_average_route(tracks, route_sample_count):
    """Simple mean route by arclength-resampling each track."""
    track_lat_medians = []
    track_lon_medians = []
    for track in tracks:
        latitude = track["lat"]
        longitude = track["lon"]
        valid_mask = np.isfinite(latitude) & np.isfinite(longitude)
        if not valid_mask.any():
            continue
        track_lat_medians.append(np.median(latitude[valid_mask]))
        track_lon_medians.append(np.median(longitude[valid_mask]))

    if not track_lat_medians or not track_lon_medians:
        raise ValueError("No valid track coordinates to compute average route.")

    reference_lat = float(np.median(track_lat_medians))
    reference_lon = float(np.median(track_lon_medians))

    route_fraction_grid = np.linspace(0, 1, route_sample_count)
    resampled_x = np.full((route_sample_count, len(tracks)), np.nan)
    resampled_y = np.full((route_sample_count, len(tracks)), np.nan)

    for idx, track in enumerate(tracks):
        latitude = track["lat"]
        longitude = track["lon"]
        valid_mask = np.isfinite(latitude) & np.isfinite(longitude)
        latitude = latitude[valid_mask]
        longitude = longitude[valid_mask]
        if latitude.size < 2:
            continue

        distance_along = cumulative_distance_meters(latitude, longitude)
        keep_mask = np.concatenate([[True], np.diff(distance_along) > 0])
        latitude = latitude[keep_mask]
        longitude = longitude[keep_mask]
        distance_along = distance_along[keep_mask]
        if distance_along.size < 2 or distance_along[-1] <= 0:
            continue

        normalized = distance_along / distance_along[-1]
        x_m, y_m = ll2xy_meters(latitude, longitude, reference_lat, reference_lon)
        resampled_x[:, idx] = np.interp(route_fraction_grid, normalized, x_m, left=np.nan, right=np.nan)
        resampled_y[:, idx] = np.interp(route_fraction_grid, normalized, y_m, left=np.nan, right=np.nan)

    valid_x = np.isfinite(resampled_x)
    valid_y = np.isfinite(resampled_y)
    sum_x = np.nansum(resampled_x, axis=1)
    sum_y = np.nansum(resampled_y, axis=1)
    count_x = np.sum(valid_x, axis=1)
    count_y = np.sum(valid_y, axis=1)
    mean_x = np.full_like(sum_x, np.nan, dtype=float)
    mean_y = np.full_like(sum_y, np.nan, dtype=float)
    np.divide(sum_x, count_x, out=mean_x, where=count_x > 0)
    np.divide(sum_y, count_y, out=mean_y, where=count_y > 0)
    mean_x = fill_missing_linear(mean_x)
    mean_y = fill_missing_linear(mean_y)

    mean_lat, mean_lon = xy2ll_meters(mean_x, mean_y, reference_lat, reference_lon)
    return {"lat": mean_lat, "lon": mean_lon, "s": route_fraction_grid}


def cumulative_distance_meters(latitude, longitude):
    """Cumulative great-circle distance along a polyline."""
    if len(latitude) < 2:
        return np.array([0.0])
    segment = haversine_meters(latitude[:-1], longitude[:-1], latitude[1:], longitude[1:])
    segment = np.where(np.isfinite(segment), segment, 0.0)
    return np.concatenate([[0.0], np.cumsum(segment)])


def ll2xy_meters(latitude, longitude, reference_latitude, reference_longitude):
    """Local equirectangular projection around (lat0,lon0)."""
    earth_radius = 6371000.0
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)
    reference_latitude = math.radians(reference_latitude)
    reference_longitude = math.radians(reference_longitude)
    x_m = earth_radius * (longitude - reference_longitude) * math.cos(reference_latitude)
    y_m = earth_radius * (latitude - reference_latitude)
    return x_m, y_m


def xy2ll_meters(x_meters, y_meters, reference_latitude, reference_longitude):
    """Inverse of ll2xy_meters."""
    earth_radius = 6371000.0
    reference_latitude = math.radians(reference_latitude)
    reference_longitude = math.radians(reference_longitude)
    latitude = np.degrees(y_meters / earth_radius + reference_latitude)
    longitude = np.degrees(x_meters / (earth_radius * math.cos(reference_latitude)) + reference_longitude)
    return latitude, longitude


def fill_missing_linear(values):
    """Fill NaNs by linear interpolation, using nearest for ends."""
    values = np.asarray(values, dtype=float)
    indices = np.arange(values.size)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return values
    filled = values.copy()
    filled[~finite_mask] = np.interp(
        indices[~finite_mask],
        indices[finite_mask],
        values[finite_mask],
        left=values[finite_mask][0],
        right=values[finite_mask][-1],
    )
    return filled


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
        line_strings.extend(coords_to_cell_of_2d(coords))
    elif geometry_type == "GeometryCollection":
        for geom in geometry.get("geometries", []):
            line_strings.extend(geojson_geometry_to_lines(geom))
    return line_strings


def coords_to_numeric_2d(coords):
    """Convert GeoJSON coordinates to Nx2 array."""
    if coords is None:
        return np.array([])
    if isinstance(coords, list):
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
            line = coords_to_numeric_2d(part)
            if line.size:
                lines.append(line)
    return lines


if __name__ == "__main__":
    main()
