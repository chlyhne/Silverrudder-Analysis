from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Manual box-plot helpers keep visual output consistent across figures and avoid
# reliance on Matplotlib's default boxplot styling, which varies across versions.

@dataclass(frozen=True)
class BoxPlotStyle:
    """Styling options for manual box plots."""

    box_half_width: float = 0.3
    box_alpha: float = 0.2
    median_linewidth: float = 1.5
    whisker_linewidth: float = 1.0
    cap_width_ratio: float = 0.8
    mean_marker: str = "o"
    mean_marker_size: float = 4.0


@dataclass(frozen=True)
class BoxPlotAxisStyle:
    """Axis styling options for box plots."""

    major_tick: float = 1.0
    minor_ticks_per_major: int = 4
    minor_tick_length: float = 2.0
    minor_grid_alpha: float = 0.5
    minor_grid_linewidth: float = 0.6


@dataclass(frozen=True)
class BoxPlotStats:
    """Precomputed stats for a Tukey-style box plot."""

    q1: float
    q3: float
    median: float
    mean: float
    lower_whisker: float
    upper_whisker: float


def apply_boxplot_axis_style(ax, style: Optional[BoxPlotAxisStyle] = None) -> None:
    """Apply y-axis grid and tick styling for box plots."""
    # Axis styling is centralized so every plot shares the same visual rhythm.
    style = style or BoxPlotAxisStyle()
    ax.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(style.major_tick))
    if style.major_tick > 0 and style.minor_ticks_per_major > 0:
        # Subdivide the major interval so minor ticks are evenly spaced.
        minor_step = style.major_tick / (style.minor_ticks_per_major + 1)
        ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    # Label minor ticks too, but skip duplicates where a major tick already labels the value.
    def minor_formatter(_value, _pos):
        return ""
    ax.yaxis.set_minor_formatter(FuncFormatter(minor_formatter))
    ax.grid(True, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", linewidth=style.minor_grid_linewidth, alpha=style.minor_grid_alpha)
    ax.grid(False, axis="x", which="both")
    ax.tick_params(axis="y", which="minor", length=style.minor_tick_length, labelleft=True)


def quantiles_no_toolbox(values: Iterable[float], quantile_points: Sequence[float]):
    """Linear-interpolated quantiles without toolboxes."""
    # Use a deterministic quantile definition to keep results stable across runtimes.
    values_list = list(values)
    points = list(quantile_points)
    sorted_values = np.sort(np.asarray(values_list, dtype=float))
    sample_count = sorted_values.size
    if sample_count == 0:
        return np.full(len(points), np.nan), np.nan

    quantiles = []
    for p in points:
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
    median_value = quantiles[points.index(0.5)] if 0.5 in points else np.nan
    return quantiles, median_value


def compute_box_plot_stats(
    samples: Iterable[float], whisker_scale: float = 1.5
) -> Optional[BoxPlotStats]:
    """Compute Tukey box plot statistics from sample values."""
    # Filter invalid entries early so whiskers and medians are well-defined.
    clean_samples = np.asarray(list(samples), dtype=float)
    clean_samples = clean_samples[np.isfinite(clean_samples)]
    if clean_samples.size == 0:
        return None

    quartiles, median_value = quantiles_no_toolbox(clean_samples, [0.25, 0.5, 0.75])
    q1 = float(quartiles[0])
    q3 = float(quartiles[2])
    if not np.isfinite(q1) or not np.isfinite(q3):
        return None

    # Tukey whiskers capture the non-outlier spread without trimming the data.
    iqr_value = q3 - q1
    lower_bound = q1 - whisker_scale * iqr_value
    upper_bound = q3 + whisker_scale * iqr_value
    lower_candidates = clean_samples[clean_samples >= lower_bound]
    upper_candidates = clean_samples[clean_samples <= upper_bound]
    lower_whisker = (
        float(np.min(lower_candidates)) if lower_candidates.size else float(np.min(clean_samples))
    )
    upper_whisker = (
        float(np.max(upper_candidates)) if upper_candidates.size else float(np.max(clean_samples))
    )
    return BoxPlotStats(
        q1=q1,
        q3=q3,
        median=float(median_value),
        mean=float(np.mean(clean_samples)),
        lower_whisker=lower_whisker,
        upper_whisker=upper_whisker,
    )


def compute_whisker_bounds(
    samples: Iterable[float], whisker_scale: float = 1.5
) -> Optional[Tuple[float, float]]:
    """Compute Tukey whisker bounds for a sample vector."""
    stats = compute_box_plot_stats(samples, whisker_scale=whisker_scale)
    if stats is None:
        return None
    return stats.lower_whisker, stats.upper_whisker


def update_whisker_range(
    current_lower: float, current_upper: float, samples: Iterable[float], whisker_scale: float = 1.5
):
    """Expand a (lower, upper) range using whisker bounds from samples."""
    # Expand plot ranges using robust whiskers so outliers do not dominate the scale.
    bounds = compute_whisker_bounds(samples, whisker_scale=whisker_scale)
    if bounds is None:
        return current_lower, current_upper
    return min(current_lower, bounds[0]), max(current_upper, bounds[1])


def compute_global_plot_range(global_lower: float, global_upper: float):
    """Expand global plot bounds with padding."""
    # Padding prevents boxes from touching the figure edges.
    if not np.isfinite(global_lower) or not np.isfinite(global_upper):
        return None
    global_range = global_upper - global_lower
    if global_range <= 0:
        return -1.0, 1.0, 2.0
    padding = 0.05 * global_range
    global_lower -= padding
    global_upper += padding
    global_range = global_upper - global_lower
    return global_lower, global_upper, global_range


def compute_local_plot_range(local_lower: float, local_upper: float, global_range: float):
    """Expand local plot bounds with padding and a minimum span."""
    # Local padding keeps each subplot readable without losing relative scale.
    local_range = local_upper - local_lower
    if local_range <= 0:
        local_range = max(global_range * 0.02, 1.0)
        mid_point = (local_lower + local_upper) / 2.0
        local_lower = mid_point - 0.5 * local_range
        local_upper = mid_point + 0.5 * local_range
        return local_lower, local_upper, local_range

    padding = 0.02 * local_range
    local_lower -= padding
    local_upper += padding
    local_range = local_upper - local_lower
    return local_lower, local_upper, local_range


def compute_units_per_inch(global_range: float):
    """Return y-axis units per inch based on default figure height."""
    # This metric lets per-boat plots share a consistent data-to-physical-height ratio.
    base_figsize = plt.rcParams.get("figure.figsize", (6.4, 4.8))
    base_height = float(base_figsize[1]) if len(base_figsize) > 1 else 4.8
    return global_range / base_height if global_range > 0 else 1.0


def resize_figure_for_range(fig, units_per_inch: float, local_range: float) -> None:
    """Resize figure height to keep units-per-inch consistent."""
    # Resize vertically so comparisons across boats are not visually skewed.
    fig_width = fig.get_size_inches()[0]
    fig_height = local_range / units_per_inch if units_per_inch > 0 else fig.get_size_inches()[1]
    fig.set_size_inches(fig_width, fig_height, forward=True)


def draw_manual_box_plot(
    ax,
    x_position: float,
    samples: Iterable[float],
    line_color,
    style: Optional[BoxPlotStyle] = None,
    whisker_scale: float = 1.5,
) -> None:
    """Draw a box plot at a fixed x position (median line + mean dot)."""
    # Manual rendering allows mean markers and color matching with track styling.
    stats = compute_box_plot_stats(samples, whisker_scale=whisker_scale)
    if stats is None:
        return

    style = style or BoxPlotStyle()
    box_half_width = style.box_half_width
    ax.add_patch(
        plt.Rectangle(
            (x_position - box_half_width, stats.q1),
            2 * box_half_width,
            stats.q3 - stats.q1,
            facecolor=line_color,
            edgecolor=line_color,
            alpha=style.box_alpha,
        )
    )
    ax.plot(
        [x_position - box_half_width, x_position + box_half_width],
        [stats.median, stats.median],
        color=line_color,
        linewidth=style.median_linewidth,
    )
    ax.plot(
        x_position,
        stats.mean,
        marker=style.mean_marker,
        markersize=style.mean_marker_size,
        markerfacecolor=line_color,
        markeredgecolor=line_color,
    )
    ax.plot(
        [x_position, x_position],
        [stats.lower_whisker, stats.upper_whisker],
        color=line_color,
        linewidth=style.whisker_linewidth,
    )
    cap_width = box_half_width * style.cap_width_ratio
    ax.plot(
        [x_position - cap_width, x_position + cap_width],
        [stats.lower_whisker, stats.lower_whisker],
        color=line_color,
        linewidth=style.whisker_linewidth,
    )
    ax.plot(
        [x_position - cap_width, x_position + cap_width],
        [stats.upper_whisker, stats.upper_whisker],
        color=line_color,
        linewidth=style.whisker_linewidth,
    )
