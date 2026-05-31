"""
Publication-quality plotting utilities for Neuro-CXG.

Provides:
    - ColorPalette: Colorblind-safe Okabe-Ito palette with semantic color mappings
    - FigureSize: Standard figure sizes for single/double-column journal layout
    - Style helpers: grid, legend, annotation utilities

Usage:
    from src.core.plotting import ColorPalette, FigureSize

    palette = ColorPalette()
    fig, ax = plt.subplots(figsize=FigureSize.DOUBLE_PANEL)
    ax.plot(x, y, color=palette.ASD, label="ASD")
    ax.plot(x, y2, color=palette.CONTROL, label="Control")
"""

import numpy as np

# ── Colorblind-safe Okabe-Ito palette ──────────────────────────────────
# Source: https://jfly.uni-koeln.de/color/
# Used by Nature, Science, and many top-tier journals

OKABE_ITO = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion (Orange)
    "#009E73",  # Bluish Green
    "#CC79A7",  # Reddish Purple (Pink)
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    "#000000",  # Black
]


class ColorPalette:
    """Colorblind-safe color palette for Neuro-CXG visualizations.

    Provides semantic color mappings for common visualization patterns:
        - ASD/Control comparison
        - Significant/Non-significant results
        - Positive/Negative values
        - Temporal/Spatial features
        - Network membership colors

    All colors are from the Okabe-Ito palette, ensuring accessibility
    for readers with color vision deficiency (8% of males).
    """

    # Core colors
    BLUE = OKABE_ITO[0]
    ORANGE = OKABE_ITO[1]
    GREEN = OKABE_ITO[2]
    PINK = OKABE_ITO[3]
    YELLOW = OKABE_ITO[4]
    SKY_BLUE = OKABE_ITO[5]
    AMBER = OKABE_ITO[6]
    BLACK = OKABE_ITO[7]

    # Semantic mappings
    ASD = ORANGE          # ASD (vermilion - stands out against blue)
    CONTROL = BLUE        # Control (standard blue)
    SIGNIFICANT = GREEN   # Statistically significant
    NEUTRAL = PINK        # Neutral/baseline
    WARNING = AMBER       # Warning/caution

    # Signed value mappings (diverging)
    POSITIVE = BLUE       # Positive values (excitatory connections)
    NEGATIVE = ORANGE     # Negative values (inhibitory connections)

    # Feature type mappings
    TEMPORAL = BLUE
    SPATIAL = ORANGE
    FREQUENCY = GREEN
    INTERNAL = PINK

    # Network colors (sync with atlas_config.py LOBE_TO_NETWORK)
    NETWORK_DMN = BLUE
    NETWORK_SALIENCE = ORANGE
    NETWORK_VISUAL = GREEN
    NETWORK_LIMBIC = ORANGE  # Same as salience for consistency

    @classmethod
    def cycle(cls, n: int = 8):
        """Return first n colors from Okabe-Ito palette."""
        return OKABE_ITO[:n]

    @classmethod
    def asd_control(cls):
        """Return (ASD, Control) color tuple."""
        return (cls.ASD, cls.CONTROL)

    @classmethod
    def positive_negative(cls):
        """Return (Positive, Negative) color tuple."""
        return (cls.POSITIVE, cls.NEGATIVE)

    # Scientific visualization palettes
    @classmethod
    def gradient_cmap(cls, name="blues"):
        """Return a gradient colormap.

        Args:
            name: Gradient name ('blues', 'oranges', 'greens', 'reds', 'purple')

        Returns:
            matplotlib colormap
        """

        gradients = {
            "blues": ["#eff3ff", "#bdd7e7", "#6baed6", "#2171b5", "#08306b"],
            "oranges": ["#fff5eb", "#fee6ce", "#fdd49e", "#fdbb84", "#e6550d"],
            "greens": ["#f7fcf5", "#c7e9c0", "#a1d99b", "#31a354", "#006d2c"],
            "reds": ["#fff5f0", "#fcbba1", "#fc9272", "#fb6a4a", "#cb181d"],
            "purple": ["#fcfcfc", "#e0e0e0", "#bcbddc", "#756bb1", "#54278f"],
        }

        import matplotlib.pyplot as _plt
        from matplotlib.colors import LinearSegmentedColormap
        colors = gradients.get(name, gradients["blues"])
        return LinearSegmentedColormap(name, {k: _plt.cm.get_cmap("viridis")(v/255) for k, v in zip(['red', 'green', 'blue'], [(int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)) for c in colors], strict=False)})


    GRADIENT_PALETTES = {
        "pub_theme": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
        "warm_theme": ["#fff5eb", "#fee6ce", "#fdd49e", "#fdbb84", "#fdaE60", "#f47b20", "#d95f0e", "#c05502"],
        "cool_theme": ["#fcfcfc", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"],
        "accent_theme": ["#7f3f00", "#d95f02", "#fe9929", "#fce090", "#fffff0", "#e5f5f9", "#99d8c9", "#2ca25f"],
    }

    FIGURE_TITLES = {
        "main": {"fontsize": 16, "fontweight": "bold", "pad": 15},
        "sub": {"fontsize": 13, "fontweight": "bold", "pad": 10},
        "panel": {"fontsize": 11, "fontweight": "semibold", "pad": 8},
    }

    LEGEND_STYLES = {
        "standard": {"frameon": True, "framealpha": 0.9, "fancybox": True, "fontsize": 10},
        "minimal": {"frameon": False, "fontsize": 9},
        "scientific": {"frameon": True, "framealpha": 0.95, "fancybox": True, "fontsize": 9, "edgecolor": "#cccccc"},
    }


# ── Standard figure sizes for journal publication ──────────────────────────────
# Based on typical journal column widths:
#   - Single column: ~3.5 inches (8.9 cm)
#   - Double column: ~7.2 inches (18.3 cm)
# With standard 300 DPI, these provide crisp output.

class FigureSize:
    """Standard figure sizes for Neuro-CXG publication figures.

    Sizes optimized for typical journal column widths with 300 DPI output.
    """

    # Single-panel figures
    SINGLE_SMALL = (6, 4)       # Small single panel (e.g., ROC curve)
    SINGLE = (8, 6)             # Standard single panel
    SINGLE_LARGE = (10, 8)      # Large single panel (e.g., heatmap)

    # Multi-panel figures
    DOUBLE_PANEL = (16, 7)      # Two panels side-by-side
    TRIPLE_PANEL = (18, 6)      # Three panels side-by-side
    QUAD_PANEL = (14, 12)       # 2x2 grid
    QUAD_WIDE = (18, 10)        # 2x2 grid, wider

    # Specialized
    BAR_CHART = (12, 6)         # Horizontal/vertical bar charts
    HEATMAP = (10, 8)           # Matrix heatmaps
    CIRCULAR = (12, 12)         # Circular connectome
    TALL = (8, 10)              # Tall narrow (e.g., lobe importance)
    WIDE = (14, 5)              # Wide short (e.g., training curves)


# ── Plotting style helpers ─────────────────────────────────────────────────────

def apply_publication_style(ax=None, grid_alpha=0.3, grid_style="--"):
    """Apply publication-quality styling to axes.

    Args:
        ax: Matplotlib axes (uses plt.gca() if None)
        grid_alpha: Grid line transparency
        grid_style: Grid line style
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    ax.grid(True, alpha=grid_alpha, linestyle=grid_style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    return ax


def add_significance_marker(ax, x1, x2, y, marker="*", text=None, color="black", fontsize=12):
    """Add statistical significance marker between two points.

    Args:
        ax: Matplotlib axes
        x1, x2: X positions to connect
        y: Y position for the marker
        marker: Symbol to display (*, **, ***, ns)
        text: Optional custom text
        color: Marker color
        fontsize: Font size
    """
    ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color=color, lw=1.0)
    label = text if text else marker
    ax.text((x1 + x2) / 2, y + 0.03, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=color)


def annotate_bar_values(ax, bars, fmt="{:.3f}", offset=0.01, fontsize=9):
    """Add value labels on top of bar chart bars.

    Args:
        ax: Matplotlib axes
        bars: Bar container from ax.bar() or ax.barh()
        fmt: Value format string
        offset: Distance from bar edge
        fontsize: Label font size
    """
    for bar in bars:
        if hasattr(bar, 'get_height'):  # Vertical bar
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                    fmt.format(height), ha="center", va="bottom", fontsize=fontsize)
        elif hasattr(bar, 'get_width'):  # Horizontal bar
            width = bar.get_width()
            ax.text(width + offset, bar.get_y() + bar.get_height() / 2,
                    fmt.format(width), ha="left", va="center", fontsize=fontsize)


def create_legend_patch(color, label, marker="o", markersize=8):
    """Create a legend patch element for custom legends.

    Args:
        color: Color for the patch
        label: Legend label text
        marker: Marker shape
        markersize: Marker size

    Returns:
        matplotlib.lines.Line2D legend element
    """
    from matplotlib.lines import Line2D
    return Line2D([0], [0], marker=marker, color="w", markerfacecolor=color,
                  markersize=markersize, label=label, markeredgecolor="#333333",
                  markeredgewidth=0.5)


# ── Statistical helper functions ────────────────────────────────────────────────

def compute_wilson_ci(successes, n_trials, z=1.96):
    """Compute Wilson score confidence interval for proportions.

    Args:
        successes: Number of successes
        n_trials: Total number of trials
        z: Z-score for confidence level (1.96 for 95%)

    Returns:
        (lower, upper) tuple
    """
    if n_trials < 1:
        return None, None
    p = successes / n_trials
    z_squared = z * z
    denom = 1 + z_squared / n_trials
    center = (p + z_squared / (2 * n_trials)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_trials + z_squared / (4 * n_trials * n_trials)) / denom
    lower = max(0, round(center - margin, 3))
    upper = min(1, round(center + margin, 3))
    return lower, upper


def format_p_value(p):
    """Format p-value for display.

    Args:
        p: P-value

    Returns:
        Formatted string (e.g., "p < 0.001", "p = 0.023")
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.2f}"
    else:
        return f"p = {p:.2f}"


def significance_stars(p):
    """Return significance stars for p-value.

    Args:
        p: P-value

    Returns:
        String with stars ("***", "**", "*", "ns")
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ── Professional visualization enhancements ─────────────────────────────────────

def apply_professional_style(ax, background_alpha=0.02, title_fontsize=14, label_fontsize=11):
    """Apply professional academic styling with subtle background.

    Args:
        ax: Matplotlib axes
        background_alpha: Alpha for subtle background gradient
        title_fontsize: Font size for titles
        label_fontsize: Font size for labels
    """

    ax.set_facecolor((1, 1, 1, 0.97))
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", which="major", labelsize=label_fontsize)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#333333")

    return ax


def add_scientific_annotation(ax, x, y, text, offset=(0, 5), fontsize=9,
                             arrow=True, arrow_color="#333333", bbox=True):
    """Add professional annotation with optional arrow and box.

    Args:
        ax: Matplotlib axes
        x: X position
        y: Y position
        text: Annotation text
        offset: Text offset (x, y)
        fontsize: Font size
        arrow: Show arrow pointing to data
        arrow_color: Arrow color
        bbox: Show bounding box around text
    """
    if bbox:
        bbox_props = {"boxstyle": "round,pad=0.3", "facecolor": "white",
                         "edgecolor": "#cccccc", "alpha": 0.9, "linewidth": 0.8}
    else:
        bbox_props = None

    if arrow:
        ax.annotate(text, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                   fontsize=fontsize, ha="center", va="bottom",
                   bbox=bbox_props, arrowprops={"arrowstyle": "->",
                   "color": arrow_color, "lw": 1.2, "connectionstyle": "arc3,rad=0"})
    else:
        ax.text(x + offset[0], y + offset[1], text, fontsize=fontsize,
               ha="center", va="bottom", bbox=bbox_props)


def create_scientific_colormap(name="scientific", reverse=False):
    """Create scientifically appropriate colormaps.

    Args:
        name: Colormap name ('scientific', 'diverging', 'sequential', 'highlight')
        reverse: Reverse the colormap

    Returns:
        matplotlib colormap
    """

    colormaps = {
        "scientific": ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"],
        "diverging": ["#d73027", "#f46d43", "#fdae61", "#fee090",
                      "#ffffbf", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"],
        "sequential": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
                       "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
        "highlight": ["#ffffff", "#fff7bc", "#feb24c", "#f03b20", "#bd0026"],
    }

    colors = colormaps.get(name, colormaps["scientific"])
    if reverse:
        colors = colors[::-1]

    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def add_confidence_band(ax, x, mean, std, color=None, alpha=0.2, label=None):
    """Add shaded confidence band to line plot.

    Args:
        ax: Matplotlib axes
        x: X values
        mean: Mean values
        std: Standard deviation
        color: Fill color (uses current line color if None)
        alpha: Transparency
        label: Label for legend
    """

    if color is None:
        color = ax.lines[-1].get_color() if ax.lines else "#1f77b4"

    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color, label=label)


def create_legend_proxy(color, marker="o", markersize=8, label=None, linewidth=2):
    """Create a professional legend proxy element.

    Args:
        color: Color for the proxy
        marker: Marker shape
        markersize: Marker size
        label: Legend label
        linewidth: Line width

    Returns:
        matplotlib.lines.Line2D
    """
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=color, marker=marker, markersize=markersize,
                   label=label, markeredgecolor="#333333", markeredgewidth=0.5,
                   linewidth=linewidth)


def style_boxplot(ax, palette=None, linewidth=1.5, flier_size=4,
                  whisker_cap_style="line", show_means=False, mean_marker="D"):
    """Apply professional styling to boxplot.

    Args:
        ax: Matplotlib axes
        palette: Color palette
        linewidth: Line width
        flier_size: Outlier marker size
        whisker_cap_style: Style of whisker caps
        show_means: Show mean markers
        mean_marker: Mean marker style
    """
    for i, box in enumerate(ax.artists if hasattr(ax, 'artists') else []):
        if palette and i < len(palette):
            box.set_facecolor(palette[i])
            box.set_alpha(0.7)
        box.set_linewidth(linewidth)

    for element in ['whiskers', 'caps', 'medians']:
        for line in getattr(ax, element, []):
            line.set_linewidth(linewidth)
            line.set_color("#333333")

    for flier in ax.get_xticks():
        if hasattr(ax, 'collections') and len(ax.collections) > flier:
            ax.collections[flier].set_sizes([flier_size])

    if show_means:
        ax.meanprops = {"marker": mean_marker, "markerfacecolor": "white",
                           "markeredgecolor": "#333333", "markersize": 6}


def add_sample_size(ax, x_data, y_data, x_pos=None, fontsize=8, color="#666666"):
    """Add sample size annotations to plot.

    Args:
        ax: Matplotlib axes
        x_data: X data for computing positions
        y_data: Y data for computing positions
        x_pos: Custom x positions (auto-computed if None)
        fontsize: Font size
        color: Text color
    """
    if x_pos is None:
        x_pos = [np.median(x_data[i]) for i in range(len(x_data))]

    for i, (x, y) in enumerate(zip(x_pos, [np.median(d) for d in y_data], strict=False)):
        ax.text(x, y - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
               f"n={len(x_data[i])}", fontsize=fontsize, color=color,
               ha="center", va="top", style="italic")


def add_statistical_annotation(
    ax,
    x_groups: list,
    means: list,
    ci_lower: list,
    ci_upper: list,
    p_value: float | None = None,
    y_offset: float = 0.02,
    fontsize: int = 10,
) -> None:
    """Add statistical annotation with confidence intervals.

    Args:
        ax: Matplotlib axes
        x_groups: X positions for each group
        means: Mean values
        ci_lower: Lower confidence interval bounds
        ci_upper: Upper confidence interval bounds
        p_value: Optional p-value for significance
        y_offset: Vertical offset for annotations
        fontsize: Font size for annotations
    """
    y_max = max(ci_upper)

    for _i, (x, mean, low, high) in enumerate(zip(x_groups, means, ci_lower, ci_upper, strict=False)):
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]],
                   fmt='o', color="#333333", capsize=5, capthick=1.5, markersize=8)

    if p_value is not None:
        stars = significance_stars(p_value)
        y_annot = y_max + y_offset
        ax.text(np.mean(x_groups), y_annot, stars, ha='center', va='bottom',
               fontsize=fontsize + 2, fontweight='bold', color="#333333")


def format_confidence_interval(mean: float, lower: float, upper: float, decimals: int = 3) -> str:
    """Format mean with confidence interval.

    Args:
        mean: Mean value
        lower: Lower CI bound
        upper: Upper CI bound
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "0.856 [0.823, 0.889]")
    """
    return f"{mean:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def add_effect_size_annotation(
    ax,
    group1_mean: float,
    group2_mean: float,
    group1_std: float,
    group2_std: float,
    x_pos: float,
    y_pos: float,
    effect_size: str = "cohen_d",
    decimals: int = 3,
) -> None:
    """Add effect size annotation (Cohen's d or Hedges' g).

    Args:
        ax: Matplotlib axes
        group1_mean: Mean of group 1
        group2_mean: Mean of group 2
        group1_std: Std of group 1
        group2_std: Std of group 2
        x_pos: X position for annotation
        y_pos: Y position for annotation
        effect_size: Type of effect size ('cohen_d' or 'hedges_g')
        decimals: Decimal places
    """
    pooled_std = np.sqrt((group1_std ** 2 + group2_std ** 2) / 2)
    if pooled_std == 0:
        return

    cohen_d = (group1_mean - group2_mean) / pooled_std

    if effect_size == "hedges_g" and (group1_std + group2_std) > 0:
        n1, n2 = 30, 30
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        cohen_d *= max(0, correction)

    label = f"d = {cohen_d:.{decimals}f}"

    if abs(cohen_d) < 0.2:
        label += " (negligible)"
    elif abs(cohen_d) < 0.5:
        label += " (small)"
    elif abs(cohen_d) < 0.8:
        label += " (medium)"
    else:
        label += " (large)"

    ax.text(x_pos, y_pos, label, fontsize=10, ha='center', va='bottom',
           style='italic', color="#333333", bbox={"boxstyle": 'round,pad=0.3',
           "facecolor": 'white', "edgecolor": '#cccccc', "alpha": 0.9})
