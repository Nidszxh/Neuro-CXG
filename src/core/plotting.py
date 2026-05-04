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
