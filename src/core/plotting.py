"""
Publication-quality plotting utilities for Neuro-CXG.

Provides:
    - ColorPalette: Colorblind-safe Okabe-Ito palette with semantic color mappings
    - FigureSize: Standard figure sizes for single/double-column journal layout
    - Style helpers: grid, legend, annotation utilities

Usage:
    from src.core.plotting import ColorPalette, FigureSize

    palette = ColorPalette()
    fig, ax = plt.subplots(figsize=FigureSize.QUAD_PANEL)
    ax.plot(x, y, color=palette.ASD, label="ASD")
    ax.plot(x, y2, color=palette.CONTROL, label="Control")
"""

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
        - Positive/Negative values
        - Temporal/Spatial features

    All colors are from the Okabe-Ito palette, ensuring accessibility
    for readers with color vision deficiency (8% of males).
    """

    # Core colors
    BLUE = OKABE_ITO[0]
    ORANGE = OKABE_ITO[1]
    GREEN = OKABE_ITO[2]
    PINK = OKABE_ITO[3]
    SKY_BLUE = OKABE_ITO[5]
    AMBER = OKABE_ITO[6]
    BLACK = OKABE_ITO[7]

    # Semantic mappings
    ASD = ORANGE  # ASD (vermilion - stands out against blue)
    CONTROL = BLUE  # Control (standard blue)
    NEUTRAL = PINK  # Neutral/baseline

    # Signed value mappings (diverging)
    NEGATIVE = ORANGE  # Negative values (inhibitory connections)

    # Feature type mappings
    TEMPORAL = BLUE
    SPATIAL = ORANGE

    @classmethod
    def cycle(cls, n: int = 8):
        """Return first n colors from Okabe-Ito palette."""
        return OKABE_ITO[:n]


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
    SINGLE = (8, 6)  # Standard single panel

    # Multi-panel figures
    QUAD_PANEL = (14, 12)  # 2x2 grid

    # Specialized
    BAR_CHART = (12, 6)  # Horizontal/vertical bar charts
    HEATMAP = (10, 8)  # Matrix heatmaps


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


# ── Professional visualization enhancements ─────────────────────────────────────


def apply_professional_style(ax, label_fontsize=11):
    """Apply professional academic styling with subtle background.

    Args:
        ax: Matplotlib axes
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
