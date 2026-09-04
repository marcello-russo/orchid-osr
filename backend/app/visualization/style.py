"""Plotting style: ggplot theme, color palette, figure defaults."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-friendly palette (Set2, 5 colors for 5 folds)
COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

FIGURE_SIZE = (8, 5)
DPI = 300


def setup_style() -> None:
    """Apply ggplot style with custom overrides for publication quality."""
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
