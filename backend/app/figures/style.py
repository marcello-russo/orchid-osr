# ruff: noqa: D100
from typing import Any

PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#F49DAA"]

# `dict[Any, Any]` e non `dict[str, Any]`: `plt.rcParams.update` accetta solo
# chiavi del `Literal` di ~400 nomi che matplotlib genera dai suoi default.
# `dict[str, Any]` non è assegnabile a quella firma, e con mypy 2.x diventa un
# errore su ogni `update` (verificato: 5 errori con 2.4.1, zero con 1.19.1 —
# cioè il pin teneva in piedi il gate per caso). Le chiavi restano validate a
# runtime da matplotlib, come prima.
MATPLOTLIB_RC: dict[Any, Any] = {
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
}
