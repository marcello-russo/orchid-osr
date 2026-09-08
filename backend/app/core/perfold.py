"""Lettura dei test per-fold (perfold_test.json) — source of truth per F1 test.

I test_metrics in results.json si riferiscono al vecchio test pre-split;
perfold_test.json contiene F1/acc per fold sul test ufficiale (300),
con medie e std. Usare sempre questi valori per deck, figure, API e audit.
"""
import glob
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parent.parent.parent
EXP_ROOT = _BACKEND / "experiments"


def load_perfold(exp_dir: str | Path) -> dict | None:
    p = Path(exp_dir) / "perfold_test.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        _log.warning("perfold illeggibile %s: %s", p, e)
        return None


def perfold_means(exp_prefix: str) -> dict[str, tuple[float, float]]:
    """{model: (f1_mean, f1_std)} per tutti gli exp che iniziano con exp_prefix."""
    out: dict[str, tuple[float, float]] = {}
    for f in sorted(glob.glob(str(EXP_ROOT / f"{exp_prefix}_*" / "perfold_test.json"))):
        if "/old" in f or "old" in Path(f).parent.name:
            continue
        model = Path(f).parent.name.replace(exp_prefix + "_", "")
        try:
            j = json.loads(Path(f).read_text())
            out[model] = (float(j["f1_mean"]), float(j["f1_std"]))
        except Exception:
            pass
    return out


def fmt_mean_std(mean: float, std: float, ndigits: int = 3) -> str:
    """'0,884 ± 0,009' in stile deck (virgola decimale)."""
    sep = ","
    m = f"{mean:.{ndigits}f}".replace(".", sep)
    s = f"{std:.{ndigits}f}".replace(".", sep)
    return f"{m} ± {s}"
