"""DeLong CIs per detector + the three spec paired tests. Prints thesis-ready numbers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.ncd.delong import delong_auc_ci, delong_paired_test

EXP = Path(__file__).resolve().parent.parent / "experiments"


def load(exp_dir: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z = np.load(EXP / exp_dir / "osr_scores.npz")
    y = np.asarray(z["y_true"]).astype(int)
    scores = {k: np.asarray(z[k], dtype=float) for k in z.files if k not in ("y_true", "class_names")}
    return y, scores


def show(title: str, exp_dir: str) -> dict[str, np.ndarray]:
    y, scores = load(exp_dir)
    print(f"== {title} ({exp_dir}) n_pos={int(y.sum())} n_neg={int((1 - y).sum())}")
    for det in ["Mahalanobis", "KNN", "Energy", "Ensemble"]:
        c = delong_auc_ci(y, scores[det])
        print(f"  {det:12s} AUC={c['auc']:.3f} 95%CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    return y, scores


if __name__ == "__main__":
    y4b, s4b = show("Exp4 DINOv2-B", "exp4_sphor_ncd_dinov2_base")
    y4t, s4t = show("Exp4 ConvNeXt-T", "exp4_sphor_ncd_convnext_tiny")
    y5t, s5t = show("Exp5 ConvNeXt-T", "exp5_sphor_ncd_convnext_tiny")
    y5b, s5b = show("Exp5 DINOv2-B", "exp5_sphor_ncd_dinov2_base")
    for title, y, a, b in [
        ("Exp4 base-KNN vs tiny-KNN", y4b, s4b["KNN"], s4t["KNN"]),
        ("Exp4 base KNN vs Energy", y4b, s4b["KNN"], s4b["Energy"]),
        ("Exp5 tiny-KNN vs base-KNN", y5t, s5t["KNN"], s5b["KNN"]),
    ]:
        r = delong_paired_test(y, a, b)
        print(f"{title}: diff={r['diff']:+.3f} 95%CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_value']:.4f}")
