"""DeLong AUROC confidence intervals + paired AUC comparison.

Reference: DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Comparing the
areas under two or more correlated ROC curves: a nonparametric approach.
Biometrics 44:837-845.

The empirical AUC is a Mann-Whitney U-statistic. Variance comes in closed
form from placement values; the paired test accounts for the covariance of
two score vectors on the same subjects. No resampling required.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

_STD = NormalDist()


def _psi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise kernel: 1.0 if a > b, 0.5 if tied, else 0.0."""
    return np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))


def _placements(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y_true must be binary 0/1")
    pos = s[y == 1][:, None]
    neg = s[y == 0][None, :]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValueError("need both classes present")
    kernel = _psi(pos, neg)
    auc = float(kernel.mean())
    v10 = kernel.mean(axis=1)
    v01 = kernel.mean(axis=0)
    return auc, v10, v01


def delong_auc_ci(y_true: np.ndarray, scores: np.ndarray, alpha: float = 0.05) -> dict:
    """Single-AUC DeLong CI. Returns dict with auc, var, ci_low, ci_high, n_pos, n_neg."""
    auc, v10, v01 = _placements(y_true, scores)
    m, n = len(v10), len(v01)
    s10 = float(v10.var(ddof=1)) if m > 1 else 0.0
    s01 = float(v01.var(ddof=1)) if n > 1 else 0.0
    var = s10 / m + s01 / n
    z = abs(_STD.inv_cdf(1.0 - alpha / 2.0))
    half = z * float(np.sqrt(var))
    return {
        "auc": auc,
        "var": float(var),
        "ci_low": float(max(0.0, auc - half)),
        "ci_high": float(min(1.0, auc + half)),
        "n_pos": m,
        "n_neg": n,
    }


def delong_paired_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired DeLong test for two score vectors on the SAME subjects."""
    auc_a, v10_a, v01_a = _placements(y_true, scores_a)
    auc_b, v10_b, v01_b = _placements(y_true, scores_b)
    m = len(v10_a)
    n = len(v01_a)
    s10_a = float(v10_a.var(ddof=1)) if m > 1 else 0.0
    s01_a = float(v01_a.var(ddof=1)) if n > 1 else 0.0
    s10_b = float(v10_b.var(ddof=1)) if m > 1 else 0.0
    s01_b = float(v01_b.var(ddof=1)) if n > 1 else 0.0
    with np.errstate(invalid="ignore"):
        c10 = float(np.nan_to_num(np.cov(v10_a, v10_b, ddof=1)[0, 1])) if m > 1 else 0.0
        c01 = float(np.nan_to_num(np.cov(v01_a, v01_b, ddof=1)[0, 1])) if n > 1 else 0.0
    var_diff = (s10_a + s10_b - 2.0 * c10) / m + (s01_a + s01_b - 2.0 * c01) / n
    var_diff = max(var_diff, 0.0)
    diff = auc_a - auc_b
    se = float(np.sqrt(var_diff))
    if se == 0.0:
        z, p = 0.0, 1.0 if diff == 0.0 else 0.0
    else:
        z = diff / se
        p = 2.0 * (1.0 - _STD.cdf(abs(z)))
    zc = abs(_STD.inv_cdf(1.0 - alpha / 2.0))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": diff,
        "se": se,
        "z": float(z),
        "p_value": float(min(max(p, 0.0), 1.0)),
        "ci_low": diff - zc * se,
        "ci_high": diff + zc * se,
        "n_pos": m,
        "n_neg": n,
    }
