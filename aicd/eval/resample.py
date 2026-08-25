"""One bootstrap protocol, used everywhere.

The manuscript currently reports intervals computed three different ways: 800
percentile resamples in the operating-point table, 1,000 in the independent
evaluation, 2,000 by default elsewhere. Nothing is wrong with any of them
individually, and a reader cannot compare two intervals that were produced by
different procedures. This module is the single procedure.

## Why BCa rather than percentile

The percentile interval assumes the bootstrap distribution of the statistic is
unbiased and symmetric. Neither holds here. Macro-F1 near a floor is bounded
below and skewed, and a false-accusation rate near 0.85 is bounded above; in
both cases the percentile interval is shifted in a direction that flatters
whichever claim the number supports. The bias-corrected and accelerated
interval of Efron (1987) corrects for median bias through z0 and for
skewness through the acceleration a, and it is the default in the statistics
literature for exactly this situation.

## The acceleration, and why it is a grouped jackknife

The acceleration term is conventionally estimated by a leave-one-out jackknife,
which needs n recomputations of the statistic. With 56,728 evaluation rows that
is prohibitive and unnecessary: the delete-d jackknife over g groups is a
standard consistent estimator, and at g = 200 the statistic is recomputed 200
times rather than 56,728. The group assignment is a fixed permutation of the
indices, so the estimate is deterministic given the seed.

    from aicd.eval.resample import bca, macro_f1
    lo, hi = bca(y, pred, macro_f1, n_boot=10000, seed=0)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

N_BOOT = 10_000
N_GROUPS = 200
ALPHA = 0.05


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int = 4) -> float:
    """Macro-F1 over classes 0..k-1, via a bincount confusion matrix.

    sklearn's f1_score is correct and far too slow to call ten thousand times.
    This agrees with it exactly, which `test_resample.py` asserts.
    """
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(k, dtype=np.float64), where=denom > 0)
    return float(f1.mean())


def macro_f1_present(y: np.ndarray, pred: np.ndarray, k: int = 4) -> float:
    """Macro-F1 over the classes the sample actually contains.

    Averaging in a class that cannot occur divides by a class the detector was
    never given a chance at, which manufactures a collapse out of arithmetic.
    """
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(k, dtype=np.float64), where=denom > 0)
    present = np.unique(y)
    return float(f1[present].mean())


def class_f1(y: np.ndarray, pred: np.ndarray, cls: int, k: int = 4) -> float:
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = float(cm[cls, cls])
    denom = 2 * tp + (cm[:, cls].sum() - tp) + (cm[cls, :].sum() - tp)
    return float(2 * tp / denom) if denom > 0 else 0.0


def human_fpr(y: np.ndarray, pred: np.ndarray, human: int = 0) -> float:
    m = y == human
    return float((pred[m] != human).mean()) if m.any() else float("nan")


def _grouped_jackknife(y, pred, stat, g: int, seed: int) -> np.ndarray:
    """Delete-d jackknife estimates of the statistic, over g groups."""
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    groups = np.array_split(idx, min(g, n))
    out = np.empty(len(groups), dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    for i, grp in enumerate(groups):
        keep[grp] = False
        out[i] = stat(y[keep], pred[keep])
        keep[grp] = True
    return out


def bca(y: np.ndarray, pred: np.ndarray, stat, n_boot: int = N_BOOT,
        alpha: float = ALPHA, seed: int = 0, n_groups: int = N_GROUPS):
    """Bias-corrected and accelerated interval for `stat(y, pred)`.

    Returns (point, lo, hi). Falls back to the percentile interval, and says so
    by returning acceleration 0, only when the bootstrap distribution is
    degenerate, which happens when every resample gives the same value.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    n = len(y)
    theta = stat(y, pred)

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        reps[b] = stat(y[i], pred[i])

    prop = float((reps < theta).mean())
    if prop <= 0.0 or prop >= 1.0:
        # Degenerate: no bias correction is estimable, so report percentiles.
        lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    z0 = norm.ppf(prop)

    jk = _grouped_jackknife(y, pred, stat, n_groups, seed)
    d = jk.mean() - jk
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def adj(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    q_lo, q_hi = adj(alpha / 2), adj(1 - alpha / 2)
    lo, hi = np.percentile(reps, [100 * q_lo, 100 * q_hi])
    return theta, float(lo), float(hi)


def bca_two_sample(ya, pa, yb, pb, stat, n_boot: int = N_BOOT,
                   alpha: float = ALPHA, seed: int = 0):
    """Interval for stat(a) - stat(b) on two *independent* samples.

    Distinct from `bca_difference`, which pairs. Use this when the two groups
    are different rows rather than the same rows scored twice: the
    contamination test compares rows the detector trained on against rows it
    did not, and those are disjoint sets, so each arm must be resampled to its
    own size independently. Pairing them would be meaningless, and treating
    them as paired would understate the variance.
    """
    ya, pa, yb, pb = map(np.asarray, (ya, pa, yb, pb))
    na, nb = len(ya), len(yb)
    theta = stat(ya, pa) - stat(yb, pb)

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = rng.integers(0, na, na)
        j = rng.integers(0, nb, nb)
        reps[b] = stat(ya[i], pa[i]) - stat(yb[j], pb[j])

    prop = float((reps < theta).mean())
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    z0 = norm.ppf(prop)

    # Jackknife over the pooled index, deleting from whichever arm the group
    # falls in, so the acceleration reflects both samples.
    r2 = np.random.default_rng(seed)
    jk = []
    for arm, (yy, pp, other) in enumerate(((ya, pa, (yb, pb)), (yb, pb, (ya, pa)))):
        n = len(yy)
        groups = np.array_split(r2.permutation(n), min(N_GROUPS // 2, n))
        keep = np.ones(n, dtype=bool)
        for grp in groups:
            keep[grp] = False
            a_val = stat(yy[keep], pp[keep])
            o_val = stat(other[0], other[1])
            jk.append(a_val - o_val if arm == 0 else o_val - a_val)
            keep[grp] = True
    jk = np.asarray(jk)
    d = jk.mean() - jk
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def adj(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo, hi = np.percentile(reps, [100 * adj(alpha / 2), 100 * adj(1 - alpha / 2)])
    return theta, float(lo), float(hi)


def bca_difference(ya, pa, yb, pb, stat, n_boot: int = N_BOOT,
                   alpha: float = ALPHA, seed: int = 0):
    """Interval for stat(b) - stat(a) when both are scored on the same rows.

    A reader cannot tell from two overlapping intervals whether a difference is
    significant, and cannot tell from two disjoint ones how large it is. The
    difference has its own interval and this computes it. The pairing matters:
    both arms are resampled with the *same* indices, so the row-level noise
    common to both cancels rather than being counted twice.
    """
    ya, pa, yb, pb = map(np.asarray, (ya, pa, yb, pb))
    if not (len(ya) == len(yb) == len(pa) == len(pb)):
        raise ValueError("paired difference needs both arms on the same rows")
    if not np.array_equal(ya, yb):
        raise ValueError("the two arms carry different labels, so they were not "
                         "scored on the same rows")
    n = len(ya)
    theta = stat(yb, pb) - stat(ya, pa)

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        reps[b] = stat(yb[i], pb[i]) - stat(ya[i], pa[i])

    prop = float((reps < theta).mean())
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    z0 = norm.ppf(prop)

    idx = np.random.default_rng(seed).permutation(n)
    groups = np.array_split(idx, min(N_GROUPS, n))
    keep = np.ones(n, dtype=bool)
    jk = np.empty(len(groups))
    for i, grp in enumerate(groups):
        keep[grp] = False
        jk[i] = stat(yb[keep], pb[keep]) - stat(ya[keep], pa[keep])
        keep[grp] = True
    d = jk.mean() - jk
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def adj(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo, hi = np.percentile(reps, [100 * adj(alpha / 2), 100 * adj(1 - alpha / 2)])
    return theta, float(lo), float(hi)
