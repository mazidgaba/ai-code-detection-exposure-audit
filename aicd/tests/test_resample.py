"""Correctness of the single bootstrap protocol.

An interval-producing routine that nobody checked is worse than reporting point
estimates, because it produces numbers that look like evidence. These tests
check the three things that could silently be wrong: the fast metric, the
interval's coverage, and the pairing in the difference estimator.

Coverage is checked against a case with a known answer rather than against
another implementation, so agreement is evidence about the method and not about
two routines sharing a bug.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from aicd.eval import resample as R


def _rand(n, k=4, seed=0, agree=0.7):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < agree, y, rng.integers(0, k, n))
    return y, pred


# --------------------------------------------------------------------------
# The fast metric must equal the slow one it replaces.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_macro_f1_matches_sklearn(seed):
    y, pred = _rand(4000, seed=seed)
    assert R.macro_f1(y, pred) == pytest.approx(
        f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0))


def test_macro_f1_present_ignores_absent_classes():
    """A class that cannot occur must not be averaged in at zero."""
    y = np.array([0, 0, 1, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])
    got = R.macro_f1_present(y, pred)
    want = f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)
    assert got == pytest.approx(want)
    # Over all four classes the same data scores lower, purely from the two
    # classes that never appear. That difference is the arithmetic artefact.
    assert R.macro_f1(y, pred) < got


def test_class_f1_matches_sklearn_per_class():
    y, pred = _rand(2000, seed=5)
    per = f1_score(y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    for c in range(4):
        assert R.class_f1(y, pred, c) == pytest.approx(per[c])


def test_human_fpr_is_the_share_of_human_rows_not_called_human():
    y = np.array([0, 0, 0, 0, 1, 2])
    pred = np.array([0, 1, 2, 3, 1, 2])
    assert R.human_fpr(y, pred) == pytest.approx(0.75)


# --------------------------------------------------------------------------
# The interval itself.
# --------------------------------------------------------------------------

def test_interval_brackets_the_point_estimate():
    y, pred = _rand(3000, seed=7)
    theta, lo, hi = R.bca(y, pred, R.macro_f1, n_boot=800, seed=1)
    assert lo <= theta <= hi
    assert theta == pytest.approx(R.macro_f1(y, pred))


def test_interval_narrows_as_the_sample_grows():
    """The one property every correct interval has."""
    widths = []
    for n in (500, 5000, 20000):
        y, pred = _rand(n, seed=3)
        _, lo, hi = R.bca(y, pred, R.macro_f1, n_boot=600, seed=2)
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2], widths


def test_bca_attains_nominal_coverage_on_a_known_case():
    """Coverage against a case whose true value is known by construction.

    The statistic is the human false-positive rate on data generated with a
    fixed true rate, so the truth is 0.30 exactly. A 95% interval should cover
    it in about 95 of 100 trials. This is the check that distinguishes a real
    interval from a plausible-looking one.
    """
    truth, trials, covered = 0.30, 100, 0
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        n_human = 400
        y = np.zeros(n_human, dtype=int)
        pred = (rng.random(n_human) < truth).astype(int)
        _, lo, hi = R.bca(y, pred, R.human_fpr, n_boot=600, seed=t, n_groups=50)
        covered += lo <= truth <= hi
    # Binomial(100, 0.95) has sd 2.2, so anything from 88 up is consistent with
    # nominal; below that the method is broken rather than unlucky.
    assert covered >= 88, f"covered {covered}/100, expected about 95"


def test_bca_differs_from_percentile_on_a_skewed_statistic():
    """If BCa never moved the endpoints it would be percentile with extra steps."""
    rng = np.random.default_rng(11)
    n = 300
    y = np.zeros(n, dtype=int)
    pred = (rng.random(n) < 0.93).astype(int)      # near the upper bound, skewed
    _, lo, hi = R.bca(y, pred, R.human_fpr, n_boot=3000, seed=4, n_groups=50)

    reps = []
    r2 = np.random.default_rng(4)
    for _ in range(3000):
        i = r2.integers(0, n, n)
        reps.append(R.human_fpr(y[i], pred[i]))
    p_lo, p_hi = np.percentile(reps, [2.5, 97.5])
    assert (abs(lo - p_lo) > 1e-6) or (abs(hi - p_hi) > 1e-6)


# --------------------------------------------------------------------------
# The paired difference.
# --------------------------------------------------------------------------

def test_difference_is_paired_not_independent():
    """Identical arms must give a difference of exactly zero with a tight band.

    If the two arms were resampled independently the interval would be wide
    even when the arms are the same, which is the error this guards.
    """
    y, pred = _rand(2000, seed=13)
    theta, lo, hi = R.bca_difference(y, pred, y, pred, R.macro_f1,
                                     n_boot=500, seed=6)
    assert theta == 0.0
    assert lo == 0.0 and hi == 0.0


def test_difference_rejects_arms_scored_on_different_rows():
    """The comparison is meaningless unless both arms saw the same rows."""
    ya, pa = _rand(500, seed=1)
    yb, pb = _rand(500, seed=2)
    with pytest.raises(ValueError):
        R.bca_difference(ya, pa, yb, pb, R.macro_f1, n_boot=50)
    with pytest.raises(ValueError):
        R.bca_difference(ya, pa, ya[:400], pa[:400], R.macro_f1, n_boot=50)


def test_difference_recovers_a_known_gap():
    y, pred_good = _rand(4000, seed=17, agree=0.85)
    rng = np.random.default_rng(18)
    # Degrade a tenth of the predictions to make a real, positive gap.
    pred_bad = pred_good.copy()
    flip = rng.random(len(y)) < 0.10
    pred_bad[flip] = rng.integers(0, 4, flip.sum())

    truth = R.macro_f1(y, pred_good) - R.macro_f1(y, pred_bad)
    theta, lo, hi = R.bca_difference(y, pred_bad, y, pred_good, R.macro_f1,
                                     n_boot=800, seed=9)
    assert theta == pytest.approx(truth)
    assert lo > 0, "a real degradation should give an interval clear of zero"
    assert lo <= truth <= hi


# --------------------------------------------------------------------------
# The two-sample (unpaired) difference, used by the contamination test.
# --------------------------------------------------------------------------

def test_two_sample_recovers_a_known_gap_between_disjoint_groups():
    ya, pa = _rand(3000, seed=21, agree=0.90)
    yb, pb = _rand(2500, seed=22, agree=0.70)
    truth = R.macro_f1(ya, pa) - R.macro_f1(yb, pb)
    theta, lo, hi = R.bca_two_sample(ya, pa, yb, pb, R.macro_f1,
                                     n_boot=800, seed=3)
    assert theta == pytest.approx(truth)
    assert lo <= truth <= hi
    assert lo > 0, "a real gap should give an interval clear of zero"


def test_two_sample_covers_zero_when_the_groups_are_alike():
    """Two independent draws from the same process must not look different."""
    ya, pa = _rand(2000, seed=31, agree=0.8)
    yb, pb = _rand(2000, seed=32, agree=0.8)
    _, lo, hi = R.bca_two_sample(ya, pa, yb, pb, R.macro_f1, n_boot=800, seed=5)
    assert lo <= 0 <= hi, f"[{lo:.4f}, {hi:.4f}] excludes zero for like samples"


def test_two_sample_is_wider_than_treating_the_same_data_as_paired():
    """Pairing disjoint groups would understate the variance, so guard it.

    Two independent samples carry both arms' noise; a paired estimator on the
    same numbers cancels noise that is not in fact shared. The unpaired
    interval must therefore be the wider of the two.
    """
    y, p_good = _rand(2000, seed=41, agree=0.9)
    rng = np.random.default_rng(42)
    p_bad = p_good.copy()
    flip = rng.random(len(y)) < 0.15
    p_bad[flip] = rng.integers(0, 4, flip.sum())

    _, ul, uh = R.bca_two_sample(y, p_good, y, p_bad, R.macro_f1, n_boot=800, seed=7)
    _, pl, ph = R.bca_difference(y, p_bad, y, p_good, R.macro_f1, n_boot=800, seed=7)
    assert (uh - ul) > (ph - pl)


def test_macro_f1_across_arms_with_a_class_absent_from_one():
    """The metric error that inverted the contamination result.

    One arm contains no rows of class 3, the other does. A four-class macro-F1
    charges the first arm for a class it cannot contain, so the comparison comes
    out backwards. Averaging over the classes present in both is the comparison
    that means something.
    """
    rng = np.random.default_rng(77)
    n = 3000
    ya = rng.choice([0, 1, 2], n)            # no class 3 at all
    pa = np.where(rng.random(n) < 0.95, ya, rng.choice([0, 1, 2], n))
    yb = rng.choice([0, 1, 2, 3], n)         # class 3 present
    pb = np.where(rng.random(n) < 0.95, yb, rng.choice([0, 1, 2, 3], n))

    # Arm A is the more accurate of the two on the classes they share.
    shared = sorted(set(np.unique(ya)) & set(np.unique(yb)))
    ma, mb = np.isin(ya, shared), np.isin(yb, shared)
    matched = (R.macro_f1_present(ya[ma], pa[ma])
               - R.macro_f1_present(yb[mb], pb[mb]))
    assert matched > 0, "arm A should lead on the classes both contain"

    # The four-class comparison reverses the sign purely from the absent class.
    naive = R.macro_f1(ya, pa) - R.macro_f1(yb, pb)
    assert naive < 0 < matched, (
        f"four-class gap {naive:+.4f}, class-matched {matched:+.4f}; the naive "
        "form should be the one that inverts")
