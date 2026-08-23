"""Invariants for the label-shift diagnosis.

The conclusion this module supports is a negative one: no post-hoc correction
recovers the collapse. A negative result is exactly the kind that a silent bug
can manufacture, because a correction that is quietly a no-op looks identical
to a correction that genuinely does not help. These tests pin the arithmetic so
that the difference stays visible.

The first test is the one that caught nothing and proved the most: temperature
scaling is monotone per row, so it cannot change which class wins, so macro-F1
must come back bit-identical. Seeing that hold in the real run is what made the
rest of the output trustworthy.
"""
from __future__ import annotations

import numpy as np

from aicd.eval.shift_diagnosis import (
    apply_prior_shift, apply_temperature, as_logits, bbse_priors, brier,
    ece, em_priors, fit_temperature, macro_f1, priors, softmax,
)


def _sample(n=600, k=4, seed=11):
    rng = np.random.default_rng(seed)
    p = softmax(rng.normal(size=(n, k)) * 2.5)
    y = np.array([rng.choice(k, p=row) for row in p])
    return y, p


def test_logit_round_trip():
    """log then softmax is the identity, which is what makes the recovery exact."""
    _, p = _sample()
    assert np.allclose(softmax(as_logits(p)), p, atol=1e-10)


def test_temperature_cannot_change_the_decision():
    """Any temperature leaves argmax, and therefore macro-F1, untouched."""
    y, p = _sample()
    base = macro_f1(y, p)
    for t in (0.2, 0.5, 1.0, 2.0, 7.5, 40.0):
        assert macro_f1(y, apply_temperature(p, t)) == base


def test_temperature_does_change_calibration():
    """The complement: it is not a no-op, it just acts on confidence only."""
    y, p = _sample()
    assert ece(y, apply_temperature(p, 6.0)) != ece(y, p)
    assert brier(y, apply_temperature(p, 6.0)) != brier(y, p)


def test_prior_shift_is_identity_when_priors_match():
    _, p = _sample()
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(apply_prior_shift(p, pi, pi), p, atol=1e-12)


def test_prior_shift_returns_a_distribution():
    _, p = _sample()
    out = apply_prior_shift(p, np.array([0.4, 0.3, 0.2, 0.1]),
                            np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all()


def test_bbse_and_em_recover_priors_when_nothing_shifted():
    """With target drawn from the source, both estimators must return source."""
    y, p = _sample(n=4000, seed=3)
    pi = priors(y)
    assert np.abs(bbse_priors(y, p, p) - pi).sum() < 0.05
    assert np.abs(em_priors(pi, p) - pi).sum() < 0.10


def test_fitted_temperature_improves_likelihood():
    """A fitted temperature is at least as good as leaving it at one."""
    y, p = _sample(n=2000, seed=5)
    t = fit_temperature(y, p)

    def nll(q):
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean()

    assert nll(apply_temperature(p, t)) <= nll(p) + 1e-9
