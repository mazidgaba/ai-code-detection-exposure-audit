"""The conditional transformation effect, and why the obvious shortcut is wrong.

The paper ranks semantics-preserving rewrites by how much macro-F1 falls when
each is applied to a whole evaluation set. That ranking is not safe, because
the rewrites fire at very different rates: renaming identifiers altered 100% of
files while stripping comments altered 81%. A rewrite that fires more often
moves the aggregate more, whatever its per-file effect, so the two deltas were
never comparable and the headline claim that renaming costs the most may
survive or may reverse.

The revision plan proposes fixing this by dividing the aggregate delta by the
application rate, giving strip-comments -0.4112 / 0.81 = -0.51 against
renaming's -0.4671, and concludes the ranking reverses.

That arithmetic is not valid, and the first test below is a counterexample.
Macro-F1 is an average of per-class F1 scores, each a ratio of counts. It does
not decompose as a weighted sum over rows, so the quotient is not the effect on
the altered subset and can err in either direction. The sound comparison scores
the baseline and the rewritten inputs on the same altered rows, which is what
the module now does.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def conditional(y, base_p, new_p, changed):
    """What the module computes: both sides scored on the altered rows."""
    before = f1_score(y[changed], base_p[changed].argmax(1),
                      average="macro", zero_division=0)
    after = f1_score(y[changed], new_p[changed].argmax(1),
                     average="macro", zero_division=0)
    return after - before


def quotient(y, base_p, new_p, changed):
    """What the plan proposes: aggregate delta divided by application rate."""
    agg = (f1_score(y, new_p.argmax(1), average="macro", zero_division=0)
           - f1_score(y, base_p.argmax(1), average="macro", zero_division=0))
    return agg / changed.mean()


def _case(n=400, k=4, rate=0.5, seed=0, break_all=True):
    """A rewrite that fires on `rate` of rows and destroys those predictions."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    base = np.zeros((n, k))
    base[np.arange(n), y] = 1.0            # baseline is perfect
    changed = np.zeros(n, dtype=bool)
    changed[rng.choice(n, int(n * rate), replace=False)] = True
    new = base.copy()
    if break_all:                          # altered rows all become class 0
        new[changed] = 0.0
        new[changed, 0] = 1.0
    return y, base, new, changed


def test_the_quotient_disagrees_with_the_real_conditional_effect():
    """The counterexample. Both numbers describe the same rewrite."""
    y, base, new, changed = _case()
    c = conditional(y, base, new, changed)
    q = quotient(y, base, new, changed)
    assert abs(c - q) > 0.05, (
        f"expected the two to diverge materially, got {c:.4f} and {q:.4f}")


def test_a_rewrite_that_changes_nothing_has_no_conditional_effect():
    y, base, new, changed = _case(break_all=False)
    assert conditional(y, base, new, changed) == 0.0


def test_application_rate_does_not_move_the_conditional_effect():
    """The whole point: two rewrites equally damaging where they fire should
    score the same conditionally, however often they fire. The aggregate
    delta cannot do this, which is why the ranking needed fixing."""
    a = conditional(*_case(rate=0.25, seed=1))
    b = conditional(*_case(rate=0.95, seed=1))
    assert abs(a - b) < 0.12, (a, b)


def test_the_aggregate_delta_does_move_with_application_rate():
    """The complement, stated so the reason for the change is on record."""
    def agg(rate):
        y, base, new, changed = _case(rate=rate, seed=2)
        return (f1_score(y, new.argmax(1), average="macro", zero_division=0)
                - f1_score(y, base.argmax(1), average="macro", zero_division=0))
    assert abs(agg(0.25)) < abs(agg(0.95))
