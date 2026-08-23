"""The properties E1's comparison depends on.

D2 minus D1_small is only the exposure effect if the two arms are identical
except for exposure. Every test here pins one half of that sentence: the arms
share their retained rows exactly, only one of them sees a withheld category,
and the conditions the result is quoted on did not move underneath it.

These run on a small synthetic corpus rather than the real one, so they stay
fast and do not need the built artifacts to be present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aicd.data.exposure_arms import DONORS, PROTECTED, build, checks


class _Splits:
    holdout_families = ["microsoft", "mistralai"]
    holdout_languages = ["go", "javascript"]
    holdout_sources = ["THEVAULT_CLASS", "ARXIV"]


class _Project:
    seed = 20260822


class _Cfg:
    splits = _Splits()
    project = _Project()


def _corpus(n_train=4000, seed=0) -> pd.DataFrame:
    """A corpus shaped like the real one: clean training rows, and conditions
    made entirely of rows carrying a withheld category."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_train):
        rows.append(dict(slice="train", model_family="qwen", language="python",
                         source="TACO", label=i % 4, problem_id=f"tr{i}"))
    for i in range(600):
        rows.append(dict(slice="val", model_family="qwen", language="python",
                         source="TACO", label=i % 4, problem_id=f"va{i}"))
    for i in range(600):
        rows.append(dict(slice="s1_in_distribution", model_family="qwen",
                         language="python", source="TACO", label=i % 4,
                         problem_id=f"s1{i}"))
    for i in range(900):
        rows.append(dict(slice="s2_unseen_generator",
                         model_family=["microsoft", "mistralai"][i % 2],
                         language="python", source="TACO", label=i % 4,
                         problem_id=f"s2{i}"))
    for i in range(900):
        rows.append(dict(slice="s3_unseen_language", model_family="qwen",
                         language=["go", "javascript"][i % 2], source="TACO",
                         label=i % 4, problem_id=f"s3{i}"))
    for i in range(900):
        rows.append(dict(slice="s4_unseen_domain", model_family="qwen",
                         language="python",
                         source=["THEVAULT_CLASS", "ARXIV"][i % 2],
                         label=i % 4, problem_id=f"s4{i}"))
    for i in range(200):
        rows.append(dict(slice="s5_compound", model_family="qwen",
                         language="go", source="THEVAULT_CLASS", label=i % 4,
                         problem_id=f"s5{i}"))
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def _built(exposure=0.356):
    df, rep = build(_corpus(), _Cfg(), exposure)
    checks(df, _Cfg(), rep)
    return df, rep


def test_arms_have_the_intended_sizes():
    df, rep = _built()
    assert rep["d2_rows"] == rep["d1_rows"]
    assert rep["d1small_rows"] == rep["d2_rows"] - rep["d2_withheld_rows"]


def test_realised_exposure_is_close_to_the_target():
    _, rep = _built(0.356)
    assert abs(rep["d2_exposure_realised"] - 0.356) < 0.02


def test_d1small_is_exactly_d2s_retained_half():
    """The arms must share these rows, not merely have the same number."""
    df, _ = _built()
    small = set(df.index[df["d1small_train"]])
    d2_clean = set(df.index[df["d2_train"] & (df["model_family"] == "qwen")
                            & (df["language"] == "python")
                            & (df["source"] == "TACO")])
    assert small == d2_clean


def test_the_unexposed_arm_sees_no_withheld_category():
    df, _ = _built()
    s = df[df["d1small_train"]]
    assert not s["model_family"].isin(_Splits.holdout_families).any()
    assert not s["language"].isin(_Splits.holdout_languages).any()
    assert not s["source"].isin(_Splits.holdout_sources).any()


def test_the_exposed_arm_sees_every_withheld_category():
    df, rep = _built()
    for v in (_Splits.holdout_families + _Splits.holdout_languages
              + _Splits.holdout_sources):
        assert rep["d2_category_rows"][str(v)] > 0, v


def test_protected_conditions_do_not_move():
    _, rep = _built()
    for s in PROTECTED:
        assert rep["conditions"][s]["before"] == rep["conditions"][s]["after"]


def test_donors_only_come_from_the_donor_conditions():
    df, _ = _built()
    moved = df[df["arm_slice"] == "train_d2"]
    assert set(moved["slice"]) <= set(DONORS)


def test_no_condition_shares_a_group_with_training():
    df, _ = _built()
    pids = set(df.loc[df["d2_train"], "problem_id"])
    for s in ("s1_in_distribution", "s5_compound") + DONORS:
        g = df[df["arm_slice"] == s]
        assert not g["problem_id"].isin(pids).any(), s


def test_it_refuses_rather_than_silently_shrinking():
    """Asking for more withheld rows than exist must fail loudly."""
    with pytest.raises(SystemExit):
        df, rep = build(_corpus(), _Cfg(), 0.95)
        checks(df, _Cfg(), rep)


def test_the_build_is_deterministic():
    a, _ = _built()
    b, _ = _built()
    assert a["d2_train"].equals(b["d2_train"])
    assert a["d1small_train"].equals(b["d1small_train"])


# --------------------------------------------------------------------------
# Reshaping an arm into the split/slice columns the trainer reads. The failure
# that matters here is silent: if the rows belonging only to the other arm were
# left as evaluation rather than marked unused, each arm would be scored partly
# on the other's training data.
# --------------------------------------------------------------------------

def _frame(arm):
    from aicd.data.exposure_arms import as_training_frame
    df, rep = _built()
    return as_training_frame(df, arm), df


@pytest.mark.parametrize("arm", ["d2", "d1small"])
def test_no_training_row_is_also_an_evaluation_row(arm):
    f, _ = _frame(arm)
    tr = set(f.index[f["split"] == "train"])
    ev = set(f.index[f["split"] == "test"])
    assert not (tr & ev)


@pytest.mark.parametrize("arm", ["d2", "d1small"])
def test_no_group_straddles_the_training_boundary(arm):
    f, _ = _frame(arm)
    tr = set(f.loc[f["split"] == "train", "problem_id"])
    ev = set(f.loc[f["split"] == "test", "problem_id"])
    assert not (tr & ev)


def test_the_other_arms_rows_become_unused_not_evaluation():
    f, raw = _frame("d1small")
    only_d2 = raw.index[raw["d2_train"] & ~raw["d1small_train"]]
    assert (f.loc[only_d2, "slice"] == "unused").all()


def test_both_arms_are_scored_on_the_same_conditions():
    """D2 minus D1_small is only a comparison if the denominator is shared."""
    a, _ = _frame("d2")
    b, _ = _frame("d1small")
    for s in ("s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound", "val"):
        assert (a["slice"] == s).sum() == (b["slice"] == s).sum(), s


def test_only_the_exposed_arm_trains_on_withheld_rows():
    from aicd.data.exposure_arms import withheld_mask
    d2, _ = _frame("d2")
    ds, _ = _frame("d1small")
    assert withheld_mask(d2[d2["split"] == "train"], _Cfg()).sum() > 0
    assert withheld_mask(ds[ds["split"] == "train"], _Cfg()).sum() == 0
