"""E8's augmentation, and the ways it could silently do nothing.

The experiment claims to train through semantics-preserving rewrites. Two
failures would leave that claim false while the run looked normal: the rewrites
firing on nothing, so the model trains on the original corpus while the report
says otherwise; and the augmentation being non-deterministic, so a resumed
session continues on a different corpus than the one its checkpoint was trained
on. Both are asserted against here.

The third property is the one that makes the experiment mean anything: labels
must not move. A renamed machine-generated file is still machine-generated, and
if augmentation touched the labels the result would be about label noise rather
than about surface cues.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aicd.models.modernbert_triplet import augment


class _Cfg:
    class modernbert:
        augment_fraction = 0.5
        augment_transforms = ["rename_identifiers", "whitespace"]

    class project:
        seed = 20260822


def _frame(n=200):
    code = ("def compute_total(values):\n"
            "    running_total = 0\n"
            "    for element in values:\n"
            "        running_total += element\n"
            "    return running_total\n")
    return pd.DataFrame({
        "code": [code + f"# row {i}\n" for i in range(n)],
        "label": [i % 4 for i in range(n)],
        "language": ["python"] * n,
        "split": ["train"] * n,
    })


def test_it_actually_rewrites_rows():
    df = _frame()
    out = augment(df, _Cfg, "t")
    differing = (out["code"].to_numpy() != df["code"].to_numpy()).sum()
    assert differing > 0
    # roughly the requested share, allowing for rewrites that no-op on a row
    assert 0.25 * len(df) <= differing <= 0.55 * len(df), differing


def test_labels_are_untouched():
    """A renamed machine-generated file is still machine-generated."""
    df = _frame()
    out = augment(df, _Cfg, "t")
    assert out["label"].equals(df["label"])


def test_row_count_is_unchanged():
    df = _frame()
    assert len(augment(df, _Cfg, "t")) == len(df)


def test_it_is_deterministic():
    """A resumed session must continue on the same corpus, not a fresh draw."""
    a = augment(_frame(), _Cfg, "t")["code"].tolist()
    b = augment(_frame(), _Cfg, "t")["code"].tolist()
    assert a == b


def test_zero_fraction_is_a_no_op():
    class Off(_Cfg):
        class modernbert(_Cfg.modernbert):
            augment_fraction = 0.0
    df = _frame()
    out = augment(df, Off, "t")
    assert out["code"].equals(df["code"])


def test_an_unknown_transform_fails_loudly():
    class Bad(_Cfg):
        class modernbert(_Cfg.modernbert):
            augment_transforms = ["definitely_not_a_transform"]
    with pytest.raises(SystemExit):
        augment(_frame(), Bad, "t")


def test_augmentation_that_changes_nothing_fails_loudly():
    """The silent failure this guards against: the run reports augmentation,
    trains on the original corpus, and nobody notices."""
    class Blank(_Cfg):
        class modernbert(_Cfg.modernbert):
            augment_transforms = ["compress_blanks"]
    df = pd.DataFrame({"code": ["x=1\n"] * 50, "label": [0] * 50,
                       "language": ["python"] * 50, "split": ["train"] * 50})
    with pytest.raises(SystemExit):
        augment(df, Blank, "t")
