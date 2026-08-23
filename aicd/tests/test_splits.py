"""Guarantees for the OOD split matrix. If these fail, every metric downstream is fiction."""
from __future__ import annotations

import pandas as pd
import pytest

from aicd import config as C
from aicd.data.splits import SLICES

CFG = C.load("base.yaml")
PATH = C.ROOT / CFG.data.cache_dir / "splits.parquet"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="run data/splits.py first")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(PATH)


def _slice(df, s):
    return df[df["slice"] == s]


def test_no_problem_id_leak(df):
    train = set(df.loc[df["slice"] == "train", "problem_id"])
    for s in SLICES:
        g = _slice(df, s)
        if not len(g):
            continue
        assert not (set(g["problem_id"]) & train), f"{s} shares problem_id with train"


def test_val_disjoint_from_train(df):
    train = set(df.loc[df["slice"] == "train", "problem_id"])
    val = set(df.loc[df["slice"] == "val", "problem_id"])
    assert not (train & val)


def test_no_model_family_leak_s2(df):
    held = set(CFG.splits.holdout_families)
    train_fams = set(df.loc[df["slice"] == "train", "model_family"])
    assert not (train_fams & held), f"held-out families present in train: {train_fams & held}"


def test_no_language_leak_s3(df):
    held = set(CFG.splits.holdout_languages)
    train_langs = set(df.loc[df["slice"] == "train", "language"])
    assert not (train_langs & held)


def test_no_source_leak_s4(df):
    held = set(CFG.splits.holdout_sources)
    train_doms = set(df.loc[df["slice"] == "train", "source"])
    assert not (train_doms & held)


def test_s5_is_compound(df):
    g = _slice(df, "s5_compound")
    if not len(g):
        pytest.skip("no compound rows in this corpus")
    assert set(g["language"]) <= set(CFG.splits.holdout_languages)
    assert set(g["source"]) <= set(CFG.splits.holdout_sources)


def test_slices_have_multiple_labels(df):
    for s in SLICES:
        g = _slice(df, s)
        if not len(g):
            continue
        assert g["label"].nunique() >= 2, f"{s} is single-class, AUC/F1 undefined"


def test_train_has_all_four_labels(df):
    train = _slice(df, "train")
    assert train["label"].nunique() == 4, (
        "training set is missing a class -- never train this binary, "
        "hybrid F1 collapses from 86 to 39"
    )
