"""Shared evaluation. Every model reports on all five slices -- never one number."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)

from aicd.config import LABEL_NAMES
from aicd.data.splits import SLICES


def slice_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = proba.argmax(axis=1)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            LABEL_NAMES[i]: float(v)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None,
                                           labels=list(range(4)), zero_division=0))
        },
    }
    # Binary human-vs-rest view: this is the number the product actually acts on.
    y_bin = (y_true != 0).astype(int)
    p_bin = 1.0 - proba[:, 0]
    if len(np.unique(y_bin)) == 2:
        out["binary_auc"] = float(roc_auc_score(y_bin, p_bin))
        out["binary_brier"] = float(brier_score_loss(y_bin, p_bin))
        # FPR on the human class at the argmax decision -- the number that
        # determines whether the system falsely accuses someone.
        human = y_true == 0
        out["human_fpr"] = float((y_pred[human] != 0).mean()) if human.any() else None
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()
    return out


def evaluate_all(df, proba_by_slice: dict[str, tuple], name: str, reports_dir: Path) -> dict:
    results = {"model": name, "slices": {}}
    for s, (y, p) in proba_by_slice.items():
        if len(y) == 0:
            continue
        results["slices"][s] = slice_metrics(np.asarray(y), np.asarray(p))
    dest = Path(reports_dir) / f"{name}.json"
    # A fresh checkout has no eval/reports directory, because the upload
    # package deliberately omits it. Without this the write fails after the
    # whole evaluation has already run.
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_table(results: dict) -> None:
    print(f"\n{'slice':22s} {'n':>7s} {'macroF1':>8s} {'wF1':>7s} {'acc':>7s} {'AUC':>7s} {'humFPR':>7s}")
    print("-" * 70)
    for s in ["s1_in_distribution"] + [x for x in SLICES if x != "s1_in_distribution"]:
        m = results["slices"].get(s)
        if not m:
            continue
        auc = m.get("binary_auc")
        fpr = m.get("human_fpr")
        print(
            f"{s:22s} {m['n']:>7,} {m['macro_f1']:>8.4f} {m['weighted_f1']:>7.4f} "
            f"{m['accuracy']:>7.4f} {auc if auc is None else round(auc,4):>7} "
            f"{fpr if fpr is None else round(fpr,4):>7}"
        )
