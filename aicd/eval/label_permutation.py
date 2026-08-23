"""Is DroidDetect-Large broken, or is its class order different from ours?

Our reconstruction of DroidDetect-Large behaves strangely: it holds a high
binary AUC while its four-way decision is degenerate, predicting one class for
most of the input. That combination is diagnostic. A model that had genuinely
failed to learn would lose the binary signal too; a model whose classifier head
indexes the four classes in a different order than we assume keeps every bit of
its discriminative power and simply attaches the wrong names to it.

The paper currently treats the model as unreconstructible. Before saying that in
print, the cheap check is to try every ordering. There are only 4! = 24 of them,
the probabilities are already scored and saved, and permuting columns of an
array costs nothing. If some permutation recovers strong four-way performance,
the artifact was never broken and the paper's claim about it has to change.

This needs no GPU. It reuses arrays produced by an earlier run.

    python -m aicd.eval.label_permutation
    python -m aicd.eval.label_permutation --variant test
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "kaggle_runs" / "results" / "capacity" / "arrays"
LABEL_NAMES = ["human", "machine", "hybrid", "adversarial"]
HUMAN = 0


def binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Human against everything else.

    Reported because it is what makes the diagnosis possible: it is invariant
    to how the other three classes are ordered among themselves, so it measures
    whether the model can separate human code at all, independent of the naming
    question.
    """
    if len(np.unique(y == HUMAN)) < 2:
        return float("nan")
    return float(roc_auc_score((y != HUMAN).astype(int), 1.0 - p[:, HUMAN]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", choices=["", "test"],
                    help="'test' uses the held-out-shard arrays")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"_{args.variant}" if args.variant else ""
    pattern = str(ARRAYS / f"proba_droiddetect_large{tag}_*.npy")
    files = sorted(f for f in glob.glob(pattern) if "_val" not in f)
    if not files:
        raise SystemExit(f"no arrays matching {pattern}")

    perms = list(itertools.permutations(range(4)))
    report = {"variant": args.variant or "default", "conditions": {}}

    print(f"{'condition':22s} {'n':>6s} {'as-is':>8s} {'best':>8s} "
          f"{'permutation':>16s} {'binaryAUC':>10s}")
    print("-" * 76)

    totals = np.zeros(len(perms))
    for f in files:
        cond = os.path.basename(f).replace(
            f"proba_droiddetect_large{tag}_", "").replace(".npy", "")
        rows = ARRAYS / f"rows_droiddetect_large{tag}_{cond}.parquet"
        if not rows.exists():
            print(f"{cond:22s}  no row file, skipped")
            continue
        y = pd.read_parquet(rows)["label"].to_numpy()
        p = np.load(f)
        if len(y) != len(p):
            print(f"{cond:22s}  {len(y)} labels against {len(p)} rows, skipped")
            continue

        scores = []
        for perm in perms:
            # perm[i] is the class our label i corresponds to in their head, so
            # reordering the columns puts their probabilities under our names.
            scores.append(f1_score(y, p[:, list(perm)].argmax(1),
                                   average="macro", zero_division=0))
        scores = np.asarray(scores)
        totals += scores
        best = int(scores.argmax())
        report["conditions"][cond] = {
            "n": int(len(y)),
            "identity_macro_f1": float(scores[0]),
            "best_macro_f1": float(scores[best]),
            "best_permutation": list(perms[best]),
            "binary_auc": binary_auc(y, p),
        }
        print(f"{cond:22s} {len(y):>6,} {scores[0]:8.4f} {scores[best]:8.4f} "
              f"{str(perms[best]):>16s} {binary_auc(y, p):10.4f}")

    if not report["conditions"]:
        raise SystemExit("nothing scored")

    # One ordering has to explain every condition. A different winner per
    # condition would mean we are fitting noise, not finding a mapping.
    k = len(report["conditions"])
    consistent = int(totals.argmax())
    report["best_overall_permutation"] = list(perms[consistent])
    report["mean_macro_f1_identity"] = float(totals[0] / k)
    report["mean_macro_f1_best"] = float(totals[consistent] / k)

    print("\n" + "=" * 76)
    print("DIAGNOSIS")
    print("=" * 76)
    print(f"  best single ordering across all {k} conditions: {perms[consistent]}")
    print(f"    as our labels are indexed : {report['mean_macro_f1_identity']:.4f} mean macro-F1")
    print(f"    under that ordering       : {report['mean_macro_f1_best']:.4f}")
    gain = report["mean_macro_f1_best"] - report["mean_macro_f1_identity"]
    print(f"    gain                      : {gain:+.4f}")
    print()
    if perms[consistent] == (0, 1, 2, 3):
        print("  The identity ordering is already the best one. The degenerate")
        print("  four-way behaviour is not a label mapping error, and the")
        print("  reproducibility note in the paper stands as written.")
    elif report["mean_macro_f1_best"] > 0.6:
        print("  A permutation recovers strong four-way performance. The")
        print("  artifact was NOT unreconstructible; our class indices were")
        print("  wrong. The paper's claim about it must be corrected, and the")
        print(f"  mapping is {dict(zip(LABEL_NAMES, perms[consistent]))}.")
    else:
        print("  No ordering recovers usable four-way performance, so the")
        print("  behaviour is not explained by a permuted head. That is worth")
        print("  stating explicitly: we tried all 24 and none worked.")

    dest = Path(args.out) if args.out else (
        ROOT / "aicd" / "eval" / "reports"
        / f"label_permutation{tag or '_default'}.json")
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
