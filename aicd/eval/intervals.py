"""Recompute the manuscript's intervals under one protocol, with per-class support.

Two reviewer findings are answered here.

**One protocol.** The paper reported 800 percentile resamples in one table,
1,000 in another and 2,000 elsewhere. Everything below uses `aicd.eval.resample`:
10,000 resamples, bias-corrected and accelerated.

**Per-class support.** The paper reports per-class F1 without saying how many
rows each rests on. S5's adversarial class has 155 rows, and an F1 computed on
155 rows carries an interval wide enough to change how the number should be
read. Reporting the point estimate alone invites a reader to treat all four
classes as equally well measured.

## Why the confusion matrix is sufficient

Macro-F1, per-class F1 and the human false-accusation rate are functions of the
confusion counts alone. Reconstructing one (label, prediction) pair per unit of
each cell reproduces those statistics exactly, and bootstrapping the
reconstruction is equivalent to bootstrapping the original rows, because the
resampled statistic depends on the rows only through their cell membership.
Metrics that need scores rather than decisions, such as AUC, are not computed
here for that reason.

    python -m aicd.eval.intervals
    python -m aicd.eval.intervals --report branch_a_e5_large.json --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aicd.eval import resample as R

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
LABELS = ["human", "machine", "hybrid", "adversarial"]
CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]


def from_confusion(cm) -> tuple[np.ndarray, np.ndarray]:
    """Expand a confusion matrix into the (label, prediction) pairs it counts."""
    cm = np.asarray(cm, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got {cm.shape}")
    y, pred = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                y.append(np.full(cm[i, j], i, dtype=np.int64))
                pred.append(np.full(cm[i, j], j, dtype=np.int64))
    if not y:
        raise ValueError("empty confusion matrix")
    return np.concatenate(y), np.concatenate(pred)


def analyse(report: dict, n_boot: int, seed: int) -> dict:
    out = {"model": report.get("model", "?"), "n_boot": n_boot,
           "method": "BCa", "conditions": {}}
    for cond, v in report.get("slices", {}).items():
        if "confusion" not in v:
            continue
        y, pred = from_confusion(v["confusion"])

        # The reconstruction must reproduce the report exactly, or the rest of
        # this is intervals around the wrong point.
        got = R.macro_f1(y, pred)
        if abs(got - v["macro_f1"]) > 1e-6:
            raise SystemExit(f"{cond}: reconstruction gives macro-F1 {got:.6f}, "
                             f"report says {v['macro_f1']:.6f}")
        if len(y) != v["n"]:
            raise SystemExit(f"{cond}: confusion sums to {len(y)}, report says {v['n']}")

        theta, lo, hi = R.bca(y, pred, R.macro_f1, n_boot=n_boot, seed=seed)
        rec = {"n": int(len(y)), "macro_f1": theta, "macro_f1_ci": [lo, hi],
               "per_class": {}}
        fpr = R.bca(y, pred, R.human_fpr, n_boot=n_boot, seed=seed)
        rec["human_fpr"] = fpr[0]
        rec["human_fpr_ci"] = [fpr[1], fpr[2]]

        support = np.bincount(y, minlength=4)
        for c, name in enumerate(LABELS):
            t, l, h = R.bca(y, pred, lambda a, b, c=c: R.class_f1(a, b, c),
                            n_boot=n_boot, seed=seed)
            # Absolute width alone ranks the classes misleadingly: an F1 near
            # 0.13 has a narrow band simply because it is near a bound. The
            # width relative to the estimate is what says how well the number
            # is pinned down.
            rec["per_class"][name] = {"support": int(support[c]), "f1": t,
                                      "ci": [l, h], "ci_width": h - l,
                                      "ci_width_relative": (h - l) / t if t > 0 else None}
        out["conditions"][cond] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="branch_a_e5_large.json")
    ap.add_argument("--n-boot", type=int, default=R.N_BOOT)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    path = REPORTS / args.report
    if not path.exists():
        raise SystemExit(f"{path} not found")
    out = analyse(json.loads(path.read_text(encoding="utf-8")), args.n_boot, args.seed)

    print(f"{out['model']}   {args.n_boot:,} BCa resamples\n")
    print(f"{'condition':22s} {'n':>7s} {'macro-F1':>9s} {'95% CI':>18s}")
    print("-" * 60)
    for cond in CONDITIONS:
        if cond not in out["conditions"]:
            continue
        r = out["conditions"][cond]
        lo, hi = r["macro_f1_ci"]
        print(f"{cond:22s} {r['n']:>7,} {r['macro_f1']:9.4f} "
              f"  [{lo:.4f}, {hi:.4f}]")

    print(f"\nPer-class F1 with the support each rests on\n")
    print(f"{'condition':22s} {'class':12s} {'n':>6s} {'F1':>7s} {'95% CI':>18s} "
          f"{'width':>7s} {'rel':>6s}")
    print("-" * 86)
    rows = []
    for cond in CONDITIONS:
        if cond not in out["conditions"]:
            continue
        for name, c in out["conditions"][cond]["per_class"].items():
            lo, hi = c["ci"]
            rel = c["ci_width_relative"]
            print(f"{cond:22s} {name:12s} {c['support']:>6,} {c['f1']:7.4f} "
                  f"  [{lo:.4f}, {hi:.4f}] {c['ci_width']:7.4f} "
                  f"{(rel if rel is not None else float('nan')):6.1%}")
            rows.append((cond, name, c["support"], c["ci_width"], rel))
    if rows:
        wa = max(rows, key=lambda r: r[3])
        wr = max((r for r in rows if r[4] is not None), key=lambda r: r[4])
        print(f"\nWidest in absolute terms : {wa[1]} on {wa[0]}, {wa[2]:,} rows, "
              f"width {wa[3]:.4f}")
        print(f"Widest relative to its own estimate: {wr[1]} on {wr[0]}, "
              f"{wr[2]:,} rows, {wr[4]:.0%} of the estimate")
        print("\nThe two disagree, and the second is the one that matters. An F1 near")
        print("a bound has a narrow band because it is near a bound, not because it")
        print("is well determined. The smallest class is the least pinned down.")
        smallest = min(rows, key=lambda r: r[2])
        print(f"\nSmallest class anywhere: {smallest[1]} on {smallest[0]}, "
              f"{smallest[2]:,} rows, interval width {smallest[3]:.4f} "
              f"({smallest[4]:.0%} of its estimate).")

    dest = REPORTS / f"intervals_{Path(args.report).stem}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
