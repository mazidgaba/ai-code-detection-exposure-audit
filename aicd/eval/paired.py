"""Paired difference intervals for the comparisons the paper rests on.

Two models scored on identical rows are compared throughout this paper, and
until now only as two point estimates. That is not enough: two overlapping
intervals do not establish that a difference is absent, and two disjoint ones do
not say how large it is. The difference is the quantity of interest and it needs
its own interval.

The pairing matters. Both arms are resampled with the *same* row indices, so
noise common to both cancels rather than being counted twice. Treating a paired
comparison as two independent samples inflates the variance and would understate
the twin control's effect, which is the comparison the paper's central claim
depends on.

## Why this could not be done before

The probability arrays from earlier runs were saved without the label vector
they align to. Aggregate statistics survived that, because the confusion
matrices were stored too, but a paired comparison needs to know which row is
which. The corpus build is deterministic, so the label vector recovered in E10
serves every arm already scored on this corpus. It is validated on entry here:
if the labels did not reproduce each arm's reported macro-F1 exactly, the
alignment would be wrong and every number below meaningless.

    python -m aicd.eval.paired
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aicd.eval import resample as R

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "aicd" / "eval" / "reports"
RUNS = ROOT / "kaggle_runs" / "results"
KEYS = ROOT / "aicd" / "artifacts" / "keys" / "eval_row_keys.parquet"
CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]

# (name, arrays dir, array prefix, report path). Both arms of each comparison
# must have been scored on the same evaluation rows.
COMPARISONS = [
    ("twin",
     ("unexposed D1_small", RUNS / "e1" / "arrays", "proba_a_e1_d1small",
      RUNS / "e1" / "branch_a_e1_d1small.json"),
     ("exposed D2", RUNS / "e1" / "arrays", "proba_a_e1_d2",
      RUNS / "e1-d2" / "results" / "reports" / "branch_a_e1_d2.json")),
]


def labels() -> pd.DataFrame:
    if not KEYS.exists():
        raise SystemExit(f"{KEYS} not found; it is produced by aicd.eval.contamination")
    return pd.read_parquet(KEYS)


def arm(key: pd.DataFrame, arrays: Path, prefix: str, report: Path):
    """Load an arm and verify the labels reproduce its reported macro-F1.

    Only the protected conditions are available for pairing. The twin's exposed
    arm draws its extra training rows from the donor conditions S2, S3 and S4,
    so those conditions have fewer evaluation rows in the arm build than in the
    standard one and the recovered label vector does not describe them. S1 and
    S5 are protected by construction: no row of either is ever moved into
    training in any arm, so their evaluation sets are identical across builds.
    A condition whose lengths disagree is skipped and recorded, never silently
    truncated to fit.
    """
    rep = json.loads(report.read_text(encoding="utf-8"))["slices"]
    out, skipped = {}, {}
    for c in CONDITIONS:
        f = arrays / f"{prefix}_{c}.npy"
        if not f.exists():
            continue
        y = key[key["slice"] == c]["label"].to_numpy()
        p = np.load(f)
        if len(y) != len(p):
            skipped[c] = (int(len(p)), int(len(y)))
            continue
        got = R.macro_f1(y, p.argmax(1))
        want = rep[c]["macro_f1"]
        if abs(got - want) > 1e-6:
            raise SystemExit(f"{prefix} {c}: labels give macro-F1 {got:.6f}, the "
                             f"report says {want:.6f}; the alignment is wrong")
        out[c] = (y, p.argmax(1))
    return out, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    key = labels()
    out = {"n_boot": args.n_boot, "method": "BCa paired difference",
           "comparisons": {}}

    for name, a_spec, b_spec in COMPARISONS:
        a_name, a_dir, a_pre, a_rep = a_spec
        b_name, b_dir, b_pre, b_rep = b_spec
        A, skip_a = arm(key, a_dir, a_pre, a_rep)
        B, skip_b = arm(key, b_dir, b_pre, b_rep)
        shared = [c for c in CONDITIONS if c in A and c in B]
        print(f"=== {name}: {b_name} minus {a_name} ===")
        print(f"  both arms verified against their own reports on "
              f"{len(shared)} conditions")
        skipped = {**skip_a, **skip_b}
        if skipped:
            print("  skipped, evaluation sets differ between builds by design:")
            for c, (n_arr, n_key) in sorted(skipped.items()):
                print(f"    {c:24s} arm build {n_arr:>7,} rows, standard {n_key:>7,}")
        print()
        print(f"{'condition':22s} {a_name[:11]:>11s} {b_name[:11]:>11s} "
              f"{'difference':>11s} {'95% CI':>20s}")
        print("-" * 80)
        rec = {}
        for c in shared:
            ya, pa = A[c]
            yb, pb = B[c]
            d, lo, hi = R.bca_difference(ya, pa, yb, pb, R.macro_f1,
                                         n_boot=args.n_boot, seed=args.seed)
            rec[c] = {"a": R.macro_f1(ya, pa), "b": R.macro_f1(yb, pb),
                      "difference": d, "ci95": [lo, hi],
                      "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(ya))}
            r = rec[c]
            print(f"{c:22s} {r['a']:11.4f} {r['b']:11.4f} {d:+11.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]"
                  f"{'' if r['excludes_zero'] else '   spans 0'}")
        out["comparisons"][name] = {"a": a_name, "b": b_name,
                                    "conditions": rec,
                                    "skipped": {k: list(v) for k, v in skipped.items()}}

        # The twin's whole point: exposure buys nothing in distribution and a
        # great deal under shift. Stating that as two intervals rather than two
        # point estimates is what makes it checkable.
        if "s1_in_distribution" in rec and "s5_compound" in rec:
            s1, s5 = rec["s1_in_distribution"], rec["s5_compound"]
            print()
            if not s1["excludes_zero"]:
                print(f"  In distribution the difference is {s1['difference']:+.4f} with an")
                print(f"  interval of [{s1['ci95'][0]:+.4f}, {s1['ci95'][1]:+.4f}] that spans zero:")
                print("  exposure buys nothing there, stated as a measurement rather")
                print("  than as an absence of evidence.")
            else:
                print(f"  In distribution the difference is {s1['difference']:+.4f}, interval")
                print(f"  [{s1['ci95'][0]:+.4f}, {s1['ci95'][1]:+.4f}], which excludes zero. The paper")
                print("  says exposure buys nothing in distribution; that needs revising.")
            print(f"\n  Under compound shift it is {s5['difference']:+.4f}, interval "
                  f"[{s5['ci95'][0]:+.4f}, {s5['ci95'][1]:+.4f}].")

    dest = REPORTS / "paired_differences.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
