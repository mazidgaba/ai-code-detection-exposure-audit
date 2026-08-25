"""Compare the generator and source axes on rows that actually carry the shift.

S2 withholds generator families. Human-written code has no generator family, so
S2's human rows are in distribution by construction and carry no shift at all;
they are 67.6% of the condition. S4 withholds a code source, which human rows do
carry, so 100% of S4 is shifted. Comparing the two axes over all rows therefore
compares a condition that is two thirds unshifted against one that is wholly
shifted, and the difference is partly composition rather than axis.

Restricting both to machine-generated rows puts them on the same footing. The
comparison needs no retraining: the twin arms' per-row probabilities are already
on disk, so this is a subset of stored arrays.

    python -m aicd.eval.matched_axis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aicd.eval import resample as R

ROOT = Path(__file__).resolve().parents[2]
KEYS = ROOT / "kaggle_runs" / "results" / "r1" / "extracted" / "arrays" / "arm_row_keys.parquet"
E1 = ROOT / "kaggle_runs" / "results" / "e1" / "arrays"
OUT = ROOT / "aicd" / "eval" / "reports" / "matched_axis.json"

AXES = {"s2_unseen_generator": "generator", "s4_unseen_domain": "source"}
HUMAN = 0
SEED, B = 20240617, 4000


def main() -> None:
    keys = pd.read_parquet(KEYS)
    report = {
        "note": ("S2's human rows carry no generator shift by construction, so the "
                 "all-rows comparison across axes is partly a comparison of "
                 "composition. Restricting both axes to machine-generated rows "
                 "removes that difference."),
        "conditions": {},
    }

    for cond, axis in AXES.items():
        sub = keys[keys["arm_slice"] == cond]
        y_all = sub["label"].to_numpy()
        a_all = np.load(E1 / f"proba_a_e1_d1small_{cond}.npy").argmax(1)
        b_all = np.load(E1 / f"proba_a_e1_d2_{cond}.npy").argmax(1)

        entry = {"axis": axis, "rows": int(len(y_all)),
                 "human_rows": int((y_all == HUMAN).sum()),
                 "human_share": round(float((y_all == HUMAN).mean()), 4)}

        for tag, mask in (("all", np.ones(len(y_all), bool)),
                          ("machine_only", y_all != HUMAN)):
            y, a, b = y_all[mask], a_all[mask], b_all[mask]
            d, lo, hi = R.bca_difference(y, a, y, b, R.macro_f1_present,
                                         n_boot=B, seed=SEED)
            entry[tag] = {
                "rows": int(len(y)),
                "d1small": round(float(R.macro_f1_present(y, a)), 4),
                "d2": round(float(R.macro_f1_present(y, b)), 4),
                "effect": round(float(d), 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
            }
        report["conditions"][cond] = entry

    s2 = report["conditions"]["s2_unseen_generator"]
    s4 = report["conditions"]["s4_unseen_domain"]
    for tag in ("all", "machine_only"):
        report.setdefault("ratio", {})[tag] = round(
            s4[tag]["effect"] / s2[tag]["effect"], 2)
    # The ordering is what the paper claims; the magnitude is what changes.
    report["ordering_survives_matching"] = bool(
        s4["machine_only"]["ci"][0] > s2["machine_only"]["ci"][1])

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"  {'axis':<12}{'basis':<14}{'rows':>7}{'effect':>10}{'95% BCa':>22}")
    for cond, e in report["conditions"].items():
        for tag in ("all", "machine_only"):
            v = e[tag]
            print(f"  {e['axis']:<12}{tag:<14}{v['rows']:>7}{v['effect']:>+10.4f}"
                  f"   [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    print(f"\n  source/generator ratio: {report['ratio']['all']}x over all rows, "
          f"{report['ratio']['machine_only']}x matched")
    print(f"  ordering survives matching (intervals disjoint): "
          f"{report['ordering_survives_matching']}")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
