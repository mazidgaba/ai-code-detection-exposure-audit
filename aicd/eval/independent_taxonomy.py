"""How much of the external collapse is the hybrid class we defined ourselves.

The independent corpus reports a macro-F1 of roughly 0.32, and almost all of the
shortfall sits in one class. That class is not native to the corpus: AIGCodeSet
labels a row hybrid when a language model repaired a failing human submission,
and we map that onto DroidCollection's machine-refined definition. The two are
close but not obviously the same construct, so a reader is entitled to ask how
much of the collapse is shift and how much is a taxonomy mismatch.

This reports both readings on the same rows: the three-class figure the corpus
supports, and the human-versus-machine figure that does not depend on the
mapping at all. Neither is the whole answer, which is why both are reported.

    python -m aicd.eval.independent_taxonomy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aicd.eval import resample as R

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "aicd" / "artifacts" / "data" / "independent" / "independent.parquet"
PROBA = ROOT / "aicd" / "artifacts" / "proba_independent.npy"
OUT = ROOT / "aicd" / "eval" / "reports" / "independent_taxonomy.json"

HUMAN, MACHINE, HYBRID = 0, 1, 2
SEED, B = 20240617, 4000


def main() -> None:
    df = pd.read_parquet(ROWS, columns=["label"])
    y = df["label"].to_numpy()
    pred = np.load(PROBA).argmax(1)
    if len(y) != len(pred):
        raise SystemExit(f"rows {len(y)} but {len(pred)} predictions")

    report = {
        "note": ("Both readings on the same predictions. 'all_classes' includes "
                 "the hybrid class, whose definition we mapped from the corpus's "
                 "own field; 'human_machine' drops it and depends on no mapping."),
        "rows_total": int(len(y)),
        "label_counts": {str(int(k)): int(v) for k, v in
                         zip(*np.unique(y, return_counts=True))},
        "views": {},
    }

    for tag, mask in (("all_classes", np.ones(len(y), bool)),
                      ("human_machine", np.isin(y, [HUMAN, MACHINE]))):
        yy, pp = y[mask], pred[mask]
        f1 = R.macro_f1_present(yy, pp)
        lo, hi = R.bca(yy, pp, R.macro_f1_present, n_boot=B, seed=SEED)[1:]
        per = {}
        for c, name in ((HUMAN, "human"), (MACHINE, "machine"), (HYBRID, "hybrid")):
            if (yy == c).any():
                tp = int(((pp == c) & (yy == c)).sum())
                fp = int(((pp == c) & (yy != c)).sum())
                fn = int(((pp != c) & (yy == c)).sum())
                den = 2 * tp + fp + fn
                per[name] = round(2 * tp / den, 4) if den else 0.0
        report["views"][tag] = {
            "rows": int(mask.sum()),
            "macro_f1": round(float(f1), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "per_class_f1": per,
        }

    a = report["views"]["all_classes"]["macro_f1"]
    b = report["views"]["human_machine"]["macro_f1"]
    report["headline"] = {
        "all_classes": a,
        "human_machine": b,
        "gap": round(b - a, 4),
        "hybrid_f1": report["views"]["all_classes"]["per_class_f1"].get("hybrid"),
        "collapse_survives_dropping_hybrid": bool(b < 0.60),
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for tag, v in report["views"].items():
        print(f"  {tag:<16} n={v['rows']:>6}  macro-F1 {v['macro_f1']:.4f} "
              f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]   {v['per_class_f1']}")
    print(f"\n  headline: {json.dumps(report['headline'], indent=2)}")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
