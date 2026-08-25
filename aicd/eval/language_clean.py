"""R1: does the exposure effect survive with no untrained language in training?

D2 draws its donors from S2, S3 and S4, so Go and JavaScript rows enter its
training set. Its gain on S4 could in principle come from general token coverage
rather than from source exposure, even though every S4 row is in a language the
model already knew. The d2nolang arm removes that possibility by construction:
its donors come only from S2 and S4, so no untrained language is ever seen.

The cost is comparability. A row used for training cannot be evaluated, so
d2nolang vacates a different set of rows than D2 does, and a naive three-way
comparison would score three arms on three different row sets. All three are
therefore restricted to the rows every arm still evaluates, which is
(arm_slice == condition) and not d2nolang_train. D1_small and D2 already have
stored per-row probabilities, so this is a subset of arrays on disk rather than
a rescoring.

    python -m aicd.eval.language_clean
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aicd.eval import resample as R

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "kaggle_runs" / "results" / "e1" / "arrays"
R1 = ROOT / "kaggle_runs" / "results" / "r1" / "extracted" / "arrays"
OUT = ROOT / "aicd" / "eval" / "reports" / "language_clean.json"

CONDS = ("s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound")
SEED = 20240617
B = 4000


def main() -> None:
    keys = pd.read_parquet(R1 / "arm_row_keys.parquet")

    # The language-clean arm must never have seen an untrained language.
    trained = set(keys.loc[keys["slice"] == "train", "language"])
    seen = set(keys.loc[keys["d2nolang_train"], "language"])
    leaked = sorted(seen - trained)
    if leaked:
        raise SystemExit(f"d2nolang saw {leaked}, which defeats the arm")

    report = {
        "note": ("All three arms restricted to rows every arm still evaluates, "
                 "so the contrasts are measured on identical rows."),
        "languages_seen_by_d2nolang": sorted(seen),
        "untrained_languages_seen": leaked,
        "conditions": {},
    }

    for cond in CONDS:
        sub = keys[keys["arm_slice"] == cond]
        keep = (~sub["d2nolang_train"]).to_numpy()
        y = sub["label"].to_numpy()[keep]

        p_small = np.load(E1 / f"proba_a_e1_d1small_{cond}.npy")[keep]
        p_d2 = np.load(E1 / f"proba_a_e1_d2_{cond}.npy")[keep]
        p_nol = np.load(R1 / f"proba_a_r1_d2nolang_{cond}.npy")
        if not (len(p_small) == len(p_d2) == len(p_nol) == len(y)):
            raise SystemExit(f"{cond}: row counts disagree after restriction")

        a_small, a_d2, a_nol = p_small.argmax(1), p_d2.argmax(1), p_nol.argmax(1)
        f_small = R.macro_f1_present(y, a_small)
        f_d2 = R.macro_f1_present(y, a_d2)
        f_nol = R.macro_f1_present(y, a_nol)

        # Paired, because every arm is scored on exactly these rows, so the
        # same y is passed for both sides of each contrast.
        # bca_difference returns stat(second) - stat(first), so the baseline
        # arm goes first for the contrast to read as arm minus baseline.
        d_exp, lo_e, hi_e = R.bca_difference(
            y, a_small, y, a_d2, R.macro_f1_present, n_boot=B, seed=SEED)
        d_cln, lo_c, hi_c = R.bca_difference(
            y, a_small, y, a_nol, R.macro_f1_present, n_boot=B, seed=SEED)

        report["conditions"][cond] = {
            "rows": int(len(y)),
            "rows_before_restriction": int(len(sub)),
            "d1small": round(float(f_small), 4),
            "d2": round(float(f_d2), 4),
            "d2nolang": round(float(f_nol), 4),
            "exposure_effect": round(float(d_exp), 4),
            "exposure_ci": [round(float(lo_e), 4), round(float(hi_e), 4)],
            "language_clean_effect": round(float(d_cln), 4),
            "language_clean_ci": [round(float(lo_c), 4), round(float(hi_c), 4)],
            "clean_excludes_zero": bool(lo_c > 0 or hi_c < 0),
        }

    s4 = report["conditions"]["s4_unseen_domain"]
    report["headline"] = {
        "condition": "s4_unseen_domain",
        "claim": ("the source-axis exposure effect survives with no untrained "
                  "language anywhere in training"),
        "exposure_effect": s4["exposure_effect"],
        "language_clean_effect": s4["language_clean_effect"],
        "survives": bool(s4["clean_excludes_zero"] and s4["language_clean_effect"] > 0),
    }

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"  {'condition':<24} {'rows':>6} {'D1_sm':>7} {'D2':>7} {'nolang':>7} "
          f"{'exposure':>9} {'clean':>9}")
    for c, v in report["conditions"].items():
        print(f"  {c:<24} {v['rows']:>6} {v['d1small']:>7.4f} {v['d2']:>7.4f} "
              f"{v['d2nolang']:>7.4f} {v['exposure_effect']:>+9.4f} "
              f"{v['language_clean_effect']:>+9.4f}")
    print(f"\n  headline: {json.dumps(report['headline'], indent=2)}")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
