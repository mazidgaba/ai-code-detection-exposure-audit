"""Full analysis of the GPU run, using the recovered probability arrays.

Branch A is the control the published detectors cannot provide: it was trained
only on our training split, so the five conditions are genuinely held out for
it. DroidDetect saw every generator family, language and source we withhold, so
running it on these slices measures nothing about shift.

The Kaggle notebook fitted calibration and a decision threshold for Branch B
only, which left Branch A's headline false-positive rates at argmax. This fits
the same 1%-target policy to Branch A, and adds the risk-coverage, bootstrap
and selective-classification analyses that need per-row probabilities.

    python -m aicd.eval.kaggle_analysis
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from aicd import config as C
from aicd.eval.selective import (aurc, risk_coverage, apply_band, policy_stats,
                                 confidence_band_for_coverage)

HUMAN = 0
ORDER = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound"]
ART = "kaggle"


def load(branch: str):
    """Yield (slice, y, proba) with the arrays aligned to their slice order."""
    base = C.artifacts(C.load("cpu.yaml")) / ART
    lab = pd.read_parquet(base / "labels.parquet")
    for s in ["val"] + ORDER:
        f = base / f"proba_{branch}_{s}.npy"
        if not f.exists():
            continue
        p = np.load(f)
        y = lab.loc[lab["slice"] == s, "label"].to_numpy()
        if len(y) != len(p):
            print(f"  [skip] {s}: {len(y)} labels vs {len(p)} rows")
            continue
        yield s, y, p


def choose_threshold(y, p_machine, target=0.01) -> float:
    """Lowest threshold whose human false-positive rate is <= target."""
    human = y == HUMAN
    if not human.any():
        return 0.5
    for t in np.unique(np.round(p_machine, 4)):
        if (p_machine[human] >= t).mean() <= target:
            return float(t)
    return 1.0


def boot(y, p, t_hi, t_lo, n=1000, seed=20260818):
    rng = np.random.default_rng(seed)
    out = {"macro_f1": [], "human_fpr_policy": [], "binary_auc": []}
    N = len(y)
    for _ in range(n):
        i = rng.integers(0, N, N)
        yb, pb = y[i], p[i]
        if len(np.unique(yb)) < 2:
            continue
        pred, pm = pb.argmax(1), 1 - pb[:, HUMAN]
        out["macro_f1"].append(f1_score(yb, pred, average="macro", zero_division=0))
        yv = (yb != HUMAN).astype(int)
        if len(np.unique(yv)) > 1:
            out["binary_auc"].append(roc_auc_score(yv, pm))
        called = (pm >= t_hi) | (pm <= t_lo)
        hc = (yb == HUMAN) & called
        if hc.any():
            out["human_fpr_policy"].append(float((pm[hc] >= t_hi).mean()))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items() if len(v) > 20}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = C.load("cpu.yaml")

    report = {"note": "GPU run; Branch A trained on our split only, so the five "
                      "conditions are genuinely held out", "branches": {}}

    for branch, name in (("a", "Branch A (ModernBERT)"), ("b", "Branch B (XGBoost)")):
        data = {s: (y, p) for s, y, p in load(branch)}
        if "val" not in data:
            print(f"[{branch}] no validation probabilities, skipping")
            continue

        # Fit the operating point on validation only, exactly as deployment would.
        yv, pv = data["val"]
        t_hi = choose_threshold(yv, 1 - pv[:, HUMAN], target=0.01)
        t_lo = max(0.0, t_hi - 0.15)
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(f"policy fitted on validation: t_hi={t_hi:.4f} t_lo={t_lo:.4f} (1% target)")
        print(f"\n{'slice':22s} {'n':>7s} {'macroF1':>9s} {'AUC':>8s} "
              f"{'AURC':>7s} {'cov':>6s} {'humFPR':>8s}")
        print("-" * 78)

        entry = {"thresholds": {"t_high": t_hi, "t_low": t_lo}, "slices": {}}
        for s in ORDER:
            if s not in data:
                continue
            y, p = data[s]
            pred, pm, conf = p.argmax(1), 1 - p[:, HUMAN], p.max(1)
            called, pmach = apply_band(pm, t_lo, t_hi)
            st = policy_stats(y, pm, called, pmach)
            cov_c, risk_c = risk_coverage(y, conf, pred != y)
            yv2 = (y != HUMAN).astype(int)

            e = {
                "n": int(len(y)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "binary_auc": float(roc_auc_score(yv2, pm)) if len(np.unique(yv2)) > 1 else float("nan"),
                "aurc": aurc(cov_c, risk_c),
                "coverage": st["coverage"],
                "human_fpr_policy": st["human_fpr"],
                "human_fpr_argmax": float((pred[y == HUMAN] != HUMAN).mean()),
                "ci": boot(y, p, t_hi, t_lo, args.boot),
            }
            # Is a confidence rule better than the fitted band at equal coverage?
            c_lo, c_hi = confidence_band_for_coverage(pm, st["coverage"], t_hi)
            c_called, c_pm = apply_band(pm, c_lo, c_hi)
            e["human_fpr_confidence"] = policy_stats(y, pm, c_called, c_pm)["human_fpr"]

            entry["slices"][s] = e
            print(f"{s:22s} {e['n']:>7,} {e['macro_f1']:>9.4f} {e['binary_auc']:>8.4f} "
                  f"{e['aurc']:>7.4f} {e['coverage']:>6.3f} {e['human_fpr_policy']:>8.4f}")
        report["branches"][branch] = entry

    dest = C.reports(cfg) / "kaggle_analysis.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- the headline, with intervals ---------------------------------------
    a = report["branches"].get("a", {}).get("slices", {})
    if a and "s1_in_distribution" in a and "s5_compound" in a:
        s1, s5 = a["s1_in_distribution"], a["s5_compound"]
        print(f"\n{'=' * 78}\nHEADLINE (Branch A, 95% bootstrap intervals)\n{'=' * 78}")
        for k, lbl in (("macro_f1", "macro-F1"), ("human_fpr_policy", "human FPR @1% policy"),
                       ("binary_auc", "binary AUC")):
            c1 = s1["ci"].get(k, (float("nan"),) * 2)
            c5 = s5["ci"].get(k, (float("nan"),) * 2)
            print(f"  {lbl:22s} S1 {s1[k]:.4f} [{c1[0]:.4f}, {c1[1]:.4f}]"
                  f"   S5 {s5[k]:.4f} [{c5[0]:.4f}, {c5[1]:.4f}]")
        sep = s5["ci"]["macro_f1"][1] < s1["ci"]["macro_f1"][0]
        print(f"\n  macro-F1 intervals {'do not overlap' if sep else 'OVERLAP'}"
              f" -> collapse is {'not' if sep else 'possibly'} a sampling artefact")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
