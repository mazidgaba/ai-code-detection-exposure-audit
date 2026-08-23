"""Decision policy: calibrated probabilities -> a decision, including abstain.

Thresholds are derived from a target false-positive rate on the HUMAN class,
not from argmax. A false accusation costs far more than a missed detection, so
the operating point is chosen on the human-FPR axis and recall is whatever it
turns out to be.

Everything between the thresholds returns `abstain`. On the compound-shift
slice the abstention rate should be high -- AICD Bench measured every published
detector at or below random there, so abstaining is the correct behaviour and
not a number to tune away.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.config import LABEL_NAMES
from aicd.data.splits import SLICES

CAVEAT_MACHINE = (
    "This is probabilistic evidence about code provenance, not proof of "
    "authorship. Detectors of this kind score at or below chance on code "
    "whose language or domain they have not seen. Corroborate with commit "
    "history, authoring telemetry, or a code walkthrough before acting."
)
CAVEAT_ABSTAIN = (
    "The evidence does not support a call either way. Treat this as no "
    "finding, not as a weak positive."
)
CAVEAT_HUMAN = (
    "No evidence of machine generation. A negative result does not rule it "
    "out; detectors miss hybrid and adversarially-styled code."
)


@dataclass
class Decision:
    decision: str
    p_machine: float
    probabilities: dict
    confidence_note: str
    threshold_used: dict


def choose_threshold(y_true: np.ndarray, p_machine: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose human false-positive rate is <= target."""
    human = y_true == 0
    if not human.any():
        return 0.5
    candidates = np.unique(np.round(p_machine, 4))
    for t in candidates:
        if (p_machine[human] >= t).mean() <= target_fpr:
            return float(t)
    return 1.0


def fit_policy(cfg, df: pd.DataFrame, proba: np.ndarray, mask) -> dict:
    y = df.loc[mask, "label"].to_numpy()
    pm = 1.0 - proba[:, 0]
    t_hi = choose_threshold(y, pm, cfg.policy.target_human_fpr)
    t_lo = max(0.0, t_hi - cfg.policy.abstain_margin)
    return {"t_high": t_hi, "t_low": t_lo,
            "target_human_fpr": cfg.policy.target_human_fpr}


def decide(proba_row: np.ndarray, thresholds: dict) -> Decision:
    p = np.asarray(proba_row, dtype=float)
    p_machine = float(1.0 - p[0])
    probs = {LABEL_NAMES[i]: float(p[i]) for i in range(len(p))}

    if p_machine >= thresholds["t_high"]:
        # Which non-human class, among machine/hybrid/adversarial
        label = LABEL_NAMES[int(np.argmax(p[1:])) + 1]
        return Decision(label, p_machine, probs, CAVEAT_MACHINE, thresholds)
    if p_machine <= thresholds["t_low"]:
        return Decision("human", p_machine, probs, CAVEAT_HUMAN, thresholds)
    return Decision("abstain", p_machine, probs, CAVEAT_ABSTAIN, thresholds)


def evaluate_policy(df, proba, mask, thresholds) -> dict:
    y = df.loc[mask, "label"].to_numpy()
    pm = 1.0 - proba[:, 0]
    hi, lo = thresholds["t_high"], thresholds["t_low"]

    abstain = (pm > lo) & (pm < hi)
    called = ~abstain
    pred_machine = pm >= hi
    human = y == 0

    decided_human = called & human
    return {
        "n": int(len(y)),
        "abstain_rate": float(abstain.mean()),
        "human_fpr_on_called": float(pred_machine[decided_human].mean())
        if decided_human.any() else None,
        "machine_recall_on_called": float(pred_machine[called & ~human].mean())
        if (called & ~human).any() else None,
        "coverage": float(called.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--branch", default="b", choices=["a", "b", "stack"])
    args = ap.parse_args()
    cfg = C.load(args.config)
    art, rep = C.artifacts(cfg), C.reports(cfg)
    prefix = {"a": "proba_a", "b": "proba_b", "stack": "proba_stack"}[args.branch]

    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    # Fit thresholds on val when present, else on the in-distribution slice.
    fit_slice = "val" if (art / f"{prefix}_val.npy").exists() else "s1_in_distribution"
    m = df["slice"] == fit_slice
    proba = np.load(art / f"{prefix}_{fit_slice}.npy")
    th = fit_policy(cfg, df, proba, m)
    print(f"[policy] fitted on '{fit_slice}' at target human FPR "
          f"{cfg.policy.target_human_fpr:.1%}")
    print(f"  t_high = {th['t_high']:.4f}   (>= this -> machine/hybrid/adversarial)")
    print(f"  t_low  = {th['t_low']:.4f}   (<= this -> human; between -> abstain)")

    out = {"thresholds": th, "slices": {}}
    print(f"\n{'slice':22s} {'n':>7s} {'coverage':>9s} {'abstain':>8s} "
          f"{'humFPR':>8s} {'recall':>8s}")
    print("-" * 68)
    for s in SLICES:
        p = art / f"{prefix}_{s}.npy"
        ms = df["slice"] == s
        if not p.exists() or not ms.any():
            continue
        pr = np.load(p)
        if len(pr) != int(ms.sum()):
            continue
        r = evaluate_policy(df, pr, ms, th)
        out["slices"][s] = r
        fpr = "-" if r["human_fpr_on_called"] is None else f"{r['human_fpr_on_called']:.4f}"
        rec = "-" if r["machine_recall_on_called"] is None else f"{r['machine_recall_on_called']:.4f}"
        print(f"{s:22s} {r['n']:>7,} {r['coverage']:>9.3f} "
              f"{r['abstain_rate']:>8.3f} {fpr:>8s} {rec:>8s}")

    with open(art / f"policy_{args.branch}.json", "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2)
    with open(rep / f"policy_{args.branch}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nA high abstain rate on s4/s5 is the intended behaviour.")


if __name__ == "__main__":
    main()
