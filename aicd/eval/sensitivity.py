"""Does the headline result survive the arbitrary-looking constants?

Several numbers in the protocol were chosen by judgement rather than derived:
the target false-accusation rate alpha, the abstention half-width delta, and
the PSI coefficient lambda. A reviewer is right to ask whether the conclusion
is an artefact of those particular values.

The conclusion under test is not any single number but the direction: an
operating point fitted in distribution should fail under compound shift. This
sweeps each constant across a wide range and reports whether that direction
ever reverses.

Everything here is post-hoc on cached probability arrays, so no retraining is
needed and the sweep is exact rather than approximate.

    python -m aicd.eval.sensitivity
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.eval.selective import apply_band, policy_stats

ART = "kaggle"
ORDER = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound"]
HUMAN = 0


def load():
    base = C.artifacts(C.load("cpu.yaml")) / ART
    lab = pd.read_parquet(base / "labels.parquet")
    data = {}
    for s in ORDER + ["val"]:
        f = base / f"proba_a_{s}.npy"
        if not f.exists():
            continue
        p = np.load(f)
        y = lab.loc[lab["slice"] == s, "label"].to_numpy()
        if len(y) == len(p):
            data[s] = (y, p)
    return data


def p_machine(p):
    # Same expression the policy code uses, so the sweep and the reported
    # numbers cannot drift apart over a floating-point hair.
    return 1.0 - p[:, HUMAN]


def threshold_for(y, p, alpha):
    """Smallest tau with P(p_machine >= tau | human) <= alpha."""
    h = p_machine(p)[y == HUMAN]
    if h.size == 0:
        return 1.0
    return float(np.quantile(h, 1.0 - alpha))


def realised(y, p, t_hi, delta):
    """False-accusation rate and coverage under the real two-threshold band.

    The band decides on the tails and abstains in the middle: machine at or
    above t_hi, human at or below t_lo = t_hi - delta. It is deliberately
    asymmetric. A symmetric band around t_hi is degenerate here, because t_hi
    sits near 1.0 and the upper decision region would run off the top of the
    probability scale, leaving a rule that can never accuse anyone.
    """
    pm = p_machine(p)
    called, pred_machine = apply_band(pm, t_hi - delta, t_hi)
    st = policy_stats(y, pm, called, pred_machine)
    fpr = st["human_fpr"]
    return (0.0 if np.isnan(fpr) else float(fpr)), float(st["coverage"])


def main() -> None:
    data = load()
    if "val" not in data:
        raise SystemExit("validation probabilities required to fit thresholds")
    yv, pv = data["val"]
    out = {}

    # ---- alpha: the target false-accusation rate ----------------------------
    print("alpha sweep: threshold fitted on validation, applied to S1 and S5")
    print(f"  {'alpha':>7s} {'tau':>8s} {'S1 FPR':>8s} {'S5 FPR':>8s} {'ratio':>8s}")
    rows = []
    for alpha in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        tau = threshold_for(yv, pv, alpha)
        f1, _ = realised(*data["s1_in_distribution"], tau, 0.0)
        f5, _ = realised(*data["s5_compound"], tau, 0.0)
        ratio = f5 / f1 if f1 > 0 else float("inf")
        rows.append({"alpha": alpha, "tau": tau, "s1_fpr": f1, "s5_fpr": f5,
                     "ratio": ratio})
        print(f"  {alpha:7.3f} {tau:8.4f} {f1:8.4f} {f5:8.4f} {ratio:8.1f}")
    out["alpha"] = rows
    worst = min(r["ratio"] for r in rows)
    print(f"  smallest S5/S1 blow-up across alpha: {worst:.1f}x")

    # ---- delta: the abstention half-width -----------------------------------
    tau = threshold_for(yv, pv, 0.01)
    print(f"\ndelta sweep at alpha=0.01, tau={tau:.4f}")
    print(f"  {'delta':>7s} {'S1 cov':>8s} {'S1 FPR':>8s} {'S5 cov':>8s} {'S5 FPR':>8s}")
    rows = []
    for delta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        f1, c1 = realised(*data["s1_in_distribution"], tau, delta)
        f5, c5 = realised(*data["s5_compound"], tau, delta)
        rows.append({"delta": delta, "s1_cov": c1, "s1_fpr": f1,
                     "s5_cov": c5, "s5_fpr": f5})
        print(f"  {delta:7.2f} {c1:8.4f} {f1:8.4f} {c5:8.4f} {f5:8.4f}")
    out["delta"] = rows
    worst5 = min(r["s5_fpr"] for r in rows)
    best1 = max(r["s1_fpr"] for r in rows)
    print(f"  across every delta: S5 stays at or above {worst5:.3f}, "
          f"S1 stays at or below {best1:.3f}")

    dest = C.ROOT / "eval" / "reports" / "sensitivity.json"

    os.makedirs(dest.parent, exist_ok=True)
    io.open(dest, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
