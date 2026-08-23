"""Selective classification: risk-coverage analysis and shift-conditioned abstention.

The paper diagnoses a failure (a threshold fitted for 1% human FPR yields 57.7%
under source shift) and then proposes conditioning abstention on measured
distribution shift, without building it. A reviewer is entitled to ask whether
the proposed fix works. This module builds it and measures it.

Three things happen here.

1. Risk-coverage curves and AURC, the standard evaluation for selective
   classifiers. The paper currently reports a single operating point, which
   says nothing about the accuracy/coverage trade-off as a whole.

2. Three abstention policies compared at MATCHED COVERAGE, which is the only
   fair comparison. Reporting a policy that abstains more and calling its lower
   error rate an improvement is circular.

      fixed        the paper's current rule: one band, fitted on validation
      confidence   per-sample thresholding, swept to any target coverage
      psi_gated    widen the band in proportion to measured population shift

3. An honest verdict on whether psi_gated actually beats confidence alone.
   PSI is derived from the score distribution, so it is entirely possible that
   gating on it merely slides along the same risk-coverage curve rather than
   reaching a better one. We test that rather than assuming it.

Usage:
    python -m aicd.eval.selective --config cpu.yaml --branch b
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval.drift import psi as psi_fn

HUMAN = 0


# --------------------------------------------------------------------- curves
def risk_coverage(y: np.ndarray, score: np.ndarray, wrong: np.ndarray):
    """Risk-coverage curve, ordered by descending confidence.

    score  per-sample confidence; higher means "more certain"
    wrong  boolean, whether that prediction is an error

    Returns coverage and selective risk at every prefix of the ranking, which
    is the standard construction from the selective-prediction literature.
    """
    order = np.argsort(-score)
    w = wrong[order].astype(np.float64)
    n = len(w)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(w) / np.arange(1, n + 1)
    return cov, risk


def aurc(cov: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    return float(np.trapezoid(risk, cov)) if len(cov) > 1 else float("nan")


def risk_at_coverage(cov: np.ndarray, risk: np.ndarray, target: float) -> float:
    """Selective risk at a target coverage, by interpolation."""
    if len(cov) == 0:
        return float("nan")
    return float(np.interp(target, cov, risk))


# --------------------------------------------------------------------- policies
def apply_band(p_machine: np.ndarray, t_lo: float, t_hi: float):
    """Return (called mask, predicted-machine mask) for a two-threshold band."""
    called = (p_machine >= t_hi) | (p_machine <= t_lo)
    pred_machine = p_machine >= t_hi
    return called, pred_machine


def policy_stats(y: np.ndarray, p_machine: np.ndarray, called: np.ndarray,
                 pred_machine: np.ndarray) -> dict:
    """Coverage, human false-positive rate and recall among covered inputs."""
    n = len(y)
    human, mach = y == HUMAN, y != HUMAN
    cov = float(called.mean()) if n else float("nan")

    hc = human & called
    fpr = float(pred_machine[hc].mean()) if hc.any() else float("nan")
    mc = mach & called
    rec = float(pred_machine[mc].mean()) if mc.any() else float("nan")
    return {"coverage": cov, "abstain_rate": 1.0 - cov,
            "human_fpr": fpr, "machine_recall": rec,
            "n_called": int(called.sum()), "n": int(n)}


def confidence_band_for_coverage(p_machine: np.ndarray, target_cov: float,
                                 t_mid: float):
    """Symmetric band around t_mid whose width hits a target coverage.

    Used to put every policy on equal footing: any policy can be made to
    abstain more, so comparisons are only meaningful at matched coverage.
    """
    d = np.abs(p_machine - t_mid)
    if target_cov >= 1.0:
        return t_mid, t_mid
    # apply_band decides on the TAILS and abstains in the middle, so the band
    # half-width is the (1 - coverage) quantile of the distance, not the
    # coverage quantile. Getting this backwards silently compares a 6%-coverage
    # policy against a 94%-coverage one.
    half = np.quantile(d, 1.0 - target_cov)
    return t_mid - half, t_mid + half


# --------------------------------------------------------------------- runner
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--branch", default="b")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="band widening per unit PSI; fixed a priori, not tuned on OOD")
    args = ap.parse_args()

    cfg = C.load(args.config)
    art, rep = C.artifacts(cfg), C.reports(cfg)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    pol = json.loads((rep / "policy_b.json").read_text(encoding="utf-8"))
    t_hi, t_lo = pol["thresholds"]["t_high"], pol["thresholds"]["t_low"]
    delta0 = t_hi - t_lo
    print(f"[selective] base band from validation: t_lo={t_lo:.3f} t_hi={t_hi:.3f} "
          f"delta={delta0:.3f}")

    drift_p = rep / "drift.json"
    drift = json.loads(drift_p.read_text(encoding="utf-8")) if drift_p.exists() else {"slices": {}}

    out = {"thresholds": {"t_high": t_hi, "t_low": t_lo, "delta0": delta0},
           "lambda": args.lam, "slices": {}}

    for s in SLICES:
        f = art / f"proba_{args.branch}_{s}.npy"
        if not f.exists():
            continue
        proba = np.load(f)
        y = df.loc[df["slice"] == s, "label"].to_numpy()
        if len(y) != len(proba):
            print(f"  [skip] {s}: {len(y)} labels vs {len(proba)} rows")
            continue

        p_machine = 1.0 - proba[:, HUMAN]
        pred = proba.argmax(1)
        wrong = pred != y
        conf = proba.max(1)

        # --- whole-curve view ------------------------------------------------
        cov, risk = risk_coverage(y, conf, wrong)
        # The deployment-relevant risk is a false accusation, not any error.
        human_wrong = (y == HUMAN) & (pred != HUMAN)
        cov_h, risk_h = risk_coverage(y, conf, human_wrong)

        entry = {
            "n": int(len(y)),
            "aurc": aurc(cov, risk),
            "aurc_human_fp": aurc(cov_h, risk_h),
            "risk_at_cov_80": risk_at_coverage(cov, risk, 0.80),
            "risk_at_cov_50": risk_at_coverage(cov, risk, 0.50),
            "human_fp_at_cov_80": risk_at_coverage(cov_h, risk_h, 0.80),
            "policies": {},
        }

        # --- policy 1: the paper's fixed band --------------------------------
        called, pm = apply_band(p_machine, t_lo, t_hi)
        fixed = policy_stats(y, p_machine, called, pm)
        entry["policies"]["fixed"] = fixed
        matched_cov = fixed["coverage"]

        # --- policy 2: per-sample confidence, matched to the same coverage ---
        c_lo, c_hi = confidence_band_for_coverage(p_machine, matched_cov, t_hi)
        called2, pm2 = apply_band(p_machine, c_lo, c_hi)
        entry["policies"]["confidence"] = policy_stats(y, p_machine, called2, pm2)

        # --- policy 3: PSI-gated band ----------------------------------------
        psi_val = drift.get("slices", {}).get(s, {}).get("overall_psi")
        if psi_val is None:
            base = np.load(art / f"proba_{args.branch}_val.npy")
            psi_val = float(psi_fn(1.0 - base[:, HUMAN], p_machine))
        delta = delta0 * (1.0 + args.lam * float(psi_val))
        g_hi = min(0.999, t_hi + 0.5 * (delta - delta0))
        g_lo = max(0.001, g_hi - delta)
        called3, pm3 = apply_band(p_machine, g_lo, g_hi)
        st3 = policy_stats(y, p_machine, called3, pm3)
        st3.update({"psi": float(psi_val), "delta": float(delta),
                    "t_lo": float(g_lo), "t_hi": float(g_hi)})
        entry["policies"]["psi_gated"] = st3

        # --- policy 4: confidence matched to psi_gated's coverage ------------
        # The fair control. If psi_gated only wins because it abstains more,
        # this row will match it and the contribution evaporates.
        m_lo, m_hi = confidence_band_for_coverage(p_machine, st3["coverage"], t_hi)
        called4, pm4 = apply_band(p_machine, m_lo, m_hi)
        entry["policies"]["confidence_matched"] = policy_stats(y, p_machine, called4, pm4)

        out["slices"][s] = entry

        print(f"\n[{s}] n={len(y):,}  AURC={entry['aurc']:.4f}  "
              f"AURC(human-FP)={entry['aurc_human_fp']:.4f}  PSI={psi_val:.3f}")
        for name in ("fixed", "confidence", "psi_gated", "confidence_matched"):
            st = entry["policies"][name]
            print(f"   {name:20s} cov={st['coverage']:.3f} "
                  f"humanFPR={st['human_fpr']:.4f} recall={st['machine_recall']:.4f}")

    dest = rep / "selective.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {dest}")

    # --- did the proposed fix actually work? ---------------------------------
    print("\n" + "=" * 68)
    print("VERDICT: psi_gated vs confidence at the SAME coverage")
    print("=" * 68)
    wins = 0
    tested = 0
    for s, e in out["slices"].items():
        a = e["policies"]["psi_gated"]["human_fpr"]
        b = e["policies"]["confidence_matched"]["human_fpr"]
        if np.isnan(a) or np.isnan(b):
            continue
        tested += 1
        better = a < b - 1e-9
        wins += better
        print(f"  {s:22s} psi_gated={a:.4f}  confidence={b:.4f}  "
              f"{'psi wins' if better else 'no gain'}")
    print(f"\n  psi_gated beats matched confidence on {wins}/{tested} slices.")
    if wins <= tested / 2:
        print("  Report this honestly: on this evidence, gating on population PSI")
        print("  mostly re-parameterises per-sample confidence rather than")
        print("  improving on it. The useful part is the coverage reduction, not")
        print("  a better risk-coverage frontier.")


if __name__ == "__main__":
    main()
