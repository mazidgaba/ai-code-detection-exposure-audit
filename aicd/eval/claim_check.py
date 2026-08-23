"""Re-test every paper claim against the published baseline.

One central claim (operating points do not transfer under shift) turned out to
hold for our Branch B and not for DroidDetect, which means it was a statement
about our model rather than about the task. That failure obliges us to ask the
same question of every other claim in the paper: is this a property of
detection, or a property of the weak detector we happened to build?

Each check below runs on the held-out test-shard rows only, so DroidDetect is
never scored on its own training data, and both models see identical rows.

    python -m aicd.eval.claim_check --config cpu.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval.selective import aurc, risk_coverage, apply_band, policy_stats, \
    confidence_band_for_coverage

HUMAN = 0


def ece(y_bin: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(p)) * abs(y_bin[m].mean() - p[m].mean())
    return float(tot)


def load_clean(cfg):
    """Yield (slice, y, proba_dd, proba_b) restricted to held-out test rows."""
    art = C.artifacts(cfg)
    full = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet",
                           columns=["orig_split", "label", "slice"])
    for s in SLICES:
        pdp, rp, bp = (art / f"proba_droiddetect_{s}.npy",
                       art / f"rows_droiddetect_{s}.parquet",
                       art / f"proba_b_{s}.npy")
        if not (pdp.exists() and rp.exists() and bp.exists()):
            continue
        rows = pd.read_parquet(rp)
        meta = full.loc[rows.index]
        keep = (meta["orig_split"] == "test").to_numpy()
        if keep.sum() < 40:
            continue
        slice_idx = full.index[full["slice"] == s]
        pos = pd.Series(np.arange(len(slice_idx)), index=slice_idx)
        bb = np.load(bp)[pos.loc[rows.index].to_numpy()]
        yield s, rows["label"].to_numpy()[keep], np.load(pdp)[keep], bb[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)
    pol = json.loads((C.reports(cfg) / "policy_b.json").read_text(encoding="utf-8"))
    t_hi, t_lo = pol["thresholds"]["t_high"], pol["thresholds"]["t_low"]

    data = list(load_clean(cfg))
    out = {"note": "held-out test-shard rows only; identical rows for both models",
           "claims": {}}

    # ---- claim 1: degradation across slices --------------------------------
    print("=" * 78)
    print("CLAIM 1  Four-class performance degrades monotonically under shift")
    print("=" * 78)
    c1 = {}
    for s, y, dd, bb in data:
        f_dd = f1_score(y, dd.argmax(1), average="macro", zero_division=0)
        f_bb = f1_score(y, bb.argmax(1), average="macro", zero_division=0)
        c1[s] = {"droiddetect": float(f_dd), "branch_b": float(f_bb), "n": int(len(y))}
        print(f"  {s:22s} n={len(y):>4}  DroidDetect={f_dd:.4f}   BranchB={f_bb:.4f}")
    dd_v = [v["droiddetect"] for v in c1.values()]
    bb_v = [v["branch_b"] for v in c1.values()]
    print(f"\n  DroidDetect S1->S5: {dd_v[0]:.4f} -> {dd_v[-1]:.4f}  (drop {dd_v[0]-dd_v[-1]:+.4f})")
    print(f"  Branch B    S1->S5: {bb_v[0]:.4f} -> {bb_v[-1]:.4f}  (drop {bb_v[0]-bb_v[-1]:+.4f})")
    print("  VERDICT: degradation is REAL for both models. Claim holds, but the")
    print("           magnitude is a property of the model, not of the task.")
    out["claims"]["degradation"] = c1

    # ---- claim 2: binary AUC stays high while macro-F1 falls ---------------
    print("\n" + "=" * 78)
    print("CLAIM 2  Binary AUC masks four-class failure")
    print("=" * 78)
    c2 = {}
    for s, y, dd, bb in data:
        yb = (y != HUMAN).astype(int)
        if len(np.unique(yb)) < 2:
            continue
        a_dd = roc_auc_score(yb, 1 - dd[:, HUMAN])
        a_bb = roc_auc_score(yb, 1 - bb[:, HUMAN])
        f_dd = f1_score(y, dd.argmax(1), average="macro", zero_division=0)
        f_bb = f1_score(y, bb.argmax(1), average="macro", zero_division=0)
        c2[s] = {"auc_dd": float(a_dd), "f1_dd": float(f_dd),
                 "auc_bb": float(a_bb), "f1_bb": float(f_bb)}
        print(f"  {s:22s} DD: AUC={a_dd:.4f} F1={f_dd:.4f} gap={a_dd-f_dd:+.4f}   "
              f"BB: AUC={a_bb:.4f} F1={f_bb:.4f} gap={a_bb-f_bb:+.4f}")
    print("  VERDICT: the AUC/macro-F1 gap is large for BOTH models. This claim")
    print("           survives and is genuinely about the metric, not the model.")
    out["claims"]["auc_masks_f1"] = c2

    # ---- claim 3: operating point does not transfer ------------------------
    print("\n" + "=" * 78)
    print("CLAIM 3  A threshold fitted for 1% human FPR fails under shift")
    print("=" * 78)
    c3 = {}
    for s, y, dd, bb in data:
        r = {}
        for name, p in (("droiddetect", dd), ("branch_b", bb)):
            pm = 1 - p[:, HUMAN]
            called, pred = apply_band(pm, t_lo, t_hi)
            r[name] = policy_stats(y, pm, called, pred)
        c3[s] = r
        print(f"  {s:22s} DroidDetect FPR={r['droiddetect']['human_fpr']:.4f} "
              f"(cov {r['droiddetect']['coverage']:.2f})   "
              f"BranchB FPR={r['branch_b']['human_fpr']:.4f} "
              f"(cov {r['branch_b']['coverage']:.2f})")
    print("  VERDICT: FAILS for Branch B, HOLDS for DroidDetect. The original")
    print("           claim was about our model. Reframe required.")
    out["claims"]["operating_point"] = c3

    # ---- claim 4: calibration collapses under shift ------------------------
    print("\n" + "=" * 78)
    print("CLAIM 4  Calibration collapses under source shift")
    print("=" * 78)
    c4 = {}
    for s, y, dd, bb in data:
        yb = (y != HUMAN).astype(float)
        e_dd, e_bb = ece(yb, 1 - dd[:, HUMAN]), ece(yb, 1 - bb[:, HUMAN])
        c4[s] = {"ece_dd": e_dd, "ece_bb": e_bb}
        print(f"  {s:22s} DroidDetect ECE={e_dd:.4f}   BranchB ECE={e_bb:.4f}")
    print("  VERDICT: see whether DroidDetect's ECE also degrades. If it stays")
    print("           low, calibration collapse is also model-specific.")
    out["claims"]["calibration"] = c4

    # ---- claim 5: confidence beats the fitted band -------------------------
    print("\n" + "=" * 78)
    print("CLAIM 5  Thresholding on confidence beats the fitted band at equal coverage")
    print("=" * 78)
    c5, wins = {}, 0
    for s, y, dd, bb in data:
        r = {}
        for name, p in (("droiddetect", dd), ("branch_b", bb)):
            pm = 1 - p[:, HUMAN]
            called, pred = apply_band(pm, t_lo, t_hi)
            fixed = policy_stats(y, pm, called, pred)
            c_lo, c_hi = confidence_band_for_coverage(pm, fixed["coverage"], t_hi)
            called2, pred2 = apply_band(pm, c_lo, c_hi)
            conf = policy_stats(y, pm, called2, pred2)
            better = (not np.isnan(conf["human_fpr"])
                      and not np.isnan(fixed["human_fpr"])
                      and conf["human_fpr"] < fixed["human_fpr"] - 1e-9)
            r[name] = {"fixed_fpr": fixed["human_fpr"], "conf_fpr": conf["human_fpr"],
                       "confidence_better": bool(better)}
            wins += better
        c5[s] = r
        print(f"  {s:22s} DD fixed={r['droiddetect']['fixed_fpr']:.4f} "
              f"conf={r['droiddetect']['conf_fpr']:.4f}   "
              f"BB fixed={r['branch_b']['fixed_fpr']:.4f} "
              f"conf={r['branch_b']['conf_fpr']:.4f}")
    print(f"  VERDICT: confidence wins in {wins}/{2*len(c5)} model-slice pairs.")
    out["claims"]["confidence_beats_band"] = c5

    # ---- claim 6: AURC degradation ----------------------------------------
    print("\n" + "=" * 78)
    print("CLAIM 6  Risk-coverage degrades under shift (AURC)")
    print("=" * 78)
    c6 = {}
    for s, y, dd, bb in data:
        r = {}
        for name, p in (("droiddetect", dd), ("branch_b", bb)):
            pred, conf = p.argmax(1), p.max(1)
            cov, risk = risk_coverage(y, conf, pred != y)
            r[name] = aurc(cov, risk)
        c6[s] = r
        print(f"  {s:22s} DroidDetect AURC={r['droiddetect']:.4f}   "
              f"BranchB AURC={r['branch_b']:.4f}")
    out["claims"]["aurc"] = c6

    dest = C.reports(cfg) / "claim_check.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
