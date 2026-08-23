"""Compare our detector against published DroidDetect, correcting for contamination.

The naive comparison is invalid, and the reason is worth stating carefully.

Our five evaluation slices are out-of-distribution *relative to our own training
split*: we withheld two generator families, two languages and three sources.
They are not out-of-distribution for DroidDetect, which was trained on the
DroidCollection train shard. And every one of our slices draws roughly a third
of its rows from that shard, because we sampled across shards to obtain all four
labels (Section III-C). So a third of every slice is literally training data for
the baseline.

This module separates the two populations by `orig_split` and reports them
apart:

    train shard   DroidDetect saw these rows while training   (contaminated)
    test shard    held out of DroidCollection training        (clean)

The gap between them measures memorisation. The clean subset is the only fair
basis for comparing the two detectors, and our own model is scored on exactly
the same rows so nothing else differs.

Usage:
    python -m aicd.eval.baseline_compare --config cpu.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from aicd import config as C
from aicd.data.splits import SLICES

HUMAN = 0


def scores(y: np.ndarray, proba: np.ndarray, t_hi: float, t_lo: float) -> dict:
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    pred = proba.argmax(1)
    p_mach = 1.0 - proba[:, HUMAN]
    human = y == HUMAN
    called = (p_mach >= t_hi) | (p_mach <= t_lo)
    hc = human & called
    yb = (y != HUMAN).astype(int)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "binary_auc": float(roc_auc_score(yb, p_mach)) if len(np.unique(yb)) > 1 else float("nan"),
        "human_fpr_argmax": float((pred[human] != HUMAN).mean()) if human.any() else float("nan"),
        "human_fpr_policy": float((p_mach[hc] >= t_hi).mean()) if hc.any() else float("nan"),
        "coverage": float(called.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    args = ap.parse_args()
    cfg = C.load(args.config)
    art, rep = C.artifacts(cfg), C.reports(cfg)

    full = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet",
                           columns=["orig_split", "label", "slice"])
    pol = json.loads((rep / "policy_b.json").read_text(encoding="utf-8"))
    t_hi, t_lo = pol["thresholds"]["t_high"], pol["thresholds"]["t_low"]

    out = {"thresholds": {"t_high": t_hi, "t_low": t_lo}, "slices": {}}
    print(f"{'slice':22s} {'shard':6s} {'n':>5s} {'  DroidDetect':>26s} "
          f"{'    Branch B':>26s}")
    print(f"{'':22s} {'':6s} {'':>5s} {'macroF1  humFPR@pol':>26s} "
          f"{'macroF1  humFPR@pol':>26s}")
    print("-" * 92)

    for s in SLICES:
        pd_path = art / f"proba_droiddetect_{s}.npy"
        rows_path = art / f"rows_droiddetect_{s}.parquet"
        b_path = art / f"proba_b_{s}.npy"
        if not (pd_path.exists() and rows_path.exists() and b_path.exists()):
            continue

        dd = np.load(pd_path)
        rows = pd.read_parquet(rows_path)
        meta = full.loc[rows.index]

        # Branch B was scored on the whole slice; line it up with the sampled
        # rows by position within the slice.
        slice_idx = full.index[full["slice"] == s]
        pos = pd.Series(np.arange(len(slice_idx)), index=slice_idx)
        bb_all = np.load(b_path)
        bb = bb_all[pos.loc[rows.index].to_numpy()]

        y = rows["label"].to_numpy()
        entry = {}
        for shard in ("train", "dev", "test"):
            m = (meta["orig_split"] == shard).to_numpy()
            if m.sum() < 20:
                continue
            d = scores(y[m], dd[m], t_hi, t_lo)
            b = scores(y[m], bb[m], t_hi, t_lo)
            entry[shard] = {"droiddetect": d, "branch_b": b}
            note = " <- contaminated" if shard == "train" else ""
            print(f"{s:22s} {shard:6s} {d['n']:>5} "
                  f"{d.get('macro_f1', float('nan')):>10.4f} "
                  f"{d.get('human_fpr_policy', float('nan')):>10.4f} "
                  f"{b.get('macro_f1', float('nan')):>14.4f} "
                  f"{b.get('human_fpr_policy', float('nan')):>10.4f}{note}")
        out["slices"][s] = entry
        print()

    dest = rep / "baseline_compare.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {dest}\n")

    # --- how much of DroidDetect's score is memorisation? --------------------
    print("=" * 72)
    print("MEMORISATION CHECK: DroidDetect on its training shard vs held-out test")
    print("=" * 72)
    gaps = []
    for s, e in out["slices"].items():
        if "train" in e and "test" in e:
            a = e["train"]["droiddetect"].get("macro_f1")
            b = e["test"]["droiddetect"].get("macro_f1")
            if a and b:
                gaps.append(a - b)
                print(f"  {s:22s} train={a:.4f}  test={b:.4f}  gap={a - b:+.4f}")
    if gaps:
        print(f"\n  mean gap {np.mean(gaps):+.4f}")
        if np.mean(gaps) < 0.02:
            print("  Small gap: DroidDetect generalises rather than memorising, so the")
            print("  contaminated rows were not inflating it much. The clean-subset")
            print("  comparison below stands as the honest one either way.")
        else:
            print("  Material gap: part of DroidDetect's advantage on these slices is")
            print("  recall of its own training data, not generalisation.")

    print("\n" + "=" * 72)
    print("HONEST COMPARISON: held-out test-shard rows only")
    print("=" * 72)
    for s, e in out["slices"].items():
        if "test" not in e:
            continue
        d, b = e["test"]["droiddetect"], e["test"]["branch_b"]
        print(f"  {s:22s} n={d['n']:>4}  DroidDetect macroF1={d.get('macro_f1', float('nan')):.4f} "
              f"humFPR={d.get('human_fpr_policy', float('nan')):.4f}   "
              f"BranchB macroF1={b.get('macro_f1', float('nan')):.4f} "
              f"humFPR={b.get('human_fpr_policy', float('nan')):.4f}")


if __name__ == "__main__":
    main()
