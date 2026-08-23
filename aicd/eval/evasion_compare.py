"""Formatter evasion measured on the published detector, not just on ours.

The paper reports that whitespace normalisation costs Branch B 26.9 macro-F1
points. Having learned that the operating-point failure was a property of our
weak model rather than of the task, the same doubt applies here: perhaps only a
detector leaning on blank-line ratio is vulnerable, and a strong one is not.

This compares DroidDetect on raw versus normalised input over identical
held-out rows, so contamination and sampling are held constant and only the
whitespace differs.

    python -m aicd.eval.evasion_compare --config cpu.yaml
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


def stats(y, proba, t_hi, t_lo) -> dict:
    pred = proba.argmax(1)
    pm = 1.0 - proba[:, HUMAN]
    human = y == HUMAN
    called = (pm >= t_hi) | (pm <= t_lo)
    hc = human & called
    yb = (y != HUMAN).astype(int)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "binary_auc": float(roc_auc_score(yb, pm)) if len(np.unique(yb)) > 1 else float("nan"),
        "human_fpr_argmax": float((pred[human] != HUMAN).mean()) if human.any() else float("nan"),
        "human_fpr_policy": float((pm[hc] >= t_hi).mean()) if hc.any() else float("nan"),
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

    out = {"note": "identical held-out test-shard rows; only whitespace differs",
           "slices": {}}
    d_f1, d_fpr = [], []

    print(f"{'slice':22s} {'n':>4s} {'  raw F1':>9s} {'norm F1':>8s} {'delta':>8s} "
          f"{'raw AUC':>8s} {'norm AUC':>9s} {'raw FPR':>8s} {'norm FPR':>9s}")
    print("-" * 92)

    for s in SLICES:
        raw_p = art / f"proba_droiddetect_{s}.npy"
        raw_r = art / f"rows_droiddetect_{s}.parquet"
        fmt_p = art / f"proba_droiddetect_fmt_test_{s}.npy"
        fmt_r = art / f"rows_droiddetect_fmt_test_{s}.parquet"
        if not all(p.exists() for p in (raw_p, raw_r, fmt_p, fmt_r)):
            continue

        rr, fr = pd.read_parquet(raw_r), pd.read_parquet(fmt_r)
        # Restrict the raw run to the same held-out rows the normalised run used.
        common = rr.index.intersection(fr.index)
        if len(common) < 40:
            continue
        rpos = pd.Series(np.arange(len(rr)), index=rr.index).loc[common].to_numpy()
        fpos = pd.Series(np.arange(len(fr)), index=fr.index).loc[common].to_numpy()

        y = full.loc[common, "label"].to_numpy()
        raw = stats(y, np.load(raw_p)[rpos], t_hi, t_lo)
        fmt = stats(y, np.load(fmt_p)[fpos], t_hi, t_lo)
        delta = fmt["macro_f1"] - raw["macro_f1"]
        d_f1.append(delta)
        if not (np.isnan(raw["human_fpr_policy"]) or np.isnan(fmt["human_fpr_policy"])):
            d_fpr.append(fmt["human_fpr_policy"] - raw["human_fpr_policy"])

        out["slices"][s] = {"raw": raw, "normalised": fmt, "delta_macro_f1": delta}
        print(f"{s:22s} {len(common):>4} {raw['macro_f1']:>9.4f} {fmt['macro_f1']:>8.4f} "
              f"{delta:>+8.4f} {raw['binary_auc']:>8.4f} {fmt['binary_auc']:>9.4f} "
              f"{raw['human_fpr_policy']:>8.4f} {fmt['human_fpr_policy']:>9.4f}")

    out["mean_delta_macro_f1"] = float(np.mean(d_f1)) if d_f1 else float("nan")
    out["mean_delta_human_fpr"] = float(np.mean(d_fpr)) if d_fpr else float("nan")
    (rep / "evasion_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  DroidDetect loses {-np.mean(d_f1):.4f} macro-F1 on average to a one-command")
    print(f"  whitespace normalisation, across {len(d_f1)} slices.")
    print(f"  Human false-positive rate rises by {np.mean(d_fpr):+.4f} on average.")
    print()
    aucs = [(v['raw']['binary_auc'], v['normalised']['binary_auc'])
            for v in out["slices"].values()]
    ra, na = np.mean([a for a, _ in aucs]), np.mean([b for _, b in aucs])
    print(f"  Meanwhile binary AUC moves only {ra:.4f} -> {na:.4f} ({na - ra:+.4f}).")
    print("  The attack is therefore close to invisible under the metric the")
    print("  literature reports, which is the same blindness documented for")
    print("  distribution shift. Reported on the published state of the art,")
    print("  not on our own weaker model.")
    print(f"\n-> {rep / 'evasion_compare.json'}")


if __name__ == "__main__":
    main()
