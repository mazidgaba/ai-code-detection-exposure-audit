"""Measure row-level contamination on the GPU build, with evidence retained.

Every one of our five conditions draws roughly a third of its rows from the
DroidCollection training shard, because a balanced four-class corpus has to be
assembled across shards. Those rows are literally training data for the
published detector. Scoring it separately on them and on held-out rows measures
the inflation an uncontrolled evaluation would report.

## Why this module exists rather than the earlier one

The earlier measurement ran on the small CPU build, at roughly 400 rows per
cell, and its per-row scores were not retained. A figure quoted from a run whose
evidence no longer exists cannot be checked by a reader, and one of the two
figures the manuscript quoted for this could not be traced to any stored
artifact at all. This module saves the per-row labels, shard provenance and
probabilities alongside the report, so the numbers remain checkable.

The two groups are disjoint sets of rows, not the same rows scored twice, so the
interval on the difference comes from the two-sample estimator rather than the
paired one.

## The label vector

The run also writes the label, condition and shard for every evaluation row.
Probability arrays from earlier runs were retained without the labels they align
to, which makes a paired comparison between two arms impossible after the fact.
Saving the vector once, from the deterministic corpus build, repairs that for
every arm scored on this corpus.

    python -m aicd.eval.contamination --config kaggle.yaml --per-cell 5000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval import resample as R

SHARDS = ("train", "test", "dev")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="kaggle.yaml")
    ap.add_argument("--model", default="base", choices=["base", "large"])
    ap.add_argument("--per-cell", type=int, default=5000,
                    help="rows per (condition, shard) cell; 0 uses everything")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    cfg = C.load(args.config)
    art, rep = C.artifacts(cfg), C.reports(cfg)
    full = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet",
                           columns=["code", "label", "slice", "orig_split"])

    # --- the label vector, saved for every evaluation row --------------------
    # Small, and it is what lets any later comparison between two arms be made
    # row by row instead of only in aggregate.
    ev = full[full["slice"].isin(list(SLICES) + ["val"])]
    keys = ev[["label", "slice", "orig_split"]].reset_index(drop=True)
    keys.to_parquet(art / "eval_row_keys.parquet", index=False)
    print(f"saved eval_row_keys.parquet: {len(keys):,} rows")
    print(keys.groupby(["slice", "orig_split"]).size().to_string())

    from aicd.models import droiddetect_baseline as DD
    import torch
    DD.select(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = DD.load(cfg, pooling="mean", device=dev)

    rng = np.random.default_rng(args.seed)
    out = {"model": f"droiddetect-{args.model}", "build": "gpu",
           "per_cell": args.per_cell, "n_boot": args.n_boot,
           "method": "BCa two-sample", "conditions": {}}

    print(f"\n{'condition':22s} {'shard':6s} {'n':>7s} {'macro-F1':>9s}")
    print("-" * 50)
    for s in SLICES:
        entry, cache = {}, {}
        for shard in SHARDS:
            m = (full["slice"] == s) & (full["orig_split"] == shard)
            sub = full[m]
            if len(sub) < 100:
                continue
            if args.per_cell and len(sub) > args.per_cell:
                sub = sub.iloc[rng.choice(len(sub), args.per_cell, replace=False)]
            p = DD.score(model, tok, sub["code"].tolist(), dev,
                         max_len=args.max_len, bs=args.batch_size,
                         tag=f"{s}/{shard}")
            y = sub["label"].to_numpy()
            pred = p.argmax(1)
            cache[shard] = (y, pred)
            np.save(art / f"proba_contam_{s}_{shard}.npy", p)
            sub[["label", "orig_split"]].to_parquet(
                art / f"rows_contam_{s}_{shard}.parquet", index=False)
            f1 = R.macro_f1(y, pred)
            entry[shard] = {"n": int(len(y)), "macro_f1": f1,
                            "human_fpr": R.human_fpr(y, pred)}
            print(f"{s:22s} {shard:6s} {len(y):>7,} {f1:9.4f}"
                  f"{'   <- contaminated' if shard == 'train' else ''}")

        # Held-out is dev and test together: both were outside DroidCollection's
        # training split, and pooling them gives the comparison more rows.
        held = [cache[k] for k in ("test", "dev") if k in cache]
        if "train" in cache and held:
            yh = np.concatenate([h[0] for h in held])
            ph = np.concatenate([h[1] for h in held])
            yt, pt = cache["train"]

            # The two arms must average over the same classes. This corpus draws
            # no adversarial rows from the training shard, so a four-class
            # macro-F1 penalises the train arm for a class it cannot contain, and
            # the resulting gap comes out negative for a reason that has nothing
            # to do with memorisation. Restrict both arms to the classes present
            # in both, and average only over those.
            shared = sorted(set(np.unique(yt)) & set(np.unique(yh)))
            mt = np.isin(yt, shared)
            mh = np.isin(yh, shared)
            yt, pt, yh, ph = yt[mt], pt[mt], yh[mh], ph[mh]

            d, lo, hi = R.bca_two_sample(yt, pt, yh, ph, R.macro_f1_present,
                                         n_boot=args.n_boot, seed=args.seed)
            entry["shared_classes"] = [int(c) for c in shared]
            entry["inflation"] = {"delta": d, "ci95": [lo, hi],
                                  "n_train": int(len(yt)), "n_held": int(len(yh)),
                                  # An interval clear of zero on either side is a
                                  # finding; testing only the positive side hid
                                  # the sign error this comment describes.
                                  "excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"{'':22s} {'gap':6s} {'':>7s} {d:+9.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]")
        out["conditions"][s] = entry
        print()

    infl = [v["inflation"]["delta"] for v in out["conditions"].values()
            if "inflation" in v]
    if infl:
        out["mean_inflation"] = float(np.mean(infl))
        out["max_inflation"] = float(np.max(infl))
        worst = max(out["conditions"].items(),
                    key=lambda kv: kv[1].get("inflation", {}).get("delta", -9))
        out["worst_condition"] = worst[0]
        print("=" * 62)
        print(f"mean inflation {out['mean_inflation']:+.4f}   "
              f"max {out['max_inflation']:+.4f} on {worst[0]}")
        n_sig = sum(1 for v in out["conditions"].values()
                    if v.get("inflation", {}).get("excludes_zero"))
        print(f"{n_sig} of {len(infl)} conditions have an interval clear of zero")
        print("\nThe shape is the part the argument uses: if the inflation is")
        print("largest on S4 and S5, an uncontrolled evaluation overstates")
        print("robustness most exactly where the benchmark is meant to be hardest.")

    dest = rep / "contamination_gpu.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
