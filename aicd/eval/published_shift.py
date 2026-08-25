"""The published detector across all five conditions, on rows it never trained on.

The paper's claim is about published detectors evaluated on benchmark-withheld
categories, but the published detector was only ever reported here on S1, and on
rows drawn from every DroidCollection shard including the one it trained on. This
scores it on all five conditions restricted to the dev and test shards, which
DroidCollection held out of training, and puts a BCa interval on each.

No new inference is needed: the per-row probabilities are already stored, so this
is a restriction of arrays on disk. The contaminated train shard is reported
alongside, because the difference between the two is the size of the effect that
scoring on training rows produces.

    python -m aicd.eval.published_shift
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aicd import config as C
from aicd.eval import resample as R

SLICES = ("s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
          "s4_unseen_domain", "s5_compound")
SEED, B = 20240617, 4000


def main() -> None:
    cfg = C.load("cpu.yaml")
    art, rep = C.artifacts(cfg), C.reports(cfg)
    full = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet",
                           columns=["orig_split", "label", "slice"])

    out = {
        "note": ("DroidDetect-Base on every condition. 'clean' is the union of "
                 "the DroidCollection dev and test shards, which its training "
                 "never saw; 'seen' is the train shard it did."),
        "conditions": {},
    }

    for s in SLICES:
        pd_path = art / f"proba_droiddetect_{s}.npy"
        rows_path = art / f"rows_droiddetect_{s}.parquet"
        if not (pd_path.exists() and rows_path.exists()):
            print(f"  {s}: arrays missing, skipped")
            continue

        proba = np.load(pd_path)
        rows = pd.read_parquet(rows_path)
        meta = full.loc[rows.index]
        y = rows["label"].to_numpy()
        pred = proba.argmax(1)

        clean = (meta["orig_split"] != "train").to_numpy()
        seen = ~clean

        entry = {"rows_clean": int(clean.sum()), "rows_seen": int(seen.sum())}
        for tag, m in (("clean", clean), ("seen", seen)):
            if m.sum() < 20:
                continue
            f1 = R.macro_f1_present(y[m], pred[m])
            lo, hi = R.bca(y[m], pred[m], R.macro_f1_present,
                           n_boot=B, seed=SEED)[1:]
            entry[tag] = {"n": int(m.sum()), "macro_f1": round(float(f1), 4),
                          "ci": [round(float(lo), 4), round(float(hi), 4)]}
        if "clean" in entry and "seen" in entry:
            d, lo, hi = R.bca_two_sample(
                y[seen], pred[seen], y[clean], pred[clean],
                R.macro_f1_present, n_boot=B, seed=SEED)
            entry["shard_effect"] = round(float(d), 4)
            entry["shard_effect_ci"] = [round(float(lo), 4), round(float(hi), 4)]
            entry["excludes_zero"] = bool(lo > 0 or hi < 0)
        out["conditions"][s] = entry

    c = out["conditions"]
    if "s1_in_distribution" in c and "s5_compound" in c:
        out["clean_drop_s1_to_s5"] = round(
            c["s1_in_distribution"]["clean"]["macro_f1"]
            - c["s5_compound"]["clean"]["macro_f1"], 4)

    (rep / "published_shift.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")

    print(f"  {'condition':<24}{'n':>6}{'clean macro-F1':>18}"
          f"{'n':>6}{'seen macro-F1':>17}{'shard effect':>15}")
    for s, e in c.items():
        cl, sn = e.get("clean"), e.get("seen")
        if not (cl and sn):
            continue
        print(f"  {s:<24}{cl['n']:>6}  {cl['macro_f1']:.4f} "
              f"[{cl['ci'][0]:.4f}, {cl['ci'][1]:.4f}]"
              f"{sn['n']:>6}  {sn['macro_f1']:.4f}"
              f"{e['shard_effect']:>+13.4f}")
    if "clean_drop_s1_to_s5" in out:
        print(f"\n  clean S1 to S5 drop: {out['clean_drop_s1_to_s5']:+.4f}")
    print(f"  -> {(rep / 'published_shift.json').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
