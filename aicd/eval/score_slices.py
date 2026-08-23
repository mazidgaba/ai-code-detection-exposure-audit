"""Score any saved Branch A checkpoint on a given build's five conditions.

Why this exists. The matched-scale run trained on three DroidCollection shards
instead of one, and its *evaluation* slices grew with the corpus: 62,283 rows
in S1 against the original 31,083. Comparing its 0.9130 with the original run's
0.8977 therefore compares two models on two different test sets, which is not a
comparison at all. Each run's own S1-to-S5 drop remains valid, but the levels
do not line up and should not be read as though they do.

The fix is to put the models on the same rows. This loads a checkpoint and
scores it against whatever split matrix is on disk, so pointing it at the
single-shard build yields numbers directly comparable with the manuscript.

Evaluation reuses aicd.models.modernbert_triplet.evaluate, the same path that
produced every other number in the paper, rather than a second implementation
that could drift from it.

    python -m aicd.eval.score_slices --weights branch_a_matched.pt --tag matched_on_original
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from aicd import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="kaggle.yaml")
    ap.add_argument("--weights", required=True,
                    help="checkpoint filename inside the artifacts directory")
    ap.add_argument("--tag", required=True,
                    help="names the report: branch_a_<tag>.json")
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--expect-train-rows", type=int, default=None,
                    help="fail unless the split matrix has this many training "
                         "rows; guards against scoring the wrong build")
    args = ap.parse_args()

    from aicd.models.modernbert_triplet import TLModel, evaluate
    from transformers import AutoModel, AutoTokenizer

    cfg = C.load(args.config)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    n_train = int((df["split"] == "train").sum())
    print(f"[data] {len(df):,} rows, {n_train:,} training")
    for s in sorted(df["slice"].dropna().unique()):
        print(f"       {s:24s} {int((df['slice'] == s).sum()):>8,}")
    if args.expect_train_rows and n_train != args.expect_train_rows:
        raise SystemExit(
            f"training split has {n_train:,} rows, expected "
            f"{args.expect_train_rows:,}. This is the wrong corpus build, and "
            "scoring it would produce numbers that look comparable and are not.")

    path = C.artifacts(cfg) / args.weights
    if not path.exists():
        raise SystemExit(f"weights not found: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    base = ck.get("base_model", cfg.modernbert.base_model)

    tok = AutoTokenizer.from_pretrained(base)
    enc = AutoModel.from_pretrained(base)
    model = TLModel(enc, enc.config.hidden_size,
                    projection_dim=ck.get("projection_dim",
                                          cfg.modernbert.projection_dim),
                    num_classes=cfg.modernbert.num_classes)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    print(f"[model] {args.weights} on {dev}: "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    # Distinct suffix so these probability arrays never overwrite the ones
    # belonging to the run the checkpoint originally came from.
    res = evaluate(cfg, df, model, tok, dev, f"branch_a_{args.tag}",
                   args.max_eval, suffix=f"_{args.tag}")

    rep = C.reports(cfg) / f"branch_a_{args.tag}.json"
    os.makedirs(rep.parent, exist_ok=True)
    print(f"\n-> {rep}")

    s = res.get("slices", res)
    if "s1_in_distribution" in s and "s5_compound" in s:
        a, b = s["s1_in_distribution"]["macro_f1"], s["s5_compound"]["macro_f1"]
        print(f"\n  S1 {a:.4f} -> S5 {b:.4f}   drop {a - b:.4f}")


if __name__ == "__main__":
    main()
