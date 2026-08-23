"""Score a detector on CodeMirage, broken down by how much it was exposed.

The point of a second external corpus is not another aggregate number. It is
that CodeMirage's generators fall into three tiers against DroidCollection's
training data, and the middle tier is the distinction this paper argues
benchmarks conflate:

  unseen_family             Anthropic, OpenAI o-series. Neither appears in
                            DroidCollection in any form.
  seen_family_unseen_model  Gemini 2.0, DeepSeek V3 and R1. The family is
                            present (google/codegemma, deepseek-coder) but
                            these particular models are not.
  seen_family_seen_model    GPT-4o-mini, Llama-3.3-70B, Qwen2.5-Coder-32B.
                            Present by name.

If accuracy falls monotonically across those tiers, the paper's claim
reproduces on data neither we nor the DroidCollection authors assembled, as a
gradient rather than a binary. If it does not, that is equally worth knowing
and the paper should say so.

The default subject is the published DroidDetect-Base, loaded from its own
repository, so this needs no checkpoint of ours and can run on any free slot.

    python -m aicd.eval.codemirage_eval
    python -m aicd.eval.codemirage_eval --max-eval 20000 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "aicd" / "artifacts" / "data" / "independent" / "codemirage_test.parquet"
LABEL_NAMES = ["human", "machine", "hybrid", "adversarial"]
HUMAN = 0
TIERS = ["human", "seen_family_seen_model", "seen_family_unseen_model",
         "unseen_family"]


def macro_present(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over the classes the corpus actually contains.

    CodeMirage has no hybrid class, so averaging over all four would divide by
    a class that cannot occur and depress the score for a reason that has
    nothing to do with the detector. The same treatment the AIGCodeSet
    evaluation uses, and for the same reason.
    """
    present = sorted(set(y.tolist()))
    return float(f1_score(y, pred, average="macro", labels=present,
                          zero_division=0))


def human_fpr(y: np.ndarray, pred: np.ndarray) -> float:
    m = y == HUMAN
    return float((pred[m] != HUMAN).mean()) if m.any() else float("nan")


def binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(set((y != HUMAN).tolist())) < 2:
        return float("nan")
    return float(roc_auc_score((y != HUMAN).astype(int), 1.0 - p[:, HUMAN]))


def stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows keeping the tier and label mix.

    Sampling uniformly would let the largest tier dominate and leave the
    unseen-family tier, the one the whole comparison rests on, thinly
    represented.
    """
    if n <= 0 or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    keep = []
    groups = list(df.groupby(["exposure", "label"], sort=True))
    per = max(n // max(len(groups), 1), 1)
    for _, g in groups:
        take = min(per, len(g))
        keep.append(g.iloc[rng.choice(len(g), size=take, replace=False)])
    return pd.concat(keep).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="kaggle.yaml")
    ap.add_argument("--model", default="base", choices=["base", "large"])
    ap.add_argument("--max-eval", type=int, default=0,
                    help="0 scores everything; otherwise a stratified sample")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} not found; run aicd.eval.codemirage_corpus first")

    from aicd import config as C
    from aicd.models import droiddetect_baseline as DD

    cfg = C.load(args.config)
    df = pd.read_parquet(CORPUS)
    df = stratified(df, args.max_eval, args.seed)
    print(f"scoring {len(df):,} rows")
    print(df.groupby(["exposure", "label"]).size().to_string())

    DD.select(args.model)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = DD.load(cfg, pooling="mean", device=dev)
    p = DD.score(model, tok, df["code"].tolist(), dev,
                 max_len=args.max_len, bs=args.batch_size, tag="codemirage")

    y = df["label"].to_numpy()
    pred = p.argmax(1)
    out = {"model": f"droiddetect-{args.model}", "rows": int(len(df)),
           "overall": {"macro_f1_present": macro_present(y, pred),
                       "binary_auc": binary_auc(y, p),
                       "human_fpr": human_fpr(y, pred)},
           "by_exposure": {}, "by_language": {}}

    # Human rows carry no generator, so they cannot belong to one tier. Each
    # machine tier is scored against the same shared human pool, which is what
    # makes the tiers comparable: the human class contributes identically to
    # every macro-F1 here.
    #
    # That also means human FPR is NOT a per-tier quantity. Reporting it in
    # this table printed the same figure on every row, which reads like a
    # finding and is nothing of the sort. It belongs to the detector and the
    # human pool, so it is reported once, above.
    print(f"\nhuman FPR on the shared pool of "
          f"{int((df['exposure'] == 'human').sum()):,} rows: "
          f"{out['overall']['human_fpr']:.4f}")
    print(f"\n{'tier':28s} {'n':>7s} {'macroF1':>9s} {'binAUC':>8s}")
    print("-" * 56)
    for t in TIERS:
        m = (df["exposure"] == t).to_numpy()
        if not m.any():
            continue
        sel = m | ((df["exposure"] == "human").to_numpy() if t != "human" else m)
        yy, pp = y[sel], p[sel]
        r = {"n": int(m.sum()),
             "scored_against": int(sel.sum()),
             "macro_f1_present": macro_present(yy, pp.argmax(1)),
             "binary_auc": binary_auc(yy, pp)}
        out["by_exposure"][t] = r
        print(f"{t:28s} {r['n']:>7,} {r['macro_f1_present']:9.4f} "
              f"{r['binary_auc']:8.4f}")

    for lang, g in df.groupby("language"):
        idx = g.index.to_numpy()
        out["by_language"][lang] = {
            "n": int(len(g)),
            "macro_f1_present": macro_present(y[idx], pred[idx]),
        }

    art = C.artifacts(cfg)
    os.makedirs(art, exist_ok=True)
    np.save(art / f"proba_codemirage_{args.model}.npy", p)
    df[["label", "exposure", "language", "generator"]].to_parquet(
        art / f"rows_codemirage_{args.model}.parquet", index=False)

    dest = ROOT / "aicd" / "eval" / "reports" / f"codemirage_eval_{args.model}.json"
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # The gradient is the reason this run exists, so it is stated whichever way
    # it comes out. A non-monotone result does not overturn the twin control,
    # which is a controlled experiment; it says that on an observational
    # comparison across generators, exposure is not the only thing driving
    # detectability. That belongs in the paper, not in a drawer.
    tiers = [out["by_exposure"].get(t, {}).get("macro_f1_present")
             for t in TIERS[1:]]
    print("\n" + "=" * 66)
    if all(v is not None for v in tiers):
        print("EXPOSURE GRADIENT  seen model -> unseen model -> unseen family")
        print("  " + "  ".join(f"{v:.4f}" for v in tiers))
        if tiers[0] >= tiers[1] >= tiers[2]:
            print("  Monotone. The paper's claim reproduces on a corpus neither")
            print("  we nor the DroidCollection authors built, as a gradient.")
        else:
            print("  NOT monotone. Report this plainly. Generator identity and")
            print("  detectability are confounded here in a way they are not in")
            print("  the twin control, so this narrows the claim rather than")
            print("  overturning it: exposure explains a large measured part of")
            print("  shift performance, and it is not the whole story.")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
