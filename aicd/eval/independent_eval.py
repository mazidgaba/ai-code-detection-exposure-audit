"""Score a trained Branch A model on the independent corpus.

The question is narrow and worth stating exactly, because a looser version of
it would be easy to answer and would prove nothing: does the in-distribution to
shifted-conditions collapse reproduce on code that was assembled by other
people, from other generators, and that we have shown does not overlap ours?

Two properties of the corpus shape how the result must be read.

Three classes, not four. AIGCodeSet contains no adversarial category, so
macro-F1 is computed over the classes that are present. Averaging a missing
class in at zero would manufacture a collapse out of arithmetic rather than
measuring one, which is precisely the sort of thing this paper objects to.

Gemini is the sharp edge. Codestral and Llama belong to families that appear in
DroidCollection; Gemini does not appear anywhere in it. The per-generator
breakdown therefore carries the cleanest unseen-generator signal available, and
it is reported separately rather than folded into the average.

    python -m aicd.eval.independent_eval --weights branch_a_matched.pt
    python -m aicd.eval.independent_eval --limit 1500      # quick look
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)

from aicd import config as C

NAMES = ["human", "machine", "hybrid", "adversarial"]
HUMAN = 0


def load_model(cfg, weights: str, device: str):
    from transformers import AutoModel, AutoTokenizer
    from aicd.models.modernbert_triplet import TLModel

    path = C.artifacts(cfg) / weights
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
    print(f"[model] {weights}  base={base}")
    print(f"        {len(missing)} missing, {len(unexpected)} unexpected tensors")
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def score(model, tok, codes, device, max_len, batch=8, cache=None):
    """Score every row, checkpointing so a killed run resumes instead of restarting.

    On CPU this is a multi-hour job, and it has already been killed once by
    memory pressure from unrelated processes. Partial results are written every
    few hundred rows and picked up on the next attempt, so an interruption
    costs minutes rather than the whole run.
    """
    done0 = 0
    out = []
    if cache is not None and os.path.exists(cache):
        prev = np.load(cache)
        if prev.ndim == 2 and prev.shape[0] <= len(codes):
            out, done0 = [prev], int(prev.shape[0])
            print(f"    resuming: {done0}/{len(codes)} already scored")

    t0 = time.time()
    for i in range(done0, len(codes), batch):
        chunk = [str(c) for c in codes[i:i + batch]]
        enc = tok(chunk, truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=(device == "cuda")):
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"])["logits"]
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        done = min(i + batch, len(codes))
        if done % 200 < batch or done == len(codes):
            rate = (done - done0) / max(time.time() - t0, 1e-9)
            print(f"    {done}/{len(codes)}  {rate:.1f} rows/s  "
                  f"eta {(len(codes)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
            if cache is not None:
                tmp = cache + ".tmp.npy"
                np.save(tmp, np.concatenate(out))
                os.replace(tmp, cache)
    return np.concatenate(out)


def bootstrap(y, pred, present, n_boot=1000, seed=0):
    """Percentile intervals for the headline quantities.

    Resamples rows with replacement, which is the right unit here: each row is
    one independent code sample. Wilson intervals cover the simple proportions
    but say nothing about macro-F1, which is a non-linear function of the whole
    confusion matrix and needs resampling.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    macro, fpr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb, pb = y[i], pred[i]
        macro.append(f1_score(yb, pb, labels=present, average="macro",
                              zero_division=0))
        h = yb == HUMAN
        fpr.append((pb[h] != HUMAN).mean() if h.any() else np.nan)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"macro_f1": q(macro), "human_false_accusation_rate": q(fpr),
            "n_boot": n_boot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--weights", default="branch_a_matched.pt")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="stratified subsample, for a quick look")
    args = ap.parse_args()

    cfg = C.load(args.config)
    df = pd.read_parquet(C.ROOT / "artifacts" / "data" / "independent" /
                         "independent.parquet")

    if args.limit and args.limit < len(df):
        idx = []
        for _, g in df.groupby("label"):
            k = max(1, int(round(args.limit * len(g) / len(df))))
            idx.extend(g.sample(n=min(k, len(g)), random_state=0).index)
        df = df.loc[idx].reset_index(drop=True)

    y = df["label"].to_numpy()
    present = sorted(set(y.tolist()))
    print(f"[data] {len(df):,} rows, classes present: "
          f"{[NAMES[i] for i in present]}")
    print(f"       counts: {df['label'].value_counts().sort_index().to_dict()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    tok, model = load_model(cfg, args.weights, device)

    cache = None if args.limit else str(C.artifacts(cfg) / "proba_independent_partial.npy")
    p = score(model, tok, df["code"].tolist(), device, args.max_len,
              args.batch_size, cache=cache)
    pred = p.argmax(1)

    # Macro-F1 over the classes the corpus actually contains. Including the
    # absent adversarial class would score it 0 and invent a collapse.
    macro = float(f1_score(y, pred, labels=present, average="macro",
                           zero_division=0))
    yb, pb = (y != HUMAN).astype(int), 1.0 - p[:, HUMAN]
    auc = float(roc_auc_score(yb, pb)) if yb.min() != yb.max() else float("nan")
    human_fpr = float((pred[y == HUMAN] != HUMAN).mean())

    print("\n" + "=" * 62)
    print(f"macro-F1 over {len(present)} present classes : {macro:.4f}")
    print(f"binary AUC (human vs rest)              : {auc:.4f}")
    print(f"human false-accusation rate             : {human_fpr:.4f}")
    print("=" * 62)
    print("\n" + classification_report(
        y, pred, labels=present, target_names=[NAMES[i] for i in present],
        zero_division=0, digits=4))

    print("confusion (rows = true, cols = predicted, all 4 columns):")
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    print("            " + "".join(f"{n[:9]:>10s}" for n in NAMES))
    for i in present:
        row = cm[i]
        tot = max(row.sum(), 1)
        print(f"  {NAMES[i]:9s} " + "".join(f"{v/tot:10.3f}" for v in row))

    # Per generator. Gemini is absent from DroidCollection entirely, so it is
    # the one column that is unambiguously an unseen generator.
    print("\nper generator (machine and hybrid rows):")
    per = {}
    for g, sub in df[df["label"] != HUMAN].groupby("generator"):
        m = df.index.isin(sub.index)
        rec = float((pred[m] != HUMAN).mean())
        per[g] = {"n": int(m.sum()), "recall_as_nonhuman": rec}
        flag = "  <- not in DroidCollection" if g == "gemini" else ""
        print(f"  {g:11s} n={m.sum():5d}  detected as non-human {rec:.4f}{flag}")

    ci = bootstrap(y, pred, present)
    print()
    print(f"bootstrap 95% CI over {ci['n_boot']} resamples:")
    print(f"  macro-F1                : [{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}]")
    print(f"  human false-accusation  : "
          f"[{ci['human_false_accusation_rate'][0]:.4f}, "
          f"{ci['human_false_accusation_rate'][1]:.4f}]")

    # Per-class figures belong in the report, not only in the printed table:
    # the paper cites them, and anything the paper cites has to be traceable
    # to a file rather than to a log that scrolled past.
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1c, sup = precision_recall_fscore_support(
        y, pred, labels=present, zero_division=0)
    per_class = {NAMES[c]: {"precision": float(pr[i]), "recall": float(rc[i]),
                            "f1": float(f1c[i]), "support": int(sup[i])}
                 for i, c in enumerate(present)}

    out = {
        "ci": ci,
        "per_class": per_class,
        "weights": args.weights,
        "n_rows": int(len(df)),
        "classes_present": [NAMES[i] for i in present],
        "macro_f1_present_classes": macro,
        "binary_auc": auc,
        "human_false_accusation_rate": human_fpr,
        "per_generator": per,
        "confusion_row_normalised": {
            NAMES[i]: (cm[i] / max(cm[i].sum(), 1)).round(4).tolist()
            for i in present},
        "note": ("AIGCodeSet has no adversarial class; macro-F1 is over the "
                 "three present classes."),
    }
    rep = C.ROOT / "eval" / "reports" / "independent_eval.json"
    # The upload zip omits eval/reports, so on a fresh Kaggle checkout the
    # directory does not exist and the write fails after all the work is done.
    os.makedirs(rep.parent, exist_ok=True)
    io.open(rep, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    np.save(C.artifacts(cfg) / "proba_independent.npy", p)
    print(f"\n-> {rep}")


if __name__ == "__main__":
    main()
