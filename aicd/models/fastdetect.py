"""Branch C: Fast-DetectGPT conditional-probability curvature, for code.

The value here is not standalone accuracy -- Droid measured Fast-DetectGPT at
67-77 weighted F1, well below a trained encoder. The value is that it needs no
training data at all. It never saw DroidCollection, so it is the control for the
competing explanation of this paper's central result: that S3 to S5 are simply
harder conditions rather than unexposed ones. A detector with no training set
cannot have had categories withheld from it, so whatever it does across the five
conditions is a property of the conditions themselves.

Score (Bao et al. 2024): with the scoring model's own distribution at each
position, compare the observed log-likelihood against the mean and variance of
the log-likelihood under that distribution:

    d = (log p(x) - mu) / sigma

Machine text sits near the conditional mean, so d is small; human text deviates,
so d is more negative. Higher d => more machine-like.

## Three properties this implementation has to get right

**Memory.** The statistic needs mu and var over the full vocabulary at every
position. Materialising those as dense tensors costs seq x vocab floats per
sample: at 512 tokens and Qwen's 151,936-token vocabulary that is 311 MB per
intermediate in fp32, and the naive expression builds about six of them, so
roughly 1.9 GB for a single sample. Batched sixteen ways that is 25-30 GB and
will not fit a 16 GB T4. The reduction is therefore chunked over positions in
`_row_stats`, which holds peak memory to `chunk x vocab` regardless of batch or
sequence length.

**Precision.** The reduction runs in fp32 even when the model runs in half
precision, because `p * logp^2` summed over 150k terms loses too much in fp16.
Weights use bf16 only on Ampere and later; on Turing, which is what a T4 is,
bf16 has no hardware support and falls back to slow paths, so fp16 is both
faster and, given the fp32 reduction, no less accurate.

**Padding.** A position is scored only when both the token doing the predicting
and the token being predicted are real. Batching without that mask would let
pad tokens contribute to the numerator and the variance, which changes the
score in a way that depends on the batch composition.

Optional refinement from Xu & Sheng (AAAI-24): weight perturbation toward
high-perplexity LINES rather than masking uniformly, which cut the samples they
needed from ~500 to ~50.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from aicd import config as C
from aicd.data.splits import SLICES

MIN_TOKENS = 8


def _dtype_for(dev: torch.device) -> torch.dtype:
    """Half precision on GPU, but bf16 only where the hardware has it.

    Turing (sm_75, the T4) exposes bf16 through emulation rather than tensor
    cores. Selecting it there costs a large multiple in wall clock for no
    accuracy gain, since the statistic itself is reduced in fp32 either way.
    """
    if dev.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def load_scorer(cfg, force_cpu: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg.fastdetect.scoring_model
    dev = torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=_dtype_for(dev)).to(dev)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding: position t predicts t+1, and the validity mask below drops
    # any position where either end of that pair is padding.
    tok.padding_side = "right"
    return tok, model, dev


@torch.no_grad()
def _row_stats(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 256):
    """Per-position observed log-prob, mean and variance, chunked over rows.

    `logits` is [N, V] and `targets` is [N]. Returns three fp32 tensors of
    length N. Peak additional memory is chunk x V floats rather than N x V,
    which is what makes batching possible at all on a 16 GB card.
    """
    n = logits.shape[0]
    obs = torch.empty(n, dtype=torch.float32, device=logits.device)
    mu = torch.empty(n, dtype=torch.float32, device=logits.device)
    var = torch.empty(n, dtype=torch.float32, device=logits.device)
    for i in range(0, n, chunk):
        lg = logits[i:i + chunk].float()
        lp = torch.log_softmax(lg, dim=-1)
        p = lp.exp()
        m = (p * lp).sum(-1)
        e2 = (p * lp.square()).sum(-1)
        obs[i:i + chunk] = lp.gather(-1, targets[i:i + chunk].unsqueeze(-1)).squeeze(-1)
        mu[i:i + chunk] = m
        # Clamped because the fp32 difference of two large sums can go a few
        # ulps negative on a near-deterministic distribution.
        var[i:i + chunk] = (e2 - m.square()).clamp(min=0.0)
        del lg, lp, p, m, e2
    return obs, mu, var


@torch.no_grad()
def curvature(code: str, tok, model, dev, max_length: int = 512,
              chunk: int = 256) -> float:
    """Fast-DetectGPT score for one sample. Analytic, no sampling needed."""
    enc = tok(code, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
    ids = enc["input_ids"]
    if ids.shape[1] < MIN_TOKENS:
        return 0.0

    logits = model(**enc).logits[0, :-1]        # predict token t+1 from t
    obs, mu, var = _row_stats(logits, ids[0, 1:], chunk)
    num = (obs - mu).sum()
    den = var.sum().clamp(min=1e-9).sqrt()
    return float((num / den).item())


@torch.no_grad()
def curvature_batch(codes, tok, model, dev, max_length: int = 512,
                    chunk: int = 256) -> np.ndarray:
    """Vectorised `curvature` over a list of sources.

    Equivalent to calling `curvature` on each element, to within fp32 rounding.
    `test_fastdetect.py` asserts that equivalence rather than assuming it.
    """
    codes = list(codes)
    out = np.zeros(len(codes), dtype=np.float32)
    if not codes:
        return out

    enc = tok(codes, truncation=True, max_length=max_length, padding=True,
              return_tensors="pt").to(dev)
    ids, am = enc["input_ids"], enc["attention_mask"]
    if ids.shape[1] < 2:
        return out

    logits = model(**enc).logits[:, :-1]
    b, t = ids.shape[0], ids.shape[1] - 1
    obs, mu, var = _row_stats(logits.reshape(-1, logits.shape[-1]),
                              ids[:, 1:].reshape(-1), chunk)

    # A position counts only if the predicting token and the predicted token
    # are both real. Padded positions are computed above and discarded here,
    # which wastes a little work and keeps the memory profile flat.
    valid = (am[:, :-1] * am[:, 1:]).float()
    num = ((obs.view(b, t) - mu.view(b, t)) * valid).sum(1)
    den = (var.view(b, t) * valid).sum(1).clamp(min=1e-9).sqrt()
    d = num / den
    d = torch.where(am.sum(1) >= MIN_TOKENS, d, torch.zeros_like(d))
    return d.float().cpu().numpy()


@torch.no_grad()
def line_perplexity(code: str, tok, model, dev, max_length: int = 512) -> list[float]:
    """Per-line perplexity, used by the AAAI-24 targeted-masking refinement.

    Lines the model finds surprising are the informative ones to perturb;
    uniform masking wastes most of its budget on formatting tokens.
    """
    out = []
    for line in code.split("\n"):
        if not line.strip():
            out.append(0.0)
            continue
        enc = tok(line, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        if enc["input_ids"].shape[1] < 2:
            out.append(0.0)
            continue
        loss = model(**enc, labels=enc["input_ids"]).loss
        out.append(float(torch.exp(loss).item()))
    return out


def score_frame(df: pd.DataFrame, tok, model, dev, cfg, tag: str = "",
                batch_size: int = 0, chunk: int = 256) -> np.ndarray:
    """Score a frame, batched and length-bucketed.

    Sorting by length before batching is what makes the batching worth having:
    on this corpus the median source is 230 tokens and the longest are 512, so
    batches drawn in corpus order pad most of their rows to the longest member
    and throw away a third of the compute.
    """
    bs = batch_size or int(getattr(cfg.fastdetect, "batch_size", 8))
    max_len = int(cfg.fastdetect.max_length)
    codes = df["code"].tolist()

    order = np.argsort([len(c) for c in codes], kind="stable")
    scores = np.zeros(len(codes), dtype=np.float32)

    t0, done = time.time(), 0
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        try:
            scores[idx] = curvature_batch([codes[j] for j in idx], tok, model,
                                          dev, max_len, chunk)
        except torch.cuda.OutOfMemoryError:
            # Fall back one sample at a time rather than losing the batch. A
            # silent zero here would look like a confident human prediction.
            torch.cuda.empty_cache()
            for j in idx:
                scores[j] = curvature(codes[j], tok, model, dev, max_len, chunk)
        done += len(idx)
        if i % (bs * 20) == 0 or done == len(order):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (len(order) - done) / max(rate, 1e-9)
            print(f"    [{tag}] {done:,}/{len(order):,}  {rate:.1f} rows/s  "
                  f"eta {eta/60:.1f} min", flush=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-per-slice", type=int, default=0,
                    help="0 scores every row; otherwise a per-slice cap")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256,
                    help="rows per vocabulary reduction; lower it if memory is tight")
    ap.add_argument("--tag", default="c")
    args = ap.parse_args()

    cfg = C.load(args.config)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")
    tok, model, dev = load_scorer(cfg, args.cpu)
    print(f"[branch_c] scorer={cfg.fastdetect.scoring_model} device={dev} "
          f"dtype={_dtype_for(dev)} batch={args.batch_size or cfg.fastdetect.batch_size}")

    from sklearn.metrics import roc_auc_score

    out = {"scorer": cfg.fastdetect.scoring_model, "slices": {}}
    rows = []
    for s in ["val"] + SLICES:
        m = df["slice"] == s
        if not m.any():
            continue
        sub = df.loc[m]
        if args.max_per_slice and len(sub) > args.max_per_slice:
            sub = sub.sample(n=args.max_per_slice, random_state=cfg.project.seed)
        sc = score_frame(sub, tok, model, dev, cfg, tag=s,
                         batch_size=args.batch_size, chunk=args.chunk)

        art = C.artifacts(cfg)
        np.save(art / f"score_{args.tag}_{s}.npy", sc)
        sub[["label"]].assign(score=sc).to_parquet(
            art / f"score_{args.tag}_{s}_index.parquet")

        y = (sub["label"].to_numpy() != 0).astype(int)
        auc = roc_auc_score(y, sc) if len(np.unique(y)) == 2 else float("nan")
        out["slices"][s] = {"n": int(len(sub)), "binary_auc": float(auc),
                            "mean_d": float(sc.mean()),
                            "mean_d_human": float(sc[y == 0].mean()) if (y == 0).any() else None,
                            "mean_d_machine": float(sc[y == 1].mean()) if (y == 1).any() else None}
        rows.append((s, len(sub), auc, float(sc.mean())))
        print(f"  {s:22s} n={len(sub):>7,} binary_AUC={auc:.4f}")

    print(f"\n{'slice':22s} {'n':>7s} {'AUC':>8s} {'mean d':>9s}")
    for s, n, auc, mu in rows:
        print(f"{s:22s} {n:>7,} {auc:>8.4f} {mu:>9.3f}")

    # The comparison this experiment exists to make. A trained detector with
    # categories withheld falls steeply from S1 to S5. A zero-shot scorer has
    # no training set, so if it holds roughly flat the conditions are not
    # intrinsically harder and the collapse belongs to the withholding.
    a = {k: v["binary_auc"] for k, v in out["slices"].items()}
    if "s1_in_distribution" in a and "s5_compound" in a:
        drop = a["s1_in_distribution"] - a["s5_compound"]
        out["auc_drop_s1_s5"] = float(drop)
        print(f"\nzero-shot binary AUC, S1 {a['s1_in_distribution']:.4f} -> "
              f"S5 {a['s5_compound']:.4f}  (drop {drop:+.4f})")
        print("Compare against the trained detector's binary AUC on the same")
        print("conditions. A flat zero-shot profile against a steep trained one")
        print("is the result; a matching decline would narrow the paper's claim.")

    # C.reports(cfg), not C.ROOT / "aicd" / "eval" / "reports". Two different
    # roots share the name in this codebase: modules under aicd/eval/ set their
    # own ROOT to the project directory via parents[2], while config.ROOT is the
    # aicd/ package directory. Joining "aicd/eval/reports" onto the latter wrote
    # to aicd/aicd/eval/reports, where the run's own report cell could not find
    # it. The helper is the only form that is correct from anywhere.
    dest = C.reports(cfg) / f"branch_{args.tag}_fastdetect.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
