"""Run the published DroidDetect detector through our five evaluation slices.

Why this exists. The paper's central claim is that operating points do not
transfer under distribution shift. Demonstrating that on our own Branch B, which
scores 0.72 macro-F1 where published work reports 0.98, invites the obvious
rebuttal: perhaps weak detectors fail this way and strong ones do not. The claim
is only general if a state-of-the-art detector fails the same way on the same
slices under the same threshold procedure.

DroidDetect is that detector, and both published sizes can be loaded here.
Base and Large name the same training data and differ only in encoder width,
which makes Large a capacity control: if the behaviour we document were merely
a small model running out of parameters, the large one would not repeat it.

Loading it takes some care. The HuggingFace repo ships a custom architecture
(`model_type: custom_model`) with no modelling code, so `AutoModel` cannot load
it. The state dict resolves to three parts:

    text_encoder.*        ModernBERT-base, 22 layers, hidden 768
    text_projection.*     Linear(768 -> 256)
    classifier.*          Linear(256 -> 4)

Note that config.json advertises `projection_dim: 128` while the weights are
256. The weights win; the config is stale.

Pooling is documented on both model cards as a masked mean, and that is what
we use. --probe re-checks it against CLS and masked-max on the validation
split, never on a reported condition, because a reconstruction choice fitted on
test data is a reconstruction tuned to flatter itself.

    python -m aicd.models.droiddetect_baseline --config cpu.yaml --probe
    python -m aicd.models.droiddetect_baseline --config cpu.yaml --max-eval 1500
    python -m aicd.models.droiddetect_baseline --model large --config kaggle.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval import metrics as M

# The published detectors we can load. Both name the same training data, the
# filtered training split of DroidCollection, and differ only in encoder size.
# That makes Large a capacity control: if the base model's behaviour on our
# conditions were simply a matter of too few parameters, the large one would
# not reproduce it.
MODELS = {
    "base": {"repo": "project-droid/DroidDetect-Base",
             "encoder": "answerdotai/ModernBERT-base",
             "weights": "droiddetect/pytorch_model.bin"},
    "large": {"repo": "project-droid/DroidDetect-Large",
              "encoder": "answerdotai/ModernBERT-large",
              "weights": "droiddetect_large/pytorch_model.bin"},
}
REPO = MODELS["base"]["repo"]
ENCODER = MODELS["base"]["encoder"]
WEIGHTS = MODELS["base"]["weights"]


def select(name: str) -> None:
    """Point the module at one of the published checkpoints."""
    global REPO, ENCODER, WEIGHTS
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    REPO = MODELS[name]["repo"]
    ENCODER = MODELS[name]["encoder"]
    WEIGHTS = MODELS[name]["weights"]
    print(f"[droiddetect] model={name} repo={REPO} encoder={ENCODER}")


class DroidDetect(nn.Module):
    """Reconstruction of the published architecture around loaded weights."""

    def __init__(self, encoder, hidden: int, proj_dim: int, n_classes: int,
                 pooling: str = "mean", normalize: bool = False):
        super().__init__()
        self.text_encoder = encoder
        self.text_projection = nn.Linear(hidden, proj_dim)
        self.classifier = nn.Linear(proj_dim, n_classes)
        self.pooling = pooling
        # DroidDetect is trained with a triplet term, and triplet objectives
        # normally operate on L2-normalised embeddings. If the published
        # inference normalises and we do not, the logits keep their direction
        # but change scale, which leaves ranking intact while skewing argmax
        # toward one class. That is exactly the pattern the large checkpoint
        # shows, so the option exists to test it rather than argue about it.
        self.normalize = normalize

    def forward(self, input_ids, attention_mask):
        h = self.text_encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        if self.pooling == "cls":
            emb = h[:, 0]
        elif self.pooling == "max":
            emb = h.masked_fill(m == 0, torch.finfo(h.dtype).min).max(1).values
        else:
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        z = self.text_projection(emb)
        if self.normalize:
            z = torch.nn.functional.normalize(z, p=2, dim=1)
        return self.classifier(z)


def load(cfg, pooling: str = "mean", device: str = "cpu", normalize: bool = False):
    from transformers import AutoModel, AutoTokenizer

    path = C.artifacts(cfg) / WEIGHTS
    if not path.exists():
        # Fetch it rather than telling the caller to. Every hosted run starts
        # with an empty working directory, so a module that only prints a curl
        # command fails eighty-five seconds in, after the corpus has already
        # been built, and wastes the session. The weights are public.
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            print(f"[droiddetect] weights absent; downloading from {REPO}")
            got = hf_hub_download(repo_id=REPO, filename="pytorch_model.bin")
            shutil.copy(got, path)
            print(f"[droiddetect] weights -> {path}")
        except Exception as e:                                   # noqa: BLE001
            raise SystemExit(
                f"weights not found at {path} and the download failed ({e})\n"
                f"  curl -L https://huggingface.co/{REPO}/resolve/main/"
                f"pytorch_model.bin -o {path}")

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        sd = sd.state_dict()

    proj_w = sd["text_projection.weight"]
    proj_dim, hidden = proj_w.shape
    n_classes = sd["classifier.weight"].shape[0]

    enc = AutoModel.from_pretrained(ENCODER)
    model = DroidDetect(enc, hidden, proj_dim, n_classes, pooling, normalize)

    enc_sd = {k[len("text_encoder."):]: v for k, v in sd.items()
              if k.startswith("text_encoder.")}
    missing, unexpected = model.text_encoder.load_state_dict(enc_sd, strict=False)
    model.text_projection.load_state_dict(
        {"weight": sd["text_projection.weight"], "bias": sd["text_projection.bias"]})
    model.classifier.load_state_dict(
        {"weight": sd["classifier.weight"], "bias": sd["classifier.bias"]})

    print(f"[droiddetect] hidden={hidden} proj={proj_dim} classes={n_classes} "
          f"pooling={pooling} l2norm={normalize}")
    print(f"  encoder tensors: {len(enc_sd)} loaded, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if len(missing) > 4:
        print(f"  WARNING first missing: {missing[:4]}")

    tok = AutoTokenizer.from_pretrained(REPO)
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def score(model, tok, codes, device, max_len=512, bs=8, tag="") -> np.ndarray:
    out, codes = [], list(codes)
    t0 = time.time()
    for i in range(0, len(codes), bs):
        enc = tok(codes[i: i + bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        logits = model(**enc)
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        if i % (bs * 25) == 0 and i:
            rate = i / (time.time() - t0)
            print(f"    [{tag}] {i:,}/{len(codes):,}  {rate:.1f} rows/s", flush=True)
    return np.vstack(out) if out else np.zeros((0, 4))


def stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n rows keeping the label mix.

    Index-based rather than groupby().apply(): pandas 3.0 excludes the grouping
    column from the applied frame, which silently drops 'label' and produces a
    KeyError far from the cause.
    """
    if n is None or len(df) <= n:
        return df
    idx = []
    for lab, g in df.groupby("label"):
        k = max(1, int(round(n * len(g) / len(df))))
        idx.extend(g.sample(n=min(k, len(g)), random_state=seed).index)
    return df.loc[idx]


def probe(cfg, device, n=400, max_len=512, slice_name="val", report=None):
    """Pick the pooling that actually reproduces published behaviour.

    The repo does not document whether the head reads the CLS token, a masked
    mean or a masked max. Guessing wrong yields a model near chance, which is
    easy to mistake for a genuine negative result, so the choice is decided
    empirically before the real evaluation runs.

    It is decided on the validation split. An earlier version of this function
    read s1_in_distribution, which is a reported evaluation condition, so the
    reconstruction was being tuned on test labels. The effect was small, one
    binary choice, but it is leakage and the split is the fix. Attention
    pooling is not a candidate: it needs learned query parameters, and the
    published checkpoint contains none, so it cannot be reconstructed.
    """
    # Push the slice filter into the reader. Loading every code column of the
    # full corpus to keep a few hundred rows needs several GB and is what gets
    # this killed on a machine that is doing anything else.
    path = C.ROOT / cfg.data.cache_dir / "splits.parquet"
    keys = pd.read_parquet(path, columns=["slice", "label"])
    want = keys.index[keys["slice"] == slice_name]
    if len(want) == 0:
        raise SystemExit(f"no rows in slice {slice_name!r}")
    picked = stratified(keys.loc[want], n, seed=0).index
    del keys, want

    full = pd.read_parquet(path, columns=["slice", "label", "code"])
    sub = full.loc[picked].copy()
    del full
    print(f"[probe] {len(sub)} rows from {slice_name!r}, "
          f"labels={sub['label'].value_counts().sort_index().tolist()}")

    from sklearn.metrics import f1_score
    y = sub["label"].to_numpy()
    rows, best = [], None
    for pooling in ("mean", "cls", "max"):
        for norm in (False, True):
            tok, model = load(cfg, pooling, device, normalize=norm)
            tag = f"{pooling}{'+l2' if norm else ''}"
            p = score(model, tok, sub["code"], device, max_len, tag=tag)
            acc = float((p.argmax(1) == y).mean())
            f1 = float(f1_score(y, p.argmax(1), average="macro",
                                zero_division=0))
            share = np.bincount(p.argmax(1), minlength=4) / len(y)
            print(f"[probe] {tag:9s} accuracy={acc:.4f} macro-F1={f1:.4f} "
                  f"pred_share={np.round(share, 3).tolist()}")
            rows.append({"pooling": pooling, "l2norm": norm,
                         "accuracy": acc, "macro_f1": f1,
                         "predicted_share": share.round(4).tolist()})
            if best is None or f1 > best[1]:
                best = (f"{pooling}{'+l2' if norm else ''}", f1)
            del model
    print(f"\n[probe] selected pooling = {best[0]} (macro-F1 {best[1]:.4f})")
    print("A published model should score well above chance here. If all three")
    print("are near chance the architecture reconstruction is wrong, not the model.")

    if report:
        out = {"selected": best[0], "selection_slice": slice_name,
               "n_rows": int(len(sub)), "max_len": max_len, "candidates": rows}
        io.open(report, "w", encoding="utf-8").write(json.dumps(out, indent=2))
        print(f"-> {report}")
    return best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--model", default="base", choices=["base", "large"],
                    help="which published DroidDetect checkpoint to evaluate")
    ap.add_argument("--pooling", default=None, choices=["mean", "cls", "max"])
    ap.add_argument("--probe", action="store_true",
                    help="decide pooling empirically, then exit")
    ap.add_argument("--probe-slice", default="val",
                    help="split the pooling probe reads; must not be a "
                         "reported evaluation condition")
    ap.add_argument("--probe-n", type=int, default=400)
    ap.add_argument("--max-eval", type=int, default=1500,
                    help="rows sampled per slice; stratified by label")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--normalize", action="store_true",
                    help="whitespace-normalise the input first (evasion test)")
    ap.add_argument("--force", action="store_true",
                    help="rescore slices even if cached output exists")
    ap.add_argument("--test-shard-only", action="store_true",
                    help="score only rows DroidDetect did not train on")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = C.load(args.config)
    select(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[droiddetect] device={device}")

    if args.probe:
        probe(cfg, device, n=args.probe_n, max_len=args.max_len,
              slice_name=args.probe_slice,
              report=C.reports(cfg) /
                     f"droiddetect_pooling_probe{'' if args.model == 'base' else '_' + args.model}.json")
        return

    pooling = args.pooling or "mean"
    tok, model = load(cfg, pooling, device)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    proba_by_slice, art = {}, C.artifacts(cfg)
    for s in ["val"] + list(SLICES):
        m = df["slice"] == s
        if not m.any():
            continue
        sub = df[m]
        # Stratify so a rare class is not sampled away; the whole point is
        # the human row of the confusion matrix.
        sub = stratified(sub, args.max_eval, cfg.project.seed)
        if args.test_shard_only:
            sub = sub[sub["orig_split"] == "test"]
            if len(sub) < 40:
                continue
        # Resume support: a completed slice is already on disk, so a restart
        # after a crash, a shutdown or an OOM kill picks up where it stopped
        # instead of repeating an hour of inference.
        # The cache key must carry the model size for the same reason the
        # output filename does. Without it the large model finds the base
        # model's arrays, reuses them, and reports base numbers under its own
        # name: a wrong answer that looks entirely plausible.
        size = "" if args.model == "base" else f"_{args.model}"
        tag_pre = size + ("_fmt" if args.normalize else "") + ("_test" if args.test_shard_only else "")
        cached = art / f"proba_droiddetect{tag_pre}_{s}.npy"
        if cached.exists() and not args.force:
            p_cached = np.load(cached)
            if len(p_cached) == len(sub):
                print(f"[droiddetect] {s}: {len(sub):,} rows -- cached, skipping")
                proba_by_slice[s] = (sub["label"].to_numpy(), p_cached)
                continue

        codes = sub["code"]
        if args.normalize:
            # Same transformation as the Branch B evasion test, so the two
            # models are attacked identically.
            from aicd.models.formatter_ablation import normalize_whitespace
            codes = codes.map(normalize_whitespace)
        print(f"[droiddetect] {s}: {len(sub):,} rows"
              f"{' (normalised)' if args.normalize else ''}"
              f"{' [test shard]' if args.test_shard_only else ''}")
        p = score(model, tok, codes, device, args.max_len, args.batch_size, s)
        proba_by_slice[s] = (sub["label"].to_numpy(), p)
        # The model size belongs in the filename. Without it, evaluating Large
        # overwrites the arrays and report belonging to Base, and the two are
        # indistinguishable afterwards.
        size = "" if args.model == "base" else f"_{args.model}"
        tag = size + ("_fmt" if args.normalize else "") + ("_test" if args.test_shard_only else "")
        np.save(art / f"proba_droiddetect{tag}_{s}.npy", p)
        sub[["label", "language", "slice"]].to_parquet(
            art / f"rows_droiddetect{tag}_{s}.parquet")

    size = "" if args.model == "base" else f"_{args.model}"
    name = "droiddetect_baseline" + size + (("_fmt" if args.normalize else "")
                                            + ("_test" if args.test_shard_only else ""))
    res = M.evaluate_all(df, proba_by_slice, name, C.reports(cfg))
    M.print_table(res)
    with open(C.reports(cfg) / f"droiddetect_meta{size}.json", "w", encoding="utf-8") as f:
        json.dump({"repo": REPO, "encoder": ENCODER, "model": args.model,
                   "pooling": pooling, "max_len": args.max_len,
                   "max_eval": args.max_eval}, f, indent=2)


if __name__ == "__main__":
    main()
