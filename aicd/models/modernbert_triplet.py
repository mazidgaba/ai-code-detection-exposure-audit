"""Branch A: ModernBERT + class-weighted CE + batch-hard triplet loss.

This reproduces the published DroidDetect recipe:
    emb    = encoder(...).last_hidden_state.mean(dim=1)
    proj   = Linear(hidden, 128)
    logits = Linear(128, 4)
    loss   = CE(weight=class_weights) + 0.1 * BatchHardSoftMarginTripletLoss

ModernBERT is chosen for its 8k context -- most real files are never truncated,
so the head/tail truncation workarounds half the SemEval field needed do not
apply here. Triplet loss matters because hybrid and adversarial code sit close
to human code in embedding space; metric learning pushes the classes apart.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aicd import config as C
from aicd.data.splits import SLICES
from aicd.eval import metrics as M


# --------------------------------------------------------------------------- data
class CodeDS(Dataset):
    def __init__(self, codes, labels, tok, max_len):
        self.codes = list(codes)
        self.labels = list(labels)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        enc = self.tok(
            self.codes[i], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


# --------------------------------------------------------------------------- loss
def batch_hard_triplet_soft_margin(emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Batch-hard mining with a soft margin: for each anchor take the furthest
    positive and the nearest negative, then softplus(d_pos - d_neg).

    Computed in fp32 even under autocast. Pairwise distances between unit
    vectors are small, and in fp16 the near-zero d_pos - d_neg differences that
    batch-hard mining depends on collapse to the same value, which silently
    turns the triplet term into a constant.
    """
    with torch.autocast(device_type=emb.device.type, enabled=False):
        return _triplet_fp32(emb.float(), labels)


def _triplet_fp32(emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    emb = F.normalize(emb, p=2, dim=1)
    dist = torch.cdist(emb, emb, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)

    pos_mask = same & ~eye
    neg_mask = ~same
    # Anchors with no positive or no negative in the batch contribute nothing.
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return emb.sum() * 0.0

    d_pos = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    d_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    return F.softplus(d_pos[valid] - d_neg[valid]).mean()


# -------------------------------------------------------------------------- model
class TLModel(nn.Module):
    def __init__(self, encoder, hidden: int, projection_dim: int = 128,
                 num_classes: int = 4, class_weights=None, triplet_weight: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Linear(hidden, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.triplet_weight = triplet_weight
        self.register_buffer(
            "class_weights",
            torch.ones(num_classes) if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # masked mean pool
        z = self.projection(emb)
        logits = self.classifier(z)
        if labels is None:
            return {"logits": logits, "embeddings": z}
        loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        loss = loss + self.triplet_weight * batch_hard_triplet_soft_margin(z, labels)
        return {"loss": loss, "logits": logits, "embeddings": z}


# ----------------------------------------------------------------------- training
def device_of(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build(cfg, class_weights=None):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.modernbert.base_model)
    enc = AutoModel.from_pretrained(cfg.modernbert.base_model)

    # Gradient checkpointing recomputes activations during the backward pass
    # instead of holding them, which is what makes ModernBERT-large trainable
    # on a 16 GB card: at 395M parameters it exhausted a T4 with 57 MiB to
    # spare. It is the right lever for an ablation because it changes nothing
    # about the result. The gradients and the updates are identical; only the
    # memory-for-time trade differs, at roughly a third more wall clock.
    # Cutting the batch size or the sequence length would have been cheaper and
    # would have changed the experiment.
    if getattr(cfg.modernbert, "gradient_checkpointing", False):
        if hasattr(enc, "gradient_checkpointing_enable"):
            enc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print(f"[build] gradient checkpointing on for "
                  f"{cfg.modernbert.base_model}")
        else:
            raise SystemExit(
                f"{cfg.modernbert.base_model} does not support gradient "
                "checkpointing, and this config asks for it. Without it the "
                "run will exhaust GPU memory rather than fail here.")

    model = TLModel(
        enc, enc.config.hidden_size,
        projection_dim=cfg.modernbert.projection_dim,
        num_classes=cfg.modernbert.num_classes,
        class_weights=class_weights,
        triplet_weight=cfg.modernbert.triplet_weight,
    )
    return tok, model


def augment(tr, cfg, tag: str):
    """E8: rewrite a share of the training rows, semantics preserved.

    The transformation battery showed that renaming identifiers costs the
    detector more than half its accuracy while changing nothing a compiler
    sees. The obvious question is whether training through those rewrites fixes
    it, and the answer is worth having either way. If it helps, the paper stops
    being purely diagnostic and gains a mitigation. If it does not, that is a
    stronger claim than the paper currently makes: augmenting away the surface
    cue does not restore transfer, so the dependence is deeper than formatting.

    The rewrite is applied once, before training, to a fixed sample. That is
    what the plan specifies and it keeps the run reproducible: the same seed
    gives the same augmented corpus, so a resumed session continues on exactly
    the data it started with. Rewriting afresh each epoch would be a different
    experiment and would break resume.

    Labels are untouched, deliberately. A renamed machine-generated file is
    still machine-generated, and that is the whole point.
    """
    frac = float(getattr(cfg.modernbert, "augment_fraction", 0.0) or 0.0)
    if frac <= 0:
        return tr

    from aicd.models.transforms import TRANSFORMS
    names = list(getattr(cfg.modernbert, "augment_transforms", None)
                 or ["rename_identifiers", "whitespace"])
    unknown = [n for n in names if n not in TRANSFORMS]
    if unknown:
        raise SystemExit(f"unknown augmentation {unknown}; "
                         f"choose from {sorted(TRANSFORMS)}")

    rng = np.random.default_rng(cfg.project.seed)
    tr = tr.copy()
    n = int(round(frac * len(tr)))
    picked = rng.choice(tr.index.to_numpy(), size=n, replace=False)
    langs = tr["language"] if "language" in tr.columns else None

    changed = 0
    codes = tr["code"].to_dict()
    for i, idx in enumerate(picked):
        fn = TRANSFORMS[names[i % len(names)]]
        lang = str(langs[idx]) if langs is not None else ""
        before = codes[idx]
        after = fn(before, lang)
        if after != before:
            codes[idx] = after
            changed += 1
    tr["code"] = tr.index.map(codes)

    print(f"[branch_a:{tag}] augmentation: {names}, target {frac:.0%} of "
          f"{len(tr):,} rows")
    print(f"  {n:,} selected, {changed:,} actually altered "
          f"({changed / max(n, 1):.1%} of those selected)")
    if changed == 0:
        raise SystemExit(
            "augmentation was requested but altered nothing, so this run would "
            "be identical to the unaugmented one while claiming otherwise")
    return tr


def train(cfg, df, drop_idx=None, tag="base", force_cpu=False, max_train=None,
          resume=False, amp=None, max_hours=None):
    """Train Branch A.

    resume: pick up from the last per-epoch checkpoint instead of restarting.
            A hosted notebook can lose its session at any point, and the final
            save only happens after the last epoch, so without this an
            interruption at epoch 2 of 3 costs the whole run.
    amp:    mixed precision. None means "on when there is a CUDA device".
    max_hours:
            stop cleanly after the last epoch that fits in the budget, rather
            than being killed partway through the next one. A hosted notebook
            that hits its wall clock loses the whole partial epoch: on the
            first matched-scale run that was 2.1 hours of GPU for nothing.
            Checkpoints are per-epoch, so stopping early costs nothing and the
            next session resumes from the same place.
    """
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    dev = device_of(force_cpu)
    use_amp = (dev.type == "cuda") if amp is None else bool(amp)
    tr = df[df["split"] == "train"]
    if drop_idx is not None:
        tr = tr.drop(index=drop_idx, errors="ignore")
    if max_train:
        tr = tr.sample(n=min(max_train, len(tr)), random_state=cfg.project.seed)
    tr = augment(tr, cfg, tag)
    print(f"[branch_a:{tag}] device={dev} train_rows={len(tr):,}")

    counts = np.bincount(tr["label"], minlength=4).astype(float)
    cw = np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0)
    print(f"  class counts={counts.astype(int).tolist()} weights={np.round(cw, 3).tolist()}")

    tok, model = build(cfg, class_weights=cw)
    model.to(dev)

    ds = CodeDS(tr["code"], tr["label"], tok, cfg.modernbert.max_length)
    dl = DataLoader(ds, batch_size=cfg.modernbert.batch_size, shuffle=True, drop_last=True)
    steps = max(len(dl) * cfg.modernbert.epochs, 1)
    opt = AdamW(model.parameters(), lr=cfg.modernbert.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * cfg.modernbert.warmup_ratio), steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    art = C.artifacts(cfg)
    ckpt_path = art / f"branch_a_{tag}_ckpt.pt"

    start_ep = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        print(f"[branch_a:{tag}] resumed from epoch {ck['epoch']}, "
              f"continuing at {start_ep}/{cfg.modernbert.epochs}")
    elif resume:
        print(f"[branch_a:{tag}] --resume given but no checkpoint at "
              f"{ckpt_path.name}, starting fresh")

    if start_ep >= cfg.modernbert.epochs:
        print(f"[branch_a:{tag}] amp={use_amp} training already complete "
              f"({cfg.modernbert.epochs} epochs), going straight to evaluation")
    else:
        print(f"[branch_a:{tag}] amp={use_amp} "
              f"running epochs {start_ep}..{cfg.modernbert.epochs - 1}")

    model.train()
    t_start = time.time()
    for ep in range(start_ep, cfg.modernbert.epochs):
        tot, n = 0.0, 0
        for i, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                enabled=use_amp):
                out = model(**batch)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += out["loss"].item()
            n += 1
            if i % 50 == 0:
                print(f"  ep{ep} step {i}/{len(dl)} loss={tot / max(n, 1):.4f}", flush=True)
        print(f"[branch_a:{tag}] epoch {ep} mean loss {tot / max(n, 1):.4f}", flush=True)

        # Checkpoint every epoch, not just at the end. Written to a temporary
        # file and renamed so a kill mid-write cannot leave a corrupt file
        # where a valid one used to be.
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": ep,
                    "base_model": cfg.modernbert.base_model,
                    "projection_dim": cfg.modernbert.projection_dim}, tmp)
        tmp.replace(ckpt_path)
        print(f"  checkpoint -> {ckpt_path.name} (epoch {ep})", flush=True)

        if max_hours is not None and ep + 1 < cfg.modernbert.epochs:
            used = (time.time() - t_start) / 3600.0
            per_epoch = used / (ep - start_ep + 1)
            if used + per_epoch > max_hours:
                print(f"[branch_a:{tag}] stopping after epoch {ep}: "
                      f"{used:.2f} h used, next epoch needs about "
                      f"{per_epoch:.2f} h, budget is {max_hours:.2f} h.")
                print(f"[branch_a:{tag}] re-run with --resume to continue from "
                      f"epoch {ep + 1}. Nothing is lost.")
                return tok, model, dev

    torch.save({"state_dict": model.state_dict(),
                "base_model": cfg.modernbert.base_model,
                "projection_dim": cfg.modernbert.projection_dim},
               art / f"branch_a_{tag}.pt")
    return tok, model, dev


@torch.no_grad()
def predict(model, tok, codes, cfg, dev, batch_size=None) -> np.ndarray:
    was_training = model.training
    model.eval()
    p = _forward_probs(model, tok, codes, cfg, dev, batch_size)
    if was_training:
        model.train()
    return p


def _forward_probs(model, tok, codes, cfg, dev, batch_size=None,
                   report_every: int = 5000) -> np.ndarray:
    """Score a sequence of code strings.

    Runs under autocast on CUDA. Training was mixed precision, and leaving
    inference in fp32 made evaluation roughly four times slower than the
    training that produced the model: a T4 has no fp32 tensor cores, so the
    whole 78k-row evaluation pass ran at a fraction of the achievable rate.
    Softmax is taken in fp32 regardless, since fp16 probabilities near 0 or 1
    lose the resolution the calibration and threshold analyses depend on.
    """
    bs = batch_size or max(cfg.modernbert.batch_size, 8)
    codes = list(codes)
    out = []
    use_amp = dev.type == "cuda"
    t0 = time.time()
    for i in range(0, len(codes), bs):
        enc = tok(codes[i: i + bs], truncation=True, max_length=cfg.modernbert.max_length,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"])["logits"]
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        # Evaluation was previously silent for its whole duration, which makes a
        # slow pass indistinguishable from a hang.
        if report_every and i and i % report_every < bs:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"    scored {i:,}/{len(codes):,}  {rate:.0f} rows/s", flush=True)
    return np.vstack(out) if out else np.zeros((0, 4))


@torch.no_grad()
def mc_dropout_confidence(model, tok, codes, cfg, dev, passes=30) -> np.ndarray:
    """Mean max-probability across stochastic forward passes.

    Low confidence on a HUMAN-labelled sample is the signal Droid used to find
    likely copilot-written code sitting in the human class. Dropping the worst
    7% and retraining bought them a real gain.
    """
    model.train()  # keep dropout active
    acc = [_forward_probs(model, tok, codes, cfg, dev) for _ in range(passes)]
    model.eval()
    return np.mean(np.stack(acc), axis=0).max(axis=1)


def evaluate(cfg, df, model, tok, dev, name: str, max_eval: int | None = None,
             suffix: str = "") -> dict:
    """max_eval caps rows scored per slice. Only for CPU smoke runs -- a capped
    evaluation is not a result, and the saved .npy will not line up with the
    stacker, which expects full-slice probabilities.

    suffix distinguishes the probability arrays of one run from another. The
    default is empty so the canonical run keeps writing proba_a_<slice>.npy,
    which is the filename every downstream analysis module already reads.
    """
    proba_by_slice = {}
    for s in ["val"] + SLICES:
        m = df["slice"] == s
        if not m.any():
            continue
        sub = df.loc[m]
        capped = max_eval is not None and len(sub) > max_eval
        if capped:
            sub = sub.sample(n=max_eval, random_state=cfg.project.seed)
        p = predict(model, tok, sub["code"], cfg, dev)
        proba_by_slice[s] = (sub["label"].to_numpy(), p)
        if not capped:
            np.save(C.artifacts(cfg) / f"proba_a{suffix}_{s}.npy", p)
    res = M.evaluate_all(df, proba_by_slice, name, C.reports(cfg))
    M.print_table(res)
    return res


def seed_everything(seed: int) -> None:
    """Seed every generator the training path draws from.

    Until this existed cfg.project.seed reached pandas sampling and nothing
    else, so weight initialisation, dropout and batch order were all left to
    torch's default entropy. Runs were not reproducible, and a seed sweep would
    have varied for reasons unrelated to the seed being swept.

    cuDNN autotuning stays enabled. Making convolution algorithm choice
    deterministic costs throughput, and the point here is that two runs
    differing only in seed differ only by the seed, not bitwise replay.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] all generators seeded with {seed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap training rows (use on CPU to verify the loop runs)")
    ap.add_argument("--resample", action="store_true",
                    help="run the MC-dropout noisy-label pass and retrain")
    ap.add_argument("--no-triplet", action="store_true", help="ablation: CE only")
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap rows scored per slice (CPU verification only)")
    # Derived from the arm registry rather than restated, so adding an arm in
    # exposure_arms.py cannot leave the trainer refusing to train it. The
    # language-clean arm was added and this list was not, which would have
    # failed the run on its first command.
    from aicd.data.exposure_arms import ARM_COLUMN
    ap.add_argument("--arm", choices=sorted(ARM_COLUMN), default=None,
                    help="train one of the exposure arms from "
                         "splits_arms.parquet instead of the ordinary split")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last per-epoch checkpoint")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable fp16 mixed precision (on by default on CUDA)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after the last epoch that fits in this "
                         "many hours of training, instead of being killed "
                         "partway through the next one. Re-run with --resume "
                         "to continue.")
    ap.add_argument("--tag", default=None,
                    help="name this run's artefacts (checkpoint, weights, "
                         "report, probability arrays). Use a distinct tag per "
                         "seed or per training-set variant, or the runs "
                         "overwrite each other and --resume continues the "
                         "wrong model. Default: 'base'.")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="seed weight initialisation, dropout and batch order "
                         "only, leaving cfg.project.seed to drive the corpus "
                         "and arm construction. Replicating a twin arm across "
                         "seeds requires exactly this: changing "
                         "cfg.project.seed instead would rebuild the arms, so "
                         "the seeds would differ in training data as well as "
                         "initialisation and could not be compared.")
    args = ap.parse_args()

    cfg = C.load(args.config)
    seed_everything(args.train_seed if args.train_seed is not None
                    else cfg.project.seed)
    if args.no_triplet:
        cfg["modernbert"]["triplet_weight"] = 0.0
    d = C.ROOT / cfg.data.cache_dir
    if args.arm:
        from aicd.data.exposure_arms import as_training_frame
        df = as_training_frame(pd.read_parquet(d / "splits_arms.parquet"), args.arm)
        n = int((df["split"] == "train").sum())
        print(f"[arm:{args.arm}] {n:,} training rows")
    else:
        df = pd.read_parquet(d / "splits.parquet")

    # Every artefact this run writes is keyed by tag: the checkpoint, the
    # weights, the report and the per-slice probability arrays. Two runs
    # sharing a tag overwrite each other, and with --resume the second silently
    # continues training the first one's weights, which would quietly turn a
    # seed sweep into one model trained twice as long. Give repeated runs
    # distinct tags.
    tag = args.tag or ("notriplet" if args.no_triplet else "base")
    amp = False if args.no_amp else None
    tok, model, dev = train(cfg, df, tag=tag, force_cpu=args.cpu,
                            max_train=args.max_train, resume=args.resume,
                            amp=amp, max_hours=args.max_hours)
    # Keep the canonical run writing proba_a_<slice>.npy; give any other run a
    # suffix so it cannot overwrite the arrays the analysis modules read.
    suffix = "" if tag in ("base", "notriplet") else f"_{tag}"
    res = evaluate(cfg, df, model, tok, dev, f"branch_a_{tag}", args.max_eval,
                   suffix=suffix)

    if args.resample:
        tr = df[df["split"] == "train"]
        human = tr[tr["label"] == 0]
        print(f"\n[resample] MC-dropout over {len(human):,} human-labelled rows "
              f"({cfg.modernbert.mc_dropout_passes} passes)...")
        conf = mc_dropout_confidence(model, tok, human["code"], cfg, dev,
                                     passes=cfg.modernbert.mc_dropout_passes)
        k = int(len(human) * cfg.modernbert.resample_drop_fraction)
        drop = human.index[np.argsort(conf)[:k]]
        print(f"[resample] dropping the {k:,} least-confident human rows "
              f"(likely copilot-written, mislabelled)")
        tok2, model2, dev2 = train(cfg, df, drop_idx=drop, tag="resampled",
                                   force_cpu=args.cpu, max_train=args.max_train,
                                   resume=args.resume, amp=amp)
        res2 = evaluate(cfg, df, model2, tok2, dev2, "branch_a_resampled", args.max_eval)
        b = res["slices"].get("s1_in_distribution", {}).get("macro_f1")
        a = res2["slices"].get("s1_in_distribution", {}).get("macro_f1")
        if b is not None and a is not None:
            print(f"\n[resample] in-distribution macro-F1 {b:.4f} -> {a:.4f} ({a - b:+.4f})")
        with open(C.reports(cfg) / "resample.json", "w", encoding="utf-8") as f:
            json.dump({"dropped": int(k)}, f)


if __name__ == "__main__":
    main()
