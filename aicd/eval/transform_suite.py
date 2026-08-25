"""Score a detector under several semantics-preserving rewrites.

The manuscript's evasion result uses one transformation. This runs the family
in aicd.models.transforms, so the robustness claim rests on a set rather than a
single command, and reports for each one both the accuracy lost and the share
of inputs it actually altered. The second number is the guard: a rewrite that
touches 5% of files and costs 0.01 macro-F1 is not evidence of robustness, and
without the applied fraction it would look like it was.

Branch B is scored from cached features and is fast. DroidDetect must be run
forward for every rewrite, which on CPU is slow, so --n keeps the sample small
enough to finish and the sample size is recorded with the result.

    python -m aicd.eval.transform_suite --model b
    python -m aicd.eval.transform_suite --model droiddetect --n 300
"""
from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from aicd import config as C
from aicd.models.transforms import TRANSFORMS

HUMAN = 0


def binary_auc(y, proba):
    yb = (y != HUMAN).astype(int)
    if yb.min() == yb.max():
        return float("nan")
    return float(roc_auc_score(yb, 1.0 - proba[:, HUMAN]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu.yaml")
    ap.add_argument("--model", default="droiddetect", choices=["droiddetect", "b"])
    ap.add_argument("--slice", default="s1_in_distribution")
    ap.add_argument("--parse-check", action="store_true",
                    help="also report the conditional effect on rows the "
                         "rewrite left syntactically valid")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = C.load(args.config)
    # Choose the rows from the small columns first, then read the code for
    # only those rows. Loading every code column of the whole corpus to keep a
    # few hundred rows needs several GB, and on a machine doing anything else
    # the process is killed before the model finishes loading.
    path = C.ROOT / cfg.data.cache_dir / "splits.parquet"
    keys = pd.read_parquet(path, columns=["slice", "label", "language"])
    pool = keys[keys["slice"] == args.slice]
    if pool.empty:
        raise SystemExit(f"no rows in slice {args.slice!r}")

    # Stratify so a small sample keeps all four classes.
    idx = []
    for _, g in pool.groupby("label"):
        k = max(1, int(round(args.n * len(g) / len(pool))))
        idx.extend(g.sample(n=min(k, len(g)), random_state=0).index)
    del keys, pool

    full = pd.read_parquet(path, columns=["slice", "label", "language", "code"])
    sub = full.loc[idx].copy()
    del full
    y = sub["label"].to_numpy()
    langs = sub["language"].astype(str).tolist()
    print(f"[suite] {len(sub)} rows from {args.slice}, "
          f"labels={np.bincount(y, minlength=4).tolist()}")

    parsers = None
    if args.parse_check:
        from aicd.eval.semantics_check import parser_for, parses
        parsers = {l: parser_for(l) for l in set(langs)}
        have = sorted(k for k, v in parsers.items() if v is not None)
        print(f"[suite] parse check on {len(have)} languages: {have}")

    if args.model == "droiddetect":
        import torch
        from aicd.models.droiddetect_baseline import load, score
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok, model = load(cfg, "mean", device)

        def predict(codes):
            return score(model, tok, pd.Series(codes), device, args.max_len,
                         tag="suite")
    else:
        raise SystemExit(
            "Branch B scoring needs the feature pipeline rebuilt per rewrite; "
            "run with --model droiddetect, or extend this to call the Branch B "
            "featuriser on transformed text.")

    rows = []
    base_codes = sub["code"].tolist()
    base_p = predict(base_codes)
    base_f1 = float(f1_score(y, base_p.argmax(1), average="macro", zero_division=0))
    base_auc = binary_auc(y, base_p)
    print(f"\n  {'transform':20s} {'applied':>8s} {'macro-F1':>9s} "
          f"{'delta':>8s} {'conditional':>11s} {'AUC':>8s} {'dAUC':>8s}")
    print(f"  {'(none)':20s} {'-':>8s} {base_f1:9.4f} {'-':>8s} "
          f"{'-':>11s} {base_auc:8.4f} {'-':>8s}")

    for name, fn in TRANSFORMS.items():
        codes = [fn(c, l) for c, l in zip(base_codes, langs)]
        changed = np.array([a != b for a, b in zip(codes, base_codes)])
        applied = float(changed.mean())
        p = predict(codes)
        f1 = float(f1_score(y, p.argmax(1), average="macro", zero_division=0))
        auc = binary_auc(y, p)

        # The conditional effect: what the rewrite costs on the rows it
        # actually touched. Comparing whole-set scores ranks rewrites by how
        # often they fire as much as by how much they matter, which is what
        # made the published ranking unsafe: renaming altered 100% of files
        # and stripping comments 81%, so the two deltas were never comparable.
        #
        # This is NOT the aggregate delta divided by the application rate.
        # Macro-F1 is an average of per-class ratios, not a sum over rows, so
        # it does not decompose that way and the quotient can be wrong in
        # either direction. The only sound comparison scores the baseline and
        # the rewrite on the same altered rows.
        if changed.any() and len(np.unique(y[changed])) > 1:
            f1_before = float(f1_score(y[changed], base_p[changed].argmax(1),
                                       average="macro", zero_division=0))
            f1_after = float(f1_score(y[changed], p[changed].argmax(1),
                                      average="macro", zero_division=0))
            cond = f1_after - f1_before
        else:
            f1_before = f1_after = cond = float("nan")

        rec = {"transform": name, "applied_fraction": applied,
               "macro_f1": f1, "delta_macro_f1": f1 - base_f1,
               "n_altered": int(changed.sum()),
               "macro_f1_altered_before": f1_before,
               "macro_f1_altered_after": f1_after,
               "delta_macro_f1_conditional": cond,
               "binary_auc": auc, "delta_auc": auc - base_auc}

        # The conditional effect restricted to rows the rewrite left
        # syntactically valid.
        #
        # Renaming breaks parsing on roughly one file in eight, because it
        # substitutes identifiers blindly and so rewrites the contents of
        # include directives and import paths. A detector reacting to broken
        # code is not evidence that it reads naming habit, so the effect has to
        # be measured where the rewrite did what it claims. This is the
        # difference between a mechanism result and a corruption rate.
        if parsers is not None and changed.any():
            ok = np.array([
                (parsers.get(l) is not None
                 and parses(parsers[l], b) and parses(parsers[l], c))
                for b, c, l in zip(base_codes, codes, langs)])
            valid = changed & ok
            rec["n_altered_parsing"] = int(valid.sum())
            rec["break_rate"] = float(1 - ok[changed].mean())
            if valid.sum() > 1 and len(np.unique(y[valid])) > 1:
                b4 = float(f1_score(y[valid], base_p[valid].argmax(1),
                                    average="macro", zero_division=0))
                af = float(f1_score(y[valid], p[valid].argmax(1),
                                    average="macro", zero_division=0))
                rec["delta_macro_f1_conditional_parsing"] = af - b4
            else:
                rec["delta_macro_f1_conditional_parsing"] = float("nan")

        rows.append(rec)
        print(f"  {name:20s} {applied:8.1%} {f1:9.4f} {f1-base_f1:+8.4f} "
              f"{cond:+11.4f} {auc:8.4f} {auc-base_auc:+8.4f}")

    out = {"model": args.model, "slice": args.slice, "n": int(len(sub)),
           "max_len": args.max_len, "baseline": {"macro_f1": base_f1,
                                                 "binary_auc": base_auc},
           "transforms": rows}
    dest = C.reports(cfg) / f"transform_suite_{args.model}.json"
    io.open(dest, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
