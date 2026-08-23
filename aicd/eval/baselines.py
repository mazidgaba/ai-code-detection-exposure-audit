"""Trivial-classifier baselines, computed per condition rather than assumed.

The manuscript compared Branch A's 0.2378 on S5 against 1/K = 0.25, described
as what random guessing achieves on four balanced classes. The conditions are
not balanced: S5 holds 1,936 human rows against 155 adversarial ones. The
identity needed checking, and checking it turns up something more interesting
than a simple correction.

Three trivial classifiers, on a condition where class c holds a fraction p_c
of N rows:

  uniform      predicts each class with probability 1/K. Recall is 1/K and
               precision is p_c, so F1_c = 2 p_c (1/K) / (p_c + 1/K). This is
               below 1/K whenever the classes are uneven, and it is the figure
               the manuscript should have quoted for "random guessing".

  stratified   predicts class c with probability p_c. Precision and recall
               both equal p_c, so F1_c = p_c and macro-F1 = mean(p_c) = 1/K
               exactly, for any class distribution whatsoever. The 0.25 in the
               manuscript is therefore a correct number attached to the wrong
               classifier: it is the prior-matched baseline, not the uniform
               one, and it does not depend on balance at all.

  majority     predicts the largest class always. Every other class scores 0.

Reporting all three is the fix. Branch A on S5 sits above uniform and below
prior-matched, which is a sharper statement than either alone.

    python -m aicd.eval.baselines
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from aicd import config as C

ART = "kaggle"
ORDER = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound"]
K = 4


def counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=K).astype(float)


def uniform_macro_f1(n: np.ndarray) -> float:
    p = n / n.sum()
    return float(np.mean(2 * p * (1 / K) / (p + 1 / K)))


def stratified_macro_f1(n: np.ndarray) -> float:
    return float(np.mean(n / n.sum()))


def majority_macro_f1(n: np.ndarray) -> float:
    pm = n.max() / n.sum()
    return float((2 * pm / (1 + pm)) / K)


def simulate(y: np.ndarray, mode: str, trials: int = 200, seed: int = 0) -> float:
    """Monte-Carlo check on the closed forms, so a slip in the algebra shows."""
    rng = np.random.default_rng(seed)
    p = counts(y) / len(y)
    vals = []
    for _ in range(trials):
        if mode == "uniform":
            yh = rng.integers(0, K, size=y.size)
        elif mode == "stratified":
            yh = rng.choice(K, size=y.size, p=p)
        else:
            yh = np.full(y.size, int(np.argmax(p)))
        vals.append(f1_score(y, yh, average="macro", zero_division=0))
    return float(np.mean(vals))


def main() -> None:
    base = C.artifacts(C.load("cpu.yaml")) / ART
    lab = pd.read_parquet(base / "labels.parquet")

    out, rows = {}, []
    for s in ORDER:
        y = lab.loc[lab["slice"] == s, "label"].to_numpy()
        if y.size == 0:
            continue
        n = counts(y)
        f = base / f"proba_a_{s}.npy"
        got = None
        if f.exists():
            p = np.load(f)
            if len(p) == len(y):
                got = float(f1_score(y, p.argmax(1), average="macro",
                                     zero_division=0))
        rec = {
            "n": int(y.size),
            "class_counts": [int(v) for v in n],
            "uniform": uniform_macro_f1(n),
            "uniform_simulated": simulate(y, "uniform"),
            "stratified": stratified_macro_f1(n),
            "stratified_simulated": simulate(y, "stratified"),
            "majority": majority_macro_f1(n),
            "branch_a": got,
        }
        out[s] = rec
        rows.append((s, rec))

    hdr = (f"{'condition':22s} {'N':>7s} {'uniform':>9s} {'(sim)':>8s} "
           f"{'strat':>8s} {'major':>8s} {'Branch A':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for s, r in rows:
        ba = f"{r['branch_a']:9.4f}" if r["branch_a"] is not None else f"{'n/a':>9s}"
        print(f"{s:22s} {r['n']:7d} {r['uniform']:9.4f} "
              f"{r['uniform_simulated']:8.4f} {r['stratified']:8.4f} "
              f"{r['majority']:8.4f} {ba}")

    # The stratified identity is the claim most worth guarding: if it ever
    # stops holding, the algebra above is wrong.
    off = max(abs(r["stratified"] - 0.25) for _, r in rows)
    print(f"\nmax |stratified - 1/K| across conditions: {off:.2e}")
    assert off < 1e-9, "stratified macro-F1 must equal 1/K exactly"

    dest = C.ROOT / "eval" / "reports" / "baselines.json"

    os.makedirs(dest.parent, exist_ok=True)
    io.open(dest, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
