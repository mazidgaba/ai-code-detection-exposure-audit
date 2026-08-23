"""How much of each evaluation slice is a near-copy of something in training?

The corpus is split by `problem_id`, and the paper leans on that: it cites
Oedingen et al.'s finding that a random row split inflates every metric by
roughly four points, and says it avoids the problem by grouping solutions to
the same problem together.

The grouping does not do that. `problem_id` is built in normalize.py as

    source + ":" + sha1(first 512 non-whitespace characters of the code)

which is a hash of the code, not an identifier for the task it solves. Two
solutions to one problem differ in their text, so they hash differently and land
in different groups. The consequence is measurable in the built corpus: 99.93%
of rows are alone in their group, and not one group contains more than one label
class, which is precisely the property the docstring claims. Problem-wise
splitting, as implemented, is a random row split.

That does not by itself mean the numbers are wrong. It means the safeguard is
absent, and whether that matters is an empirical question about how much
near-duplication DroidCollection actually contains. This module answers it, by
asking of every evaluation row: is there a training row that is nearly the same
code?

Token-level MinHash with LSH, banded at the loosest threshold and then bucketed
exactly from the signatures, so one index answers every threshold at once.

    python -m aicd.eval.duplication_audit
    python -m aicd.eval.duplication_audit --perms 256 --shingle 7
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "aicd" / "artifacts" / "data" / "splits.parquet"
SLICES = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
          "s4_unseen_domain", "s5_compound", "val"]

# Identifiers, numbers, strings collapsed, and operators kept as single tokens.
# Comments are deliberately NOT stripped: two files that differ only in their
# comments are still near-duplicates for a detector that reads surface text,
# and the paper's own transformation results show comments carry signal.
TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|[^\s\w]")


def shingles(code: str, k: int) -> set[bytes]:
    toks = TOKEN.findall(code)
    if len(toks) < k:
        return {" ".join(toks).encode("utf-8", "ignore")} if toks else set()
    return {" ".join(toks[i:i + k]).encode("utf-8", "ignore")
            for i in range(len(toks) - k + 1)}


def signature(code: str, k: int, perms: int) -> MinHash:
    """One signature. update_batch hashes the whole shingle set in C rather
    than looping in Python, which is the difference between this audit taking
    minutes and taking hours."""
    m = MinHash(num_perm=perms)
    sh = shingles(code, k)
    if sh:
        m.update_batch(list(sh))
    return m


def self_test(df: pd.DataFrame, k: int, perms: int, n: int = 400,
              pool: int = 4000, seed: int = 7) -> dict:
    """Does the detector find duplicates it is handed deliberately?

    The audit's result is a near-zero, and a near-zero is exactly what a broken
    index also produces: every query returning no candidate is indistinguishable
    from every query having no near-duplicate. So the finding is worth nothing
    without this control, which indexes a sample of training rows and then asks
    three questions with known answers. Identical text must come back at 1.0.
    Text with a variable renamed and a comment bolted on is the same program and
    must still come back high. Unrelated code must not come back at all.
    """
    rng = np.random.default_rng(seed)
    tr = df[df["slice"] == "train"]["code"].astype(str).tolist()
    sample = [tr[i] for i in rng.choice(len(tr), min(pool, len(tr)), replace=False)]

    lsh = MinHashLSH(threshold=0.5, num_perm=perms)
    sigs = []
    with lsh.insertion_session() as session:
        for i, c in enumerate(sample):
            m = signature(c, k, perms)
            sigs.append(m)
            session.insert(str(i), m)

    def best(code: str) -> float:
        m = signature(code, k, perms)
        return max((m.jaccard(sigs[int(c)]) for c in lsh.query(m)), default=0.0)

    def perturb(c: str) -> str:
        c = re.sub(r"\bresult\b", "outcome_value", c)
        c = re.sub(r"\bi\b", "idx", c)
        return "# adapted\n" + c

    unrelated = df[df["slice"] == "s5_compound"]["code"].astype(str).tolist()[:n]
    cases = {
        "identical": [best(c) for c in sample[:n]],
        "renamed_and_commented": [best(perturb(c)) for c in sample[:n]],
        "unrelated": [best(c) for c in unrelated],
    }
    out = {}
    print(f"\n{'control':24s} {'n':>5s} {'mean':>8s} {'median':>8s} "
          f"{'>=0.85':>8s} {'>=0.5':>8s}")
    print("-" * 66)
    for name, v in cases.items():
        a = np.array(v) if v else np.zeros(1)
        out[name] = {"n": len(v), "mean": float(a.mean()),
                     "median": float(np.median(a)),
                     "share_above_085": float((a >= 0.85).mean()),
                     "share_above_05": float((a >= 0.5).mean())}
        print(f"{name:24s} {len(v):>5} {a.mean():>8.4f} {np.median(a):>8.4f} "
              f"{(a >= 0.85).mean():>7.1%} {(a >= 0.5).mean():>7.1%}")
    ok = (out["identical"]["share_above_085"] > 0.99
          and out["renamed_and_commented"]["share_above_05"] > 0.90
          and out["unrelated"]["share_above_05"] < 0.01)
    out["verdict"] = "ok" if ok else "THE DETECTOR IS NOT WORKING"
    print(f"\n  control verdict: {out['verdict']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="run the positive control and stop")
    ap.add_argument("--perms", type=int, default=128)
    ap.add_argument("--shingle", type=int, default=5,
                    help="token n-gram size; 5 is the usual choice for code")
    ap.add_argument("--thresholds", default="0.5,0.7,0.85")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ths = sorted(float(x) for x in args.thresholds.split(","))
    df = pd.read_parquet(SPLITS, columns=["code", "slice", "label", "language"])

    if args.self_test:
        self_test(df, args.shingle, args.perms)
        return

    train = df[df["slice"] == "train"]
    print(f"training rows      : {len(train):,}")
    print(f"signatures         : {args.perms} permutations, "
          f"{args.shingle}-token shingles\n")

    # Index the training side at the loosest threshold; anything above a
    # stricter one is necessarily a candidate here too, so a single index is
    # enough and the exact bucketing happens afterwards from the signatures.
    lsh = MinHashLSH(threshold=min(ths), num_perm=args.perms)
    tr_sig: list[MinHash] = []
    with lsh.insertion_session() as session:
        for i, code in enumerate(train["code"].astype(str).tolist()):
            m = signature(code, args.shingle, args.perms)
            tr_sig.append(m)
            session.insert(str(i), m)
            if (i + 1) % 10000 == 0:
                print(f"  indexed {i + 1:,}", flush=True)
    print(f"  indexed {len(tr_sig):,}\n")

    report = {"perms": args.perms, "shingle": args.shingle,
              "thresholds": ths, "train_rows": int(len(train)), "slices": {}}

    hdr = f"{'slice':22s} {'rows':>8s}" + "".join(f"{'>=' + str(t):>12s}" for t in ths)
    print(hdr)
    print("-" * len(hdr))

    for sl in SLICES:
        g = df[df["slice"] == sl]
        if not len(g):
            continue
        hits = {t: 0 for t in ths}
        best_all = []
        for code in g["code"].astype(str).tolist():
            m = signature(code, args.shingle, args.perms)
            best = 0.0
            for cid in lsh.query(m):
                j = m.jaccard(tr_sig[int(cid)])
                if j > best:
                    best = j
            best_all.append(best)
            for t in ths:
                if best >= t:
                    hits[t] += 1
        n = len(g)
        report["slices"][sl] = {
            "rows": int(n),
            "above": {str(t): {"rows": int(hits[t]), "share": hits[t] / n}
                      for t in ths},
            "max_similarity_mean": float(np.mean(best_all)),
            "max_similarity_median": float(np.median(best_all)),
        }
        print(f"{sl:22s} {n:>8,}"
              + "".join(f"{hits[t]:>7,} {hits[t]/n:>4.1%}" for t in ths))

    report["control"] = self_test(df, args.shingle, args.perms)

    dest = Path(args.out) if args.out else (
        ROOT / "aicd" / "eval" / "reports" / "duplication_audit.json")
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}")

    s1 = report["slices"].get("s1_in_distribution")
    if s1:
        worst = s1["above"][str(min(ths))]["share"]
        print("\n" + "=" * 72)
        print("WHAT THIS MEANS FOR THE IN-DISTRIBUTION NUMBER")
        print("=" * 72)
        print(f"  {worst:.2%} of S1 rows have a training row at Jaccard "
              f">= {min(ths)}")
        print("  S1 is the condition the paper's 0.8977 is measured on, and it "
              "is\n  the reference every degradation is quoted against.")


if __name__ == "__main__":
    main()
