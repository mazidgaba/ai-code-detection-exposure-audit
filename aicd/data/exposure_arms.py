"""E1: build the three training arms that isolate what exposure costs.

The paper's central question is what happens when withholding is defined
relative to the detector's training data rather than the benchmark's partition.
It currently answers that by comparing our withheld-category model against
someone else's non-withheld model, trained at a different scale on a different
corpus, through a reconstruction. That is a confound with three terms in it,
not a measurement.

The measurement needs a twin: the same architecture, the same recipe, the same
seed, differing only in whether the withheld categories were present. Building
that twin runs into a fact about this corpus that is not obvious until you look:
the withheld-category rows are not sitting in reserve, they ARE the evaluation
conditions. splits.py routes every row carrying a withheld family, language or
source into S2 through S5. So training on those categories means taking rows out
of evaluation, and the design question is which rows and how many.

Three arms, all at the same total row count as the existing model:

  D1        the existing model. 196,854 rows, no withheld category present.
  D2        the same count, of which a measured fraction carry a withheld
            category. The default fraction is 0.356, which is what the exposure
            audit found in the training split the published detector names, so
            D2 is exposed to roughly what that detector was exposed to.
  D1_small  exactly D2's retained rows and nothing else.

The third arm is the one that makes the comparison clean, and it is why this is
worth an extra training run:

  D2 - D1_small   identical retained data, plus withheld rows. Pure exposure.
  D1 - D1_small   identical composition, more of it. Pure scale.
  D2 - D1         equal budget. What a benchmark-relative evaluation reports.

Without D1_small, D2 - D1 confounds exposure with dilution, because holding the
row count fixed while adding withheld rows necessarily removes retained ones.

Donor rows are drawn only from S2, S3 and S4. S1 and S5 are left exactly as
they are: S5 is the headline condition and the smallest at 4,168 rows, and
leaving both untouched means the existing model's saved probabilities stay
directly comparable there with no rescoring at all. D2 still sees every
withheld category, because all seven of them occur in S2 through S4.

    python -m aicd.data.exposure_arms --config kaggle.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from aicd import config as C

PROTECTED = ("s1_in_distribution", "s5_compound")
DONORS = ("s2_unseen_generator", "s3_unseen_language", "s4_unseen_domain")
# The language-clean arm draws from the generator-novel and source-novel
# conditions only, so no row in an untrained language enters its training set.
NOLANG_DONORS = ("s2_unseen_generator", "s4_unseen_domain")


def withheld_mask(df: pd.DataFrame, cfg) -> pd.Series:
    """Rows carrying at least one withheld category, on any of the three axes."""
    fams = set(cfg.splits.holdout_families)
    langs = set(cfg.splits.holdout_languages)
    srcs = set(cfg.splits.holdout_sources)
    return (df["model_family"].isin(fams)
            | df["language"].isin(langs)
            | df["source"].isin(srcs))


def draw_by_problem(df: pd.DataFrame, pool_idx: pd.Index, target: int,
                    rng: np.random.Generator, forbidden: set) -> pd.Index:
    """Take whole problem_id groups until `target` rows are reached.

    Drawing by group rather than by row keeps the guarantee the splitter was
    trying to give, whatever the key is actually worth: a group never straddles
    the training boundary. Groups that touch a protected condition are refused
    outright, so S1 and S5 cannot lose a row to this.
    """
    sub = df.loc[pool_idx]
    groups = sub.groupby("problem_id").indices
    names = np.array([g for g in groups if g not in forbidden], dtype=object)
    rng.shuffle(names)

    taken, total = [], 0
    for g in names:
        rows = sub.index[groups[g]]
        if total + len(rows) > target and total > 0:
            continue
        taken.append(rows)
        total += len(rows)
        if total >= target:
            break
    return pd.Index(np.concatenate(taken)) if taken else pd.Index([])


def build(df: pd.DataFrame, cfg, exposure: float) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg.project.seed)
    df = df.copy()

    train_idx = df.index[df["slice"] == "train"]
    n_total = len(train_idx)
    n_withheld = int(round(exposure * n_total))
    n_retained = n_total - n_withheld

    # Problem ids that any protected condition depends on. A donor group
    # sharing one of these would put an S1 or S5 row's group into training.
    forbidden = set(df.loc[df["slice"].isin(PROTECTED), "problem_id"])

    carries = withheld_mask(df, cfg)
    pool = df.index[df["slice"].isin(DONORS) & carries]
    donor_idx = draw_by_problem(df, pool, n_withheld, rng, forbidden)
    if len(donor_idx) < n_withheld * 0.9:
        raise SystemExit(
            f"only {len(donor_idx):,} withheld rows available for a target of "
            f"{n_withheld:,}. Lower --exposure or widen the donor conditions.")

    # D2's retained half is a subsample of the existing training set; D1_small
    # is exactly that half, so the two arms share their retained data by
    # construction rather than by coincidence.
    retained_idx = pd.Index(rng.choice(train_idx, size=n_retained, replace=False))

    df["d2_train"] = False
    df.loc[retained_idx, "d2_train"] = True
    df.loc[donor_idx, "d2_train"] = True
    df["d1small_train"] = False
    df.loc[retained_idx, "d1small_train"] = True

    # A fourth arm, exposed to everything except an unseen language.
    #
    # D2 draws donors from S2, S3 and S4, so it sees Go and JavaScript rows in
    # training. Its gain on S4 could therefore in principle come from general
    # token coverage rather than from source exposure, even though every row of
    # S4 is in a language the model already knew. This arm removes that
    # possibility by construction: its donors come only from the
    # generator-novel and source-novel conditions, so no Go or JavaScript row
    # ever enters training, and S3 and S5 remain untouched held-out language
    # conditions in both arms.
    #
    # It shares D1_small's retained rows exactly, so d2nolang minus d1small is
    # the same contrast as d2 minus d1small with the language axis removed.
    nolang_pool = df.index[df["slice"].isin(NOLANG_DONORS) & carries]
    nolang_idx = draw_by_problem(df, nolang_pool, n_withheld, rng, forbidden)
    if len(nolang_idx) < n_withheld * 0.9:
        raise SystemExit(
            f"only {len(nolang_idx):,} language-clean withheld rows for a target "
            f"of {n_withheld:,}; lower --exposure or widen NOLANG_DONORS.")
    seen_langs = set(df.loc[nolang_idx, "language"])
    trained_langs = set(df.loc[df["slice"] == "train", "language"])
    leaked = seen_langs - trained_langs
    if leaked:
        raise SystemExit(f"language-clean arm would see {sorted(leaked)}, which "
                         "defeats the purpose of the arm")
    df["d2nolang_train"] = False
    df.loc[retained_idx, "d2nolang_train"] = True
    df.loc[nolang_idx, "d2nolang_train"] = True

    # Donors have left evaluation. Anything sharing their group goes too.
    donor_pids = set(df.loc[donor_idx, "problem_id"])
    arm_slice = df["slice"].copy()
    arm_slice.loc[donor_idx] = "train_d2"
    also = df.index[df["slice"].isin(DONORS) & df["problem_id"].isin(donor_pids)
                    & ~df.index.isin(donor_idx)]
    arm_slice.loc[also] = "unused"
    df["arm_slice"] = arm_slice

    report = {
        "exposure_target": exposure,
        "d2_rows": int(df["d2_train"].sum()),
        "d2_withheld_rows": int(len(donor_idx)),
        "d2_exposure_realised": float(len(donor_idx) / max(df["d2_train"].sum(), 1)),
        "d1small_rows": int(df["d1small_train"].sum()),
        "d2nolang_rows": int(df["d2nolang_train"].sum()),
        "d2nolang_withheld_rows": int(len(nolang_idx)),
        "d1_rows": int(n_total),
        "dropped_for_group_hygiene": int(len(also)),
        "conditions": {},
    }
    for s in ("s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound", "val"):
        before = int((df["slice"] == s).sum())
        after = int((df["arm_slice"] == s).sum())
        report["conditions"][s] = {"before": before, "after": after,
                                   "kept": after / before if before else 0.0}
    return df, report


def checks(df: pd.DataFrame, cfg, report: dict) -> None:
    """Refuse to emit a corpus that would make the comparison meaningless."""
    carries = withheld_mask(df, cfg)
    bad = int((df["d1small_train"] & carries).sum())
    if bad:
        raise SystemExit(f"{bad} withheld-category rows leaked into D1_small, "
                         "which is supposed to be the unexposed arm")

    d2_pids = set(df.loc[df["d2_train"], "problem_id"])
    for s in ("s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"):
        overlap = df.index[(df["arm_slice"] == s)
                           & df["problem_id"].isin(d2_pids)]
        if len(overlap):
            raise SystemExit(f"{len(overlap)} rows of {s} share a problem_id "
                             "with D2's training set")

    for s in PROTECTED:
        c = report["conditions"][s]
        if c["before"] != c["after"]:
            raise SystemExit(f"{s} was supposed to be untouched but went from "
                             f"{c['before']:,} to {c['after']:,}")

    for arm in ("d2_train", "d1small_train"):
        present = sorted(df.loc[df[arm], "label"].unique())
        if len(present) != 4:
            raise SystemExit(f"{arm} has labels {present}, expected all four")

    # The whole claim is that D2 was exposed to the categories D1 was not. A
    # category the draw happened to miss would leave that claim false for one
    # axis while every aggregate still looked right, so check them one by one.
    absent = []
    d2 = df[df["d2_train"]]
    for col, values in (("model_family", cfg.splits.holdout_families),
                        ("language", cfg.splits.holdout_languages),
                        ("source", cfg.splits.holdout_sources)):
        for v in values:
            n = int((d2[col] == v).sum())
            report.setdefault("d2_category_rows", {})[str(v)] = n
            if n == 0:
                absent.append(str(v))
    if absent:
        raise SystemExit("D2 was not exposed to " + ", ".join(absent)
                         + "; the draw missed a withheld category entirely")


ARM_COLUMN = {"d2": "d2_train", "d1small": "d1small_train",
              "d2nolang": "d2nolang_train"}


def as_training_frame(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Reshape the arm table into the split/slice columns the trainer reads.

    The trainer knows about `split` and `slice`, not about arms, and teaching it
    about arms would spread the concept through code that has no business
    knowing it. So the arm is resolved here instead: the chosen arm's rows
    become training, and rows belonging only to the other arm become unused
    rather than evaluation, which is the part that would otherwise be a silent
    and severe leak.
    """
    col = ARM_COLUMN[arm]
    out = df.copy()
    out["slice"] = out["arm_slice"]
    held = out["arm_slice"].isin(["train", "train_d2"])
    out.loc[held & ~out[col], "slice"] = "unused"
    out.loc[out[col], "slice"] = "train"
    out["split"] = np.where(
        out["slice"].isin(["train", "val"]), out["slice"],
        np.where(out["slice"] == "unused", "unused", "test"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="kaggle.yaml")
    ap.add_argument("--exposure", type=float, default=0.356,
                    help="share of D2's training rows carrying a withheld "
                         "category; the default matches the exposure audit")
    args = ap.parse_args()

    cfg = C.load(args.config)
    d = C.artifacts(cfg) / "data"
    df = pd.read_parquet(d / "splits.parquet")

    df, report = build(df, cfg, args.exposure)
    checks(df, cfg, report)

    df.to_parquet(d / "splits_arms.parquet", index=False)

    # Masks let the existing model be re-scored on the reduced conditions by
    # subsetting arrays it has already produced, so D1 needs no retraining and
    # no GPU at all.
    md = d / "arm_masks"
    os.makedirs(md, exist_ok=True)
    for s in ("s2_unseen_generator", "s3_unseen_language", "s4_unseen_domain"):
        orig = df["slice"] == s
        keep = (df.loc[orig, "arm_slice"] == s).to_numpy()
        np.save(md / f"keep_{s}.npy", keep)

    (d / "arm_report.json").write_text(json.dumps(report, indent=2),
                                       encoding="utf-8")

    print(f"D1        {report['d1_rows']:>9,} rows")
    print(f"D2        {report['d2_rows']:>9,} rows  "
          f"({report['d2_withheld_rows']:,} withheld, "
          f"{report['d2_exposure_realised']:.1%})")
    print(f"D1_small  {report['d1small_rows']:>9,} rows")
    print(f"\ndropped for group hygiene: {report['dropped_for_group_hygiene']:,}\n")
    print(f"{'condition':22s} {'before':>9s} {'after':>9s} {'kept':>7s}")
    for s, c in report["conditions"].items():
        print(f"{s:22s} {c['before']:>9,} {c['after']:>9,} {c['kept']:>6.1%}")
    print(f"\nwritten -> {d / 'splits_arms.parquet'}")


if __name__ == "__main__":
    main()
