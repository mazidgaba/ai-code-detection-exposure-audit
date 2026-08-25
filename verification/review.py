"""Independent review pass over the manuscript.

Runs the checks a reviewer would run by hand, and a few they would not have
time for. Four categories:

  NUMBERS   every table cell traced to the report file that produced it
  CORPUS    GPU-build and CPU-build figures must not be interchanged, which is
            the failure mode most likely to produce a plausible falsehood here
  STYLE     sentence-length distribution against the reference papers, plus
            the phrasings that mark machine-written prose
  STRUCTURE LaTeX validity, citations, floats, em dashes

Exit code is non-zero if anything in NUMBERS, CORPUS or STRUCTURE fails; STYLE
findings are advisory and printed without failing the run.

    python verification/review.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics as st
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "aicd", "eval", "reports")
ORDER = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound"]
ROW = {"s1 in-distribution": "s1_in_distribution",
       "s2 unseen generator": "s2_unseen_generator",
       "s3 unseen language": "s3_unseen_language",
       "s4 unseen source": "s4_unseen_domain",
       "s5 compound": "s5_compound",
       "train": "train", "validation": "val"}


def _load_rel(rel):
    """Load a report that lives outside aicd/eval/reports, by repo-relative path.

    The twin control's two arms were produced by separate Kaggle runs and are
    kept under kaggle_runs/results/ with their run evidence, rather than being
    copied into the reports directory where they would lose that association.
    """
    f = os.path.join(HERE, "..", *rel.split("/"))
    return json.load(io.open(f, encoding="utf-8")) if os.path.exists(f) else None


def load(*p):
    f = os.path.join(REP, *p)
    return json.load(io.open(f, encoding="utf-8")) if os.path.exists(f) else None


def clean(c):
    s = re.sub(r"\\(best|bad|textbf|emph|textit)\{([^{}]*)\}", r"\2", c.strip())
    s = s.replace("{,}", "").replace("$", "").replace("\\%", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return s.replace("{", "").replace("}", "").strip()


def num(c):
    s = clean(c).replace("\u2212", "-")
    return float(s) if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s) else None


def tables(src):
    out = {}
    for blk in re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", src, re.S):
        m = re.search(r"\\label\{([^}]+)\}", blk)
        b = re.search(r"\\begin\{tabular\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(.*?)"
                      r"\\end\{tabular\}", blk, re.S)
        if not (m and b):
            continue
        rows = []
        for raw in b.group(1).split("\\\\"):
            line = re.sub(r"\\(top|mid|bottom)rule", " ", raw)
            line = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", " ", line)
            line = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", line).strip()
            if line:
                rows.append(line.split("&"))
        out[m.group(1)] = rows
    return out


def main() -> int:
    tex = os.path.join(HERE, "main.tex")
    if not os.path.exists(tex):
        # The public artifact ships the code and the measured results but not
        # the manuscript, which is unpublished. Say so rather than dying on a
        # traceback that reads like a broken repository.
        print("main.tex is not present, so there is nothing to audit against.\n"
              "This repository publishes the code and the measured results; the\n"
              "manuscript is not distributed until the paper is published. Every\n"
              "figure the manuscript quotes is recorded in\n"
              "research_state/numbers_ledger.csv against the file it came from,\n"
              "and each analysis under aicd/eval/ regenerates its own report.")
        return 0
    src = io.open(tex, encoding="utf-8").read()
    T = tables(src)
    fail, warn = [], []
    ledger = []
    checked = 0

    def chk(label, got, want, tol=6e-4):
        nonlocal checked
        checked += 1
        status = "ok"
        if got is None or want is None:
            fail.append(f"{label}: unresolved (file={got} paper={want})")
            status = "unresolved"
        elif abs(got - want) > tol:
            fail.append(f"{label}: file={got:.4f} paper={want}")
            status = "mismatch"
        # Recorded as well as counted, so the audit can be exported as a ledger
        # rather than only summarised as a total.
        ledger.append({"label": label, "source_value": got, "paper_value": want,
                       "tolerance": tol, "status": status})

    def rs(row):
        h = clean(row[0]).lower()
        for k, v in ROW.items():
            if h.startswith(k):
                return v
        return None

    # ---------------- NUMBERS: GPU build ------------------------------------
    ka, kb = load("kaggle", "branch_a_base.json"), load("kaggle", "branch_b_xgb.json")
    kan = load("kaggle_analysis.json")

    for row in T.get("tab:control", []):
        s = rs(row)
        if not s or not ka or s not in ka["slices"]:
            continue
        v = [num(x) for x in row[1:]]
        if len(v) < 5:
            continue
        chk(f"tab:control {s} n", ka["slices"][s]["n"], v[0], tol=0.6)
        chk(f"tab:control {s} A macro_f1", ka["slices"][s]["macro_f1"], v[1])
        chk(f"tab:control {s} A auc", ka["slices"][s]["binary_auc"], v[2])
        chk(f"tab:control {s} B macro_f1", kb["slices"][s]["macro_f1"], v[3])
        chk(f"tab:control {s} B auc", kb["slices"][s]["binary_auc"], v[4])

    if kan and "a" in kan.get("branches", {}):
        A = kan["branches"]["a"]["slices"]
        for row in T.get("tab:policy", []):
            s = rs(row)
            if not s or s not in A:
                continue
            v = [num(x) for x in row[1:]]
            chk(f"tab:policy {s} coverage", A[s]["coverage"], v[0], tol=6e-4)
            chk(f"tab:policy {s} human_fpr", A[s]["human_fpr_policy"], v[1])
            m = re.findall(r"\[([\d.]+), ([\d.]+)\]", " ".join(row))
            if m and "human_fpr_policy" in A[s].get("ci", {}):
                lo, hi = A[s]["ci"]["human_fpr_policy"]
                chk(f"tab:policy {s} CI lo", lo, float(m[0][0]), tol=2e-3)
                chk(f"tab:policy {s} CI hi", hi, float(m[0][1]), tol=2e-3)
        B = kan["branches"].get("b", {}).get("slices", {})
        for row in T.get("tab:aurc", []):
            s = rs(row)
            if not s or s not in A:
                continue
            v = [num(x) for x in row[1:]]
            chk(f"tab:aurc {s} A", A[s]["aurc"], v[0])
            if s in B:
                chk(f"tab:aurc {s} B", B[s]["aurc"], v[1])

    ksp = load("kaggle", "splits.json")
    if ksp:
        for row in T.get("tab:slices", []):
            s = rs(row)
            if not s or s not in ksp:
                continue
            v = [num(x) for x in row[1:]]
            chk(f"tab:slices {s} rows", ksp[s]["rows"], v[0], tol=0.6)
            labs = ksp[s].get("labels")
            if labs:
                for i, nm in enumerate(["hum", "mach", "hyb", "adv"]):
                    chk(f"tab:slices {s} {nm}", labs[i], v[i + 1], tol=0.6)
            if v[0] is not None and None not in v[1:5] and sum(v[1:5]) != v[0]:
                fail.append(f"tab:slices {s}: labels sum {sum(v[1:5])} != rows {v[0]}")

    # ---------------- NUMBERS: CPU build ------------------------------------
    bc, ev = load("baseline_compare.json"), load("evasion_compare.json")
    if bc:
        for row in T.get("tab:contamination", []):
            s = rs(row)
            if not s or s not in bc["slices"]:
                continue
            e = bc["slices"][s]
            if "train" not in e or "test" not in e:
                continue
            tr = e["train"]["droiddetect"]["macro_f1"]
            te = e["test"]["droiddetect"]["macro_f1"]
            v = [num(x) for x in row[1:]]
            chk(f"tab:contam {s} train", tr, v[0])
            chk(f"tab:contam {s} test", te, v[1])
            chk(f"tab:contam {s} inflation", tr - te, v[2])
    if ev:
        for row in T.get("tab:evasion", []):
            s = rs(row)
            if not s or s not in ev["slices"]:
                continue
            e, v = ev["slices"][s], [num(x) for x in row[1:]]
            if len(v) < 5:
                continue
            chk(f"tab:evasion {s} raw", e["raw"]["macro_f1"], v[0])
            chk(f"tab:evasion {s} norm", e["normalised"]["macro_f1"], v[1])
            chk(f"tab:evasion {s} delta", e["delta_macro_f1"], v[2])
            chk(f"tab:evasion {s} raw_auc", e["raw"]["binary_auc"], v[3])
            chk(f"tab:evasion {s} norm_auc", e["normalised"]["binary_auc"], v[4])

    # ---------------- NUMBERS: exposure audit -------------------------------
    # Table tab:exposure carries the measurement that replaced the paper's
    # weakest inference, so it needs the same guard as everything else.
    ex = load("exposure_audit.json")
    if ex:
        flat = {}
        for axis, cats in ex["axes"].items():
            for cat, v in cats.items():
                flat[cat.replace("_", "")] = v
        for row in T.get("tab:exposure", []):
            key = clean(row[1]).lower().replace("\\_", "").replace("_", "")
            key = re.sub(r"[^a-z]", "", key)
            v = [num(x) for x in row[2:]]
            if len(v) < 2:
                continue
            if key.startswith("unionof"):
                chk("tab:exposure union rows",
                    ex["union_rows_touching_any_withheld_category"], v[0], tol=0.6)
                chk("tab:exposure union share",
                    ex["union_share"] * 100, v[1], tol=6e-3)
            elif key == "withheldcategory":
                continue  # header row, not data
            elif key in flat:
                chk(f"tab:exposure {key} rows", flat[key]["rows"], v[0], tol=0.6)
                chk(f"tab:exposure {key} share", flat[key]["share"] * 100, v[1],
                    tol=6e-3)
            else:
                warn.append(f"tab:exposure: unmatched category {key!r}")

    # ---------------- NUMBERS: trivial baselines ----------------------------
    bl = load("baselines.json")
    if bl:
        for row in T.get("tab:baselines", []):
            s = rs(row)
            if not s or s not in bl:
                continue
            b, v = bl[s], [num(x) for x in row[1:]]
            if len(v) < 4:
                continue
            chk(f"tab:baselines {s} uniform", b["uniform"], v[0])
            chk(f"tab:baselines {s} stratified", b["stratified"], v[1])
            chk(f"tab:baselines {s} majority", b["majority"], v[2])
            chk(f"tab:baselines {s} branch_a", b["branch_a"], v[3])

    # ---------------- NUMBERS: matched-scale control ------------------------
    # Both columns must come from runs scored on identical evaluation rows.
    # An earlier draft compared a model scored on 62,283 S1 rows against one
    # scored on 31,083, which is not a comparison at all.
    lf = load("branch_a_matched_on_original.json")
    if lf and ka:
        lf = lf["slices"]
        for row in T.get("tab:matched", []):
            s = rs(row)
            if not s or s not in lf:
                continue
            v = [num(x) for x in row[1:]]
            if len(v) < 3:
                continue
            chk(f"tab:matched {s} original", ka["slices"][s]["macro_f1"], v[0])
            chk(f"tab:matched {s} matched", lf[s]["macro_f1"], v[1])
            chk(f"tab:matched {s} delta",
                lf[s]["macro_f1"] - ka["slices"][s]["macro_f1"], v[2])

    # ---------------- NUMBERS: the twin control -----------------------------
    # The paper's centrepiece, so every cell of it is traced. The arms are
    # only comparable if both were scored on the same rows, which is what the
    # row count column asserts; a mismatch there would mean the two models
    # were evaluated on different conditions and the difference column is
    # meaningless.
    tw = load("twin_control.json")
    if tw:
        for row in T.get("tab:twin", []):
            head = clean(row[0]).lower()
            key = next((k for k in tw["conditions"]
                        if k.split("_")[0] == head.split()[0]), None)
            if not key:
                continue
            c = tw["conditions"][key]
            v = [num(x) for x in row[1:]]
            if len(v) < 4:
                continue
            chk(f"tab:twin {key} rows", c["rows"], v[0], tol=0.5)
            chk(f"tab:twin {key} d1small", c["d1small"], v[1])
            chk(f"tab:twin {key} d2", c["d2"], v[2])
            chk(f"tab:twin {key} difference", c["difference"], v[3])
        for label, val in (("drop_d1small", tw["drop_d1small"]),
                           ("drop_d2", tw["drop_d2"])):
            if f"{val:.4f}" not in src:
                warn.append(f"twin control {label}={val:.4f} not quoted in the text")
        n = tw["s1_argmax_disagreements"]
        if f"{n:,}".replace(",", "{,}") not in src:
            warn.append(f"the {n:,} S1 disagreements are not quoted; without "
                        "them the identical S1 scores look like a copied number")

    # ---------------- NUMBERS: label-shift corrections ----------------------
    # The section's claim is that nothing recovers accuracy, so the guard is on
    # the relation rather than the cells: if any correction ever did recover
    # meaningfully, every cell could stay correct while the section became
    # false. The oracle row is the one that matters, because it is an upper
    # bound on what any estimator could achieve.
    sd = load("shift_diagnosis_a.json")
    if sd:
        s5 = sd["conditions"]["s5_compound"]
        src_map = {
            "none (uncorrected)": (s5["macro_f1"], s5["human_fpr_at_threshold"]),
            "oracle priors": (s5["oracle_prior_correction"]["macro_f1"],
                              s5["oracle_prior_correction"]["human_fpr_at_threshold"]),
            "oracle temperature + priors":
                (s5["oracle_temperature_and_prior"]["macro_f1"], None),
            "bbse": (s5["bbse_correction"]["macro_f1"],
                     s5["bbse_correction"]["human_fpr_at_threshold"]),
            "em/sld": (s5["em_correction"]["macro_f1"],
                       s5["em_correction"]["human_fpr_at_threshold"]),
            "threshold refit, k=10":
                (s5["kshot"]["10"]["none"]["macro_f1_mean"],
                 s5["kshot"]["10"]["none"]["human_fpr_refit_threshold"]),
            "threshold refit, k=100":
                (s5["kshot"]["100"]["none"]["macro_f1_mean"],
                 s5["kshot"]["100"]["none"]["human_fpr_refit_threshold"]),
        }
        def norm(x):
            # Row labels carry \cite{...} and $...$; strip both, and collapse
            # whitespace so "k=10" and "k=100" stay distinguishable. A prefix
            # match here silently scored the k=100 row against the k=10 entry.
            x = re.sub(r"\\cite\{[^}]*\}", "", x)
            return re.sub(r"[^a-z0-9=]+", "", x.lower())

        norm_map = {norm(k): k for k in src_map}
        for row in T.get("tab:shift", []):
            key = norm_map.get(norm(clean(row[0])))
            if not key:
                continue
            want_f1, want_fpr = src_map[key]
            v = [num(x) for x in row[1:]]
            if not v:
                continue
            chk(f"tab:shift {key} macro_f1", want_f1, v[0], tol=1.2e-3)
            if len(v) > 1 and v[1] is not None and want_fpr is not None:
                chk(f"tab:shift {key} human_fpr", want_fpr, v[1], tol=1.2e-3)

        base = s5["macro_f1"]
        best = max(s5["oracle_prior_correction"]["macro_f1"],
                   s5["oracle_temperature_and_prior"]["macro_f1"])
        if best - base > 0.02:
            fail.append(f"a label-shift correction now recovers {best - base:+.4f} on "
                        "S5; the section says nothing recovers accuracy")
        # Temperature alone is a monotone map of the logits and cannot move an
        # argmax, so this equality is an identity. If it ever breaks, the
        # temperature path is doing something other than scaling.
        for k in ("10", "50", "100", "500"):
            a = s5["kshot"][k]["none"]["macro_f1_mean"]
            b = s5["kshot"][k]["temperature"]["macro_f1_mean"]
            if abs(a - b) > 1e-9:
                fail.append(f"k={k}: temperature changed macro-F1 from {a} to {b}; "
                            "a positive temperature cannot move an argmax")

    # ---------------- NUMBERS: augmentation mitigation ----------------------
    # The claim is that augmentation does not measurably change the collapse.
    # The cells are traced, and the reading is guarded: the arm's drop must stay
    # inside the six-seed range, because the moment it leaves, "inside seed
    # range" in the table becomes false while every other cell stays right.
    ag = load("e8_augment.json")
    if ag:
        for row in T.get("tab:augment", []):
            s_ = rs(row)
            if not s_ or s_ not in ag["slices"]:
                continue
            v = [num(x) for x in row[1:]]
            if len(v) < 3:
                continue
            chk(f"tab:augment {s_} augmented", ag["slices"][s_]["macro_f1"], v[0])
            chk(f"tab:augment {s_} reference", ag["reference"]["by_condition"][s_], v[1])
            chk(f"tab:augment {s_} delta",
                ag["slices"][s_]["delta_vs_reference"], v[2])
        lo = ag["reference"]["drop_min"]
        hi = ag["reference"]["drop_max"]
        if not (lo <= ag["drop"] <= hi):
            fail.append(f"augmented drop {ag['drop']:.4f} is outside the seed range "
                        f"[{lo:.4f}, {hi:.4f}]; the table says it is inside")
        if ag["vs_seed_range"] != "inside":
            fail.append("e8_augment reports the arm as "
                        f"{ag['vs_seed_range']}, not inside the seed range")
        # The augmentation must actually have fired. A run that selected rows and
        # altered none would produce these numbers and mean nothing.
        a = ag["augmentation"]
        if a["altered_fraction_of_selected"] < 0.5:
            fail.append(f"augmentation altered only "
                        f"{a['altered_fraction_of_selected']:.1%} of selected rows")
        for val in (a["selected"], a["actually_altered"]):
            if f"{val:,}".replace(",", "{,}") not in src:
                warn.append(f"augmentation count {val:,} is not quoted; without it a "
                            "reader cannot tell the rewrites fired at all")

    # ---------------- NUMBERS: CodeMirage tiers -----------------------------
    # The only experiment whose result does not support the paper. The guard is
    # therefore inverted: if the tiers ever became monotone, the section's
    # careful hedging would be understating the evidence, and the text would
    # need rewriting in the paper's favour rather than against it.
    cm = load("codemirage_eval_base.json")
    cmt = load("codemirage_test.json")
    if cm:
        label = {"family and model seen": "seen_family_seen_model",
                 "family seen, model new": "seen_family_unseen_model",
                 "family unseen": "unseen_family"}
        for row in T.get("tab:codemirage", []):
            head = clean(row[0]).lower()
            key = label.get(head)
            v = [num(x) for x in row[1:]]
            if head.startswith("overall"):
                if len(v) >= 3:
                    chk("tab:codemirage overall n", cm["rows"], v[0], tol=0.6)
                    chk("tab:codemirage overall f1",
                        cm["overall"]["macro_f1_present"], v[1])
                    chk("tab:codemirage overall auc", cm["overall"]["binary_auc"], v[2])
                continue
            if not key or key not in cm["by_exposure"] or len(v) < 3:
                continue
            e = cm["by_exposure"][key]
            chk(f"tab:codemirage {key} n", e["n"], v[0], tol=0.6)
            chk(f"tab:codemirage {key} f1", e["macro_f1_present"], v[1])
            chk(f"tab:codemirage {key} auc", e["binary_auc"], v[2])

        if cm.get("monotone_decreasing"):
            fail.append("the CodeMirage tiers are now monotone; the section says "
                        "they are not, and the hedging around it is no longer "
                        "warranted")
        # The section rests on Ruby being the worst language and on Ruby and
        # HTML being the two absent from training. Both are checkable.
        if cm.get("by_language"):
            worst = min(cm["by_language"], key=cm["by_language"].get)
            if worst != "ruby":
                fail.append(f"the worst CodeMirage language is now {worst}, not ruby; "
                            "the language argument in the text assumes ruby")
        if cmt and sorted(cmt.get("languages_absent_from_droid", [])) != ["html", "ruby"]:
            fail.append("the set of CodeMirage languages absent from DroidCollection "
                        "has changed; the text names HTML and Ruby")

    # ---------------- NUMBERS: statistical comparisons ----------------------
    # The seed effect size and the Holm-corrected generator tests. The guard on
    # the latter is the one that matters: if a comparison ever stopped
    # surviving correction, the text would still read "both survive" while the
    # underlying claim had failed.
    cp = load("comparisons.json")
    if cp:
        sv = cp["seed_drop"]
        for label, val, fmt in (("mean", sv["mean"], "{:.4f}"),
                                ("sd", sv["sd"], "{:.4f}"),
                                ("ci lo", sv["ci95"][0], "{:.4f}"),
                                ("ci hi", sv["ci95"][1], "{:.4f}"),
                                ("dz", sv["cohens_dz"], "{:.1f}"),
                                ("t", sv["t"], "{:.1f}")):
            if fmt.format(val) not in src:
                warn.append(f"seed drop {label}={fmt.format(val)} not quoted in the text")
        if sv["df"] != sv["n_seeds"] - 1:
            fail.append("seed drop df does not match the number of seeds")
        if f"t({sv['df']})" not in src:
            warn.append(f"the degrees of freedom t({sv['df']}) are not stated")

        gc = cp["generator_comparison"]
        if not gc["all_survive_holm"]:
            fail.append("a generator comparison no longer survives Holm correction; "
                        "the text says both do")
        for g, d in gc["tests"].items():
            if f"{d['z']:.2f}" not in src:
                warn.append(f"generator test z={d['z']:.2f} not quoted")
            if d["p_holm"] >= 0.05:
                fail.append(f"gemini vs {g}: Holm-adjusted p={d['p_holm']:.2e} "
                            "is not significant")
        # A family of two tests must be corrected as a family of two. If a third
        # generator ever entered, the adjusted values in the text would be wrong.
        if gc["family_size"] != len(gc["tests"]):
            fail.append("Holm family size does not match the number of tests run")

    # ---------------- NUMBERS: contamination on the larger build ------------
    # The metric guard is the important one here. This corpus contributes no
    # adversarial rows from the training shard, so a four-class macro-F1 makes
    # the gap come out negative for a reason unrelated to memorisation. The
    # report must say it restricted both arms to shared classes.
    cg = load("contamination_gpu.json")
    if cg:
        if "three classes present in both" not in cg.get("comparison", ""):
            fail.append("contamination_gpu.json no longer reports a class-matched "
                        "comparison; a four-class gap on this corpus is an artefact")
        for row in T.get("tab:contamgpu", []):
            s_ = rs(row)
            if not s_ or s_ not in cg["conditions"]:
                continue
            e = cg["conditions"][s_]
            v = [num(x) for x in row[1:]]
            if len(v) < 3:
                continue
            chk(f"tab:contamgpu {s_} train", e["train_macro_f1"], v[0])
            chk(f"tab:contamgpu {s_} held", e["held_macro_f1"], v[1])
            # The third cell carries the estimate and its interval together, so
            # take the leading signed number rather than the whole cell.
            lead = re.match(r"([+-]?\d+\.\d+)", clean(row[3]))
            chk(f"tab:contamgpu {s_} inflation", e["inflation"],
                float(lead.group(1)) if lead else None)
            m = re.findall(r"\[([\d.]+), ([\d.]+)\]", " ".join(row))
            if m:
                chk(f"tab:contamgpu {s_} CI lo", e["ci95"][0], float(m[0][0]), tol=2e-3)
                chk(f"tab:contamgpu {s_} CI hi", e["ci95"][1], float(m[0][1]), tol=2e-3)
        if cg["n_excluding_zero"] < len(cg["conditions"]):
            fail.append(f"only {cg['n_excluding_zero']} of {len(cg['conditions'])} "
                        "contamination intervals exclude zero; the text says all do")
        # The shape carries the argument: inflation must be smallest on S1 and
        # largest on S5, or the sentence about where it concentrates is false.
        seq = [cg["conditions"][c]["inflation"] for c in ORDER
               if c in cg["conditions"]]
        if seq and (seq[0] != min(seq) or seq[-1] != max(seq)):
            fail.append("contamination is no longer smallest on S1 and largest on "
                        "S5; the text claims that ordering")
        for label, val in (("mean", cg["mean_inflation"]),):
            if f"{val:.4f}" not in src:
                warn.append(f"contamination {label}={val:.4f} not quoted in the text")

    # ---------------- NUMBERS: paired difference intervals ------------------
    # The twin's in-distribution null is the paper's most quotable claim, so the
    # guard is that it stays a *measured* null: the interval must span zero. If
    # it ever stopped spanning zero, "exposure is worth nothing at all in
    # distribution" would be false while the point estimate stayed 0.0000.
    pdf_ = load("paired_differences.json")
    if pdf_:
        tw = pdf_["comparisons"].get("twin", {}).get("conditions", {})
        s1 = tw.get("s1_in_distribution")
        s5 = tw.get("s5_compound")
        if s1:
            if s1["excludes_zero"]:
                fail.append("the twin's in-distribution paired interval no longer "
                            f"spans zero ({s1['ci95']}); the text calls it a null")
            for v in s1["ci95"]:
                if f"{v:+.4f}".replace("+", "") not in src.replace("$", ""):
                    warn.append(f"twin S1 interval bound {v:+.4f} not quoted")
        if s5:
            if not s5["excludes_zero"]:
                fail.append("the twin's compound paired interval now spans zero; "
                            "the central effect is no longer established")
        # A pairing is only valid on rows both arms actually share. Any condition
        # the module skipped must not appear as a paired interval in the text.
        for c in pdf_["comparisons"].get("twin", {}).get("skipped", {}):
            if c in tw:
                fail.append(f"{c} was skipped for length mismatch yet carries a "
                            "paired interval")

    # ---------------- NUMBERS: axis decomposition ---------------------------
    # The reviewer-facing point of this table. Two conditions are novel in their
    # programming language, where a detector lacks token coverage, so the twin's
    # effect on them cannot be attributed to exposure alone. S4 is the condition
    # that carries the argument precisely because it is language-clean, and if
    # that ever stopped being true the section's central sentence would be false
    # while every cell in the table stayed correct.
    ax = load("axis_decomposition.json")
    if ax:
        col = {"language": "language", "source": "source", "family": "model_family"}
        for row in T.get("tab:axes", []):
            s_ = rs(row)
            if not s_ or s_ not in ax["conditions"]:
                continue
            e = ax["conditions"][s_]
            v = [num(x) for x in row[1:]]
            if len(v) < 4:
                continue
            for i, key in enumerate(["language", "source", "family"]):
                chk(f"tab:axes {s_} {key}",
                    e[col[key]]["fraction_novel"] * 100, v[i], tol=6e-2)
            chk(f"tab:axes {s_} exposure", e["twin_effect"], v[3])
        s4 = ax["conditions"]["s4_unseen_domain"]
        if s4["language"]["fraction_novel"] > 0.01:
            fail.append("S4 is no longer language-clean; the argument that it "
                        "isolates exposure from token coverage fails")
        if s4["source"]["fraction_novel"] < 0.99:
            fail.append("S4 is no longer wholly novel in source")
        # The uncomfortable finding must stay in the text: the generator axis,
        # which benchmarks withhold most often, is the one worth least.
        gen = ax["conditions"]["s2_unseen_generator"]["twin_effect"]
        if abs(gen) > 0.10:
            fail.append(f"the generator-axis effect is now {gen:+.4f}; the text "
                        "describes it as small")
        for label, val in (("S4 exposure", s4["twin_effect"]),
                           ("S2 exposure", gen)):
            if f"{abs(val):.4f}" not in src:
                warn.append(f"{label} {val:+.4f} is not quoted in the text")

    # ---------------- NUMBERS: the third-party audit ------------------------
    # This is the audit that does not choose its own categories, so it carries
    # the non-circular version of the paper's central claim. If any AICD Bench
    # language ever stopped being present, the section would be overstating.
    ab = load("aicdbench_audit.json")
    if ab:
        for row in T.get("tab:aicdaudit", []):
            cells = [clean(c) for c in row]
            if not cells or not cells[0]:
                continue
            name = cells[0].replace("\\#", "#").strip()
            key = next((k for k in ab["languages"] if k.lower() == name.lower()),
                       None)
            if key is None:
                continue
            v = [num(x) for x in row[1:]]
            if len(v) >= 1:
                chk(f"tab:aicdaudit {key} rows", ab["languages"][key]["rows"],
                    v[0], tol=0.5)
        if not ab["every_withheld_language_present"]:
            fail.append("an AICD Bench withheld language is no longer present in "
                        "the training split; the text says every one of the six is")
        u = ab["languages_union"]
        chk("aicdbench union rows", u["rows"],
            u["rows"] if f"{u['rows']:,}".replace(",", "{,}") in src else None,
            tol=0.5)

    # ---------------- NUMBERS: variance components --------------------------
    # Design versus seed variation, compared on standard deviations rather than
    # ranges. If design ever came to dominate, the claim that the collapse is
    # not a property of the architecture would no longer follow.
    vc = load("variance_components.json")
    if vc:
        for k in ("design", "seed"):
            sd = vc[k]["sd"]
            chk(f"variance {k} sd", sd, sd if f"{sd:.4f}" in src else None)
        r = vc["sd_ratio_seed_over_design"]
        if f"{r}" not in src:
            warn.append(f"seed/design sd ratio {r} is not quoted in the text")
        if not vc["seed_dominates"]:
            fail.append("design variation now exceeds seed variation; the text "
                        "says design moves the collapse less than the seed does")

    # ---------------- NUMBERS: the external corpus taxonomy -----------------
    # The external collapse is partly the hybrid class, whose definition we
    # supplied. Both readings must stay in the text, and the restricted one
    # must stay well below the in-distribution score, or the section claims a
    # collapse the mapping-free view does not support.
    it = load("independent_taxonomy.json")
    if it:
        for tag in ("all_classes", "human_machine"):
            v = it["views"][tag]
            chk(f"independent {tag} macro_f1", v["macro_f1"],
                v["macro_f1"] if f"{v['macro_f1']:.4f}" in src else None)
        if not it["headline"]["collapse_survives_dropping_hybrid"]:
            fail.append("dropping the hybrid class removes the external "
                        "collapse; the text says the fall survives it")
        hm = it["views"]["human_machine"]
        for b in hm["ci"]:
            chk("independent human_machine ci", b,
                b if f"{b:.4f}" in src else None)

    # ---------------- NUMBERS: the twin across seeds ------------------------
    # Replication is what separates the effect from initialisation noise. Two
    # things must keep holding or the section overstates what three runs show:
    # the compound effect must stay many standard deviations from zero, and the
    # in-distribution null must stay within about one.
    sw = load("seed_twin.json")
    if sw:
        for row in T.get("tab:seedtwin", []):
            s_ = rs(row)
            if not s_ or s_ not in sw["conditions"]:
                continue
            e = sw["conditions"][s_]
            v = [num(x) for x in row[1:]]
            if len(v) < 6:
                continue
            for i, d in enumerate(e["differences"]):
                chk(f"tab:seedtwin {s_} seed{i+1}", d, v[i])
            chk(f"tab:seedtwin {s_} mean", e["mean"], v[3])
            chk(f"tab:seedtwin {s_} sd", e["sd"], v[4], tol=6e-5)
        s5 = sw["conditions"]["s5_compound"]
        s1 = sw["conditions"]["s1_in_distribution"]
        if s5["sd_from_zero"] is None or s5["sd_from_zero"] < 5:
            fail.append("the compound twin effect is no longer far from zero "
                        "relative to its seed spread; the text says it is not "
                        "initialisation noise")
        if s1["sd_from_zero"] is not None and s1["sd_from_zero"] > 2:
            fail.append(f"the in-distribution null is now "
                        f"{s1['sd_from_zero']} SD from zero; the text calls it "
                        "a null")

    # ---------------- NUMBERS: the published detector under shift -----------
    # The claim is about published detectors, so this table has to keep saying
    # two things: the detector degrades on rows it never trained on, and the
    # advantage from scoring on training rows grows with condition severity.
    ps = load("published_shift.json")
    if ps:
        for row in T.get("tab:published", []):
            s_ = rs(row)
            if not s_ or s_ not in ps["conditions"]:
                continue
            e = ps["conditions"][s_]
            v = [num(x) for x in row[1:]]
            if len(v) < 2:
                continue
            chk(f"tab:published {s_} n", e["clean"]["n"], v[0], tol=0.5)
            chk(f"tab:published {s_} clean", e["clean"]["macro_f1"], v[1])
            if len(v) >= 4:
                chk(f"tab:published {s_} seen", e["seen"]["macro_f1"], v[2])
                chk(f"tab:published {s_} effect", e["shard_effect"], v[3])
        mild = ps["conditions"]["s2_unseen_generator"]["shard_effect"]
        hard = ps["conditions"]["s5_compound"]["shard_effect"]
        if hard <= mild:
            fail.append(f"the shard advantage no longer grows with severity "
                        f"({mild:+.4f} on S2 against {hard:+.4f} on S5); the text "
                        "says it flatters most where the benchmark tests hardest")
        if f"{ps['clean_drop_s1_to_s5']:.4f}" not in src:
            warn.append(f"published detector clean drop "
                        f"{ps['clean_drop_s1_to_s5']:+.4f} is not quoted")

    # ---------------- NUMBERS: the axes on comparable rows ------------------
    # S2 is mostly human rows, which carry no generator shift. If that ever
    # stopped being corrected for, the axis comparison would silently revert to
    # comparing composition rather than axes.
    ma = load("matched_axis.json")
    if ma:
        if not ma["ordering_survives_matching"]:
            fail.append("the axis ordering no longer survives matching on "
                        "machine-generated rows; the text says it does")
        # These live in prose rather than a table, so the check is that the
        # value the report computes is the value the text quotes.
        for cond, e in ma["conditions"].items():
            v = e["machine_only"]["effect"]
            chk(f"matched {e['axis']} effect", v,
                v if f"{v:.4f}" in src else None)
            lo, hi = e["machine_only"]["ci"]
            for b in (lo, hi):
                chk(f"matched {e['axis']} ci", b,
                    b if f"{b:.4f}" in src else None)
        for tag in ("all", "machine_only"):
            r = ma["ratio"][tag]
            if f"{r}" not in src and f"{r:.1f}" not in src:
                warn.append(f"axis ratio {r}x ({tag}) is not quoted in the text")

    # ---------------- NUMBERS: the language-clean arm -----------------------
    # R1. The arm exists to show the source-axis effect is not token coverage,
    # so two things must hold or the section claims more than it measured: the
    # arm must never have seen an untrained language, and its S4 effect must
    # still be positive with an interval clear of zero.
    lc = load("language_clean.json")
    if lc:
        if lc["untrained_languages_seen"]:
            fail.append("the language-clean arm saw "
                        f"{lc['untrained_languages_seen']}, which defeats it")
        for row in T.get("tab:nolang", []):
            s_ = rs(row)
            if not s_ or s_ not in lc["conditions"]:
                continue
            e = lc["conditions"][s_]
            v = [num(x) for x in row[1:]]
            if len(v) < 3:
                continue
            chk(f"tab:nolang {s_} rows", e["rows"], v[0], tol=0.5)
            chk(f"tab:nolang {s_} exposure", e["exposure_effect"], v[1])
            chk(f"tab:nolang {s_} clean", e["language_clean_effect"], v[2])
        s4 = lc["conditions"]["s4_unseen_domain"]
        if not (s4["clean_excludes_zero"] and s4["language_clean_effect"] > 0):
            fail.append("the source-axis effect no longer survives the "
                        "language-clean arm; the text says it does")
        if f"{s4['language_clean_effect']:.4f}" not in src:
            warn.append(f"language-clean S4 effect "
                        f"{s4['language_clean_effect']:+.4f} is not quoted")

    # ---------------- NUMBERS: the third twin arm ---------------------------
    # The scale control. If scale ever stopped running against exposure, the
    # paragraph claiming the effect survives in spite of scale would be false
    # while both point estimates stayed correct.
    ta = load("twin_three_arms.json")
    if ta:
        arms = ta["arms"]
        for k in ("d1", "d1small", "d2"):
            if f"{arms[k]['rows_trained']:,}".replace(",", "{,}") not in src:
                warn.append(f"twin arm {k} row count {arms[k]['rows_trained']:,} "
                            "is not quoted in the text")
        if not ta["scale_runs_against_exposure"]:
            fail.append(f"scale effect {ta['scale_effect']:+.4f} no longer runs "
                        f"against exposure {ta['exposure_effect']:+.4f}; the text "
                        "says the gain survives in spite of scale")
        for label, val in (("scale", ta["scale_effect"]),
                           ("exposure", ta["exposure_effect"])):
            if f"{abs(val):.4f}" not in src:
                warn.append(f"twin {label} effect {val:+.4f} not quoted")

    # ---------------- NUMBERS: semantics preservation -----------------------
    # The mechanism claim depends on the rewrites doing what they say. The
    # guard is that the breakage rate stays quoted and that no rewrite silently
    # starts merging identifiers, which a parse check cannot see.
    sc = load("semantics_check.json")
    if sc:
        ren = sc["transforms"].get("rename_identifiers", {})
        if "collision_rate" in ren and ren["collision_rate"] > 0:
            fail.append(f"renaming now collides identifiers at "
                        f"{ren['collision_rate']:.4%}; the text says the mapping "
                        "is injective")
        br = ren.get("break_rate")
        if br is not None and f"{br:.2%}".rstrip("%") not in src.replace(r"\%", ""):
            warn.append(f"rename breakage {br:.2%} is not quoted in the text")
        # Any rewrite whose breakage grows into the same range as renaming's
        # would need reporting too, and silence would be the failure.
        for name, t in sc["transforms"].items():
            if name == "rename_identifiers":
                continue
            if t.get("break_rate", 0) > 0.01:
                warn.append(f"{name} now breaks {t['break_rate']:.2%} of the files "
                            "it alters and the text describes it as clean")

    # ---------------- NUMBERS: the exposure-audit null ----------------------
    # The audit's share is ordinary against a random draw of the same shape, and
    # the text now says so. If it ever became extreme the paper would be
    # understating its own evidence, so the guard fires in both directions.
    en = load("exposure_null.json")
    if en:
        # The text states these as percentages, so check both renderings before
        # complaining; a check that only knows one form reports a false absence.
        for label, val in (("observed", en["observed"]),
                           ("null median", en["null_median"]),
                           ("null p05", en["null_p05"]),
                           ("null p95", en["null_p95"])):
            forms = {f"{val:.4f}", f"{val:.3f}", f"{val*100:.2f}", f"{val*100:.1f}"}
            if not any(f in src for f in forms):
                warn.append(f"exposure null {label}={val:.4f} not quoted in any "
                            "of its decimal or percentage forms")
        pct = en["percentile_of_observed"]
        if not (0.05 <= pct <= 0.95) and en.get("reading") == "typical":
            fail.append(f"exposure share is at the {pct:.1%} percentile but the "
                        "report still calls it typical")
        if en.get("reading") != "typical":
            warn.append(f"the exposure share is now {en['reading']} at the "
                        f"{pct:.1%} percentile; the text describes it as ordinary")

    # ---------------- NUMBERS: the encoder panel ----------------------------
    # This table is what licenses the word "detectors" in the plural, so the
    # guard is on the spread rather than only on the cells: if the encoders ever
    # stopped agreeing, the generality claim would fail while every individual
    # number stayed correct.
    pn = load("e3_panel.json")
    if pn:
        cols = ["reference", "codebert", "unixcoder"]
        for row in T.get("tab:panel", []):
            s_ = rs(row)
            if not s_:
                continue
            v = [num(x) for x in row[1:]]
            if len(v) < 3:
                continue
            for i, key in enumerate(cols):
                src_val = (pn["reference"]["by_condition"][s_] if key == "reference"
                           else pn[key]["slices"][s_]["macro_f1"])
                chk(f"tab:panel {s_} {key}", src_val, v[i])
        drops = [pn["reference"]["drop"], pn["codebert"]["drop"], pn["unixcoder"]["drop"]]
        spread = max(drops) - min(drops)
        seeds = load("seed_sweep_six.json")
        if seeds:
            sd = [v[0] - v[-1] for v in seeds["seeds"].values()]
            seed_spread = max(sd) - min(sd)
            if spread > seed_spread:
                fail.append(f"encoder spread {spread:.4f} now exceeds the seed spread "
                            f"{seed_spread:.4f}; the panel no longer supports the "
                            "plural claim the section draws from it")
        if f"{spread:.4f}" not in src:
            warn.append(f"encoder drop spread {spread:.4f} is not quoted in the text")

    # ---------------- NUMBERS: the zero-shot control ------------------------
    # Every cell, plus the three drops. The zero-shot arm and the twin arms come
    # from different files, and the comparison is only meaningful if all three
    # were scored on the same five conditions, so the condition keys are checked
    # to line up rather than assumed to.
    zs = load("branch_c_fastdetect.json")
    d1 = _load_rel("kaggle_runs/results/e1/branch_a_e1_d1small.json")
    d2 = _load_rel("kaggle_runs/results/e1-d2/results/reports/branch_a_e1_d2.json")
    if zs and d1 and d2:
        z, a1, a2 = zs["slices"], d1["slices"], d2["slices"]
        for row in T.get("tab:zeroshot", []):
            s_ = rs(row)
            if not s_ or s_ not in z:
                continue
            if s_ not in a1 or s_ not in a2:
                fail.append(f"tab:zeroshot {s_}: condition missing from a twin arm, "
                            "so the row compares different evaluations")
                continue
            v = [num(x) for x in row[1:]]
            if len(v) < 3:
                continue
            chk(f"tab:zeroshot {s_} zeroshot", z[s_]["binary_auc"], v[0])
            chk(f"tab:zeroshot {s_} unexposed", a1[s_]["binary_auc"], v[1])
            chk(f"tab:zeroshot {s_} exposed", a2[s_]["binary_auc"], v[2])
        drop = lambda r: r["s1_in_distribution"]["binary_auc"] - r["s5_compound"]["binary_auc"]
        for label, val in (("zero-shot", drop(z)), ("unexposed", drop(a1)),
                           ("exposed", drop(a2))):
            if f"{abs(val):.4f}" not in src:
                warn.append(f"zero-shot control: {label} drop {val:+.4f} is not "
                            "quoted in the text")
        # The claim the subsection rests on. If the zero-shot scorer ever did
        # decline like the unexposed arm, the section's conclusion would be
        # false while every individual cell stayed correct.
        if abs(drop(z)) > 0.5 * abs(drop(a1)):
            fail.append(f"zero-shot drop {drop(z):+.4f} is no longer small against "
                        f"the unexposed arm's {drop(a1):+.4f}; the intrinsic-difficulty "
                        "control no longer supports what the section says")

    # ---------------- NUMBERS: the architecture ablation --------------------
    # Every arm's five conditions and its drop are traced. Two further checks
    # guard the reading rather than the arithmetic. First, the two spreads must
    # be quoted, because the whole subsection turns on the arm spread being the
    # smaller of the two and a reader cannot check that claim without both.
    # Second, no arm may be described as reducing the collapse: three of the
    # four sit inside the seed range, so any such sentence would be comparing
    # against the maximum of a noisy baseline, which is the error the
    # subsection exists to avoid.
    ab = load("e5_ablation.json")
    if ab:
        rowmap = {"no triplet term": "triplet0", "triplet weight 0.2": "triplet02",
                  "projection 256": "proj256", "modernbert-large": "large"}
        conds = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
                 "s4_unseen_domain", "s5_compound"]
        seen = set()
        for row in T.get("tab:ablation", []):
            tag = rowmap.get(clean(row[0]).lower())
            if not tag or tag not in ab["arms"]:
                continue
            seen.add(tag)
            arm = ab["arms"][tag]
            v = [num(x) for x in row[1:]]
            if len(v) < 6:
                continue
            for i, c in enumerate(conds):
                chk(f"tab:ablation {tag} {c}", arm["by_condition"][c], v[i])
            chk(f"tab:ablation {tag} drop", arm["drop"], v[5])
        missing = set(ab["arms"]) - seen
        if missing:
            warn.append(f"tab:ablation omits arm(s) {sorted(missing)}; a table of "
                        "ablations that drops an arm invites the question of why")
        for label, val in (("arm spread", ab["arm_spread"]),
                           ("seed spread", ab["seed_spread"])):
            if f"{val:.4f}" not in src:
                warn.append(f"ablation {label}={val:.4f} not quoted; the claim that "
                            "design choices move the drop less than seeds do "
                            "cannot be checked without both spreads")
        if ab["arm_spread"] > ab["seed_spread"]:
            fail.append("ablation arm spread now exceeds the seed spread; the "
                        "subsection's central sentence is no longer true")
        # A sentence that credits an arm with reducing the collapse is only a
        # finding if that arm sits outside the seed range. Hedged sentences are
        # exempt: the subsection has to describe the flattering reading in
        # order to reject it, and "every arm appears to reduce the collapse"
        # is the rejection, not the claim.
        hedge = re.compile(r"\b(appear|seem|artefact|artifact|would|flatter|"
                           r"until|not\b)", re.I)
        claim = re.compile(r"\b(reduc\w+|shrink\w+|improv\w+)\b[^.]{0,80}collapse", re.I)
        inside = [k for k, a in ab["arms"].items() if a["vs_seed_range"] == "inside"]
        if inside:
            for sent in re.split(r"(?<=[.!?])\s+", src):
                if claim.search(sent) and not hedge.search(sent):
                    warn.append("the text credits something with reducing the "
                                f"collapse while {len(inside)} arm(s) sit inside "
                                f"the seed range: {clean(sent)[:90]}")
                    break

    # ---------------- NUMBERS: seed variance --------------------------------
    sv = load("seed_variance.json")
    if sv:
        want = {"seed 20260818": "paper (seed 20260818)",
                "seed 1": "seed1", "seed 2": "seed2"}
        for row in T.get("tab:seeds", []):
            head = clean(row[0]).lower()
            v = [num(x) for x in row[1:]]
            if len(v) < 5:
                continue
            key = next((k for lbl, k in want.items() if head.startswith(lbl)), None)
            if key and key in sv["runs"]:
                for i, s in enumerate(ORDER):
                    chk(f"tab:seeds {key} {s}", sv["runs"][key][s], v[i])
            elif head.startswith("mean"):
                for i, s in enumerate(ORDER):
                    chk(f"tab:seeds mean {s}", sv["mean"][s], v[i])
            elif head.startswith("sd"):
                for i, s in enumerate(ORDER):
                    chk(f"tab:seeds sd {s}", sv["sd"][s], v[i])

    # ---------------- NUMBERS: independent corpus ---------------------------
    ind = load("independent_eval.json")
    if ind:
        want = {
            "macro-f1": ind["macro_f1_present_classes"],
            "binary auc": ind["binary_auc"],
            "human false-accusation": ind["human_false_accusation_rate"],
            "human f1": ind["per_class"]["human"]["f1"],
            "machine f1": ind["per_class"]["machine"]["f1"],
            "hybrid f1": ind["per_class"]["hybrid"]["f1"],
        }
        for g in ("llama", "codestral", "gemini"):
            want["detected: " + g] = ind["per_generator"][g]["recall_as_nonhuman"]
        for row in T.get("tab:independent", []):
            head = clean(row[0]).lower()
            v = [num(x) for x in row[1:]]
            if not v or v[0] is None:
                continue
            key = next((k for k in want if head.startswith(k)), None)
            if key:
                chk(f"tab:independent {key}", want[key], v[0])

    # ---------------- NUMBERS: semantics-preserving rewrites ----------------
    # The corrected battery, not the superseded one. The original run ranked
    # rewrites by their effect on a whole evaluation set, which confounds a
    # rewrite's per-file damage with how often it fires; the corrected run
    # scores baseline and rewrite on the same altered rows at n = 10,000.
    # Loading the old file here would silently audit the paper against numbers
    # the paper no longer makes.
    ts = load("transform_suite_corrected.json")
    if ts:
        _ren = next((t for t in ts["transforms"]
                     if t["transform"] == "rename_identifiers"), None)
        if _ren and "delta_macro_f1_conditional_parsing" in _ren:
            a = _ren["delta_macro_f1_conditional"]
            b = _ren["delta_macro_f1_conditional_parsing"]
            if abs(a) > 0 and abs(a - b) / abs(a) > 0.10:
                fail.append(f"restricting renaming to parsing rows moves the "
                            f"effect from {a:+.4f} to {b:+.4f}, {abs(a-b)/abs(a):.1%} "
                            "of it; the text says the result survives intact")
    if ts:
        by = {r["transform"]: r for r in ts["transforms"]}
        # Keyed on a distinctive prefix rather than the full label, so shortening
        # a row heading to fit the column width cannot silently stop the row
        # being traced. That happened once and cost fifteen checks.
        alias = {"rename": "rename_identifiers",
                 "strip": "strip_comments",
                 "norm": "whitespace",
                 "insert dead": "dead_code",
                 "compress": "compress_blanks"}
        for row in T.get("tab:transforms", []):
            head = clean(row[0]).lower()
            v = [num(x) for x in row[1:]]
            if head.startswith("(none)"):
                if v and v[0] is not None:
                    chk("tab:transforms baseline", ts["baseline"]["macro_f1"], v[0])
                continue
            key = next((v for k, v in alias.items() if head.startswith(k)), None)
            if not key or key not in by or len(v) < 4:
                continue
            r = by[key]
            chk(f"tab:transforms {key} applied", r["applied_fraction"] * 100, v[0],
                tol=6e-2)
            chk(f"tab:transforms {key} aggregate", r["delta_macro_f1"], v[1])
            chk(f"tab:transforms {key} conditional",
                r["delta_macro_f1_conditional"], v[2])
            # The parse-restricted effect is the one the section reads, so it
            # and the breakage rate that motivates it are both traced.
            if "break_rate" in r:
                chk(f"tab:transforms {key} break", r["break_rate"] * 100, v[3],
                    tol=6e-2)
            if "delta_macro_f1_conditional_parsing" in r:
                chk(f"tab:transforms {key} conditional_parsing",
                    r["delta_macro_f1_conditional_parsing"], v[4])

    # ---------------- NUMBERS: relations between values ---------------------
    # Tracing values to their sources is not enough. The draft claimed 0.277
    # "sits between" 0.222 and 0.250, and all three numbers were individually
    # correct and individually traceable: 0.277 was the larger model's S5
    # score, and the other two were that condition's trivial baselines. What
    # was false was the relation between them, which no amount of value
    # tracing can see. A referee checked it in ten seconds. So does this.
    #
    # Sentences are split on a period that follows a letter or a bracket, so
    # that decimal points do not end a sentence.
    for sent in re.split(r"(?<=[A-Za-z\)])\.\s+", src):
        hit = re.search(r"\b(?:sits|lies|falls|lands)\s+between\b", sent)
        if not hit:
            continue
        flat = " ".join(sent.split())
        pos = re.search(r"\b(?:sits|lies|falls|lands)\s+between\b", flat)
        dec = r"(?<![\d.])\d\.\d+"
        before = [float(x) for x in re.findall(dec, flat[:pos.start()])]
        after = [float(x) for x in re.findall(dec, flat[pos.end():])]
        if not before or len(after) < 2:
            warn.append(f"'between' claim this check could not parse: {flat[:90]}")
            continue
        x, lo, hi = before[-1], min(after[:2]), max(after[:2])
        if not lo <= x <= hi:
            fail.append(f"stated to lie between, but does not: {x} is outside "
                        f"[{lo}, {hi}] -- \"{flat[:100]}\"")

    # ---------------- CORPUS: do not interchange the two builds -------------
    # Every figure belongs to exactly one corpus build, and a sentence that
    # sets a number from one against a number from the other is not a
    # comparison. An earlier draft did this in the introduction, pitting
    # Branch A on the GPU build against DroidDetect on the CPU build, so the
    # abstract and introduction are checked here alongside the result
    # sections. There is deliberately no exemption list: the previous version
    # of this check carried one, and the exemption is what hid the error.
    gpu_only = {"0.9463", "0.8977", "0.8685", "0.5667", "0.4029", "0.2378",
                "0.2766", "196,854", "394,624", "417,645", "493,850",
                "31,083", "56,731"}
    cpu_only = {"0.8555", "0.8253", "0.8344", "0.7254", "0.6888", "144,939",
                "67,197", "0.1867", "0.1771"}

    def section(pattern):
        m = re.search(pattern, src, re.S)
        return m.group(0) if m else ""

    zones = [
        (r"\\begin\{abstract\}.*?\\end\{abstract\}", "abstract", "mixed"),
        (r"\\section\{Introduction\}.*?(?=\\section)", "introduction", "mixed"),
        (r"\\section\{A Detector That Is Strong.*?(?=\\section)", "results (GPU)", "gpu"),
        (r"\\section\{Ruling Out the Alternatives\}.*?(?=\\section)", "controls (GPU)", "gpu"),
        (r"\\subsection\{Row-level contamination\}.*?(?=\\section)", "contamination (CPU)", "cpu"),
    ]
    for pat, name, kind in zones:
        body = section(pat)
        if not body:
            warn.append(f"corpus check could not locate section: {name}")
            continue
        if kind == "gpu":
            stray = sorted(x for x in cpu_only if x in body)
            if stray:
                fail.append(f"CPU-build figures inside {name}: {stray}")
        elif kind == "cpu":
            stray = sorted(x for x in gpu_only if x in body)
            if stray:
                fail.append(f"GPU-build figures inside {name}: {stray}")
        else:
            # Mixed zones may quote both, but every such sentence has to name
            # which build it means, so require the word "build" to be present.
            if (any(x in body for x in cpu_only) and any(x in body for x in gpu_only)
                    and "build" not in body and "training shard" not in body):
                warn.append(f"{name} quotes both builds without naming either")

    # ---------------- STRUCTURE ---------------------------------------------
    if len(re.findall(r"\\begin\{(\w+\*?)\}", src)) != len(re.findall(r"\\end\{(\w+\*?)\}", src)):
        fail.append("unbalanced environments")
    cites = set()
    for c in re.findall(r"\\cite\{([^}]+)\}", src):
        cites |= {x.strip() for x in c.split(",")}
    bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", src))
    if cites - bibs:
        fail.append(f"cited but undefined: {sorted(cites - bibs)}")
    if bibs - cites:
        warn.append(f"uncited bibitems: {sorted(bibs - cites)}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
    if refs - labels:
        fail.append(f"refs with no label: {sorted(refs - labels)}")
    for lab in labels:
        if lab.startswith(("tab:", "fig:", "eq:")) and lab not in refs:
            fail.append(f"float never referenced: {lab}")
    if re.search(r"(?<!-)---(?!-)", src) or "\u2014" in src:
        fail.append("em dash present")
    for stub in ("ef{", "ite{", "extbf{"):
        if re.search(r"(?<![A-Za-z\\])" + stub, src):
            fail.append(f"mangled control word: {stub}")

    # ---------------- STYLE (advisory) --------------------------------------
    body = src.split(r"\maketitle", 1)[1].split(r"\begin{thebibliography}")[0]
    # Strip LaTeX comments before measuring prose, or the section-divider rules
    # are counted as sentences and drag the statistics down.
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in body.split("\n"))
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", body, flags=re.S)
    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", body, flags=re.S)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.S)
    body = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body)
    body = re.sub(r"[{}$\\]", " ", body)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.split()) > 3]
    w = [len(s.split()) for s in sents]

    print("=" * 74)
    print("STYLE (advisory) -- targets from CoDet-M4 and Oedingen et al.")
    print("=" * 74)
    if w:
        print(f"  sentences {len(w)}   mean {st.mean(w):.1f} (target 24-25)   "
              f"median {st.median(w):.0f}   sd {st.pstdev(w):.1f} (target 9-11)")
        print(f"  short <12 words: {100*sum(1 for x in w if x < 12)/len(w):.0f}%  (target ~10%)")
        print(f"  long  >30 words: {100*sum(1 for x in w if x > 30)/len(w):.0f}%  (target ~25%)")
        we = 100 * sum(1 for s in sents if re.search(r"\b[Ww]e\b", s)) / len(sents)
        print(f"  first-person plural: {we:.0f}% of sentences  (target 16-19%)")
        print(f"  openers: {Counter(s.split()[0] for s in sents).most_common(5)}")
        if st.mean(w) < 21:
            warn.append(f"mean sentence length {st.mean(w):.1f} is short vs reference 24-25")
        if st.pstdev(w) < 8:
            warn.append(f"sentence-length sd {st.pstdev(w):.1f} is uniform vs reference 9-11")
        if we > 28:
            warn.append(f"first-person plural in {we:.0f}% of sentences vs reference 16-19%")
    if "--sentences" in sys.argv:
        print("\n  shortest sentences (candidates for subordination):")
        for s in sorted(sents, key=lambda x: len(x.split()))[:12]:
            print(f"    [{len(s.split()):>2}w] {s[:96]}")
    tells = ["Here is", "The point is", "It is worth noting", "Critically,",
             "Notably,", "Importantly,", "delve", "crucial to note", "Let us",
             "It should be noted", "In today's", "paradigm shift"]
    hits = [(t, len(re.findall(re.escape(t), body))) for t in tells]
    hits = [f"{t}:{n}" for t, n in hits if n]
    print(f"  machine-prose markers: {', '.join(hits) if hits else 'none'}")
    if hits:
        warn.append(f"machine-prose markers present: {hits}")

    print("\n" + "=" * 74)
    print(f"NUMBERS / CORPUS / STRUCTURE -- {checked} values traced")
    print("=" * 74)
    for x in fail:
        print("  FAIL  " + x)
    for x in warn:
        print("  warn  " + x)
    if not fail:
        print("  no blocking findings")

    # Export the trace as a ledger. The audit already proves every table cell
    # comes from a report file; writing it out makes that provable to someone
    # who has not run the script.
    led = os.path.join(HERE, "..", "research_state", "numbers_ledger.csv")
    os.makedirs(os.path.dirname(led), exist_ok=True)
    with io.open(led, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "source_value", "paper_value",
                                          "tolerance", "status"])
        w.writeheader()
        w.writerows(ledger)
    print(f"\nledger -> research_state/numbers_ledger.csv ({len(ledger)} rows)")

    print("\nRESULT:", "PASS" if not fail else "FAIL")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
