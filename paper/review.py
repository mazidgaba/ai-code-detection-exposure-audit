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

    python paper/review.py
"""
from __future__ import annotations

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
    src = io.open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
    T = tables(src)
    fail, warn = [], []
    checked = 0

    def chk(label, got, want, tol=6e-4):
        nonlocal checked
        checked += 1
        if got is None or want is None:
            fail.append(f"{label}: unresolved (file={got} paper={want})")
        elif abs(got - want) > tol:
            fail.append(f"{label}: file={got:.4f} paper={want}")

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
    ts = load("transform_suite_droiddetect.json")
    if ts:
        by = {r["transform"]: r for r in ts["transforms"]}
        alias = {"rename identifiers": "rename_identifiers",
                 "strip comments": "strip_comments",
                 "normalise whitespace": "whitespace",
                 "insert dead code": "dead_code",
                 "compress blank lines": "compress_blanks"}
        for row in T.get("tab:transforms", []):
            head = clean(row[0]).lower()
            v = [num(x) for x in row[1:]]
            if head.startswith("(none)"):
                if v and v[0] is not None:
                    chk("tab:transforms baseline", ts["baseline"]["macro_f1"], v[0])
                continue
            key = alias.get(head)
            if not key or key not in by or len(v) < 4:
                continue
            r = by[key]
            chk(f"tab:transforms {key} applied", r["applied_fraction"] * 100, v[0],
                tol=6e-2)
            chk(f"tab:transforms {key} f1", r["macro_f1"], v[1])
            chk(f"tab:transforms {key} delta", r["delta_macro_f1"], v[2])
            chk(f"tab:transforms {key} dauc", r["delta_auc"], v[3])

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
    print("\nRESULT:", "PASS" if not fail else "FAIL")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
