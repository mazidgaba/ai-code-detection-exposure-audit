"""Extend the corpus with your own generations.

Four modes, matching the four labels. The adversarial mode is the
highest-value part of this file: published detectors score ~0.10 recall on
evasion attempts without adversarial training data and ~0.92 with it, so a few
thousand adversarial samples matter more than tens of thousands of ordinary
generations.

Backends: an OpenAI-compatible HTTP endpoint (works with vLLM, Ollama's
OpenAI shim, or a hosted API). Nothing here is called during training -- this
is a corpus-building tool you run deliberately.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

import pandas as pd

from aicd import config as C

# ------------------------------------------------------------------ prompts
# 10 templates so the detector does not overfit one prompting style. A corpus
# built from a single template teaches the model to detect that template.
MACHINE_TEMPLATES = [
    "Write a {lang} solution to the following problem. Return code only.\n\n{spec}",
    "Implement this in {lang}. No explanation, code only.\n\n{spec}",
    "You are a {lang} programmer. Solve this and return only the code.\n\n{spec}",
    "{spec}\n\nProvide your answer as {lang} code. Answer:",
    "Complete the task below in {lang}. Output nothing but the source.\n\n{spec}",
    "Task: {spec}\nLanguage: {lang}\nRespond with code only, no commentary.",
    "Write clean, idiomatic {lang} for this requirement:\n\n{spec}",
    "Produce a working {lang} implementation. Code only.\n\n{spec}",
    "As a senior {lang} engineer, implement:\n\n{spec}\n\nReturn only code.",
    "Solve in {lang}. Do not explain. Do not use markdown fences.\n\n{spec}",
]

HYBRID_REWRITE = (
    "You are an experienced {lang} programmer. Given the code snippet, rewrite "
    "it so that it does the same thing, but is written differently.\n\n"
    "Code snippet:\n{code}\n\nReturn code only."
)
HYBRID_INFILL = (
    "Given the following code, fill in the <add your code here> lines. You can "
    "add more than a single line for each of these blanks.\n\n"
    "Code snippet:\n{code}\n\nReturn code only."
)

# Evasion prompts. These mirror the strategies the AAAI-24 robustness study and
# the Droid adversarial split used, which together drove bypass rates as high
# as 0.75 against detectors with no adversarial training data.
ADVERSARIAL_TEMPLATES = [
    "Write a {lang} answer to this in a casual, human-like style with high "
    "randomness, as if a beginner or hobbyist wrote it quickly. Make it less "
    "formal, like something someone jotted down.\n\n{spec}",
    "Solve this in {lang}. Replace all variable names with single characters. "
    "Return code only.\n\n{spec}",
    "Solve this in {lang}, but write it the way a tired developer would at 2am: "
    "inconsistent spacing, a couple of terse names, no docstrings.\n\n{spec}",
    "Write {lang} code for this. Match the style of this human-written sample "
    "as closely as you can:\n\n{example}\n\nTask: {spec}\nReturn code only.",
    "Solve this in {lang}. Do not add comments. Do not add blank lines between "
    "logical blocks. Use short names.\n\n{spec}",
]

FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)

# --------------------------------------------------------- post-hoc mutations
HUMAN_NAMES = ["tmp", "res", "val", "buf", "acc", "cnt", "idx", "arr", "ans",
               "cur", "prev", "nxt", "aux", "lst", "dct", "x1", "n2", "t"]


def strip_comments(code: str, lang: str) -> str:
    if lang == "python":
        code = re.sub(r"#.*?$", "", code, flags=re.M)
        code = re.sub(r'("""|\'\'\').*?\1', "", code, flags=re.S)
    else:
        code = re.sub(r"//.*?$", "", code, flags=re.M)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return "\n".join(l for l in code.split("\n") if l.strip())


def rename_identifiers(code: str, rng: random.Random, rate: float = 0.6) -> str:
    idents = sorted({m for m in re.findall(r"\b[a-z_][a-z_0-9]{3,}\b", code)})
    idents = [i for i in idents if i not in {"self", "return", "import", "print", "range", "class"}]
    rng.shuffle(idents)
    for name in idents[: int(len(idents) * rate)]:
        code = re.sub(rf"\b{re.escape(name)}\b", rng.choice(HUMAN_NAMES), code)
    return code


def perturb_spacing(code: str, rng: random.Random) -> str:
    out = []
    for line in code.split("\n"):
        if not line.strip():
            if rng.random() < 0.6:
                continue
        elif rng.random() < 0.15:
            line = line.rstrip() + " " * rng.randint(1, 3)
        out.append(line)
    return "\n".join(out)


MUTATIONS = {
    "strip_comments": lambda c, l, r: strip_comments(c, l),
    "rename_identifiers": lambda c, l, r: rename_identifiers(c, r),
    "perturb_spacing": lambda c, l, r: perturb_spacing(c, r),
}


# ------------------------------------------------------------------- backend
def call_llm(prompt: str, model: str, base_url: str, api_key: str | None,
             temperature: float, max_tokens: int = 1024, retries: int = 3) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"  [warn] generation failed: {type(e).__name__}")
                return None
            time.sleep(2 ** attempt)
    return None


def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    m = FENCE.search(text)
    code = m.group(1) if m else text
    code = code.strip()
    return code if len(code) >= 40 else None


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--mode", required=True,
                    choices=["machine", "hybrid", "adversarial", "mutate"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--base-url", default=os.environ.get(
        "AICD_LLM_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--api-key", default=os.environ.get("AICD_LLM_API_KEY"))
    ap.add_argument("--language", default="python")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = C.load(args.config)
    rng = random.Random(cfg.project.seed)
    df = pd.read_parquet(C.ROOT / cfg.data.cache_dir / "splits.parquet")

    # Seed pool: human code, restricted to training-eligible rows so generated
    # samples can never be derived from an evaluation slice.
    pool = df[(df["label"] == 0) & (df["language"] == args.language) &
              (df["split"] == "train")]
    if pool.empty:
        raise SystemExit(f"no human {args.language} training rows to seed from")
    pool = pool.sample(n=min(args.n, len(pool)), random_state=cfg.project.seed)

    rows = []
    for i, (_, r) in enumerate(pool.iterrows()):
        code, spec = r["code"], r["code"][:600]
        temp = round(rng.uniform(0.4, 1.0), 2)

        if args.mode == "mutate":
            # No LLM needed: post-hoc evasion applied to existing machine code.
            src = df[(df["label"] == 1) & (df["language"] == args.language) &
                     (df["split"] == "train")].sample(1, random_state=i)
            name = rng.choice(list(MUTATIONS))
            out = MUTATIONS[name](src.iloc[0]["code"], args.language, rng)
            rows.append({"code": out, "label": 3, "language": args.language,
                         "generator": f"mutation:{name}", "template_id": name,
                         "temperature": None, "mode": "mutate"})
            continue

        if args.mode == "machine":
            tid = rng.randrange(len(MACHINE_TEMPLATES))
            prompt = MACHINE_TEMPLATES[tid].format(lang=args.language, spec=spec)
            label = 1
        elif args.mode == "hybrid":
            tid = rng.randrange(2)
            tmpl = HYBRID_REWRITE if tid == 0 else HYBRID_INFILL
            body = code
            if tid == 1:
                lines = code.split("\n")
                if len(lines) > 6:
                    cut = len(lines) // 2
                    body = "\n".join(lines[:cut] + ["<add your code here>"] + lines[cut + 3:])
            prompt = tmpl.format(lang=args.language, code=body)
            label = 2
        else:
            tid = rng.randrange(len(ADVERSARIAL_TEMPLATES))
            prompt = ADVERSARIAL_TEMPLATES[tid].format(
                lang=args.language, spec=spec, example=code[:800])
            label = 3

        out = extract_code(call_llm(prompt, args.model, args.base_url,
                                    args.api_key, temp))
        if out:
            rows.append({"code": out, "label": label, "language": args.language,
                         "generator": args.model, "template_id": tid,
                         "temperature": temp, "mode": args.mode})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pool)} requested, {len(rows)} kept", flush=True)

    if not rows:
        raise SystemExit("nothing generated -- check --base-url and --model")

    out_df = pd.DataFrame(rows)
    dest = C.ROOT / cfg.data.cache_dir / (args.out or f"generated_{args.mode}_{args.language}.parquet")
    out_df.to_parquet(dest, index=False)
    print(f"\n[done] {dest.name}: {len(out_df):,} rows")
    print(out_df["template_id"].value_counts().to_string())
    print("\nNext: re-run data/filter.py and data/splits.py to fold these in.")
    print("Generated rows must go through the same quality gate as the corpus.")


if __name__ == "__main__":
    main()
