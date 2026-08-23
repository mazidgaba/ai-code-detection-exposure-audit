"""Check a Kaggle notebook before it costs GPU hours.

Two runs have already been wasted, and both failures were mechanically
detectable before launch. The first was a missing `machine_shape` in the kernel
metadata, which let Kaggle pick a P100 rather than the T4 the workload wants.
The second was `--train-shards 3` in a notebook that asserts it built the
196,854-row split: three shards is the matched-scale corpus, so the run would
have spent an hour building the wrong thing and then tripped its own assertion.

Neither needed a GPU to catch. This runs the checks that would have caught them:

  1. Every `python -m module` in the notebook is importable.
  2. Every flag passed to it actually exists in that module's parser, and every
     choice-valued flag is given one of its allowed values.
  3. Every config file named in the notebook exists.
  4. Every module and config the notebook touches is present in the packaged
     dataset, so the Kaggle session will find it.
  5. The kernel metadata names an accelerator, is private, and lists the code
     dataset as an input.

    python kaggle_runs/preflight.py 5_e1_d2.ipynb
    python kaggle_runs/preflight.py --all
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import pathlib
import re
import subprocess
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
ZIP = ROOT / "kaggle" / "aicd-code.zip"


def notebook_code(path: pathlib.Path) -> str:
    nb = json.load(io.open(path, encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")


def commands(src: str):
    """Every command list in the notebook, with simple variables resolved.

    Resolving them matters more than it sounds. Nearly every command in these
    notebooks passes the config as a variable, `--config CFG`, so a parser that
    only accepts fully literal lists files almost everything under "dynamic,
    not checked" and reports success. That is precisely where the wrong shard
    count hid. So constant string assignments are substituted first, and only
    genuinely loop-built commands are left unchecked.
    """
    consts = dict(re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*$',
                             src, re.M))
    out, dynamic = [], 0
    pattern = r"(?:run\(\s*|cmd\s*=\s*)(\[[^\]]*\])"
    for m in re.finditer(pattern, src, re.S):
        body = m.group(1)
        resolved = body
        for name, val in consts.items():
            resolved = re.sub(r"(?<![\w\"])" + re.escape(name) + r"(?![\w\"])",
                              json.dumps(val), resolved)
        try:
            out.append([str(x) for x in ast.literal_eval(resolved)])
        except (ValueError, SyntaxError):
            dynamic += 1
    return out, dynamic


def check_resume_is_real(src: str) -> list[str]:
    """A notebook that says --resume must be able to resume.

    This is the bug that nearly cost seven hours twice. The trainer's --resume
    looks for a checkpoint in the working directory, and a fresh Kaggle session
    starts with that directory empty. Passing the flag without restoring the
    checkpoint from the previous run's output does not fail: it prints
    "starting fresh" and quietly retrains from epoch 0.

    The corpus has to come back too. Rebuilding is deterministic given the seed
    but not across library versions, and we have measured the same pipeline
    producing 417,645 rows on one day and 417,431 on another. Resuming a
    checkpoint onto a corpus it was never trained on would be silently wrong.
    """
    if "--resume" not in src:
        return []
    fails = []
    if "_ckpt.pt" not in src:
        fails.append("passes --resume but never restores a checkpoint; a fresh "
                     "session would retrain from epoch 0 and say nothing")
    if "splits.parquet" not in src:
        fails.append("passes --resume but never restores splits.parquet; a "
                     "rebuild can differ across library versions")
    return fails


def check_shard_consistency(src: str) -> list[str]:
    """The shard count and the row count the notebook ENFORCES must agree.

    One train shard plus dev and test is 493,850 raw rows, filtering to the
    417,645 of the original GPU build with 196,854 training. Three shards is
    the matched-scale corpus, roughly 545,000 training rows. A notebook that
    downloads three shards and then asserts 196,854 spends an hour building the
    wrong corpus before tripping its own assertion.

    The first version of this check scanned the whole notebook for row counts
    and passed as soon as any of them matched, which meant the comment
    explaining the difference was enough to satisfy it. Prose is not a
    guarantee. Only the executable assertion is, so that is what is read.
    """
    m = re.search(r'"--train-shards",\s*"(\d+)"', src)
    if not m:
        return []
    shards = int(m.group(1))
    expected = {1: 196854, 3: 545000}.get(shards)
    if expected is None:
        return [f"--train-shards {shards} is neither the GPU build (1) nor "
                "the matched-scale build (3)"]

    guards = [int(x) for x in re.findall(r"abs\(\s*rows\s*-\s*(\d+)\s*\)", src)]
    for g in guards:
        if abs(g - expected) > 30000:
            return [f"--train-shards {shards} builds about {expected:,} "
                    f"training rows, but the notebook asserts {g:,}"]

    lower = [int(x) for x in re.findall(r"rows\s*<\s*(\d+)", src)]
    for lo in lower:
        if expected < lo:
            return [f"--train-shards {shards} builds about {expected:,} "
                    f"training rows, but the notebook requires more than {lo:,}"]

    if not guards and not lower:
        return [f"--train-shards {shards} is used but nothing checks the "
                "resulting row count"]
    return []


def parser_help(module: str) -> str | None:
    r = subprocess.run([PY, "-m", module, "--help"], capture_output=True,
                       text=True, cwd=ROOT)
    return (r.stdout + r.stderr) if r.returncode == 0 else None


def check_notebook(path: pathlib.Path) -> list[str]:
    fails: list[str] = []
    src = notebook_code(path)
    cmds, dynamic = commands(src)
    if not cmds and not dynamic:
        fails.append("no commands found at all; the parser may be wrong")

    seen_help: dict[str, str] = {}
    for cmd in cmds:
        if len(cmd) < 2 or cmd[0] != "-m":
            continue
        mod = cmd[1]
        if mod == "pytest":
            continue
        if mod not in seen_help:
            h = parser_help(mod)
            if h is None:
                fails.append(f"{mod}: will not run (--help failed)")
                seen_help[mod] = ""
                continue
            seen_help[mod] = h
        help_text = seen_help[mod]
        if not help_text:
            continue

        i = 2
        while i < len(cmd):
            tok = cmd[i]
            if tok.startswith("--"):
                if tok not in help_text:
                    fails.append(f"{mod}: flag {tok} does not exist")
                else:
                    nxt = cmd[i + 1] if i + 1 < len(cmd) else None
                    if nxt is not None and not nxt.startswith("--"):
                        # If the parser advertises a choice set, the value has
                        # to be in it. This is what catches a plausible but
                        # wrong value.
                        m = re.search(re.escape(tok) + r"\s+\{([^}]*)\}", help_text)
                        if m:
                            allowed = [c.strip() for c in m.group(1).split(",")]
                            if nxt not in allowed:
                                fails.append(
                                    f"{mod}: {tok}={nxt} not in {allowed}")
                        i += 1
            i += 1

    fails.extend(check_resume_is_real(src))
    fails.extend(check_shard_consistency(src))

    for cfg in sorted(set(re.findall(r'"([a-z0-9_]+\.yaml)"', src))):
        if not (ROOT / "aicd" / "configs" / cfg).exists():
            fails.append(f"config {cfg} does not exist")

    # Everything the notebook invokes must be inside the packaged dataset, or
    # the Kaggle session will not find it.
    if ZIP.exists():
        names = set(zipfile.ZipFile(ZIP).namelist())
        for cmd in cmds:
            if len(cmd) > 1 and cmd[0] == "-m" and cmd[1].startswith("aicd."):
                rel = cmd[1].replace(".", "/") + ".py"
                if rel not in names:
                    fails.append(f"{rel} is missing from {ZIP.name}")
        for cfg in sorted(set(re.findall(r'"([a-z0-9_]+\.yaml)"', src))):
            if f"aicd/configs/{cfg}" not in names:
                fails.append(f"aicd/configs/{cfg} is missing from {ZIP.name}")
    else:
        fails.append(f"{ZIP.name} not built; run kaggle/prepare_upload.py")

    print(f"  {len(cmds)} literal commands"
          + (f", {dynamic} built dynamically (not checked)" if dynamic else ""))
    return fails


def check_remote_dataset(slug: str, src: str) -> list[str]:
    """The dataset on Kaggle must contain what the notebook calls.

    Checking the local zip is not the same thing. The zip can be rebuilt and
    the upload forgotten, and each account carries its own copy, so a module
    that exists here and on one account can be absent on another. The session
    would then die on an import it had no way to satisfy.
    """
    r = subprocess.run(["kaggle", "datasets", "files", slug,
                        "--page-size", "500"], capture_output=True, text=True)
    if r.returncode != 0:
        return [f"could not list {slug}: {(r.stderr or r.stdout).strip()[:120]}"]
    listed = r.stdout
    fails = []
    cmds, _ = commands(src)
    wanted = {c[1].replace(".", "/") + ".py"
              for c in cmds if len(c) > 1 and c[0] == "-m"
              and c[1].startswith("aicd.")}
    wanted |= {f"aicd/configs/{c}"
               for c in re.findall(r'"([a-z0-9_]+\.yaml)"', src)}
    for w in sorted(wanted):
        if w not in listed:
            fails.append(f"{slug} does not contain {w}")
    if not fails:
        print(f"  {slug}: all {len(wanted)} required files present")
    return fails


def check_metadata(meta: dict) -> list[str]:
    fails = []
    if not meta.get("machine_shape"):
        fails.append("kernel metadata does not name an accelerator "
                     "(machine_shape); Kaggle will pick one, and it picked a "
                     "P100 last time")
    if meta.get("enable_gpu") is not True:
        fails.append("enable_gpu is not true")
    if meta.get("is_private") is not True:
        fails.append("is_private is not true; this is unpublished work")
    if not meta.get("dataset_sources"):
        fails.append("no dataset_sources; the session will not find aicd/")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--metadata", help="kernel-metadata.json to check too")
    ap.add_argument("--dataset", help="verify this Kaggle dataset actually "
                                      "contains what the notebook calls")
    args = ap.parse_args()

    if args.all:
        # Two-digit prefixes exist (11_e6, 12_e9). The single-character class
        # silently skipped them, so --all reported success on a subset.
        books = sorted(HERE.glob("[0-9]*_*.ipynb"))
    elif args.notebook:
        books = [HERE / args.notebook]
    else:
        ap.error("give a notebook or --all")

    bad = 0
    for b in books:
        print(f"\n=== {b.name} ===")
        if not b.exists():
            print("  MISSING")
            bad += 1
            continue
        fails = check_notebook(b)
        for f in fails:
            print(f"  FAIL  {f}")
        bad += len(fails)
        if not fails:
            print("  ok")

    if args.dataset:
        print(f"\n=== remote dataset {args.dataset} ===")
        src = notebook_code(books[0])
        fails = check_remote_dataset(args.dataset, src)
        for f in fails:
            print(f"  FAIL  {f}")
        bad += len(fails)

    if args.metadata:
        print(f"\n=== {args.metadata} ===")
        fails = check_metadata(json.load(io.open(args.metadata, encoding="utf-8")))
        for f in fails:
            print(f"  FAIL  {f}")
        bad += len(fails)
        if not fails:
            print("  ok")

    print("\n" + ("PREFLIGHT FAILED" if bad else "PREFLIGHT PASSED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
