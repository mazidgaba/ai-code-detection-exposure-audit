"""How much of branch B's signal is merely code formatting?

Oedingen et al. ran every sample through Black and lost only ~4 accuracy
points on embeddings (~8 on the white-box features), which is the reassuring
answer: formatting is a real signal but not the main one. If your delta is
much larger, your detector is a whitespace detector wearing a hat.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from aicd import config as C

EXT = {"python": ".py", "java": ".java", "cpp": ".cpp", "c": ".c",
       "csharp": ".cs", "go": ".go", "javascript": ".js"}


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def format_code(code: str, language: str, tmp: Path) -> str:
    """Return formatted source, or the original when no formatter is available."""
    try:
        if language == "python" and have("black"):
            p = tmp / f"s{EXT[language]}"
            p.write_text(code, encoding="utf-8")
            subprocess.run(["black", "-q", "--fast", str(p)], capture_output=True, timeout=20)
            return p.read_text(encoding="utf-8")
        if language == "go" and have("gofmt"):
            r = subprocess.run(["gofmt"], input=code, capture_output=True, text=True, timeout=20)
            return r.stdout or code
        if language in ("cpp", "c", "csharp", "java", "javascript") and have("clang-format"):
            r = subprocess.run(["clang-format"], input=code, capture_output=True, text=True, timeout=20)
            return r.stdout or code
    except Exception:
        pass
    return code


def normalize_whitespace(code: str) -> str:
    """Fallback when no external formatter exists: collapse the formatting
    degrees of freedom the white-box features key on."""
    lines = [l.rstrip() for l in code.replace("\t", "    ").split("\n")]
    return "\n".join(l for l in lines if l.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--use-external", action="store_true",
                    help="shell out to black/gofmt/clang-format when installed")
    args = ap.parse_args()
    cfg = C.load(args.config)
    d = C.ROOT / cfg.data.cache_dir

    df = pd.read_parquet(d / "splits.parquet")
    tools = {t: have(t) for t in ("black", "gofmt", "clang-format")}
    print(f"[ablation] external formatters available: {tools}")
    mode = "external" if (args.use_external and any(tools.values())) else "whitespace-normalize"
    print(f"[ablation] mode: {mode}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if mode == "external":
            df["code"] = [format_code(c, l, tmp) for c, l in zip(df["code"], df["language"])]
        else:
            df["code"] = df["code"].map(normalize_whitespace)

    dest = d / "splits_formatted.parquet"
    df.to_parquet(dest, index=False)
    print(f"[done] {dest.name} written. Now run:")
    print("  python -m aicd.features.build --input splits_formatted.parquet")
    print("  python -m aicd.models.xgb   # compare macro-F1 against the unformatted run")


if __name__ == "__main__":
    main()
