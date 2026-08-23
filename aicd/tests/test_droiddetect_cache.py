"""The cache key must distinguish the published model sizes.

A run of DroidDetect-Large once reused DroidDetect-Base's cached probability
arrays, because the cache filename omitted the model size while the output
filename included it. The result was a report full of base numbers under the
large model's name: silently wrong, and entirely plausible looking.

The check reads the scoring loop specifically rather than the whole file. An
earlier version of this test searched globally for the assignment of whatever
variable the cache line used, and broke when an unrelated local called `tag`
appeared earlier in the module. A test that fails for the wrong reason is
barely better than no test.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "models" / "droiddetect_baseline.py"
CACHE_LINE = r'cached = art / f"proba_droiddetect\{(\w+)\}_\{s\}\.npy"'
SAVE_LINE = r'np\.save\(art / f"proba_droiddetect\{(\w+)\}_\{s\}\.npy"'


def scoring_loop() -> str:
    """The body of main() from the slice loop to the report name."""
    text = SRC.read_text(encoding="utf-8")
    start = text.index("proba_by_slice, art = {}")
    end = text.index("res = M.evaluate_all", start)
    return text[start:end]


def assignment_of(var: str, body: str) -> str:
    """The last assignment of `var` before it is used, within this body."""
    hits = re.findall(rf"^\s*{var} = (.+)$", body, re.M)
    assert hits, f"no assignment of {var} in the scoring loop"
    return hits[0]


def test_cache_key_includes_model_size():
    body = scoring_loop()
    m = re.search(CACHE_LINE, body)
    assert m, "cache read line not found in the scoring loop"
    assert "size" in assignment_of(m.group(1), body), (
        "the cache key must include the model size, or large reuses base's "
        "arrays and reports them under its own name")


def test_read_and_write_keys_agree():
    """Whatever names the cached file must also name the saved file."""
    body = scoring_loop()
    r, w = re.search(CACHE_LINE, body), re.search(SAVE_LINE, body)
    assert r and w, "cache read/write lines not found in the scoring loop"
    for var in (r.group(1), w.group(1)):
        assert "size" in assignment_of(var, body), (
            f"{var} must include the model size")


def test_size_is_empty_for_base():
    """Base keeps its historical filenames, so published paths do not move."""
    body = scoring_loop()
    assert re.search(r'size = "" if args\.model == "base"', body), (
        "base must map to an empty suffix, or every existing artefact path "
        "changes and previously published numbers become unreachable")
