"""Semantics-preserving rewrites of source text, for robustness testing.

The evasion result in the manuscript rests on one transformation, whitespace
normalisation, and one transformation is a thin basis for a claim about
robustness. These are the obvious neighbours: each is something a formatter, a
minifier, a linter or a bored undergraduate could apply without understanding
the code, and none of them changes what the program does.

Deliberately shallow. These are regex rewrites, not parsers, so they will not
handle every construct in nine languages correctly. They are chosen so that
their failure mode is to leave text unchanged rather than to corrupt it, which
keeps the measurement conservative: a transformation that silently no-ops
understates the attack rather than inventing one. `applied_fraction` reports
how often each actually changed the input, so a no-op cannot masquerade as a
negative result.

    from aicd.models.transforms import TRANSFORMS
    TRANSFORMS["strip_comments"](code, "python")
"""
from __future__ import annotations

import hashlib
import re

from aicd.models.formatter_ablation import normalize_whitespace

# Languages whose line comments start with # rather than //.
HASH_COMMENT = {"python", "ruby", "shell", "bash", "perl", "r"}

# Reserved words we must never rename. Not exhaustive per language, but the
# union is safe: renaming is skipped for anything in this set.
KEYWORDS = {
    "if", "else", "elif", "for", "while", "return", "def", "class", "import",
    "from", "as", "try", "except", "finally", "raise", "with", "lambda",
    "pass", "break", "continue", "global", "nonlocal", "assert", "yield",
    "del", "in", "is", "not", "and", "or", "none", "true", "false", "self",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple", "len",
    "range", "print", "public", "private", "protected", "static", "void",
    "new", "this", "null", "func", "var", "let", "const", "package", "type",
    "struct", "interface", "map", "chan", "go", "defer", "switch", "case",
    "default", "do", "double", "long", "short", "char", "byte", "boolean",
    "extends", "implements", "throws", "throw", "catch", "final", "abstract",
    "using", "namespace", "template", "typename", "auto", "unsigned",
    "include", "define", "ifdef", "endif", "main", "printf", "std", "string",
}

IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def strip_comments(code: str, lang: str = "") -> str:
    """Remove line comments and the common block-comment forms."""
    out = []
    hash_style = lang.lower() in HASH_COMMENT
    for line in code.split("\n"):
        # Only strip a comment marker that is not inside an obvious string.
        if hash_style:
            q = line.find("#")
        else:
            q = line.find("//")
        if q >= 0 and line.count('"', 0, q) % 2 == 0 and line.count("'", 0, q) % 2 == 0:
            line = line[:q].rstrip()
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return "\n".join(l for l in text.split("\n") if l.strip())


def rename_identifiers(code: str, lang: str = "") -> str:
    """Rename user identifiers to opaque but stable names.

    Stability matters: the same identifier must map to the same replacement
    within a file, or the code stops being equivalent. Hashing the name gives
    that without carrying a counter around.
    """
    seen: dict[str, str] = {}

    def sub(m):
        name = m.group(0)
        if name.lower() in KEYWORDS:
            return name
        if name not in seen:
            h = hashlib.md5(name.encode()).hexdigest()[:6]
            seen[name] = f"v_{h}"
        return seen[name]

    return IDENT.sub(sub, code)


def insert_dead_code(code: str, lang: str = "") -> str:
    """Append an unreachable no-op block in the file's comment style."""
    hash_style = lang.lower() in HASH_COMMENT
    tail = ("\n# _unused = 0\n" if hash_style else "\n// int _unused = 0;\n")
    return code.rstrip("\n") + "\n" + tail


def compress_blank_lines(code: str, lang: str = "") -> str:
    """Collapse runs of blank lines to one, leaving indentation untouched.

    Weaker than full normalisation, and included to separate the contribution
    of blank-line structure from that of indentation.
    """
    return re.sub(r"\n{3,}", "\n\n", code)


TRANSFORMS = {
    "whitespace": lambda c, l="": normalize_whitespace(c),
    "strip_comments": strip_comments,
    "rename_identifiers": rename_identifiers,
    "compress_blanks": compress_blank_lines,
    "dead_code": insert_dead_code,
}


def applied_fraction(codes, name: str, langs=None) -> float:
    """Share of inputs the transformation actually altered."""
    fn = TRANSFORMS[name]
    langs = langs if langs is not None else [""] * len(codes)
    n = sum(1 for c, l in zip(codes, langs) if fn(c, l) != c)
    return n / max(1, len(codes))
