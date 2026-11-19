"""Text clean-up and one list parser per list-column format."""
from __future__ import annotations

import ast
import html
import json
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_BBCODE_RE = re.compile(r"\[/?[a-zA-Z0-9_]+(?:=[^\]]*)?\]")


def strip_html(text: str | None) -> str:
    """Strip HTML tags, unescape entities, normalize whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def normalize_ws(text: str | None) -> str:
    return _WS_RE.sub(" ", text).strip() if text else ""


def clean_review(text: str | None) -> str:
    """Review clean-up: unescape entities, drop Steam BBCode tags ([b], [h1], ...), normalize whitespace."""
    if not text:
        return ""
    return normalize_ws(_BBCODE_RE.sub(" ", html.unescape(text)))


# --- list-column parsers: one per format, no guessing -------------------------------------------


def parse_semicolon(value: str | None) -> list[str]:
    """'Action;FPS' (nikdavis steam.csv)."""
    return _dedup([p.strip() for p in value.split(";")]) if value else []


def parse_comma(value: str | None) -> list[str]:
    """'Action, FPS'."""
    return _dedup([p.strip() for p in value.split(",")]) if value else []


def parse_json(value: str | None) -> list[str]:
    """'["Action", "FPS"]' (double quotes)."""
    if not value:
        return []
    parsed = json.loads(value)
    return _dedup([str(p).strip() for p in parsed])


def parse_pylist(value: str | None) -> list[str]:
    """"['Action', 'FPS']" (Python repr)."""
    if not value:
        return []
    parsed = ast.literal_eval(value)
    return _dedup([str(p).strip() for p in parsed])


_PARSERS = {"semicolon": parse_semicolon, "comma": parse_comma, "json": parse_json, "pylist": parse_pylist}


def parse_list(value: str | None, fmt: str) -> list[str]:
    """Parse a list cell using the format named in the config. Raises on malformed cells."""
    if fmt not in _PARSERS:
        raise ValueError(f"unknown list format {fmt!r}; expected one of {sorted(_PARSERS)}")
    if value is None or (isinstance(value, float) and value != value):
        return []
    value = str(value).strip()
    if value in ("", "[]", "nan", "None"):
        return []
    return _PARSERS[fmt](value)


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
