"""Canonicalisation / de-obfuscation with a character-level offset map.

The offset map is what lets an explanation produced on the cleaned text be
projected back onto the request exactly as it arrived (Section 3.3 of the thesis).

Each transform rewrites ``text`` and keeps ``src`` in step, where ``src[i]`` is
the index, in the *original raw* string, that cleaned character ``i`` came from.
"""
from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import List


@dataclass
class CanonResult:
    text: str                       # cleaned, case-preserving canonical text
    src: List[int]                  # src[i] -> index in the raw input
    layers: int = 0                 # number of decoding rounds that changed the text
    raw: str = ""                   # the original input

    def raw_span(self, start: int, end: int):
        """Map a [start, end) span in canonical text back to the raw request."""
        if not self.src or start >= len(self.src):
            return (0, 0)
        end = min(end, len(self.src))
        idxs = self.src[start:end]
        if not idxs:
            return (0, 0)
        return (min(idxs), max(idxs) + 1)


_ENTITY_RE = re.compile(r"&(#x?[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]{1,31});")
_PCT_RE = re.compile(r"%([0-9a-fA-F]{2})")
_UNI_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})")
_WS_RE = re.compile(r"\s+")


def _rewrite(text: str, src: List[int], pattern: re.Pattern, repl):
    """Apply ``pattern`` to ``text`` carrying ``src`` along.

    ``repl(match)`` returns the replacement string; every replacement character
    is mapped back to the first raw index covered by the match.
    """
    out_chars: List[str] = []
    out_src: List[int] = []
    pos = 0
    for m in pattern.finditer(text):
        # unchanged slice before the match
        for i in range(pos, m.start()):
            out_chars.append(text[i])
            out_src.append(src[i])
        rep = repl(m)
        anchor = src[m.start()] if m.start() < len(src) else (src[-1] if src else 0)
        for c in rep:
            out_chars.append(c)
            out_src.append(anchor)
        pos = m.end()
    for i in range(pos, len(text)):
        out_chars.append(text[i])
        out_src.append(src[i])
    return "".join(out_chars), out_src


def _decode_percent(text, src):
    return _rewrite(text, src, _PCT_RE, lambda m: _safe_chr(int(m.group(1), 16)))


def _decode_entities(text, src):
    def repl(m):
        try:
            return html.unescape(m.group(0))
        except Exception:
            return m.group(0)
    return _rewrite(text, src, _ENTITY_RE, repl)


def _decode_unicode(text, src):
    def repl(m):
        hexpart = m.group(1) or m.group(2)
        return _safe_chr(int(hexpart, 16))
    return _rewrite(text, src, _UNI_RE, repl)


def _safe_chr(code: int) -> str:
    try:
        ch = chr(code)
        # keep control characters out of the canonical form
        return ch if ch.isprintable() or ch in "\t\n\r " else " "
    except (ValueError, OverflowError):
        return " "


def _collapse_ws(text, src):
    def repl(_m):
        return " "
    return _rewrite(text, src, _WS_RE, repl)


def canonicalize(raw: str, max_rounds: int = 5, max_len: int = 4096) -> CanonResult:
    """Decode layered encodings and normalise whitespace, keeping an offset map.

    Case is preserved so the result can feed both the transformer and the
    dashboard display; the feature extractor lowercases internally.
    """
    if raw is None:
        raw = ""
    raw = str(raw)
    text = raw
    src = list(range(len(raw)))
    layers = 0
    for _ in range(max_rounds):
        before = text
        text, src = _decode_percent(text, src)
        text, src = _decode_entities(text, src)
        text, src = _decode_unicode(text, src)
        if text == before:
            break
        layers += 1
    text, src = _collapse_ws(text, src)
    text = text.strip()
    # keep src aligned with the stripped text
    lead = len(text) and 0
    # recompute src for strip (strip removes leading/trailing ws)
    full, fullsrc = _collapse_ws(*_decode_unicode(*_decode_entities(*_decode_percent(raw, list(range(len(raw)))))))
    stripped = full.strip()
    offset = full.find(stripped) if stripped else 0
    src = fullsrc[offset: offset + len(stripped)]
    text = stripped
    if len(text) > max_len:
        text = text[:max_len]
        src = src[:max_len]
    return CanonResult(text=text, src=src, layers=layers, raw=raw)
