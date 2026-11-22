"""Named, human-readable engineered features (Section 3.4 / Table 3.1).

Every feature has a plain-language name so that a SHAP attribution can be read
without a lookup table. ``FEATURE_NAMES`` is the canonical ordering used by the
tree model and by the explanation layer.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from .canonicalize import CanonResult, canonicalize

SQL_KEYWORDS = [
    "select", "union", "insert", "update", "delete", "drop", "from", "where",
    "or", "and", "having", "group", "order", "exec", "execute", "declare",
    "cast", "convert", "waitfor", "sleep", "benchmark", "information_schema",
    "table_name", "column_name", "concat", "char", "hex", "load_file",
]
SQL_OPERATORS = ["=", "<", ">", "!=", "<>", "||", "--", "#", "/*", "*/"]
HTML_TAGS = [
    "script", "img", "svg", "iframe", "body", "input", "a", "div", "video",
    "audio", "object", "embed", "style", "form", "math", "details", "marquee",
]
EVENT_HANDLERS = [
    "onerror", "onload", "onclick", "onmouseover", "onfocus", "onstart",
    "onanimationstart", "ontoggle", "onpointerover", "onbegin",
]
SCRIPT_SCHEMES = ["javascript:", "data:", "vbscript:"]
DOM_SINKS = ["alert", "prompt", "confirm", "eval", "document.cookie",
             "document.write", "innerhtml", "fromcharcode", "settimeout"]

# ordering matters: the tree model and explainer rely on it
FEATURE_NAMES: List[str] = [
    # lexical
    "length", "single_quote_count", "double_quote_count", "bracket_count",
    "angle_bracket_count", "semicolon_count", "comment_marker_count",
    "special_char_ratio", "char_entropy",
    # sql vocabulary
    "sql_keyword_count", "sql_operator_count", "union_present", "tautology_pattern",
    # markup / script vocabulary
    "html_tag_count", "event_handler_count", "script_scheme_count", "dom_sink_count",
    # structural
    "quote_balance", "tag_balance", "comment_terminator", "statement_separator",
    "nesting_depth",
    # encoding
    "decoding_layers", "mixed_encoding", "non_ascii_ratio",
    # request context
    "field_is_query", "field_is_body", "field_is_cookie", "field_is_header",
]

_TAUTOLOGY_RE = re.compile(r"(\bor\b|\band\b)\s*['\"]?\s*\d+\s*=\s*\d+|['\"]\s*(or|and)\s*['\"]?\d", re.I)
_WORD_RE = re.compile(r"[a-zA-Z_]+")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_features(canon: CanonResult, field: str = "query") -> Dict[str, float]:
    """Compute the feature dictionary for one canonicalised input."""
    t = canon.text
    low = t.lower()
    n = max(len(t), 1)
    words = _WORD_RE.findall(low)
    wordset = set(words)

    specials = sum(1 for c in t if not c.isalnum() and not c.isspace())
    non_ascii = sum(1 for c in t if ord(c) > 127)

    f: Dict[str, float] = {}
    # lexical
    f["length"] = float(len(t))
    f["single_quote_count"] = float(t.count("'"))
    f["double_quote_count"] = float(t.count('"'))
    f["bracket_count"] = float(sum(t.count(c) for c in "()[]{}"))
    f["angle_bracket_count"] = float(t.count("<") + t.count(">"))
    f["semicolon_count"] = float(t.count(";"))
    f["comment_marker_count"] = float(low.count("--") + t.count("#") + low.count("/*"))
    f["special_char_ratio"] = specials / n
    f["char_entropy"] = _entropy(t)
    # sql vocabulary
    f["sql_keyword_count"] = float(sum(1 for w in words if w in SQL_KEYWORDS))
    f["sql_operator_count"] = float(sum(low.count(op) for op in SQL_OPERATORS))
    f["union_present"] = 1.0 if "union" in wordset else 0.0
    f["tautology_pattern"] = 1.0 if _TAUTOLOGY_RE.search(t) else 0.0
    # markup / script vocabulary
    f["html_tag_count"] = float(sum(low.count("<" + tag) for tag in HTML_TAGS))
    f["event_handler_count"] = float(sum(low.count(h) for h in EVENT_HANDLERS))
    f["script_scheme_count"] = float(sum(low.count(s) for s in SCRIPT_SCHEMES))
    f["dom_sink_count"] = float(sum(low.count(d) for d in DOM_SINKS))
    # structural
    f["quote_balance"] = float(abs(t.count("'") % 2) + abs(t.count('"') % 2))
    f["tag_balance"] = float(abs(t.count("<") - t.count(">")))
    f["comment_terminator"] = 1.0 if ("--" in low or low.rstrip().endswith("#") or "/*" in low) else 0.0
    f["statement_separator"] = 1.0 if ";" in t else 0.0
    f["nesting_depth"] = float(_max_nesting(t))
    # encoding
    f["decoding_layers"] = float(canon.layers)
    f["mixed_encoding"] = 1.0 if (canon.layers > 0 and ("%" in canon.raw and "&" in canon.raw)) else 0.0
    f["non_ascii_ratio"] = non_ascii / n
    # request context
    field = (field or "query").lower()
    f["field_is_query"] = 1.0 if field in ("query", "url", "path") else 0.0
    f["field_is_body"] = 1.0 if field in ("body", "form", "post") else 0.0
    f["field_is_cookie"] = 1.0 if field == "cookie" else 0.0
    f["field_is_header"] = 1.0 if field == "header" else 0.0
    return f


def _max_nesting(t: str) -> int:
    depth = cur = 0
    for c in t:
        if c in "([{":
            cur += 1
            depth = max(depth, cur)
        elif c in ")]}":
            cur = max(0, cur - 1)
    return depth


def features_to_vector(f: Dict[str, float]) -> List[float]:
    return [f[name] for name in FEATURE_NAMES]


def featurize(raw: str, field: str = "query") -> List[float]:
    """Convenience: raw string -> feature vector in FEATURE_NAMES order."""
    return features_to_vector(extract_features(canonicalize(raw), field))
