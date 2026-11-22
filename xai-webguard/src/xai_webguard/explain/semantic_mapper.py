"""Map highlighted evidence to syntactic roles and CWE / CAPEC identifiers."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# role -> (matcher, cwe, capec)
ROLE_RULES = [
    ("tautology / boolean clause", re.compile(r"(\bor\b|\band\b)\s*['\"]?\s*\d+\s*=\s*\d+", re.I), "CWE-89", "CAPEC-66"),
    ("comment terminator", re.compile(r"(--|#|/\*)"), "CWE-89", "CAPEC-66"),
    ("union operator", re.compile(r"\bunion\b", re.I), "CWE-89", "CAPEC-66"),
    ("statement separator", re.compile(r";"), "CWE-89", "CAPEC-66"),
    ("string delimiter", re.compile(r"['\"]"), "CWE-89", "CAPEC-66"),
    ("event-handler attribute", re.compile(r"on[a-z]+\s*=", re.I), "CWE-79", "CAPEC-63"),
    ("script tag", re.compile(r"<\s*script", re.I), "CWE-79", "CAPEC-63"),
    ("script-bearing scheme", re.compile(r"(javascript:|data:|vbscript:)", re.I), "CWE-79", "CAPEC-63"),
    ("dom sink / script call", re.compile(r"(alert|prompt|eval|document\.(cookie|write)|innerhtml)", re.I), "CWE-79", "CAPEC-63"),
    ("markup tag", re.compile(r"<\s*[a-z]", re.I), "CWE-79", "CAPEC-63"),
]


def classify_span(text: str) -> Optional[Dict]:
    for role, rx, cwe, capec in ROLE_RULES:
        if rx.search(text or ""):
            return {"role": role, "cwe": cwe, "capec": capec}
    return None


def map_spans(spans: List[Dict]) -> List[Dict]:
    out = []
    for s in spans:
        info = classify_span(s.get("raw_text") or s.get("token", ""))
        if info:
            out.append({**s, **info})
        else:
            out.append({**s, "role": "unclassified", "cwe": None, "capec": None})
    return out


def dominant_standard(mapped: List[Dict], predicted_class: str) -> Dict:
    """Pick the CWE/CAPEC that best matches the predicted class and evidence."""
    default = {"sqli": ("CWE-89", "CAPEC-66"), "xss": ("CWE-79", "CAPEC-63")}.get(predicted_class, (None, None))
    cwes = [m["cwe"] for m in mapped if m.get("cwe")]
    if cwes:
        top = max(set(cwes), key=cwes.count)
        capec = next((m["capec"] for m in mapped if m.get("cwe") == top), default[1])
        return {"cwe": top, "capec": capec}
    return {"cwe": default[0], "capec": default[1]}
