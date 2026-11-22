"""
Feature Extraction Module for XAI-WebGuard
Implements:
1. TF-IDF Vectorization customized for Web Attack Payloads (preserving syntax tokens).
2. Domain-Specific Cybersecurity Feature Extractor (keywords, tags, operators, length, entropy).
3. Hybrid Feature Pipeline combining TF-IDF and Domain Features with human-readable names.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Union
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


# Domain vocabulary definitions
SQL_KEYWORDS = [
    "select", "union", "insert", "update", "delete", "drop", "from", "where",
    "or", "and", "having", "group", "order", "exec", "execute", "declare",
    "cast", "convert", "waitfor", "sleep", "benchmark", "information_schema",
    "table_name", "column_name", "concat", "char", "hex", "load_file",
    "schema", "database", "version", "truncate", "create", "alter"
]

SQL_OPERATORS = ["=", "<", ">", "!=", "<>", "||", "--", "#", "/*", "*/", "@@"]

HTML_TAGS = [
    "script", "img", "svg", "iframe", "body", "input", "a", "div", "video",
    "audio", "object", "embed", "style", "form", "math", "details", "marquee"
]

EVENT_HANDLERS = [
    "onerror", "onload", "onclick", "onmouseover", "onfocus", "onstart",
    "onanimationstart", "ontoggle", "onpointerover", "onbegin", "onloadstart"
]

SCRIPT_SCHEMES = ["javascript:", "data:", "vbscript:"]

DOM_SINKS = [
    "alert", "prompt", "confirm", "eval", "document.cookie",
    "document.write", "innerhtml", "fromcharcode", "settimeout", "window.location"
]

TAUTOLOGY_PATTERN = re.compile(
    r"(\bor\b|\band\b)\s*['\"]?\s*\d+\s*=\s*\d+|['\"]\s*(or|and)\s*['\"]?\d|1\s*=\s*1|'a'\s*=\s*'a'",
    re.IGNORECASE
)


def calculate_entropy(text: str) -> float:
    """Calculates Shannon entropy of the character distribution in a string."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts handcrafted, explainable cybersecurity domain features from raw HTTP requests:
    - Lexical (length, special character count & ratio, Shannon entropy)
    - SQL attack markers (keyword count, operator count, tautology patterns)
    - XSS attack markers (HTML tag count, JS event handlers, script schemes, DOM sinks)
    - Structural syntax indicators (quote imbalance, tag imbalance, comment terminators)
    """

    FEATURE_NAMES = [
        "domain_length",
        "domain_special_char_count",
        "domain_special_char_ratio",
        "domain_char_entropy",
        "domain_sql_keyword_count",
        "domain_sql_operator_count",
        "domain_sql_tautology_flag",
        "domain_html_tag_count",
        "domain_event_handler_count",
        "domain_script_scheme_count",
        "domain_dom_sink_count",
        "domain_quote_imbalance",
        "domain_tag_imbalance",
        "domain_comment_terminator_flag",
        "domain_semicolon_count"
    ]

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        features = []
        for text in X:
            t = str(text) if text is not None else ""
            low = t.lower()
            n = max(len(t), 1)

            # Lexical features
            length = float(len(t))
            specials = float(sum(1 for c in t if not c.isalnum() and not c.isspace()))
            special_ratio = specials / n
            entropy = calculate_entropy(t)

            # SQL features
            sql_kw_cnt = float(sum(1 for kw in SQL_KEYWORDS if re.search(r"\b" + kw + r"\b", low)))
            sql_ops_cnt = float(sum(low.count(op) for op in SQL_OPERATORS))
            tautology = 1.0 if TAUTOLOGY_PATTERN.search(low) else 0.0

            # XSS features
            html_tags_cnt = float(sum(low.count("<" + tag) for tag in HTML_TAGS))
            event_cnt = float(sum(low.count(ev) for ev in EVENT_HANDLERS))
            scheme_cnt = float(sum(low.count(sc) for sc in SCRIPT_SCHEMES))
            sink_cnt = float(sum(low.count(sink) for sink in DOM_SINKS))

            # Structural syntax features
            quote_imbal = float((t.count("'") % 2) + (t.count('"') % 2))
            tag_imbal = float(abs(t.count("<") - t.count(">")))
            comment_term = 1.0 if ("--" in low or low.rstrip().endswith("#") or "/*" in low) else 0.0
            semicolon_cnt = float(t.count(";"))

            features.append([
                length, specials, special_ratio, entropy,
                sql_kw_cnt, sql_ops_cnt, tautology,
                html_tags_cnt, event_cnt, scheme_cnt, sink_cnt,
                quote_imbal, tag_imbal, comment_term, semicolon_cnt
            ])

        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.FEATURE_NAMES)


def build_tfidf_vectorizer(max_features: int = 2500) -> TfidfVectorizer:
    """
    Constructs a TfidfVectorizer customized for web attack payloads:
    Uses subword character n-grams with word boundaries (char_wb, 3-5 ngrams)
    which captures exact syntactic fragments like '<script', 'alert(', '1=1--',
    'union sel', and 'onerror=' without losing punctuation.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        sublinear_tf=True
    )


class HybridFeaturePipeline:
    """
    Unified Pipeline combining TF-IDF and Domain Features.
    Provides transform() that outputs a single matrix with tracked feature names.
    """
    def __init__(self, max_tfidf_features: int = 2000):
        self.tfidf = build_tfidf_vectorizer(max_features=max_tfidf_features)
        self.domain = DomainFeatureExtractor()
        self.feature_names: List[str] = []

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.domain.fit(X)
        tfidf_names = [f"tfidf_{name}" for name in self.tfidf.get_feature_names_out()]
        domain_names = list(self.domain.get_feature_names_out())
        self.feature_names = tfidf_names + domain_names
        return self

    def transform(self, X):
        X_tfidf = self.tfidf.transform(X)
        X_domain = self.domain.transform(X)
        X_domain_sparse = sp.csr_matrix(X_domain)
        return sp.hstack([X_tfidf, X_domain_sparse], format="csr")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        return self.feature_names
