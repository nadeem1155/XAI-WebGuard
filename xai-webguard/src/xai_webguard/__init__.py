"""XAI-WebGuard: explainable detection of SQL injection and cross-site scripting.

Reference implementation accompanying the M.Tech thesis
"An Explainable Artificial Intelligence Framework for the Automated Detection
of SQL Injection and Cross-Site Scripting Attacks" (Fauzia Khatoon).
"""
from .canonicalize import canonicalize, CanonResult
from .features import extract_features, featurize, FEATURE_NAMES
from .pipeline import XAIWebGuard, Report

__version__ = "0.1.0"
__all__ = ["canonicalize", "CanonResult", "extract_features", "featurize",
           "FEATURE_NAMES", "XAIWebGuard", "Report"]
