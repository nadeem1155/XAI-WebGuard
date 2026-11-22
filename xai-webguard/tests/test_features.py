import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.canonicalize import canonicalize
from xai_webguard.features import extract_features, FEATURE_NAMES, featurize


def test_vector_length_matches_names():
    assert len(featurize("select 1")) == len(FEATURE_NAMES)


def test_tautology_detected():
    f = extract_features(canonicalize("id=1 OR 1=1--"))
    assert f["tautology_pattern"] == 1.0
    assert f["comment_terminator"] == 1.0


def test_xss_signals():
    f = extract_features(canonicalize("<img src=x onerror=alert(1)>"))
    assert f["event_handler_count"] >= 1
    assert f["dom_sink_count"] >= 1
    assert f["html_tag_count"] >= 1


def test_benign_is_quiet():
    f = extract_features(canonicalize("search=blue running shoes"))
    assert f["tautology_pattern"] == 0.0
    assert f["event_handler_count"] == 0.0
