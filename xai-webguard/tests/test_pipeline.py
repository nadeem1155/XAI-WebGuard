import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.features import FEATURE_NAMES, featurize
from xai_webguard.models.tree_model import TreeModel
from xai_webguard.models.fusion import FusionModel, fit_temperature
from xai_webguard.pipeline import XAIWebGuard


def _toy_model():
    rng = np.random.default_rng(0)
    texts = (["id=1 OR 1=1--"] * 30 + ["<img src=x onerror=alert(1)>"] * 30 + ["search=hello world"] * 30)
    labels = np.array([1] * 30 + [2] * 30 + [0] * 30)
    X = np.array([featurize(t) for t in texts], float)
    return TreeModel({"n_estimators": 60, "max_depth": 4}).fit(X, labels), X, labels


def test_tree_trains_and_shap_shape():
    model, X, y = _toy_model()
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    sv = model.shap_values(X[:3], target_class=1)
    assert sv.shape == (3, len(FEATURE_NAMES))


def test_temperature_positive():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1, 1], size=200)
    y = p.argmax(1)
    assert fit_temperature(p, y) > 0


def test_inspect_flags_attack():
    model, _, _ = _toy_model()
    guard = XAIWebGuard(model, None, FusionModel())
    r = guard.inspect("id=7 OR 1=1--")
    assert r.verdict in ("sqli", "xss")
    assert r.action in ("flag", "block")
    assert r.standards["cwe"] in ("CWE-89", "CWE-79")
