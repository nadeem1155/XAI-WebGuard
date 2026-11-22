"""End-to-end smoke test on synthetic data (tree model only, no GPU needed).

Confirms the whole pipeline runs: canonicalise -> features -> train Model A ->
calibrated decision -> detection metrics -> explanation report. It does NOT
produce results to report; it verifies the code works before you plug in the
real datasets.

    python scripts/make_synthetic_data.py
    python scripts/smoke_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.data import build_dataset, group_split
from xai_webguard.features import extract_features, features_to_vector
from xai_webguard.canonicalize import canonicalize
from xai_webguard.models.tree_model import TreeModel
from xai_webguard.models.fusion import FusionModel
from xai_webguard.evaluate.detection_metrics import detection_report
from xai_webguard.pipeline import XAIWebGuard


def vec(ds):
    X = np.array([features_to_vector(extract_features(canonicalize(t), f))
                  for t, f in zip(ds.texts, ds.fields)], float)
    return X, ds.labels


def main():
    src = [{"name": "synthetic", "path": "data/synthetic.csv", "text_col": "Query",
            "label_col": "Label", "label_map": {"0": "benign", "1": "sqli", "2": "xss"},
            "default_field": "query"}]
    if not os.path.exists("data/synthetic.csv"):
        print("Run: python scripts/make_synthetic_data.py first")
        sys.exit(1)

    ds = build_dataset(src)
    train, val, test = group_split(ds, (0.7, 0.15, 0.15), 0)
    print(f"[data] total={len(ds)} train={len(train)} val={len(val)} test={len(test)}")

    Xtr, ytr = vec(train)
    Xva, yva = vec(val)
    Xte, yte = vec(test)

    model = TreeModel({"n_estimators": 150, "max_depth": 6}).fit(Xtr, ytr, Xva, yva)
    fusion = FusionModel().calibrate(model.predict_proba(Xva), yva)
    rep = detection_report(yte, fusion.fuse(model.predict_proba(Xte), None))
    print(f"[detection] macro-F1={rep['macro_f1']:.3f} MCC={rep['mcc']:.3f} "
          f"PR-AUC={rep['pr_auc']} ECE={rep['ece']:.3f}")

    guard = XAIWebGuard(model, None, fusion)
    for probe in ["id=5 OR 1=1--", "q=<img src=x onerror=alert(1)>", "search=blue shoes"]:
        r = guard.inspect(probe)
        top = r.evidence[0] if r.evidence else {}
        print(f"[inspect] {probe!r:40s} -> {r.verdict:7s} conf={r.confidence:.2f} "
              f"action={r.action} cwe={r.standards.get('cwe')} "
              f"top_evidence={top.get('feature', '-')}")

    assert rep["macro_f1"] > 0.7, "synthetic macro-F1 unexpectedly low"
    print("\nSMOKE TEST PASSED — pipeline runs end to end.")


if __name__ == "__main__":
    main()
