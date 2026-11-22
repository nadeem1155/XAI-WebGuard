"""Evaluate detection (Table 5.1 / 5.2). Loads Model A, optionally Model B, fuses,
and reports metrics on the in-distribution test split and the cross-dataset set.

    python scripts/run_detection_eval.py --config configs/default.yaml \\
        --tree results/model_a.pkl [--transformer results/model_b]
"""
import argparse
import json
import os
import sys

import numpy as np

from _common import load_config, vectorize, build_dataset, group_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.models.tree_model import TreeModel
from xai_webguard.models.fusion import FusionModel
from xai_webguard.evaluate.detection_metrics import detection_report


def proba(tree, transformer, ds):
    X, y = vectorize(ds)
    pa = tree.predict_proba(X)
    pb = transformer.predict_proba(ds.texts) if transformer else None
    return pa, pb, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tree", default="results/model_a.pkl")
    ap.add_argument("--transformer", default=None)
    ap.add_argument("--out", default="results/detection_metrics.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    tree = TreeModel.load(args.tree)
    transformer = None
    if args.transformer:
        from xai_webguard.models.transformer_model import TransformerModel
        transformer = TransformerModel.load(args.transformer)

    ds = build_dataset(cfg["sources"])
    train, val, test = group_split(ds, cfg["split"], cfg["seed"])

    # calibrate + tune fusion weight on validation
    fusion = FusionModel(uncertainty_band=cfg["fusion"]["uncertainty_band"])
    pav, pbv, yv = proba(tree, transformer, val)
    fusion.calibrate(pav, yv, pbv, yv if pbv is not None else None)
    if pbv is not None:
        fusion.tune_weight(pav, pbv, yv)

    results = {}
    pat, pbt, yt = proba(tree, transformer, test)
    results["in_distribution"] = detection_report(yt, fusion.fuse(pat, pbt))

    if cfg.get("cross_dataset"):
        try:
            cds = build_dataset(cfg["cross_dataset"], dedup=False)
            pac, pbc, yc = proba(tree, transformer, cds)
            results["cross_dataset"] = detection_report(yc, fusion.fuse(pac, pbc))
        except Exception as e:
            results["cross_dataset_error"] = str(e)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
