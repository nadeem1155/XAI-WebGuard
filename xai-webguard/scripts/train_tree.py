"""Train Model A (feature-based tree) and save it. Usage:
    python scripts/train_tree.py --config configs/default.yaml --out results/model_a.pkl
"""
import argparse
import os

from _common import load_config, vectorize, build_dataset, group_split
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.models.tree_model import TreeModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/model_a.pkl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = build_dataset(cfg["sources"])
    train, val, test = group_split(ds, cfg["split"], cfg["seed"])
    Xtr, ytr = vectorize(train)
    Xva, yva = vectorize(val)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = TreeModel(cfg["tree"]).fit(Xtr, ytr, Xva, yva)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
