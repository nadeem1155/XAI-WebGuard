"""Fine-tune Model B (code-aware transformer). Needs torch + transformers + GPU.
    python scripts/train_transformer.py --config configs/default.yaml --out results/model_b
"""
import argparse
import os
import sys

from _common import load_config, build_dataset, group_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.models.transformer_model import TransformerModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/model_b")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["transformer"]

    ds = build_dataset(cfg["sources"])
    train, val, _ = group_split(ds, cfg["split"], cfg["seed"])
    model = TransformerModel(tc["model_name"], tc["max_len"])
    model.fit(train.texts, train.labels, val.texts, val.labels,
              epochs=tc["epochs"], lr=float(tc["lr"]), batch_size=tc["batch_size"], out_dir=args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
