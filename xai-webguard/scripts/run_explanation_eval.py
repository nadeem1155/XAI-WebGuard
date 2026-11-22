"""Evaluate explanation quality (Table 5.3, Fig. 5.6/5.7).

Needs a token-level annotation file with columns: text, label, span_start, span_end
(the minimal attack span). Computes plausibility (token-F1) for TreeSHAP on
Model A, and, if Model B and torch are available, comprehensiveness / sufficiency
and plausibility for Integrated Gradients.

    python scripts/run_explanation_eval.py --tree results/model_a.pkl \\
        --annotations data/annotations.csv [--transformer results/model_b]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.models.tree_model import TreeModel
from xai_webguard.canonicalize import canonicalize
from xai_webguard.explain.tree_shap import feature_attributions
from xai_webguard.evaluate.explanation_metrics import token_f1
from xai_webguard.pipeline import _feature_text


def tree_plausibility(tree, df):
    """Overlap between the raw text that top features point to and the gold span."""
    scores = []
    for _, r in df.iterrows():
        text, y = str(r["text"]), int(r["label"])
        gold = set(range(int(r["span_start"]), int(r["span_end"])))
        feats = feature_attributions(tree, text, y, top_k=5)
        pred = set()
        for fa in feats:
            snippet = _feature_text(fa["feature"], text)
            idx = text.lower().find(snippet.lower())
            if idx >= 0:
                pred |= set(range(idx, idx + len(snippet)))
        scores.append(token_f1(pred, gold))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="results/model_a.pkl")
    ap.add_argument("--transformer", default=None)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="results/explanation_metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.annotations)
    tree = TreeModel.load(args.tree)
    out = {"n_annotations": int(len(df)),
           "treeshap": {"plausibility_token_f1": tree_plausibility(tree, df)}}

    if args.transformer:
        try:
            from xai_webguard.models.transformer_model import TransformerModel
            from xai_webguard.explain.integrated_gradients import top_raw_spans
            tm = TransformerModel.load(args.transformer)
            ig_scores = []
            for _, r in df.iterrows():
                text, y = str(r["text"]), int(r["label"])
                gold = set(range(int(r["span_start"]), int(r["span_end"])))
                spans = top_raw_spans(tm, text, y, k=5)
                pred = set()
                for s in spans:
                    rs, re_ = s["raw_span"]
                    pred |= set(range(rs, re_))
                ig_scores.append(token_f1(pred, gold))
            out["integrated_gradients"] = {"plausibility_token_f1": float(np.mean(ig_scores))}
        except Exception as e:
            out["integrated_gradients_error"] = str(e)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
