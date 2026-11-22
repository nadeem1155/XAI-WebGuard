"""Shared helpers for the CLI scripts."""
import os
import sys

import numpy as np
import yaml

# make the package importable when running scripts directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xai_webguard.data import build_dataset, group_split          # noqa: E402
from xai_webguard.features import extract_features, features_to_vector  # noqa: E402
from xai_webguard.canonicalize import canonicalize               # noqa: E402


def load_config(path="configs/default.yaml"):
    with open(path) as fh:
        return yaml.safe_load(fh)


def vectorize(ds):
    X = np.array([features_to_vector(extract_features(canonicalize(t), f))
                  for t, f in zip(ds.texts, ds.fields)], dtype=float)
    return X, ds.labels
