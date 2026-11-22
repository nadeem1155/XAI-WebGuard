"""Dataset loading, label harmonisation, de-duplication and group-aware splits.

The public datasets ship with different column names and label conventions, so
each source is described by a small mapping in the config file (see
``configs/default.yaml``). Edit those mappings to point at your local copies;
nothing here downloads data.

Classes are harmonised to three integer labels:
    0 = benign, 1 = sqli, 2 = xss
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .canonicalize import canonicalize

LABELS = {"benign": 0, "sqli": 1, "xss": 2}
LABEL_NAMES = {0: "benign", 1: "sqli", 2: "xss"}


@dataclass
class Dataset:
    texts: List[str]
    labels: np.ndarray            # int array in {0,1,2}
    fields: List[str]
    groups: np.ndarray            # near-duplicate group ids (leak control)
    source: List[str]

    def __len__(self):
        return len(self.texts)

    def subset(self, idx):
        idx = np.asarray(idx)
        return Dataset(
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
            fields=[self.fields[i] for i in idx],
            groups=self.groups[idx],
            source=[self.source[i] for i in idx],
        )


def _norm_label(value, mapping: Dict[str, str]) -> Optional[int]:
    key = str(value).strip().lower()
    if key in mapping:
        return LABELS[mapping[key]]
    if key in LABELS:
        return LABELS[key]
    return None


def load_source(cfg: dict) -> Tuple[List[str], List[int], List[str], str]:
    """Load one dataset source from its config mapping.

    cfg keys: path, text_col, label_col, label_map, field_col (optional),
              default_field (optional), sep (optional).
    """
    df = pd.read_csv(cfg["path"], sep=cfg.get("sep", ","), dtype=str,
                     encoding=cfg.get("encoding", "utf-8"), on_bad_lines="skip")
    text_col = cfg["text_col"]
    label_col = cfg["label_col"]
    label_map = cfg.get("label_map", {})
    field_col = cfg.get("field_col")
    default_field = cfg.get("default_field", "query")

    texts, labels, fields = [], [], []
    for _, row in df.iterrows():
        text = row.get(text_col)
        if text is None or (isinstance(text, float) and np.isnan(text)):
            continue
        y = _norm_label(row.get(label_col), label_map)
        if y is None:
            continue
        texts.append(str(text))
        labels.append(y)
        fields.append(str(row.get(field_col)) if field_col else default_field)
    return texts, labels, fields, cfg.get("name", cfg["path"])


# ----------------------------- de-duplication -----------------------------

def _shingles(text: str, k: int = 5) -> set:
    t = re.sub(r"\s+", " ", text.lower())
    if len(t) < k:
        return {t}
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _minhash(shingles: set, num_perm: int = 32, seed: int = 1) -> Tuple[int, ...]:
    """Tiny MinHash (no external dependency) for near-duplicate grouping."""
    if not shingles:
        return tuple([0] * num_perm)
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 31 - 1, size=num_perm)
    b = rng.integers(0, 2 ** 31 - 1, size=num_perm)
    prime = (1 << 31) - 1
    sig = [prime] * num_perm
    for sh in shingles:
        h = int(hashlib.md5(sh.encode("utf-8")).hexdigest()[:8], 16)
        for j in range(num_perm):
            v = (int(a[j]) * h + int(b[j])) % prime
            if v < sig[j]:
                sig[j] = v
    return tuple(sig)


def assign_groups(texts: List[str], bands: int = 8, rows: int = 4) -> np.ndarray:
    """Group near-duplicates via LSH banding so a group never spans a split."""
    num_perm = bands * rows
    sigs = [_minhash(_shingles(canonicalize(t).text), num_perm) for t in texts]
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    buckets: Dict[Tuple, List[int]] = {}
    for i, sig in enumerate(sigs):
        for bnd in range(bands):
            band = (bnd,) + sig[bnd * rows:(bnd + 1) * rows]
            buckets.setdefault(band, []).append(i)
    for members in buckets.values():
        for j in range(1, len(members)):
            union(members[0], members[j])
    roots = {}
    groups = np.empty(len(texts), dtype=int)
    for i in range(len(texts)):
        r = find(i)
        groups[i] = roots.setdefault(r, len(roots))
    return groups


def build_dataset(sources: List[dict], dedup: bool = True) -> Dataset:
    texts, labels, fields, source = [], [], [], []
    for cfg in sources:
        t, y, fl, name = load_source(cfg)
        texts += t
        labels += y
        fields += fl
        source += [name] * len(t)
    # exact de-dup on canonical text
    if dedup:
        seen, keep = set(), []
        for i, t in enumerate(texts):
            key = canonicalize(t).text
            if key not in seen:
                seen.add(key)
                keep.append(i)
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]
        fields = [fields[i] for i in keep]
        source = [source[i] for i in keep]
    groups = assign_groups(texts) if dedup else np.arange(len(texts))
    return Dataset(texts, np.asarray(labels), fields, groups, source)


def group_split(ds: Dataset, ratios=(0.7, 0.15, 0.15), seed: int = 0):
    """Split so that whole near-duplicate groups stay on one side (no leakage)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ds.groups)
    rng.shuffle(uniq)
    n = len(uniq)
    a = int(ratios[0] * n)
    b = int((ratios[0] + ratios[1]) * n)
    train_g, val_g, test_g = set(uniq[:a]), set(uniq[a:b]), set(uniq[b:])
    def idx(gset):
        return np.array([i for i in range(len(ds)) if ds.groups[i] in gset])
    return ds.subset(idx(train_g)), ds.subset(idx(val_g)), ds.subset(idx(test_g))
