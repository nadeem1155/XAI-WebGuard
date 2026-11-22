"""Model B: fine-tuned code-aware transformer over token sequences.

Wraps a Hugging Face sequence classifier (CodeBERT by default; DistilBERT as a
lighter alternative). Torch and transformers are imported lazily so the rest of
the package works without them installed.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..canonicalize import canonicalize


def _require_torch():
    try:
        import torch  # noqa
        import transformers  # noqa
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Model B needs torch and transformers. Install with:\n"
            "    pip install torch transformers\n"
            f"(original error: {e})")


class TransformerModel:
    def __init__(self, model_name: str = "microsoft/codebert-base", max_len: int = 256,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.max_len = max_len
        self.device = device
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self, num_labels: int = 3):
        _require_torch()
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels).to(self.device)

    def fit(self, texts: List[str], labels, val_texts=None, val_labels=None,
            epochs: int = 3, lr: float = 2e-5, batch_size: int = 16, out_dir: str = "results/model_b"):
        _require_torch()
        import torch
        from torch.utils.data import DataLoader, Dataset as TorchDS
        self._ensure_loaded()
        canon = [canonicalize(t).text for t in texts]
        tok = self.tokenizer

        class DS(TorchDS):
            def __init__(self, X, y):
                self.enc = tok(X, truncation=True, max_length=256, padding="max_length")
                self.y = list(y)
            def __len__(self):
                return len(self.y)
            def __getitem__(self, i):
                item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(int(self.y[i]))
                return item

        loader = DataLoader(DS(canon, labels), batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        self.model.train()
        for ep in range(epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    out = self.model(**batch)
                scaler.scale(out.loss).backward()
                scaler.step(opt)
                scaler.update()
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        return self

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        _require_torch()
        import torch
        self._ensure_loaded()
        canon = [canonicalize(t).text for t in texts]
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(canon), batch_size):
                chunk = canon[i:i + batch_size]
                enc = self.tokenizer(chunk, truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(probs) if probs else np.zeros((0, 3))

    @classmethod
    def load(cls, path: str, max_len: int = 256) -> "TransformerModel":
        m = cls(model_name=path, max_len=max_len)
        m._ensure_loaded()
        return m
