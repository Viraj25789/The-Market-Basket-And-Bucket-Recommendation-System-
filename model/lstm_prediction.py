"""
lstm_prediction.py
==================
Predicts the next item in a purchase sequence using an LSTM neural network.

How it works (plain English):
  1. Build a vocabulary: assign each unique item a number.
  2. Slide a window of size `seq_length` over every transaction to create
     (context → next_item) training pairs.
  3. Train: Embedding layer encodes items → LSTM learns sequence patterns
     → Dense layer outputs probabilities for every item.
  4. predict(items) → returns the top items most likely to come next.

TensorFlow is optional: if not installed, all methods return [] gracefully.
"""

import os
import json
import numpy as np
import pandas as pd

# Optional TensorFlow – project works without it
try:
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Embedding  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[LSTM] TensorFlow not found – SequencePredictor disabled.")


class SequencePredictor:
    """
    Parameters
    ----------
    data_path  : path to the transactions CSV
    seq_length : number of preceding items used as context window (default 3)
    epochs     : training epochs (default 30)
    """

    def __init__(self, data_path: str, seq_length: int = 3, epochs: int = 30):
        self.data_path  = data_path
        self.seq_length = seq_length
        self.epochs     = epochs
        self.model      = None
        self.item2idx   = {}   # "Milk" → 3
        self.idx2item   = {}   # 3 → "Milk"
        self.vocab_size = 0

        if not TF_AVAILABLE:
            return

        self._build_vocab()
        if self.vocab_size > 1:
            self._prepare_and_train()
        else:
            print("[LSTM] Vocabulary too small to train.")

    # ── Vocabulary ────────────────────────────────────────────────────────────

    def _build_vocab(self):
        df  = pd.read_csv(self.data_path)
        col = {c.lower(): c for c in df.columns}.get("items")
        if col is None:
            raise ValueError("[LSTM] No 'Items' column found.")

        all_items = sorted({
            item.strip()
            for row in df[col].dropna()
            for item in str(row).split(",")
            if item.strip() and item.strip().lower() != "nan"
        })

        # Index 0 is reserved for padding
        self.item2idx   = {item: idx + 1 for idx, item in enumerate(all_items)}
        self.idx2item   = {idx: item for item, idx in self.item2idx.items()}
        self.vocab_size = len(all_items) + 1
        print(f"[LSTM] Vocabulary: {self.vocab_size - 1} items")

    # ── Training data ─────────────────────────────────────────────────────────

    def _prepare_sequences(self):
        """Convert transactions into fixed-length (context, target) pairs."""
        df  = pd.read_csv(self.data_path)
        col = {c.lower(): c for c in df.columns}.get("items")

        X_list, y_list = [], []
        for row in df[col].dropna():
            items = [i.strip() for i in str(row).split(",")]
            idxs  = [self.item2idx[i] for i in items if i in self.item2idx]

            for start in range(len(idxs) - 1):
                # Left-pad context so it always has length seq_length
                context = idxs[max(0, start - self.seq_length + 1): start + 1]
                context = [0] * (self.seq_length - len(context)) + context
                X_list.append(context)
                y_list.append(idxs[start + 1])

        if not X_list:
            return None, None
        return np.array(X_list, dtype="int32"), to_categorical(y_list, num_classes=self.vocab_size)

    # ── Model ─────────────────────────────────────────────────────────────────

    def _build_model(self):
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=min(32, self.vocab_size),
                input_length=self.seq_length,
            ),
            LSTM(64),
            Dense(self.vocab_size, activation="softmax"),
        ])
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _prepare_and_train(self):
        X, y = self._prepare_sequences()
        if X is None or len(X) == 0:
            print("[LSTM] No sequences – skipping training.")
            return
        print(f"[LSTM] Training on {len(X)} sequences …")
        self.model = self._build_model()
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=min(64, max(8, len(X) // 10)),
            verbose=0,
        )
        print("[LSTM] Training complete.")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, items: list, top_n: int = 3) -> list:
        """
        Given recent items (e.g. ['Milk', 'Bread']), return top_n likely next items.
        Returns [] if model is unavailable or items are unknown.
        """
        if not TF_AVAILABLE or self.model is None:
            return []

        idxs = [self.item2idx[i.strip()] for i in items if i.strip() in self.item2idx]
        if not idxs:
            return []

        context = idxs[-self.seq_length:]
        context = [0] * (self.seq_length - len(context)) + context
        probs   = self.model.predict(np.array([context], dtype="int32"), verbose=0)[0]

        # Zero out already-seen items and padding
        probs[list(set(idxs) | {0})] = 0.0
        top_idxs = probs.argsort()[::-1][:top_n]
        return [self.idx2item[i] for i in top_idxs if i in self.idx2item]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_path: str = "data/lstm_model", vocab_path: str = "data/lstm_vocab.json"):
        if self.model is None:
            print("[LSTM] Nothing to save.")
            return
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        self.model.save(model_path)
        with open(vocab_path, "w") as f:
            json.dump({"item2idx": self.item2idx, "seq_length": self.seq_length}, f)
        print(f"[LSTM] Saved → {model_path}  |  vocab → {vocab_path}")

    @classmethod
    def load(cls, data_path: str, model_path: str = "data/lstm_model",
             vocab_path: str = "data/lstm_vocab.json") -> "SequencePredictor":
        """Load a pre-trained model instead of retraining from scratch."""
        if not TF_AVAILABLE:
            return cls.__new__(cls)

        inst = cls.__new__(cls)
        inst.data_path = data_path
        with open(vocab_path) as f:
            meta = json.load(f)
        inst.item2idx   = meta["item2idx"]
        inst.idx2item   = {int(v): k for k, v in meta["item2idx"].items()}
        inst.seq_length = meta["seq_length"]
        inst.vocab_size = len(inst.item2idx) + 1
        inst.model      = load_model(model_path)
        print("[LSTM] Loaded pre-trained model.")
        return inst