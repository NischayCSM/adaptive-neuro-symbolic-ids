"""
Baseline classifiers for UNSW-NB15 intrusion detection.

Three models are provided — all sharing a common ``BaselineModel`` interface
so the training notebook can iterate over them uniformly.

Models
------
- **RandomForestBaseline** — scikit-learn ``RandomForestClassifier``
- **XGBoostBaseline** — XGBoost ``XGBClassifier``
- **MLPBaseline** — small PyTorch feed-forward network

Each exposes:
    model.fit(X_train, y_train)
    model.predict(X_test)           -> np.ndarray
    model.predict_proba(X_test)     -> np.ndarray  (n, n_classes)
    model.save(path)
    model.load(path)                -> cls
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class BaselineModel(ABC):
    """Minimal interface shared by all baseline classifiers."""

    name: str = "BaselineModel"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Random Forest
# ═══════════════════════════════════════════════════════════════════════════

class RandomForestBaseline(BaselineModel):
    """Wrapper around ``sklearn.ensemble.RandomForestClassifier``."""

    name = "RandomForest"

    def __init__(self, **kwargs: Any) -> None:
        from sklearn.ensemble import RandomForestClassifier

        defaults = dict(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        defaults.update(kwargs)
        self.model = RandomForestClassifier(**defaults)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        logger.info("[%s] Training on %s ...", self.name, X.shape)
        self.model.fit(X, y)
        logger.info("[%s] Training complete.", self.name)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("[%s] Saved to %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestBaseline":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
# XGBoost
# ═══════════════════════════════════════════════════════════════════════════

class XGBoostBaseline(BaselineModel):
    """Wrapper around ``xgboost.XGBClassifier``."""

    name = "XGBoost"

    def __init__(self, **kwargs: Any) -> None:
        from xgboost import XGBClassifier

        defaults = dict(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        defaults.update(kwargs)
        self.model = XGBClassifier(**defaults)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        logger.info("[%s] Training on %s ...", self.name, X.shape)
        self.model.fit(X, y)
        logger.info("[%s] Training complete.", self.name)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("[%s] Saved to %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostBaseline":
        from xgboost import XGBClassifier

        obj = cls.__new__(cls)
        obj.model = XGBClassifier()
        obj.model.load_model(str(path))
        return obj


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch MLP
# ═══════════════════════════════════════════════════════════════════════════

class _MLPNet:
    """Small feed-forward network built with PyTorch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        import torch
        import torch.nn as nn
        from src.device import DEVICE

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary output (logit)

        self.net = nn.Sequential(*layers)
        self.device = DEVICE
        self.net.to(self.device)

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class MLPBaseline(BaselineModel):
    """Small PyTorch MLP for binary classification."""

    name = "MLP"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 30,
        patience: int = 5,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self._net: Optional[_MLPNet] = None
        self._input_dim: Optional[int] = None
        self._threshold: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._input_dim = X.shape[1]
        self._net = _MLPNet(self._input_dim, self.hidden_dims, self.dropout)
        device = self._net.device

        # Class-weight for imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        # Train / validation split (90/10) for early stopping
        n_val = max(1, int(0.1 * len(X)))
        idx = np.random.RandomState(42).permutation(len(X))
        X_t, y_t = X[idx[n_val:]], y[idx[n_val:]]
        X_v, y_v = X[idx[:n_val]], y[idx[:n_val]]

        train_ds = TensorDataset(
            torch.tensor(X_t, dtype=torch.float32),
            torch.tensor(y_t, dtype=torch.float32),
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        X_v_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        y_v_t = torch.tensor(y_v, dtype=torch.float32).to(device)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        logger.info("[%s] Training on %s  (device=%s) ...", self.name, X.shape, device)
        for epoch in range(1, self.epochs + 1):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = self._net(xb).squeeze(-1)
                loss = criterion(logits, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X_t)

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_logits = self._net(X_v_t).squeeze(-1)
                val_loss = criterion(val_logits, y_v_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 5 == 0 or wait == 0:
                logger.info(
                    "  Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f%s",
                    epoch, self.epochs, epoch_loss, val_loss,
                    " *" if wait == 0 else "",
                )

            if wait >= self.patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net.eval()
        logger.info("[%s] Training complete.", self.name)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self._threshold).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("MLP is not trained. Call fit() first.")

        self._net.eval()
        device = self._net.device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self._net(X_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Return (n, 2) for compatibility with sklearn convention
        return np.column_stack([1 - probs, probs])

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "input_dim": self._input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "threshold": self._threshold,
            "net_state_dict": self._net.state_dict() if self._net else None,
        }
        torch.save(state, path)
        logger.info("[%s] Saved to %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLPBaseline":
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(hidden_dims=state["hidden_dims"], dropout=state["dropout"])
        obj._input_dim = state["input_dim"]
        obj._threshold = state["threshold"]
        obj._net = _MLPNet(
            state["input_dim"], state["hidden_dims"], state["dropout"]
        )
        obj._net.load_state_dict(state["net_state_dict"])
        obj._net.eval()
        return obj


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation utilities
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Compute a comprehensive set of binary classification metrics.

    Returns a dict with keys:
        model, accuracy, precision_macro, recall_macro, f1_macro,
        precision_per_class, recall_per_class, f1_per_class,
        false_positive_rate, confusion_matrix
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"], output_dict=True
    )

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "fpr": fpr,
        "precision_normal": report["Normal"]["precision"],
        "recall_normal": report["Normal"]["recall"],
        "f1_normal": report["Normal"]["f1-score"],
        "precision_attack": report["Attack"]["precision"],
        "recall_attack": report["Attack"]["recall"],
        "f1_attack": report["Attack"]["f1-score"],
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ── Convenience ──────────────────────────────────────────────────────────

ALL_BASELINES: list[type[BaselineModel]] = [
    RandomForestBaseline,
    XGBoostBaseline,
    MLPBaseline,
]

if __name__ == "__main__":
    import os
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline

    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break

    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = RESULTS_DIR / "models"
    NEURAL_DIR = RESULTS_DIR / "neural_model"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        X_train = np.load(PROCESSED_DIR / "X_train.npy")
        y_train = np.load(PROCESSED_DIR / "y_train.npy")
        y_cat_train = np.load(PROCESSED_DIR / "y_cat_train.npy")
        X_test = np.load(PROCESSED_DIR / "X_test.npy")
        y_test = np.load(PROCESSED_DIR / "y_test.npy")
    except FileNotFoundError:
        print("ERROR: Required data files not found in data/processed/.", file=sys.stderr)
        sys.exit(1)

    print("Training Baseline Models...")
    for model_cls in ALL_BASELINES:
        model = model_cls()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_binary(y_test, y_pred, model_name=model.name)
        print(f"[{model.name}] Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
        model.save(MODELS_DIR / f"{model.name.lower()}.joblib")

    print("\nTraining NeuralIDS (Multi-Class)...")
    from src.models.neural_ids import NeuralIDS
    
    # Load canonical class list (single source of truth saved by run_pipeline.py)
    import json
    classes_path = PROCESSED_DIR / "label_classes.json"
    if not classes_path.exists():
        print("ERROR: label_classes.json not found. Re-run preprocessing.", file=sys.stderr)
        sys.exit(1)
    with open(classes_path) as f:
        all_classes = json.load(f)
    num_classes = len(all_classes)
    print(f"  Loaded {num_classes} classes from label_classes.json: {all_classes}")
    
    # Build mapping from the pipeline's encoded integers to contiguous 0-based indices.
    # The pipeline's LabelEncoder includes <UNKNOWN> at index 0, so the real classes
    # are encoded as 1..10.  We need to map those to 0..9 for the neural network.
    pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
    le = pipe.label_encoder
    # Map: encoded_int -> 0-based class index (using the canonical class order)
    encoded_to_idx = {}
    for i, cls_name in enumerate(all_classes):
        encoded_val = le.transform([cls_name])[0]
        encoded_to_idx[encoded_val] = i
    
    y_cat_train_idx = np.array([encoded_to_idx[c] for c in y_cat_train])
    
    print("\nTraining XGBoost (Multi-Class) for fusion...")
    xgb_multi = XGBoostBaseline()
    xgb_multi.fit(X_train, y_cat_train_idx)
    xgb_multi.save(MODELS_DIR / "xgboost_multi.joblib")
    
    from src.device import DEVICE, print_device_info
    device = DEVICE
    print_device_info()
    neural_model = NeuralIDS(input_dim=X_train.shape[1], output_dim=num_classes).to(device)
    
    optimizer = optim.Adam(neural_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_cat_train_idx, dtype=torch.long).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    neural_model.train()
    epochs = 10
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = neural_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  NeuralIDS Epoch {epoch}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        
    weights_path = NEURAL_DIR / "neural_ids_weights.pth"
    torch.save(neural_model.state_dict(), weights_path)
    print(f"NeuralIDS weights saved to {weights_path}")
    print("\nAll baseline and neural models trained and saved successfully.")
