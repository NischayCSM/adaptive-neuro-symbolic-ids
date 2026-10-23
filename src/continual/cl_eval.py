"""
Shared helpers for continual-learning runs (replay, EWC, naive seeds).

Accuracy matrix convention: a[i][j] = accuracy on Exp (j+1)'s test set after
training on Exp (i+1). With T experiences:
    ACC       = mean(a[T-1][:])
    BWT       = mean_{i<T-1} (a[T-1][i] - a[i][i])
    Retention = mean_{i<T-1} a[T-1][i]   (accuracy on Exp 1..T-1 after the last one)
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.continual.experience_builder import ExperienceBuilder


def find_project_root() -> Path:
    root = Path(os.getcwd())
    if root.name != "adaptive-neuro-symbolic-ids":
        for parent in root.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                return parent
    return root


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_continual_data(project_root: Path, processed_dir=None) -> dict:
    """Load processed data with decoded labels. Exits with an error if anything is missing.
    processed_dir: directory with the preprocessed arrays (default None = data/processed)."""
    processed = Path(processed_dir) if processed_dir is not None else project_root / "data" / "processed"
    required = ["X_train.npy", "y_cat_train.npy", "X_test.npy", "y_cat_test.npy",
                "pipeline.joblib", "label_classes.json"]
    missing = [f for f in required if not (processed / f).exists()]
    if missing:
        print(f"ERROR: missing required files in {processed}: {missing}. "
              "Run 'python src/preprocessing/run_pipeline.py' first.", file=sys.stderr)
        sys.exit(1)

    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    if pipe.label_encoder is None:
        print("ERROR: pipeline.joblib has no label encoder; cannot decode labels.", file=sys.stderr)
        sys.exit(1)

    X_train = np.load(processed / "X_train.npy")
    X_test = np.load(processed / "X_test.npy")
    y_cat_train = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_train.npy"))
    y_cat_test = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))

    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    builder = ExperienceBuilder(config_path=str(project_root / "configs" / "continual.yaml"))
    num_exps = builder.get_num_experiences()

    def split(X, y_cat):
        out = []
        for eid in range(1, num_exps + 1):
            X_k, y_k_cat = builder.build_experience(X, y_cat, exp_id=eid, cumulative=False)
            if len(X_k) == 0:
                print(f"ERROR: experience {eid} is empty.", file=sys.stderr)
                sys.exit(1)
            out.append((X_k, np.array([class_to_idx[c] for c in y_k_cat])))
        return out

    return {
        "input_dim": X_train.shape[1],
        "all_classes": all_classes,
        "class_to_idx": class_to_idx,
        "feature_names": pipe.feature_names,
        "num_exps": num_exps,
        "train_splits": split(X_train, y_cat_train),
        "test_splits": split(X_test, y_cat_test),
    }


def predict_proba(model, X, device, batch_size=65536) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device))
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out)


def evaluate_on_experiences(model, test_splits, device):
    """Returns ([acc per exp], [macro-F1 per exp]) for the neural model alone."""
    accs, f1s = [], []
    for X_k, y_k in test_splits:
        preds = np.argmax(predict_proba(model, X_k, device), axis=-1)
        accs.append(float(accuracy_score(y_k, preds)))
        f1s.append(float(f1_score(y_k, preds, average="macro", zero_division=0)))
    return accs, f1s


def cl_metrics(acc_matrix) -> dict:
    a = np.asarray(acc_matrix)
    T = a.shape[0]
    return {
        "ACC": float(a[T - 1].mean()),
        "BWT": float(np.mean([a[T - 1][i] - a[i][i] for i in range(T - 1)])),
        "retention": float(a[T - 1][:T - 1].mean()),
    }


def matrix_rows(acc_matrix, f1_matrix):
    """CSV rows in the same layout as sequential_finetune.csv."""
    rows = []
    for k, (acc_row, f1_row) in enumerate(zip(acc_matrix, f1_matrix), start=1):
        row = {"Phase": f"After Exp{k}"}
        for j, (acc, f1) in enumerate(zip(acc_row, f1_row), start=1):
            row[f"Exp{j}_Acc"] = acc
            row[f"Exp{j}_F1"] = f1
        rows.append(row)
    return rows


def symbolic_results(kb_path: Path, feature_names, X) -> list:
    """Run SymbolicReasoner over every row of X (active rules cached; same semantics)."""
    from src.rules.knowledge_base import KnowledgeBase
    from src.reasoning.reasoner import SymbolicReasoner

    kb = KnowledgeBase(db_path=kb_path)
    cached_rules = kb.get_active_rules()
    if not cached_rules:
        print(f"ERROR: no ACTIVE rules in {kb_path}.", file=sys.stderr)
        sys.exit(1)
    kb.get_active_rules = lambda: cached_rules
    reasoner = SymbolicReasoner(kb, feature_names)
    return [reasoner.predict(x) for x in X]


def fuse_predictions(probs, prob_classes, sym_results, all_classes, class_to_idx, alpha) -> np.ndarray:
    """Fuse per-row neural probabilities with symbolic results via NeuroSymbolicFuser."""
    from src.reasoning.fusion import NeuroSymbolicFuser

    fuser = NeuroSymbolicFuser(alpha=alpha)
    preds = np.empty(len(probs), dtype=np.int64)
    for i in range(len(probs)):
        n_probs = {all_classes[int(c)]: probs[i][j] for j, c in enumerate(prob_classes)}
        res = fuser.fuse(n_probs, sym_results[i])
        preds[i] = class_to_idx[res["final_predicted_class"]]
    return preds
