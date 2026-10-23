"""
Accuracy, macro-F1 and macro FPR for the saved static baselines on the test split.

- randomforest.joblib and mlp.joblib are BINARY models (Normal vs Attack, trained on y_train)
  and are scored against y_test (binary).
- xgboost_multi.joblib is the 10-class model, scored against the 10 attack categories.

Macro FPR = mean over classes of FP / (FP + TN) (one-vs-rest).
"""

import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.models.baselines import RandomForestBaseline, MLPBaseline, XGBoostBaseline
from src.continual.cl_eval import find_project_root
from src.evaluation.evaluate_milestone2 import calculate_fpr


def macro_fpr(y_true, y_pred, labels):
    return float(np.mean(list(calculate_fpr(y_true, y_pred, labels).values())))


def main():
    root = find_project_root()
    processed = root / "data" / "processed"
    models_dir = root / "results" / "models"

    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_test = np.load(processed / "X_test.npy")
    y_bin = np.load(processed / "y_test.npy")
    y_cat = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))
    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    y_multi = np.array([all_classes.index(c) for c in y_cat])

    specs = [
        ("RandomForest", "randomforest.joblib", RandomForestBaseline, "binary", y_bin, [0, 1]),
        ("MLP", "mlp.joblib", MLPBaseline, "binary", y_bin, [0, 1]),
        ("XGBoost_multi", "xgboost_multi.joblib", XGBoostBaseline, "10-class", y_multi, list(range(len(all_classes)))),
    ]
    rows = []
    for name, fname, cls, task, y_true, labels in specs:
        path = models_dir / fname
        if not path.exists():
            print(f"ERROR: {path} not found. Run src/models/baselines.py first.", file=sys.stderr)
            sys.exit(1)
        y_pred = cls.load(path).predict(X_test)
        rows.append({
            "Model": name, "File": fname, "Task": task, "N_test": len(y_true),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro_FPR": macro_fpr(y_true, y_pred, labels),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    out = root / "results" / "static_baselines.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
