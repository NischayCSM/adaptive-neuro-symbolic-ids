"""
Alpha sweep for STATIC fusion: XGBoost (results/models/xgboost_multi.joblib) +
static KB (data/processed/knowledge_base.db), on the full test set.

Same fusion rule as evaluate_milestone2.py:
    score(c) = alpha * P_xgb(c) + (1 - alpha) * max precision of matched rules predicting c
    prediction = argmax_c score(c)

Note: NeuroSymbolicFuser breaks exact ties by Python set iteration order, which depends on
PYTHONHASHSEED. Exact ties occur when no rule fires and alpha = 0 (all scores are 0).
The number of such rows is reported; run with PYTHONHASHSEED=0 for a reproducible value.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.models.baselines import XGBoostBaseline
from src.evaluation.evaluate_milestone2 import calculate_fpr
from src.continual.cl_eval import find_project_root, symbolic_results, fuse_predictions

ALPHAS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    root = find_project_root()
    processed = root / "data" / "processed"
    out_csv = root / "results" / "static_alpha_sweep.csv"

    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_test = np.load(processed / "X_test.npy")
    y_cat_test = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))
    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_true = np.array([class_to_idx[c] for c in y_cat_test])

    model_path = root / "results" / "models" / "xgboost_multi.joblib"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run src/models/baselines.py first.", file=sys.stderr)
        sys.exit(1)
    xgb = XGBoostBaseline.load(model_path).model
    probs = xgb.predict_proba(X_test)

    kb_path = processed / "knowledge_base.db"
    sym = symbolic_results(kb_path, pipe.feature_names, X_test)
    fired = np.array([bool(s.get("all_matches")) for s in sym])
    print(f"Static KB rule coverage: {fired.mean():.4f} ({fired.sum()}/{len(fired)})")

    rows = []
    for alpha in ALPHAS:
        preds = fuse_predictions(probs, xgb.classes_, sym, all_classes, class_to_idx, alpha)
        rows.append({
            "Alpha": alpha,
            "Accuracy": accuracy_score(y_true, preds),
            "Macro_F1": f1_score(y_true, preds, average="macro", zero_division=0),
            "Macro_FPR": float(np.mean(list(calculate_fpr(y_true, preds, list(range(len(all_classes)))).values()))),
            "Rule_Coverage": fired.mean(),
            "Tied_Rows_No_Rule": int((~fired).sum()) if alpha == 0.0 else 0,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset"),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
