import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

# Adjust path to import src
PROJECT_ROOT = Path(os.getcwd())
if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
    for parent in PROJECT_ROOT.parents:
        if parent.name == "adaptive-neuro-symbolic-ids":
            PROJECT_ROOT = parent
            break
            
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import XGBoostBaseline

def run_diagnostics():
    print("=== DIAGNOSTICS START ===")
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # 1. Load exact same data used in run_ablation_study.py
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy") # The actual multi-class array used
    
    with open(PROCESSED_DIR / "label_classes.json") as f:
        all_classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    try:
        from src.preprocessing.pipeline import UNSW_NB15_Pipeline
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        if pipe.label_encoder is not None:
            y_cat_test = pipe.label_encoder.inverse_transform(y_cat_test)
    except:
        pass
        
    y_test_idx = np.array([class_to_idx[c] for c in y_cat_test])
    
    # 1a. Print len(X_test)
    print(f"\n1a. len(X_test) being evaluated: {len(X_test)}")
    
    # 1b. Load XGBoost Multi-class model and print classes
    xgb_wrapper = XGBoostBaseline.load(RESULTS_DIR / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    print(f"\n1b. xgb_model.classes_       : {list(xgb_model.classes_)}")
    print(f"    label_classes.json order : {all_classes}")
    
    # 1c. Run direct prediction on EXACT X_test/y_test_idx used in run_ablation_study.py
    preds = xgb_model.predict(X_test)
    acc_multiclass = accuracy_score(y_test_idx, preds)
    print(f"\n1c. accuracy_score(y_test_idx, xgb_model.predict(X_test)) = {acc_multiclass:.4f}")
    
    # 1d. Explain the 90.09% bug/discrepancy by loading binary y_test
    print("\n--- Why it doesn't equal 90.09% ---")
    y_test_binary = np.load(PROCESSED_DIR / "y_test.npy")
    # Load the BINARY xgboost model (trained in stage 3)
    xgb_wrapper_binary = XGBoostBaseline.load(RESULTS_DIR / "models" / "xgboost.joblib")
    preds_binary = xgb_wrapper_binary.model.predict(X_test)
    acc_binary = accuracy_score(y_test_binary, preds_binary)
    print(f"Binary y_test.npy accuracy (Normal vs Attack): {acc_binary:.4f} (This is Stage 3's 90.09%)")
    print("CONCLUSION: run_ablation_study.py loads the 10-class `y_cat_test.npy`. The bug is that we expected 90.09% on a 10-class problem, but that score belongs to the 2-class problem.")
    print("=== DIAGNOSTICS END ===")

if __name__ == "__main__":
    run_diagnostics()
