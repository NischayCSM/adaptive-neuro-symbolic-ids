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
from src.preprocessing.pipeline import UNSW_NB15_Pipeline

def run_diagnostics():
    print("--- Diagnostics ---")
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # 1. Load Data
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy")
    
    # Load canonical class list
    classes_path = PROCESSED_DIR / "label_classes.json"
    with open(classes_path) as f:
        all_classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    # Decode pipeline
    try:
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        if pipe.label_encoder is not None:
            y_cat_test = pipe.label_encoder.inverse_transform(y_cat_test)
    except Exception as e:
        pass
        
    y_test_idx = np.array([class_to_idx[c] for c in y_cat_test])
    
    print(f"1. Evaluating on FULL test set? Rows = {len(X_test)}")
    
    # 2. Load Model & Check Classes
    xgb_wrapper = XGBoostBaseline.load(RESULTS_DIR / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    
    print(f"\n2. xgb_model.classes_ : {xgb_model.classes_}")
    print(f"   label_classes.json : {all_classes}")
    
    # 3. Direct Accuracy Check
    preds_direct = xgb_model.predict(X_test)
    
    acc = accuracy_score(y_test_idx, preds_direct)
    print(f"\n3. Direct xgb_model.predict(X_test) accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
if __name__ == "__main__":
    run_diagnostics()
