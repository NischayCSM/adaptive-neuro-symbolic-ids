import os
import json
import numpy as np
from pathlib import Path

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
from src.rules.knowledge_base import KnowledgeBase
from src.reasoning.reasoner import SymbolicReasoner
from src.reasoning.fusion import NeuroSymbolicFuser
from src.preprocessing.pipeline import UNSW_NB15_Pipeline

def run_sanity_check():
    print("--- Sanity Check: XGBoost + Fusion ---")
    
    # Paths
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # 1. Load Data
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy")
    
    try:
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        if pipe.label_encoder is not None:
            y_cat_test = pipe.label_encoder.inverse_transform(y_cat_test)
        feature_names = pipe.feature_names
    except Exception as e:
        feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]
        
    # Canonical Classes
    classes_path = PROCESSED_DIR / "label_classes.json"
    with open(classes_path) as f:
        all_classes = json.load(f)
        
    print(f"Total Canonical Classes: {len(all_classes)}")
    
    # 2. Load Models
    xgb_wrapper = XGBoostBaseline.load(RESULTS_DIR / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    
    kb = KnowledgeBase(db_path=PROCESSED_DIR / "continual_kb.db")
    reasoner = SymbolicReasoner(kb, feature_names)
    fuser = NeuroSymbolicFuser(alpha=0.5)
    
    # 3. Process 5 sample rows
    print("\nEvaluating 5 Sample Rows...")
    probs = xgb_model.predict_proba(X_test[:5])
    xgb_classes = xgb_model.classes_
    
    for i in range(5):
        print(f"\nRow {i+1} - True Class: {y_cat_test[i]}")
        
        # XGBoost Probabilities
        n_probs = {all_classes[int(cls_idx)]: probs[i][j] for j, cls_idx in enumerate(xgb_classes)}
        
        # Top 2 XGBoost Predictions
        sorted_xgb = sorted(n_probs.items(), key=lambda x: x[1], reverse=True)
        print(f"  XGBoost Top-2 Probs:")
        for cls_name, prob in sorted_xgb[:2]:
            print(f"    - {cls_name}: {prob:.4f}")
            
        # Symbolic Reasoner
        sym_res = reasoner.predict(X_test[i])
        print(f"  Symbolic Prediction: {sym_res.get('predicted_class')}")
            
        # Fusion
        fuse_res = fuser.fuse(n_probs, sym_res)
        print(f"  Final Fused Prediction: {fuse_res['final_predicted_class']}")

if __name__ == "__main__":
    run_sanity_check()
