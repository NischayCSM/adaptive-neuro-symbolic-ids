import os
import pandas as pd
from pathlib import Path

import sys

def run_ablation_study():
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = RESULTS_DIR / "milestone3_ablation.csv"
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    if not (PROCESSED_DIR / "X_train.npy").exists():
        print("ERROR: Required data file 'data/processed/X_train.npy' not found.", file=sys.stderr)
        print("Please run 'python src/preprocessing/run_pipeline.py' first to generate this file.", file=sys.stderr)
        sys.exit(1)
        
    import numpy as np
    import torch
    from src.device import DEVICE, print_device_info
    from sklearn.metrics import accuracy_score, f1_score
    from src.models.baselines import XGBoostBaseline
    from src.continual.experience_builder import ExperienceBuilder
    from src.rules.knowledge_base import KnowledgeBase
    from src.reasoning.reasoner import SymbolicReasoner
    from src.reasoning.fusion import NeuroSymbolicFuser
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline

    print("Running Milestone 3 Ablation Study...")
    
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_cat_train = np.load(PROCESSED_DIR / "y_cat_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy")
    
    try:
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        if pipe.label_encoder is not None:
            y_cat_train = pipe.label_encoder.inverse_transform(y_cat_train)
            y_cat_test = pipe.label_encoder.inverse_transform(y_cat_test)
        feature_names = pipe.feature_names
    except Exception as e:
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
        
    # Load canonical class list (single source of truth)
    import json
    classes_path = PROCESSED_DIR / "label_classes.json"
    if not classes_path.exists():
        print("ERROR: label_classes.json not found. Re-run preprocessing.", file=sys.stderr)
        sys.exit(1)
    with open(classes_path) as f:
        all_classes = json.load(f)
    num_classes = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    xgb_wrapper = XGBoostBaseline.load(PROJECT_ROOT / "results" / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    print("Loaded trained XGBoost (Multi-Class) model.")
    
    # Use populated KB from continual pipeline
    kb = KnowledgeBase(db_path=PROCESSED_DIR / "continual_kb.db")
    reasoner = SymbolicReasoner(kb, feature_names)
    
    y_test_idx = np.array([class_to_idx[c] for c in y_cat_test])
    
    probs = xgb_model.predict_proba(X_test)
    xgb_classes = xgb_model.classes_
        
    results = []
    
    for alpha in [0.0, 0.5, 1.0]:
        print(f"Evaluating Alpha = {alpha}...")
        fuser = NeuroSymbolicFuser(alpha=alpha)
        
        preds = []
        rule_coverage_count = 0
        for i in range(len(X_test)):
            n_probs = {all_classes[int(cls_idx)]: probs[i][j] for j, cls_idx in enumerate(xgb_classes)}
            sym_res = reasoner.predict(X_test[i])
            if sym_res.get("all_matches"):
                rule_coverage_count += 1
            fuse_res = fuser.fuse(n_probs, sym_res)
            preds.append(class_to_idx.get(fuse_res["final_predicted_class"], np.argmax(probs[i])))
            
        print(f"Rule coverage: {rule_coverage_count} / {len(X_test)} rows matched at least one rule ({(rule_coverage_count/len(X_test))*100:.2f}%)")
        acc = accuracy_score(y_test_idx, preds)
        f1 = f1_score(y_test_idx, preds, average='macro', zero_division=0)
        
        results.append({
            "Alpha": alpha,
            "Configuration": "Symbolic Only" if alpha == 0.0 else ("Neural Only" if alpha == 1.0 else "Neuro-Symbolic"),
            "Accuracy": acc,
            "Macro_F1": f1
        })
        
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved ablation study results to {csv_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_ablation_study()
