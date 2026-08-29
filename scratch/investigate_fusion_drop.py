import os
import sys
import sqlite3
import numpy as np
import json
import pandas as pd
from pathlib import Path

# Adjust path to import src
PROJECT_ROOT = Path(os.getcwd())
if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
    for parent in PROJECT_ROOT.parents:
        if parent.name == "adaptive-neuro-symbolic-ids":
            PROJECT_ROOT = parent
            break
            
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import XGBoostBaseline
from src.rules.knowledge_base import KnowledgeBase
from src.reasoning.reasoner import SymbolicReasoner
from src.reasoning.fusion import NeuroSymbolicFuser
from src.preprocessing.pipeline import UNSW_NB15_Pipeline

def run_investigation():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    DB_PATH = PROCESSED_DIR / "knowledge_base.db"
    
    print("=== 1. ALL 31 STATIC RULES ===")
    conn = sqlite3.connect(DB_PATH)
    rules_df = pd.read_sql_query("SELECT rule_id, predicted_class, conditions, precision, coverage FROM rules WHERE status = 'ACTIVE'", conn)
    conn.close()
    
    for _, row in rules_df.iterrows():
        print(f"[{row['predicted_class']}] Prec: {row['precision']:.4f}, Cov: {row['coverage']:.4f} | Cond: {row['conditions']}")
        
    print("\n=== 2. ALPHA VALUE ===")
    import src.evaluation.evaluate_milestone2 as eval_m2
    # Quick inspection of the file to see what alpha is initialized to
    with open(PROJECT_ROOT / "src" / "evaluation" / "evaluate_milestone2.py") as f:
        lines = f.readlines()
        for line in lines:
            if "fuser = NeuroSymbolicFuser" in line:
                print(f"evaluate_milestone2.py initialization: {line.strip()}")
                break
                
    print("\n=== 3. 5 WORKED EXAMPLES (XGBoost Correct, Fusion Wrong) ===")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy")
    
    with open(PROCESSED_DIR / "label_classes.json") as f:
        class_names = json.load(f)
        
    try:
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        feature_names = pipe.feature_names
        if pipe.label_encoder is not None:
            y_cat_test = pipe.label_encoder.inverse_transform(y_cat_test)
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]
        
    xgb_wrapper = XGBoostBaseline.load(RESULTS_DIR / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    
    kb = KnowledgeBase(db_path=DB_PATH)
    reasoner = SymbolicReasoner(kb, feature_names)
    fuser = NeuroSymbolicFuser(alpha=0.5) # Hardcoding what we expect, to simulate it
    
    probs = xgb_model.predict_proba(X_test)
    xgb_classes = xgb_model.classes_
    
    count = 0
    # Let's sub-sample heavily to not run the reasoner on 175k rows just to find 5
    # Find indices where XGBoost is correct
    xgb_preds = [class_names[np.argmax(p)] for p in probs]
    correct_indices = [i for i, (p, t) in enumerate(zip(xgb_preds, y_cat_test)) if p == t]
    
    for i in correct_indices:
        if count >= 5:
            break
            
        true_class = y_cat_test[i]
        xgb_prob_dist = probs[i]
        xgb_class = class_names[np.argmax(xgb_prob_dist)]
        
        n_probs = {class_names[int(cls_idx)]: xgb_prob_dist[j] for j, cls_idx in enumerate(xgb_classes)}
        
        sym_result = reasoner.predict(X_test[i])
        fusion_result = fuser.fuse(n_probs, sym_result)
        final_class = fusion_result["final_predicted_class"]
        
        if final_class != true_class:
            print(f"\n--- Example {count+1} ---")
            print(f"True Label: {true_class}")
            print(f"XGBoost Prediction: {xgb_class} (Prob: {n_probs[xgb_class]:.4f})")
            print(f"Symbolic Prediction: {sym_result['predicted_class']} (Matched {len(sym_result['all_matches'])} rules)")
            for r in sym_result['all_matches']:
                target = r.get('predicted_class', r.get('attack_type', 'Unknown'))
                print(f"  -> Rule: {r['rule_id']} | Class: {target} | Prec: {r['precision']:.4f}")
            print(f"Final Fused Prediction: {final_class}")
            pass
            count += 1
            
if __name__ == "__main__":
    run_investigation()
