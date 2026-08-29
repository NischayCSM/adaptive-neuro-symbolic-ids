import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

def main():
    parser = argparse.ArgumentParser(description="Run the Adaptive Neuro-Symbolic IDS Demo")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to simulate")
    parser.add_argument("--speed", type=float, default=0.0, help="Delay in seconds between batches for readability")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = RESULTS_DIR / "demo_run.jsonl"
    md_path = RESULTS_DIR / "demo_summary.md"
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    if not (PROCESSED_DIR / "X_test.npy").exists():
        print("ERROR: Required data file 'data/processed/X_test.npy' not found.", file=sys.stderr)
        sys.exit(1)
        
    from src.device import DEVICE
    from src.models.baselines import XGBoostBaseline
    from src.rules.knowledge_base import KnowledgeBase
    from src.reasoning.reasoner import SymbolicReasoner
    from src.reasoning.fusion import NeuroSymbolicFuser
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline

    print("Loading test data and models for demo...")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    
    pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
    features = pipe.feature_names
    
    classes_path = PROCESSED_DIR / "label_classes.json"
    with open(classes_path) as f:
        all_classes = json.load(f)
    xgb_wrapper = XGBoostBaseline.load(PROJECT_ROOT / "results" / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    print("Loaded trained XGBoost (Multi-Class) model.")
    
    kb = KnowledgeBase(db_path=PROCESSED_DIR / "continual_kb.db")
    reasoner = SymbolicReasoner(kb, features)
    fuser = NeuroSymbolicFuser(alpha=0.5)
    
    with open(jsonl_path, "w") as f:
        pass # Clear file
        
    print(f"Running {args.batches} demonstration batches...")
    
    rules_fired = 0
    total_samples = 0
    
    for batch_num in range(args.batches):
        idx = np.random.randint(0, len(X_test))
        sample_x = X_test[idx]
        
        # Demo predictions
        X_demo = sample_x.reshape(1, -1)
        probs = xgb_model.predict_proba(X_demo)
        xgb_classes = xgb_model.classes_
            
        n_probs = {all_classes[int(cls_idx)]: probs[0][j] for j, cls_idx in enumerate(xgb_classes)}
        sym_res = reasoner.predict(sample_x)
        
        fuse_res = fuser.fuse(n_probs, sym_res)
        
        if sym_res.get("all_matches"):
            rules_fired += 1
        total_samples += 1
            
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch": batch_num + 1,
            "neural_prediction": all_classes[np.argmax(probs)],
            "neural_confidence": float(np.max(probs)),
            "symbolic_matches": len(sym_res.get("all_matches", [])),
            "fused_prediction": fuse_res["final_predicted_class"],
            "fused_confidence": float(fuse_res["final_confidence"]),
            "explanation": fuse_res["symbolic_explanation"]
        }
        
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
        if args.speed > 0:
            time.sleep(args.speed)
            
    # Write summary
    md_content = f"""# Demo Run Summary
    
- **Batches Processed:** {args.batches}
- **Rules Fired:** {rules_fired}
- **Symbolic Intervention Rate:** {(rules_fired/total_samples):.2%}

The structured output traces of each inference step have been logged to `demo_run.jsonl`.
"""
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"Demo complete. Logs saved to {jsonl_path} and {md_path}")

if __name__ == "__main__":
    main()
