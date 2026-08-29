import os
import pandas as pd
from pathlib import Path

import sys

def run_final_comparison():
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = RESULTS_DIR / "milestone3_comparison.csv"
    md_path = RESULTS_DIR / "milestone3_interpretation.md"
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    if not (PROCESSED_DIR / "X_train.npy").exists():
        print("ERROR: Required data file 'data/processed/X_train.npy' not found.", file=sys.stderr)
        print("Please run 'python src/preprocessing/run_pipeline.py' first to generate this file.", file=sys.stderr)
        sys.exit(1)
        
    import numpy as np
    import torch
    from src.device import DEVICE, print_device_info
    from sklearn.metrics import accuracy_score, f1_score
    from src.models.neural_ids import NeuralIDS
    from src.models.baselines import XGBoostBaseline
    from src.continual.experience_builder import ExperienceBuilder
    from src.rules.knowledge_base import KnowledgeBase
    from src.reasoning.reasoner import SymbolicReasoner
    from src.reasoning.fusion import NeuroSymbolicFuser
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline

    print("Running Final Milestone 3 Comparison...")
    
    # Load Data
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
    
    builder = ExperienceBuilder(config_path=str(PROJECT_ROOT / "configs" / "continual.yaml"))
    num_exps = builder.get_num_experiences()
    device = DEVICE
    print_device_info()
    
    # Pre-load Static Neural Model
    static_neural = NeuralIDS(input_dim=X_train.shape[1], output_dim=num_classes).to(device)
    try:
        static_neural.load_state_dict(torch.load(PROJECT_ROOT / "results" / "neural_model" / "neural_ids_weights.pth", map_location=device))
    except:
        print("Warning: Static neural weights not found. Using untrained static model.")
    static_neural.eval()
    
    xgb_wrapper = XGBoostBaseline.load(PROJECT_ROOT / "results" / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    print("Loaded trained XGBoost (Multi-Class) model for fusion.")
    
    # Pre-load Static KB
    static_kb = KnowledgeBase(db_path=PROCESSED_DIR / "knowledge_base.db")
    static_reasoner = SymbolicReasoner(static_kb, feature_names)
    fuser = NeuroSymbolicFuser(alpha=0.5)

    # Note: To fulfill the requirement of using the continually updated models, 
    # we load the checkpoints saved by run_milestone3_forgetting.py (sequential_finetune.py).
    
    continual_neural = NeuralIDS(input_dim=X_train.shape[1], output_dim=num_classes).to(device)
    
    continual_kb = KnowledgeBase(db_path=PROCESSED_DIR / "continual_kb.db")
    continual_reasoner = SymbolicReasoner(continual_kb, feature_names)
    
    print("\nColumn Definitions (Which models feed which columns):")
    print(" - Static_Neural: Frozen NeuralIDS MLP (results/neural_model/neural_ids_weights.pth)")
    print(" - Static_NeuroSym: Frozen XGBoost Baseline (results/models/xgboost_multi.joblib) + Frozen KB (data/processed/knowledge_base.db)")
    print(" - Continual_Neural: Iteratively trained NeuralIDS MLP (results/continual/continual_checkpoints/neural_ids_exp_{eid}.pth)")
    print(" - Continual_NeuroSym: Iteratively trained NeuralIDS MLP + Continually updated KB (data/processed/continual_kb.db)\n")
    
    results = []
    
    for eid in range(1, num_exps + 1):
        print(f"Evaluating Experience {eid}...")
        # Get data
        X_test_k, y_test_k_cat = builder.build_experience(X_test, y_cat_test, exp_id=eid, cumulative=False)
        y_test_k = np.array([class_to_idx[c] for c in y_test_k_cat])
        
        if len(X_test_k) == 0:
            continue
            
        # 1. Static Neural
        with torch.no_grad():
            static_logits = static_neural(torch.tensor(X_test_k, dtype=torch.float32).to(device))
            static_probs = torch.softmax(static_logits, dim=-1).cpu().numpy()
            static_preds = np.argmax(static_probs, axis=-1)
        acc_sn = accuracy_score(y_test_k, static_preds)
        
        xgb_probs = xgb_model.predict_proba(X_test_k)
        xgb_classes = xgb_model.classes_
        
        # 2. Static Neuro-Symbolic (Using XGBoost for fusion)
        sns_preds = []
        for i in range(len(X_test_k)):
            n_probs = {all_classes[int(cls_idx)]: xgb_probs[i][j] for j, cls_idx in enumerate(xgb_classes)}
            sym_res = static_reasoner.predict(X_test_k[i])
            fuse_res = fuser.fuse(n_probs, sym_res)
            sns_preds.append(class_to_idx.get(fuse_res["final_predicted_class"], static_preds[i]))
        acc_sns = accuracy_score(y_test_k, sns_preds)
        
        # Load continual model checkpoint for this experience
        checkpoint_path = PROJECT_ROOT / "results" / "continual" / "continual_checkpoints" / f"neural_ids_exp_{eid}.pth"
        if checkpoint_path.exists():
            continual_neural.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Warning: Continual Neural checkpoint not found at {checkpoint_path}. Using initial weights.")
                
        # 3. Continual Neural
        continual_neural.eval()
        with torch.no_grad():
            cont_logits = continual_neural(torch.tensor(X_test_k, dtype=torch.float32).to(device))
            cont_probs = torch.softmax(cont_logits, dim=-1).cpu().numpy()
            cont_preds = np.argmax(cont_probs, axis=-1)
        acc_cn = accuracy_score(y_test_k, cont_preds)
        
        # 4. Continual Neuro-Symbolic (Using continually updated MLP for fusion)
        cns_preds = []
        for i in range(len(X_test_k)):
            # BUGFIX: We must use cont_probs (from continual_neural) instead of xgb_probs
            n_probs = {all_classes[j]: cont_probs[i][j] for j in range(num_classes)}
            sym_res = continual_reasoner.predict(X_test_k[i])
            fuse_res = fuser.fuse(n_probs, sym_res)
            cns_preds.append(class_to_idx.get(fuse_res["final_predicted_class"], cont_preds[i]))
        acc_cns = accuracy_score(y_test_k, cns_preds)
        
        results.append({
            "Experience": eid,
            "Static_Neural": acc_sn,
            "Static_NeuroSym": acc_sns,
            "Continual_Neural": acc_cn,
            "Continual_NeuroSym": acc_cns
        })
        
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Generate MD interpretation
    with open(md_path, "w") as f:
        f.write("# Milestone 3 Final Comparison\n\n")
        f.write("Evaluation of the 4 system variants across continual experiences.\n\n")
        f.write(df.to_csv(index=False, sep="|"))
        
    print(f"Saved final comparison to {csv_path} and {md_path}")
    
if __name__ == "__main__":
    run_final_comparison()
