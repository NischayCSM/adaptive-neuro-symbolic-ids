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
    
    from src.continual.cl_eval import symbolic_results, fuse_predictions, predict_proba

    def load_neural(path):
        if not path.exists():
            print(f"ERROR: checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)
        m = NeuralIDS(input_dim=X_train.shape[1], output_dim=num_classes).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        return m

    CONT_DIR = PROJECT_ROOT / "results" / "continual"
    REPLAY_SEED = 42
    static_neural = load_neural(PROJECT_ROOT / "results" / "neural_model" / "neural_ids_weights.pth")
    naive_ckpts = [CONT_DIR / "continual_checkpoints" / f"neural_ids_exp_{k}.pth" for k in range(1, num_exps + 1)]
    replay_ckpts = [CONT_DIR / "replay_checkpoints" / f"neural_ids_exp_{k}_seed{REPLAY_SEED}.pth" for k in range(1, num_exps + 1)]
    naive_models = [load_neural(p) for p in naive_ckpts]
    replay_models = [load_neural(p) for p in replay_ckpts]

    xgb_wrapper = XGBoostBaseline.load(PROJECT_ROOT / "results" / "models" / "xgboost_multi.joblib")
    xgb_model = xgb_wrapper.model
    print("Loaded trained XGBoost (Multi-Class) model for fusion.")

    static_kb_path = PROCESSED_DIR / "knowledge_base.db"
    continual_kb_path = PROCESSED_DIR / "continual_kb.db"
    alpha = 0.5

    print("\nColumn Definitions (Which models feed which columns):")
    print(" - Static_Neural: Frozen NeuralIDS trained on all classes (results/neural_model/neural_ids_weights.pth)")
    print(" - Static_NeuroSym: Frozen XGBoost (results/models/xgboost_multi.joblib) + static KB (data/processed/knowledge_base.db)")
    print(" - Naive_*: NAIVE sequential fine-tuning, no replay (results/continual/continual_checkpoints/neural_ids_exp_{k}.pth)")
    print(f" - Replay_*: replay fine-tuning, seed {REPLAY_SEED} (results/continual/replay_checkpoints/neural_ids_exp_{{k}}_seed{REPLAY_SEED}.pth)")
    print(" - *_AfterExpK: checkpoint saved right after training on Exp K, evaluated on Exp K (plasticity, NOT retention)")
    print(" - *_Final: checkpoint after the last experience, evaluated on Exp K (retention for K < last)")
    print(" - *_NeuroSym (continual): that NeuralIDS + continually updated KB (data/processed/continual_kb.db)")
    print(f" - All fusion uses NeuroSymbolicFuser(alpha={alpha})\n")

    results = []

    for eid in range(1, num_exps + 1):
        print(f"Evaluating Experience {eid}...")
        X_test_k, y_test_k_cat = builder.build_experience(X_test, y_cat_test, exp_id=eid, cumulative=False)
        y_test_k = np.array([class_to_idx[c] for c in y_test_k_cat])

        if len(X_test_k) == 0:
            print(f"ERROR: Experience {eid} test set is empty.", file=sys.stderr)
            sys.exit(1)

        static_sym = symbolic_results(static_kb_path, feature_names, X_test_k)
        cont_sym = symbolic_results(continual_kb_path, feature_names, X_test_k)

        def neural_acc(model):
            return accuracy_score(y_test_k, np.argmax(predict_proba(model, X_test_k, device), axis=-1))

        def fused_acc(probs, prob_classes, sym):
            return accuracy_score(y_test_k, fuse_predictions(probs, prob_classes, sym, all_classes, class_to_idx, alpha))

        def cont_fused_acc(model):
            return fused_acc(predict_proba(model, X_test_k, device), range(num_classes), cont_sym)

        row = {"Experience": eid}
        row["Static_Neural"] = neural_acc(static_neural)
        row["Static_NeuroSym"] = fused_acc(xgb_model.predict_proba(X_test_k), xgb_model.classes_, static_sym)
        row["Naive_Neural_AfterExpK"] = neural_acc(naive_models[eid - 1])
        row["Naive_NeuroSym_AfterExpK"] = cont_fused_acc(naive_models[eid - 1])
        row["Naive_Neural_Final"] = neural_acc(naive_models[-1])
        row["Naive_NeuroSym_Final"] = cont_fused_acc(naive_models[-1])
        row["Replay_Neural_AfterExpK"] = neural_acc(replay_models[eid - 1])
        row["Replay_NeuroSym_AfterExpK"] = cont_fused_acc(replay_models[eid - 1])
        row["Replay_Neural_Final"] = neural_acc(replay_models[-1])
        row["Replay_NeuroSym_Final"] = cont_fused_acc(replay_models[-1])
        results.append({k: float(v) if k != "Experience" else v for k, v in row.items()})
        print(row)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Generate MD interpretation
    with open(md_path, "w") as f:
        f.write("# Milestone 3 Final Comparison\n\n")
        f.write("Evaluation of system variants across continual experiences.\n\n")
        f.write("- Static_*: frozen models (Static_NeuroSym = XGBoost + static KB)\n")
        f.write("- Naive_*: naive sequential fine-tuning (no replay); Replay_*: replay fine-tuning, seed 42\n")
        f.write("- *_AfterExpK: model right after Exp K, on Exp K (plasticity). *_Final: model after the last experience, on Exp K (retention)\n")
        f.write("- *_NeuroSym for Naive/Replay: fused with the continual KB (data/processed/continual_kb.db), alpha=0.5\n\n")
        f.write(df.to_csv(index=False, sep="|"))
        
    print(f"Saved final comparison to {csv_path} and {md_path}")
    
if __name__ == "__main__":
    run_final_comparison()
