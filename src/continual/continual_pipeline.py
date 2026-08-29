import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.continual.experience_builder import ExperienceBuilder
from src.continual.drift_detection import DriftDetector
from src.rules.knowledge_base import KnowledgeBase
from src.rules.validator import RuleValidator
# from src.rules.merge import pool_and_merge_rules  # Assuming this exists based on 2.4

import sys

def run_pipeline():
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results" / "continual"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_path = RESULTS_DIR / "pipeline_active_rules.png"
    
    if not (PROCESSED_DIR / "X_train.npy").exists() or not (PROCESSED_DIR / "y_cat_train.npy").exists():
        print("ERROR: Required data files ('X_train.npy', 'y_cat_train.npy') not found in data/processed/.", file=sys.stderr)
        print("Please run 'python src/preprocessing/run_pipeline.py' first to generate these files.", file=sys.stderr)
        sys.exit(1)
        
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_cat_train = np.load(PROCESSED_DIR / "y_cat_train.npy")
    
    try:
        from src.preprocessing.pipeline import UNSW_NB15_Pipeline
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        if pipe.label_encoder is not None:
            y_cat_train = pipe.label_encoder.inverse_transform(y_cat_train)
    except Exception as e:
        print(f"Warning: could not decode y_cat_train: {e}")
        
    print("Loaded data successfully. Running full pipeline...")
    
    # Initialization
    builder = ExperienceBuilder(config_path=str(PROJECT_ROOT / "configs" / "continual.yaml"))
    kb_path = PROCESSED_DIR / "continual_kb.db"
    if kb_path.exists():
        deleted = False
        for attempt in range(3):
            try:
                kb_path.unlink()
                deleted = True
                break
            except PermissionError as e:
                print(f"Warning: Failed to delete {kb_path} (attempt {attempt + 1}/3) - file lock issue? {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning: Unexpected error when deleting {kb_path}: {e}")
                break
        
        if not deleted:
            print(f"Could not delete {kb_path} due to file locks (e.g. OneDrive). Falling back to SQL DROP...")
            import sqlite3
            try:
                conn = sqlite3.connect(kb_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                conn.close()
                print("Successfully cleared database tables via SQL fallback.")
            except Exception as e:
                print(f"Critical error: SQL fallback failed: {e}", file=sys.stderr)
                sys.exit(1)
        
    kb = KnowledgeBase(db_path=kb_path)
    detector = DriftDetector(threshold=0.15)
    
    # Setup reference on Exp1
    X_exp1, _ = builder.build_experience(X_train, y_cat_train, exp_id=1, cumulative=False)
    detector.update_reference(X_exp1)
    
    from src.rules.tree_extraction import train_rule_tree, extract_rules_from_tree
    from src.rules.association_rules import generate_association_rules
    from src.rules.importance_threshold import generate_importance_rules
    from src.rules.shap_rules import generate_shap_rules
    from src.rules.merge import merge_rules
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    import pandas as pd
    import json
    
    # Try to load feature names from pipeline
    try:
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
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
        class_names = json.load(f)

    validator = RuleValidator(config_path=str(PROJECT_ROOT / "configs" / "rules.yaml"))
    
    num_exps = builder.get_num_experiences()
    
    results = []
    
    for eid in range(1, num_exps + 1):
        print(f"\n--- Processing Experience {eid} ---")
        
        # Build experience
        X_exp, y_exp_cat = builder.build_experience(X_train, y_cat_train, exp_id=eid, cumulative=False)
        
        # Drift Detection
        if eid == 1:
            drift_detected = True # Force initial rule extraction
        else:
            drift_detected = detector.detect(X_exp)
            if drift_detected:
                print(f"Drift detected at experience {eid}! Triggering rule induction.")
            else:
                print(f"No significant drift at experience {eid}.")
                
        if drift_detected:
            # 1. Rule Extraction
            print("Generating rules from multiple extractors...")
            dt_model = train_rule_tree(X_exp, y_exp_cat, max_depth=8)
            tree_class_names = list(dt_model.classes_)
            
            tree_rules = extract_rules_from_tree(dt_model, feature_names, tree_class_names)
            
            try:
                assoc_rules = generate_association_rules(X_exp, y_exp_cat, feature_names, tree_class_names)
            except Exception as e:
                print(f"Warning: Association rule generator failed: {e}")
                assoc_rules = []
                
            try:
                import_rules = generate_importance_rules(X_exp, y_exp_cat, feature_names, tree_class_names)
            except Exception as e:
                print(f"Warning: Importance rule generator failed: {e}")
                import_rules = []
                
            try:
                shap_rules = generate_shap_rules(X_exp, y_exp_cat, feature_names, tree_class_names)
            except Exception as e:
                print(f"Warning: SHAP rule generator failed: {e}")
                shap_rules = []
                
            all_generated_rules = tree_rules + assoc_rules + import_rules + shap_rules
            print(f"Generated {len(all_generated_rules)} candidate rules.")
            
            # Merge
            merged_rules = merge_rules([all_generated_rules])
            
            # 2. Validation
            print("Validating rules...")
            # We split the experience data to validate rules
            val_size = len(X_exp) // 2
            X_val1, y_val1 = X_exp[:val_size], y_exp_cat[:val_size]
            X_val2, y_val2 = X_exp[val_size:], y_exp_cat[val_size:]
            
            accepted_rules = validator.score_and_filter(
                merged_rules, X_val1, y_val1, X_val2, y_val2, feature_names
            )
            
            # 3. KB Update — set attack_type and status=ACTIVE so rules
            #    are actually returned by get_active_rules()
            print(f"Adding {len(accepted_rules)} accepted rules to Knowledge Base.")
            for rule in accepted_rules:
                rule["status"] = "ACTIVE"
                kb.add_rule(rule)
            
            # Update reference for future drift detection
            detector.update_reference(X_exp)
            
        stats = kb.export_summary()
        results.append({
            "Experience": eid,
            "Drift_Detected": drift_detected,
            "Active_Rules": stats["by_status"].get("ACTIVE", 0),
            "Class_Distribution": json.dumps(stats["by_class"])
        })
        
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "continual_pipeline_metrics.csv", index=False)
    
    # Plot active rules over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Experience"], df["Active_Rules"], marker='o')
    plt.title("Number of Active Rules in Knowledge Base over Continual Experiences")
    plt.xlabel("Experience Batch")
    plt.ylabel("Total Active Rules")
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"\nContinual Pipeline completed. Saved metrics to {RESULTS_DIR}")
if __name__ == "__main__":
    run_pipeline()
