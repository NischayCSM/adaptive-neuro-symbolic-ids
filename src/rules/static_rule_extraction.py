import os
import sys
import numpy as np
import json
from pathlib import Path

# Adjust path to import src
PROJECT_ROOT = Path(os.getcwd())
if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
    for parent in PROJECT_ROOT.parents:
        if parent.name == "adaptive-neuro-symbolic-ids":
            PROJECT_ROOT = parent
            break
            
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.knowledge_base import KnowledgeBase
from src.rules.validator import RuleValidator
from src.rules.tree_extraction import train_rule_tree, extract_rules_from_tree
from src.rules.association_rules import generate_association_rules
from src.rules.importance_threshold import generate_importance_rules
from src.rules.shap_rules import generate_shap_rules
from src.rules.merge import merge_rules
from src.preprocessing.pipeline import UNSW_NB15_Pipeline

def run_static_extraction():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    
    print("--- Running Static Rule Extraction (Milestone 2 Baseline) ---")
    
    # Load Data
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_cat_train = np.load(PROCESSED_DIR / "y_cat_train.npy")
    
    # Try to load feature names from pipeline
    try:
        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
        feature_names = pipe.feature_names
        if pipe.label_encoder is not None:
            y_cat_train = pipe.label_encoder.inverse_transform(y_cat_train)
    except Exception as e:
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
        print(f"Warning: could not decode y_cat_train: {e}")
        
    # Load canonical class list (single source of truth)
    classes_path = PROCESSED_DIR / "label_classes.json"
    if not classes_path.exists():
        print("ERROR: label_classes.json not found. Re-run preprocessing.", file=sys.stderr)
        sys.exit(1)
    with open(classes_path) as f:
        class_names = json.load(f)
        
    # Setup KB
    kb_path = PROCESSED_DIR / "knowledge_base.db"
    if kb_path.exists():
        kb_path.unlink() # Start fresh for Milestone 2
    kb = KnowledgeBase(db_path=kb_path)
    
    validator = RuleValidator(config_path=str(PROJECT_ROOT / "configs" / "rules.yaml"))
    
    print("Generating rules from multiple extractors on the full training set...")
    dt_model = train_rule_tree(X_train, y_cat_train, max_depth=8)
    tree_class_names = list(dt_model.classes_)
    
    tree_rules = extract_rules_from_tree(dt_model, feature_names, tree_class_names)
    
    try:
        assoc_rules = generate_association_rules(X_train, y_cat_train, feature_names, tree_class_names)
    except Exception as e:
        print(f"Warning: Association rule generator failed: {e}")
        assoc_rules = []
        
    try:
        import_rules = generate_importance_rules(X_train, y_cat_train, feature_names, tree_class_names)
    except Exception as e:
        print(f"Warning: Importance rule generator failed: {e}")
        import_rules = []
        
    try:
        shap_rules = generate_shap_rules(X_train, y_cat_train, feature_names, tree_class_names)
    except Exception as e:
        print(f"Warning: SHAP rule generator failed: {e}")
        shap_rules = []
        
    all_generated_rules = tree_rules + assoc_rules + import_rules + shap_rules
    print(f"Generated {len(all_generated_rules)} candidate rules.")
    
    # Merge
    merged_rules = merge_rules([all_generated_rules])
    
    print("Validating static rules...")
    # Split the full training data to validate rules
    val_size = len(X_train) // 2
    X_val1, y_val1 = X_train[:val_size], y_cat_train[:val_size]
    X_val2, y_val2 = X_train[val_size:], y_cat_train[val_size:]
    
    accepted_rules = validator.score_and_filter(
        merged_rules, X_val1, y_val1, X_val2, y_val2, feature_names
    )
    
    print(f"Adding {len(accepted_rules)} accepted rules to static Knowledge Base ({kb_path.name}).")
    for rule in accepted_rules:
        rule["status"] = "ACTIVE"
        kb.add_rule(rule)
        
    print("Static Rule Extraction Complete.")

if __name__ == "__main__":
    run_static_extraction()
