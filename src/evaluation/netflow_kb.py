"""
NetFlow-v2 knowledge base: the continual pipeline's rule induction, run once on NF-UNSW-NB15-v2.

Same steps and settings as continual_pipeline.py's rule-induction block (KB continual_kb_v1_gain_clean: --gain-check,
min_gain 0.5, max_match_fraction 0.5, configs/rules.yaml, no lift check, no mixed validation):
  generators tree_extraction (depth 8), association_rules, importance_threshold, shap_rules -> merge_rules ->
  RuleValidator.score_and_filter.
Generators are trained on NF-UNSW train; the validator uses the NF-UNSW validation split, cut into two stratified
halves (seed 42) for its val1 (precision, coverage, gain, match fraction) and val2 (stability) inputs. NF-UNSW test
and NF-ToN are not touched. np.random.seed(42) before induction, as in the pipeline (SHAP subsampling).

Writes data/processed_netflow/kb_netflow_gain.db and results/external_netflow/kb_build_netflow.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.continual.cl_eval import find_project_root
from src.rules.tree_extraction import train_rule_tree, extract_rules_from_tree
from src.rules.association_rules import generate_association_rules
from src.rules.importance_threshold import generate_importance_rules
from src.rules.shap_rules import generate_shap_rules
from src.rules.merge import merge_rules
from src.rules.validator import RuleValidator
from src.rules.knowledge_base import KnowledgeBase

SEED = 42
MIN_GAIN, MAX_MATCH_FRACTION = 0.5, 0.5
KB_NAME = "kb_netflow_gain.db"


def main():
    root = find_project_root()
    data_dir, res_dir = root / "data" / "processed_netflow", root / "results" / "external_netflow"
    kb_path, log_path = data_dir / KB_NAME, res_dir / "kb_build_netflow.json"
    for p in (kb_path, log_path):
        if p.exists():
            sys.exit(f"ERROR: {p.relative_to(root)} already exists; not overwriting")
    features = json.loads((data_dir / "feature_names.json").read_text())
    classes = np.array(json.loads((data_dir / "label_classes.json").read_text()))
    X_tr, y_tr = np.load(data_dir / "X_train.npy"), classes[np.load(data_dir / "y_train.npy")]
    X_va, y_va = np.load(data_dir / "X_val.npy"), classes[np.load(data_dir / "y_val.npy")]
    h1, h2 = train_test_split(np.arange(len(y_va)), test_size=0.5, stratify=y_va, random_state=SEED)
    h1, h2 = np.sort(h1), np.sort(h2)

    np.random.seed(SEED)
    dt_model = train_rule_tree(X_tr, y_tr, max_depth=8)
    names = list(dt_model.classes_)
    generated = {"tree_extraction": extract_rules_from_tree(dt_model, features, names)}
    for gen, fn in (("association_rules", generate_association_rules), ("importance_threshold", generate_importance_rules),
                    ("shap_rules", generate_shap_rules)):
        try:
            generated[gen] = fn(X_tr, y_tr, features, names)
        except Exception as e:  # same fallback as the pipeline
            print(f"Warning: {gen} generator failed: {e}")
            generated[gen] = []
    candidates = [r for g in generated.values() for r in g]
    merged = merge_rules([candidates])
    print(f"Generated {len(candidates)} candidate rules ({ {g: len(r) for g, r in generated.items()} }); "
          f"{len(merged)} after merging")

    validator = RuleValidator(config_path=str(root / "configs" / "rules.yaml"), gain_check=True, min_gain=MIN_GAIN,
                              max_match_fraction=MAX_MATCH_FRACTION)
    accepted = validator.score_and_filter(merged, X_va[h1], y_va[h1], X_va[h2], y_va[h2], features)

    kb = KnowledgeBase(db_path=kb_path)
    for rule in accepted:
        rule["status"] = "ACTIVE"
        kb.add_rule(rule)
    active = kb.get_active_rules()
    by_class = Counter(r["predicted_class"] for r in active)
    by_gen = Counter(g for r in accepted for g in r["generators"])
    log = {"kb": str(kb_path.relative_to(root)), "seed": SEED, "gain_check": True, "min_gain": MIN_GAIN,
           "max_match_fraction": MAX_MATCH_FRACTION, "n_train": int(len(y_tr)), "val1_size": int(len(h1)),
           "val2_size": int(len(h2)), "effective_min_support": int(validator.effective_min_support(len(h1))),
           "candidates_per_generator": {g: len(r) for g, r in generated.items()}, "candidates": len(candidates),
           "merged": len(merged), "accepted": len(accepted), "conflicts": int(validator.n_conflicts),
           "gain_rejected": int(validator.n_gain_rejected), "accepted_per_generator": dict(by_gen),
           "active_per_class": {c: by_class.get(c, 0) for c in classes.tolist()}}
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"KB {kb_path.relative_to(root)}: {len(active)} active rules; per class {log['active_per_class']}")


if __name__ == "__main__":
    main()
