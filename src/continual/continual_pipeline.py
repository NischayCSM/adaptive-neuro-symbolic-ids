import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.continual.experience_builder import ExperienceBuilder
from src.continual.drift_detection import DriftDetector
from src.rules.knowledge_base import KnowledgeBase
from src.rules.validator import RuleValidator, _get_rule_mask as _rule_mask
# from src.rules.merge import pool_and_merge_rules  # Assuming this exists based on 2.4

import sys
import argparse

MEMORY_PER_CLASS = 1000
DEFAULT_DRIFT_THRESHOLD = 0.15


def run_pipeline(mixed_validation=False, retire=False, seed=42, kb_name="continual_kb.db",
                 lift_check=False, min_lift=1.5, max_match_fraction=0.5,
                 gain_check=False, min_gain=0.5, exclude_train_rows=None,
                 drift_threshold=DEFAULT_DRIFT_THRESHOLD, processed_dir=None):
    """
    drift_threshold: KS drift threshold of the DriftDetector (default 0.15). A non-default value is
        recorded in the build log.
    processed_dir: directory with the preprocessed arrays and pipeline.joblib; the KB is written there too
        (default None = data/processed). A non-default value is recorded in the build log.
    gain_check: also reject candidate rules whose normalised precision gain
        (precision - base rate) / (1 - base rate) in the validation half is below min_gain or that match
        more than max_match_fraction of it (see RuleValidator). Default off.
    exclude_train_rows: optional list of files with row indices into X_train.npy (.npz with a
        "global_index" array, e.g. replay_finetune.py validation memories, or .npy). The union of those
        rows is removed from the training data before anything else, so they are used for neither rule
        induction, validation, the validation memory nor drift detection. Default None (use all rows).
    lift_check: also reject candidate rules whose lift (precision / class base rate in the validation
        half) is below min_lift or that match more than max_match_fraction of the validation half
        (see RuleValidator). Default off.
    mixed_validation: at experience k > 1, each validation half is the current experience's half
        plus half of a class-balanced memory of earlier experiences' training data (up to
        MEMORY_PER_CLASS random samples per class), and candidates are also conflict-checked
        against the rules already in the KB.
    retire: at experience k > 1, re-validate the KB's ACTIVE rules on the same mixed first half
        and retire those with precision < 0.85 (rules below the validator's min support there are kept).
    """
    np.random.seed(seed)  # SHAP subsampling and drift-detector subsampling
    memory_rng = np.random.default_rng(seed)
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    PROCESSED_DIR = Path(processed_dir).resolve() if processed_dir else PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results" / "continual"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # The default KB keeps the original output names; other KBs get their own suffix.
    suffix = "" if kb_name == "continual_kb.db" else "_" + Path(kb_name).stem
    plot_path = RESULTS_DIR / f"pipeline_active_rules{suffix}.png"
    
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
        
    n_excluded = 0
    if exclude_train_rows:
        excl = np.unique(np.concatenate([
            np.load(f)["global_index"] if str(f).endswith(".npz") else np.load(f) for f in exclude_train_rows]))
        excluded_by_class = dict(zip(*[a.tolist() for a in np.unique(y_cat_train[excl], return_counts=True)]))
        keep = np.ones(len(X_train), dtype=bool)
        keep[excl] = False
        X_train, y_cat_train = X_train[keep], y_cat_train[keep]
        n_excluded = int(len(excl))
        print(f"Excluding {n_excluded} training rows (union of {len(exclude_train_rows)} files): {excluded_by_class}")

    print("Loaded data successfully. Running full pipeline...")
    
    # Initialization
    builder = ExperienceBuilder(config_path=str(PROJECT_ROOT / "configs" / "continual.yaml"))
    kb_path = PROCESSED_DIR / kb_name
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
    detector = DriftDetector(threshold=drift_threshold)
    
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

    validator = RuleValidator(config_path=str(PROJECT_ROOT / "configs" / "rules.yaml"), lift_check=lift_check,
                              min_lift=min_lift, max_match_fraction=max_match_fraction,
                              gain_check=gain_check, min_gain=min_gain)

    num_exps = builder.get_num_experiences()
    
    results = []
    # Class-balanced memory of earlier experiences' training data: class -> (half A, half B)
    memory = {}
    build_log = {"kb": str(kb_path.relative_to(PROJECT_ROOT)), "seed": seed,
                 "mixed_validation": mixed_validation, "retire": retire,
                 "memory_per_class": MEMORY_PER_CLASS, "experiences": [], "rule_experience": {}}
    if lift_check:
        build_log.update({"lift_check": True, "min_lift": min_lift, "max_match_fraction": max_match_fraction})
    if gain_check:
        build_log.update({"gain_check": True, "min_gain": min_gain, "max_match_fraction": max_match_fraction})
    if exclude_train_rows:
        build_log.update({"exclude_train_rows": [str(f) for f in exclude_train_rows],
                          "n_excluded_train_rows": n_excluded, "excluded_by_class": excluded_by_class})
    if drift_threshold != DEFAULT_DRIFT_THRESHOLD:
        build_log["drift_threshold"] = drift_threshold
    if processed_dir:
        build_log["processed_dir"] = str(PROCESSED_DIR.relative_to(PROJECT_ROOT))

    def memory_half(h):
        if not memory:
            return np.empty((0, X_train.shape[1]), dtype=X_train.dtype), np.empty(0, dtype=y_cat_train.dtype)
        return (np.concatenate([memory[c][h][0] for c in memory]),
                np.concatenate([memory[c][h][1] for c in memory]))

    for eid in range(1, num_exps + 1):
        print(f"\n--- Processing Experience {eid} ---")

        # Build experience
        X_exp, y_exp_cat = builder.build_experience(X_train, y_cat_train, exp_id=eid, cumulative=False)

        # Validation halves: the current experience split in half, plus (mixed validation /
        # retirement) half of the memory of earlier experiences in each.
        val_size = len(X_exp) // 2
        X_val1, y_val1 = X_exp[:val_size], y_exp_cat[:val_size]
        X_val2, y_val2 = X_exp[val_size:], y_exp_cat[val_size:]
        X_mix1, y_mix1, X_mix2, y_mix2 = X_val1, y_val1, X_val2, y_val2
        if eid > 1 and (mixed_validation or retire):
            (mX1, my1), (mX2, my2) = memory_half(0), memory_half(1)
            X_mix1, y_mix1 = np.concatenate([X_val1, mX1]), np.concatenate([y_val1, my1])
            X_mix2, y_mix2 = np.concatenate([X_val2, mX2]), np.concatenate([y_val2, my2])
            print(f"Mixed validation halves: {len(X_mix1)} / {len(X_mix2)} samples "
                  f"({len(mX1)} / {len(mX2)} from memory of {sorted(memory)})")
        if mixed_validation and eid > 1:
            X_val1, y_val1, X_val2, y_val2 = X_mix1, y_mix1, X_mix2, y_mix2

        exp_log = {"experience": eid, "n_train": int(len(X_exp)), "val1_size": int(len(X_val1)),
                   "memory_classes": sorted(memory), "retired": 0, "accepted": 0, "conflicts": 0}

        # (c) Re-validate existing ACTIVE rules on the mixed first half and retire weak ones.
        if retire and eid > 1:
            min_sup = validator.effective_min_support(len(X_mix1))
            retired_by_class = {}
            for rule in kb.get_active_rules():
                mask = _rule_mask(rule, X_mix1, feature_names)
                support = int(mask.sum())
                if support < min_sup:
                    continue
                precision = float(np.sum(y_mix1[mask] == rule["predicted_class"]) / support)
                if precision < 0.85:
                    kb.retire_rule(rule["rule_id"])
                    retired_by_class[rule["predicted_class"]] = retired_by_class.get(rule["predicted_class"], 0) + 1
            exp_log["retired"] = sum(retired_by_class.values())
            exp_log["retired_by_class"] = retired_by_class
            print(f"Retired {exp_log['retired']} ACTIVE rules with precision < 0.85 on mixed data "
                  f"(min support {min_sup}): {retired_by_class}")

        # Drift Detection
        drift_result = None  # detector output dict; None at experience 1 (forced)
        if eid == 1:
            drift_detected = True # Force initial rule extraction
        else:
            drift_result = detector.detect(X_exp)
            drift_detected = drift_result["drift_detected"]
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
            # We split the experience data to validate rules (halves built above)
            accepted_rules = validator.score_and_filter(
                merged_rules, X_val1, y_val1, X_val2, y_val2, feature_names,
                existing_rules=kb.get_active_rules() if mixed_validation else None,
            )
            exp_log["conflicts"] = int(validator.n_conflicts)
            if lift_check:
                exp_log["lift_rejected"] = int(validator.n_lift_rejected)
            if gain_check:
                exp_log["gain_rejected"] = int(validator.n_gain_rejected)

            # 3. KB Update — set attack_type and status=ACTIVE so rules
            #    are actually returned by get_active_rules()
            print(f"Adding {len(accepted_rules)} accepted rules to Knowledge Base.")
            for rule in accepted_rules:
                rule["status"] = "ACTIVE"
                build_log["rule_experience"][kb.add_rule(rule)] = eid
            exp_log["accepted"] = len(accepted_rules)

            # Update reference for future drift detection
            detector.update_reference(X_exp)
            
        # (a) Add up to MEMORY_PER_CLASS random training samples per class of this experience.
        for c in np.unique(y_exp_cat):
            idx = np.flatnonzero(y_exp_cat == c)
            idx = memory_rng.choice(idx, size=min(MEMORY_PER_CLASS, len(idx)), replace=False)
            h = len(idx) // 2
            memory[c] = ((X_exp[idx[:h]], y_exp_cat[idx[:h]]), (X_exp[idx[h:]], y_exp_cat[idx[h:]]))

        stats = kb.export_summary()
        active_rules = kb.get_active_rules()
        # Class breakdown of ACTIVE rules only (export_summary's by_class also counts RETIRED ones)
        active_by_class = {}
        for rule in active_rules:
            active_by_class[rule["predicted_class"]] = active_by_class.get(rule["predicted_class"], 0) + 1
        exp_log["active_after"] = len(active_rules)
        build_log["experiences"].append(exp_log)
        results.append({
            "Experience": eid,
            "Drift_Detected": bool(drift_detected),
            "Drift_Score": drift_result["drift_score"] if drift_result else None,
            "P_Value_Avg": drift_result["p_value_avg"] if drift_result else None,
            "Active_Rules": stats["by_status"].get("ACTIVE", 0),
            "Class_Distribution": json.dumps(dict(sorted(active_by_class.items())))
        })
        
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / f"continual_pipeline_metrics{suffix}.csv", index=False)
    with open(RESULTS_DIR / f"kb_build{suffix or '_continual_kb'}.json", "w") as f:
        json.dump(build_log, f, indent=2)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-validation", action="store_true",
                        help="validate candidates at experience k > 1 against a memory of earlier experiences too")
    parser.add_argument("--retire", action="store_true",
                        help="re-validate ACTIVE rules on mixed data at each new experience; retire if precision < 0.85")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kb-name", default="continual_kb.db", help="KB file name inside data/processed/")
    parser.add_argument("--lift-check", action="store_true",
                        help="reject rules with lift < --min-lift or match fraction > --max-match-fraction")
    parser.add_argument("--min-lift", type=float, default=1.5)
    parser.add_argument("--max-match-fraction", type=float, default=0.5,
                        help="match-fraction cap used by --lift-check and --gain-check")
    parser.add_argument("--gain-check", action="store_true",
                        help="reject rules with (precision - base rate) / (1 - base rate) < --min-gain "
                             "or match fraction > --max-match-fraction")
    parser.add_argument("--min-gain", type=float, default=0.5)
    parser.add_argument("--exclude-train-rows", nargs="+", default=None, metavar="FILE",
                        help=".npz (global_index) or .npy files of X_train row indices to leave out entirely")
    parser.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD,
                        help="KS drift threshold (mean KS statistic) that triggers rule induction")
    parser.add_argument("--processed-dir", default=None,
                        help="directory with the preprocessed arrays; the KB is written there too "
                             "(default: data/processed)")
    args = parser.parse_args()
    run_pipeline(mixed_validation=args.mixed_validation, retire=args.retire, seed=args.seed, kb_name=args.kb_name,
                 lift_check=args.lift_check, min_lift=args.min_lift, max_match_fraction=args.max_match_fraction,
                 gain_check=args.gain_check, min_gain=args.min_gain, exclude_train_rows=args.exclude_train_rows,
                 drift_threshold=args.drift_threshold, processed_dir=args.processed_dir)
