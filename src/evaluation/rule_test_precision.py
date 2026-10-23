"""
Test-set quality of every ACTIVE rule in a continual KB (full 10-class test set).

Per rule: test precision = matched test flows of the rule's class / matched test flows
(undefined when the rule matches nothing), test coverage = matched flows of the rule's
class / test flows of that class. A rule's experience is the experience its class belongs
to in configs/continual.yaml (rules are only extracted from that experience's data).

KB-level: active rules per experience / class, rules retired per experience (from the
pipeline's build log, when there is one), mean validation precision (as stored in the KB),
mean conditions per rule, fidelity (share of test flows where SymbolicReasoner's prediction
equals xgboost_multi.joblib's; flows where no rule fires count as disagreement, and the
share among fired flows is reported too) and test coverage (flows matched by >= 1 rule).

Usage: python -m src.evaluation.rule_test_precision --kb continual_kb_v1.db --tag v1
Writes results/rule_test_precision_{tag}.csv and results/kb_report_{tag}.json.
"""

import sys
import json
import argparse

import numpy as np
import pandas as pd

from src.models.baselines import XGBoostBaseline
from src.rules.knowledge_base import KnowledgeBase
from src.rules.validator import _get_rule_mask
from src.continual.experience_builder import ExperienceBuilder
from src.continual.cl_eval import find_project_root, symbolic_results


def precision_stats(p, n_total):
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return {"n_rules": int(n_total), "n_matching_test": 0}
    return {
        "n_rules": int(n_total), "n_matching_test": int(len(p)),
        "mean": float(p.mean()), "median": float(np.median(p)), "min": float(p.min()),
        "n_ge_0.80": int((p >= 0.80).sum()), "pct_ge_0.80": float((p >= 0.80).sum() / n_total * 100),
        "n_ge_0.50": int((p >= 0.50).sum()), "pct_ge_0.50": float((p >= 0.50).sum() / n_total * 100),
    }


def main(kb_name, tag):
    root = find_project_root()
    processed = root / "data" / "processed"
    kb_path = processed / kb_name
    if not kb_path.exists():
        print(f"ERROR: {kb_path} not found.", file=sys.stderr)
        sys.exit(1)

    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    names = pipe.feature_names
    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    X_test = np.load(processed / "X_test.npy")
    y_test = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))

    builder = ExperienceBuilder(config_path=str(root / "configs" / "continual.yaml"))
    class_exp = {c: e for e in range(1, builder.get_num_experiences() + 1)
                 for c in builder.get_classes_for_experience(e)}

    rules = KnowledgeBase(db_path=kb_path).get_active_rules()
    if not rules:
        print(f"ERROR: no ACTIVE rules in {kb_path}.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for r in rules:
        m = _get_rule_mask(r, X_test, names)
        cls = r["predicted_class"]
        correct = int((y_test[m] == cls).sum())
        rows.append({
            "rule_id": r["rule_id"], "predicted_class": cls, "experience": class_exp[cls],
            "n_conditions": len(r["conditions"]),
            "conditions": " AND ".join(f"{c['feature']} {c['operator']} {c['threshold']:.4f}" for c in r["conditions"]),
            "val_precision": r["precision"], "val_coverage": r["coverage"],
            "test_matched": int(m.sum()), "test_correct": correct,
            "test_precision": correct / m.sum() if m.any() else np.nan,
            "test_coverage": correct / (y_test == cls).sum(),
        })
    df = pd.DataFrame(rows)
    csv_path = root / "results" / f"rule_test_precision_{tag}.csv"
    df.to_csv(csv_path, index=False)

    out = {"kb": str(kb_path.relative_to(root)), "tag": tag, "n_active": len(rules)}
    out["test_precision"] = precision_stats(df["test_precision"].to_numpy(float), len(df))
    out["test_precision_by_class"] = {c: precision_stats(g["test_precision"].to_numpy(float), len(g))
                                      for c, g in df.groupby("predicted_class")}
    out["test_precision_by_experience"] = {int(e): precision_stats(g["test_precision"].to_numpy(float), len(g))
                                           for e, g in df.groupby("experience")}
    out["active_per_class"] = {c: int((df["predicted_class"] == c).sum()) for c in all_classes}
    out["active_per_experience"] = {int(e): int(n) for e, n in df["experience"].value_counts().sort_index().items()}

    log_path = root / "results" / "continual" / f"kb_build_{kb_path.stem}.json"
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        out["build_log"] = str(log_path.relative_to(root))
        out["retired_per_experience"] = {e["experience"]: e["retired"] for e in log["experiences"]}
        out["accepted_per_experience"] = {e["experience"]: e["accepted"] for e in log["experiences"]}
        out["conflicts_per_experience"] = {e["experience"]: e["conflicts"] for e in log["experiences"]}
    out["status_counts"] = KnowledgeBase(db_path=kb_path).export_summary()["by_status"]

    out["mean_val_precision"] = float(df["val_precision"].mean())
    out["conditions_per_rule"] = {"mean": float(df["n_conditions"].mean()), "median": float(df["n_conditions"].median())}

    sym = symbolic_results(kb_path, names, X_test)
    fired = np.array([bool(s.get("all_matches")) for s in sym])
    sym_pred = np.array([s["predicted_class"] for s in sym])
    xgb = XGBoostBaseline.load(root / "results" / "models" / "xgboost_multi.joblib").model
    xgb_pred = np.array(all_classes)[xgb.predict(X_test).astype(int)]
    agree = sym_pred == xgb_pred
    out["fidelity"] = {"all_test_flows": float(agree.mean()),
                       "fired_flows_only": float(agree[fired].mean()) if fired.any() else None,
                       "n_fired": int(fired.sum()), "n_test": len(X_test)}
    out["test_coverage"] = float(fired.mean())
    out["test_coverage_per_class"] = {c: float(fired[y_test == c].mean()) for c in all_classes}

    with open(root / "results" / f"kb_report_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", required=True, help="KB file name inside data/processed/")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    main(args.kb, args.tag)
