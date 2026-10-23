"""
Rule-quality metrics for ACTIVE rules in data/processed/continual_kb.db, plus example
rules converted back to original feature units.

- Conditions per rule (mean, median).
- Fidelity: among test flows where >= 1 rule fires, fraction where SymbolicReasoner's
  predicted class equals XGBoost's (xgboost_multi.joblib) predicted class.
- Test coverage per class: fraction of each class's test flows matched by >= 1 rule.
- Example rules: thresholds inverted with the pipeline's StandardScaler
  (value * scale_ + mean_) for numeric features; categorical features (ordinal-encoded,
  not scaled) are shown as the set of categories satisfying the condition.
  Precision/coverage are reported both as stored in the KB (validation half of that
  experience's training data, which only contains that experience's classes) and on the
  full 10-class test set (precision = matched flows of the rule's class / matched flows;
  coverage = matched flows of the rule's class / all test flows of that class).
"""

import json

import numpy as np
import pandas as pd

from src.models.baselines import XGBoostBaseline
from src.rules.knowledge_base import KnowledgeBase
from src.reasoning.reasoner import evaluate_condition
from src.continual.cl_eval import find_project_root, symbolic_results

QUOTED = [  # (class, [(feature, operator, threshold rounded to 4 dp), ...])
    ("Normal", [("ackdat", ">", -0.4834), ("ackdat", "<=", -0.3600)]),
    ("Generic", [("sbytes", "<=", -0.0459), ("smean", ">", -0.4079)]),
]


def rule_mask(rule, X, feat_idx):
    m = np.ones(len(X), dtype=bool)
    for c in rule["conditions"]:
        col = X[:, feat_idx[c["feature"]]]
        m &= np.array([evaluate_condition(v, c["operator"], c["threshold"]) for v in col])
    return m


def main():
    root = find_project_root()
    processed = root / "data" / "processed"
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    names = pipe.feature_names
    feat_idx = {n: i for i, n in enumerate(names)}
    num_pos = {names[i]: j for j, i in enumerate(pipe._num_indices)}
    cat_pos = {names[i]: j for j, i in enumerate(pipe._cat_indices)}
    scaler, encoder = pipe._scaler, pipe._encoder

    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    X_test = np.load(processed / "X_test.npy")
    y_test = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))

    kb_path = processed / "continual_kb.db"
    rules = KnowledgeBase(db_path=kb_path).get_active_rules()
    out = {"kb": str(kb_path.relative_to(root)), "n_active_rules": len(rules)}

    # --- Conditions per rule
    n_conds = np.array([len(r["conditions"]) for r in rules])
    out["conditions_per_rule"] = {"mean": float(n_conds.mean()), "median": float(np.median(n_conds)),
                                  "min": int(n_conds.min()), "max": int(n_conds.max())}

    # --- Fidelity and per-class coverage
    sym = symbolic_results(kb_path, names, X_test)
    fired = np.array([bool(s.get("all_matches")) for s in sym])
    sym_pred = np.array([s["predicted_class"] for s in sym])
    xgb = XGBoostBaseline.load(root / "results" / "models" / "xgboost_multi.joblib").model
    xgb_pred = np.array(all_classes)[xgb.predict(X_test).astype(int)]
    out["fidelity"] = {"n_fired": int(fired.sum()), "n_test": len(X_test),
                       "agree_with_xgboost": float((sym_pred[fired] == xgb_pred[fired]).mean()),
                       "symbolic_correct_when_fired": float((sym_pred[fired] == y_test[fired]).mean())}
    out["test_coverage_per_class"] = {
        c: {"n_test": int((y_test == c).sum()), "coverage": float(fired[y_test == c].mean()),
            "n_active_rules_for_class": sum(r["predicted_class"] == c for r in rules)}
        for c in all_classes}
    out["test_coverage_overall"] = float(fired.mean())

    # --- Example rules
    def pick_best(cls):
        cands = [r for r in rules if r["predicted_class"] == cls]
        best_p = max(r["precision"] for r in cands)
        tied = [r for r in cands if r["precision"] == best_p]
        r = max(tied, key=lambda r: r["coverage"])
        return r, f"highest KB precision for {cls} ({len(tied)} rule(s) tied at {best_p:.4f}; tie broken by KB coverage)"

    def find_quoted(cls, conds):
        for r in rules:
            got = sorted((c["feature"], c["operator"], round(c["threshold"], 4)) for c in r["conditions"])
            if r["predicted_class"] == cls and got == sorted(conds):
                return r
        raise SystemExit(f"ERROR: quoted rule for {cls} {conds} not found among ACTIVE rules in {kb_path}")

    examples = [pick_best("Exploits"), pick_best("Fuzzers")]
    examples += [(find_quoted(cls, conds), "quoted rule") for cls, conds in QUOTED]

    def to_units(c):
        f, op, th = c["feature"], c["operator"], c["threshold"]
        if f in num_pos:
            j = num_pos[f]
            return f"{f} {op} {th * scaler.scale_[j] + scaler.mean_[j]:.6g}"
        cats = list(encoder.categories_[cat_pos[f]])
        ok = [cat for code, cat in enumerate(cats) if evaluate_condition(code, op, th)]
        return f"{f} in {{{', '.join(ok[:12])}{', ...' if len(ok) > 12 else ''}}} ({len(ok)}/{len(cats)} categories; ordinal {op} {th:.4f})"

    out["example_rules"] = []
    for r, why in examples:
        m = rule_mask(r, X_test, feat_idx)
        cls = r["predicted_class"]
        out["example_rules"].append({
            "why": why, "rule_id": r["rule_id"], "class": cls,
            "scaled": " AND ".join(f"{c['feature']} {c['operator']} {c['threshold']:.4f}" for c in r["conditions"]),
            "original_units": " AND ".join(to_units(c) for c in r["conditions"]),
            "kb_precision": r["precision"], "kb_coverage": r["coverage"],
            "test_matched": int(m.sum()),
            "test_precision": float((y_test[m] == cls).mean()) if m.any() else None,
            "test_coverage": float((y_test[m] == cls).sum() / (y_test == cls).sum()),
        })

    with open(root / "results" / "rule_quality.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
