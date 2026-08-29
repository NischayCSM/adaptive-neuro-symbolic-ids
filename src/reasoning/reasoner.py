from typing import Dict, Any, Tuple, List
import operator
import numpy as np

def evaluate_condition(feat_value, op: str, thresh: float) -> bool:
    if op == ">": return feat_value > thresh
    if op == "<=": return feat_value <= thresh
    if op == ">=": return feat_value >= thresh
    if op == "<": return feat_value < thresh
    if op == "==": return feat_value == thresh
    return True

class SymbolicReasoner:
    def __init__(self, kb, feature_names: List[str]):
        """
        Initializes the SymbolicReasoner.
        
        Args:
            kb: An instance of KnowledgeBase
            feature_names: List of feature names corresponding to the indices of the feature vector.
        """
        self.kb = kb
        self.feature_names = feature_names

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates a single feature vector against all ACTIVE rules in the KB.
        
        Returns a dict:
            - predicted_class: str
            - matched_rule_ids: list
            - confidence: float
            - evidence: list of human-readable conditions
        """
        # Convert feature vector to dict for easier lookup
        x_dict = {name: x[i] for i, name in enumerate(self.feature_names)}
        
        active_rules = self.kb.get_active_rules()
        matched_rules = []
        
        for rule in active_rules:
            match = True
            for cond in rule["conditions"]:
                feat_val = x_dict.get(cond["feature"])
                if feat_val is None or not evaluate_condition(feat_val, cond["operator"], cond["threshold"]):
                    match = False
                    break
            
            if match:
                matched_rules.append(rule)
                
        if not matched_rules:
            return {
                "predicted_class": "no symbolic prediction",
                "matched_rule_ids": [],
                "confidence": 0.0,
                "evidence": []
            }
            
        # If there are matches, group them by predicted class
        # and find the single rule with the highest precision to determine the final class
        best_rule = max(matched_rules, key=lambda r: r.get("precision", 0.0))
        
        if "predicted_class" not in best_rule:
            raise ValueError(f"Schema Error: Rule '{best_rule.get('rule_id', 'UNKNOWN')}' is missing expected key 'predicted_class'. Keys found: {list(best_rule.keys())}")
            
        predicted_class = best_rule["predicted_class"]
        
        # We might have matched rules predicting other classes, but we'll report the top prediction's evidence
        class_matched_rules = [r for r in matched_rules if r.get("predicted_class") == predicted_class]
        
        rule_ids = [r["rule_id"] for r in class_matched_rules]
        confidence = best_rule.get("precision", 0.0)
        
        # Format evidence from the best rule
        evidence = []
        for cond in best_rule["conditions"]:
            evidence.append(f"{cond['feature']} {cond['operator']} {cond['threshold']:.4f} (Actual: {x_dict[cond['feature']]:.4f})")
            
        return {
            "predicted_class": predicted_class,
            "matched_rule_ids": rule_ids,
            "confidence": confidence,
            "evidence": evidence,
            "all_matches": matched_rules # useful for fusion per-class scoring
        }
