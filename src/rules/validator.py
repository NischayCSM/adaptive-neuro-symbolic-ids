import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

def _eval_condition(cond, X, feature_names):
    feat_idx = feature_names.index(cond["feature"])
    op = cond["operator"]
    thresh = cond["threshold"]
    
    if op == ">":
        return X[:, feat_idx] > thresh
    elif op == "<=":
        return X[:, feat_idx] <= thresh
    elif op == ">=":
        return X[:, feat_idx] >= thresh
    elif op == "<":
        return X[:, feat_idx] < thresh
    elif op == "==":
        return X[:, feat_idx] == thresh
    return np.ones(len(X), dtype=bool)

def _get_rule_mask(rule, X, feature_names):
    mask = np.ones(len(X), dtype=bool)
    for cond in rule["conditions"]:
        mask &= _eval_condition(cond, X, feature_names)
    return mask

def _check_conflict(rule, accepted_rules):
    """
    Returns 1 if the rule conflicts with an existing accepted rule.
    A conflict is defined as having the exact same conditions but a different predicted class.
    (Simplified conflict check for now).
    """
    # Create canonical condition string
    r_conds = sorted(rule["conditions"], key=lambda x: (x["feature"], x["operator"], x["threshold"]))
    
    for ar in accepted_rules:
        ar_conds = sorted(ar["conditions"], key=lambda x: (x["feature"], x["operator"], x["threshold"]))
        if r_conds == ar_conds and rule["predicted_class"] != ar["predicted_class"]:
            return 1
    return 0

class RuleValidator:
    def __init__(self, config_path="configs/rules.yaml", lift_check=False, min_lift=1.5, max_match_fraction=0.5,
                 gain_check=False, min_gain=0.5):
        """
        lift_check: optional extra filter (default off, so the original behaviour is unchanged).
            When on, a rule is rejected unless BOTH
                lift           = precision / base_rate(class in the same validation half) >= min_lift
                match_fraction = support / validation size                               <= max_match_fraction
        gain_check: optional extra filter (default off). When on, a rule is rejected unless BOTH
                gain           = (precision - base_rate) / (1 - base_rate)               >= min_gain
                match_fraction                                                           <= max_match_fraction
            (max_match_fraction is shared with the lift check.)
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.lift_check = lift_check
        self.min_lift = min_lift
        self.max_match_fraction = max_match_fraction
        self.gain_check = gain_check
        self.min_gain = min_gain

        self.w1 = self.config["weights"].get("precision", 0.4)
        self.w2 = self.config["weights"].get("coverage", 0.3)
        self.w3 = self.config["weights"].get("stability", 0.2)
        self.w4 = self.config["weights"].get("conflict", 0.2)
        self.min_support = self.config.get("min_support", 50)
        self.threshold = self.config.get("acceptance_threshold", 0.6)
        
        self.accepted_rules = []

    def effective_min_support(self, n_val):
        # Dynamic min support based on validation set size (0.5%)
        return min(self.min_support, max(10, int(n_val * 0.005)))

    def score_and_filter(self, candidate_rules, X_val1, y_cat_val1, X_val2, y_cat_val2, feature_names,
                         existing_rules=None):
        """
        existing_rules: optional list of rules already in the KB. When given, a candidate also
        counts as conflicting if it has the same conditions as an existing rule but a different class.
        """
        self.accepted_rules = []
        existing_rules = existing_rules or []
        self.n_conflicts = 0
        self.n_lift_rejected = 0
        self.n_gain_rejected = 0
        
        actual_min_support = self.effective_min_support(len(X_val1))
        
        accepted_counts = defaultdict(int)
        rejected_counts = defaultdict(int)
        rejected_samples = []  # collect a few rejected-rule diagnostics
        
        # Sort candidates by number of generators (descending) as a heuristic proxy for confidence
        # so higher confidence rules are processed first for conflict checking.
        sorted_candidates = sorted(candidate_rules, key=lambda r: len(r["generators"]), reverse=True)
        
        for rule in sorted_candidates:
            # 1. Evaluate on val1
            mask1 = _get_rule_mask(rule, X_val1, feature_names)
            support1 = np.sum(mask1)
            
            if support1 < actual_min_support:
                for gen in rule["generators"]: rejected_counts[gen] += 1
                if len(rejected_samples) < 3:
                    rejected_samples.append({
                        "class": rule["predicted_class"],
                        "reason": f"low support ({support1} < {actual_min_support})",
                        "support": int(support1),
                    })
                continue
                
            pred_class = rule["predicted_class"]
            correct1 = np.sum((y_cat_val1 == pred_class) & mask1)
            precision = correct1 / support1
            
            total_class_val1 = np.sum(y_cat_val1 == pred_class)
            coverage = correct1 / total_class_val1 if total_class_val1 > 0 else 0
            
            # 2. Evaluate on val2 (Stability)
            mask2 = _get_rule_mask(rule, X_val2, feature_names)
            support2 = np.sum(mask2)
            if support2 > 0:
                correct2 = np.sum((y_cat_val2 == pred_class) & mask2)
                stability = correct2 / support2
            else:
                stability = 0
                
            # 3. Conflict Check
            conflict = _check_conflict(rule, self.accepted_rules + existing_rules)
            self.n_conflicts += conflict
            
            # 4. Score
            # Use effective coverage to prevent catch-all rules from inflating the score
            effective_coverage = min(coverage, 0.5)
            score = (self.w1 * precision) + (self.w2 * effective_coverage) + (self.w3 * stability) - (self.w4 * conflict)
            rule["validation_score"] = float(score)
            rule["precision"] = float(precision)
            rule["coverage"] = float(coverage)
            
            # Hard precision floor: Reject any rule with < 85% precision
            if precision < 0.85:
                for gen in rule["generators"]: rejected_counts[gen] += 1
                if len(rejected_samples) < 3:
                    rejected_samples.append({
                        "class": pred_class,
                        "reason": f"precision {precision:.4f} < floor 0.85",
                        "precision": float(precision),
                        "coverage": float(coverage),
                        "stability": float(stability),
                        "conflict": int(conflict),
                        "score": float(score),
                    })
                continue

            # Optional lift check. Absolute precision alone is not enough: when one class dominates
            # the validation data (Normal is ~86% of experience 1), a near-trivial rule that fires on
            # almost every row "inherits" that base rate as its precision and clears the 0.85 floor
            # while carrying no information (e.g. "is_sm_ips_ports <= 4.66 -> Normal" fires on 98% of
            # test rows). Requiring precision to beat the class base rate by a factor (lift) and capping
            # the fraction of rows a rule may match rejects such rules. Note that a class whose base
            # rate exceeds 1 / min_lift can then get no rules at all from this validation data.
            if self.lift_check:
                base_rate = total_class_val1 / len(X_val1)
                lift = precision / base_rate if base_rate > 0 else 0.0
                match_fraction = support1 / len(X_val1)
                rule["lift"] = float(lift)
                rule["match_fraction"] = float(match_fraction)
                if lift < self.min_lift or match_fraction > self.max_match_fraction:
                    self.n_lift_rejected += 1
                    for gen in rule["generators"]: rejected_counts[gen] += 1
                    if len(rejected_samples) < 3:
                        rejected_samples.append({
                            "class": pred_class,
                            "reason": f"lift {lift:.3f} (min {self.min_lift}) / match fraction "
                                      f"{match_fraction:.3f} (max {self.max_match_fraction})",
                        })
                    continue

            # Optional normalised precision gain. gain = (precision - base_rate) / (1 - base_rate) is the
            # share of the possible improvement over the base rate that the rule achieves: 0 means the rule
            # is no more precise than always predicting its class, 1 means it is perfectly precise, and a
            # negative value means it is worse than the base rate. Unlike lift, whose maximum 1 / base_rate
            # is close to 1 for a dominant class (about 1.03-1.18 per experience here), gain can reach 1 for
            # every class, so dominant classes can still get rules. With min_gain = 0.5, a class at base
            # rate 0.86 needs precision >= 0.93. The match-fraction cap still rejects near-trivial rules.
            if self.gain_check:
                base_rate = total_class_val1 / len(X_val1)
                gain = (precision - base_rate) / (1 - base_rate) if base_rate < 1 else 0.0
                match_fraction = support1 / len(X_val1)
                rule["gain"] = float(gain)
                rule["match_fraction"] = float(match_fraction)
                if gain < self.min_gain or match_fraction > self.max_match_fraction:
                    self.n_gain_rejected += 1
                    for gen in rule["generators"]: rejected_counts[gen] += 1
                    if len(rejected_samples) < 3:
                        rejected_samples.append({
                            "class": pred_class,
                            "reason": f"gain {gain:.3f} (min {self.min_gain}) / match fraction "
                                      f"{match_fraction:.3f} (max {self.max_match_fraction})",
                        })
                    continue

            if score >= self.threshold:
                self.accepted_rules.append(rule)
                for gen in rule["generators"]: accepted_counts[gen] += 1
            else:
                for gen in rule["generators"]: rejected_counts[gen] += 1
                if len(rejected_samples) < 3:
                    rejected_samples.append({
                        "class": pred_class,
                        "reason": f"score {score:.4f} < threshold {self.threshold}",
                        "precision": float(precision),
                        "coverage": float(coverage),
                        "stability": float(stability),
                        "conflict": int(conflict),
                        "score": float(score),
                    })
                
        # Logging
        print("=== Rule Validation Summary ===")
        all_gens = set(list(accepted_counts.keys()) + list(rejected_counts.keys()))
        for gen in all_gens:
            acc = accepted_counts[gen]
            rej = rejected_counts[gen]
            print(f"Generator '{gen}': Accepted={acc}, Rejected={rej}")
        print(f"  Threshold: {self.threshold}  |  Weights: prec={self.w1}, cov={self.w2}, stab={self.w3}, confl={self.w4}")
        if self.lift_check:
            print(f"  Lift check: min lift {self.min_lift}, max match fraction {self.max_match_fraction} "
                  f"-> rejected {self.n_lift_rejected} rules that passed the precision floor")
        if self.gain_check:
            print(f"  Gain check: min gain {self.min_gain}, max match fraction {self.max_match_fraction} "
                  f"-> rejected {self.n_gain_rejected} rules that passed the earlier checks")
        if rejected_samples:
            print("  Sample rejected rules:")
            for s in rejected_samples:
                print(f"    -> class={s['class']}, {s['reason']}")
                if "precision" in s:
                    print(f"      prec={s['precision']:.4f}, cov={s['coverage']:.4f}, "
                          f"stab={s['stability']:.4f}, conflict={s['conflict']}")
            
        return self.accepted_rules
