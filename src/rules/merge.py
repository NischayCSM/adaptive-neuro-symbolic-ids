import hashlib
import json

def _hash_conditions(conditions, predicted_class):
    """
    Create a canonical hash for a set of conditions and predicted class
    so duplicate rules can be identified.
    """
    # Sort conditions by feature, then operator, then threshold to ensure consistency
    sorted_conds = sorted(conditions, key=lambda x: (x["feature"], x["operator"], x["threshold"]))
    cond_str = json.dumps(sorted_conds, sort_keys=True)
    combined = f"{cond_str}_{predicted_class}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def merge_rules(rule_lists):
    """
    Pool rules from multiple generators and deduplicate them.
    If a rule is discovered by multiple generators, combine their generator tags.
    """
    merged = {}
    
    for rule_list in rule_lists:
        for rule in rule_list:
            rule_hash = _hash_conditions(rule["conditions"], rule["predicted_class"])
            
            if rule_hash in merged:
                # Merge generators
                existing = merged[rule_hash]
                for gen in rule["generators"]:
                    if gen not in existing["generators"]:
                        existing["generators"].append(gen)
                # Keep the max sample size
                existing["samples"] = max(existing["samples"], rule["samples"])
            else:
                merged[rule_hash] = dict(rule) # copy
                
    return list(merged.values())
