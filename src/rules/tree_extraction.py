import numpy as np
from sklearn.tree import DecisionTreeClassifier
import uuid

def train_rule_tree(X, y, max_depth=8):
    """Train a decision tree for rule extraction."""
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model

def extract_rules_from_tree(model, feature_names, class_names):
    """
    Traverse the Decision Tree and extract root-to-leaf paths as JSON-serializable rules.
    """
    tree_ = model.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    
    rules = []
    
    def recurse(node, current_conditions):
        if tree_.children_left[node] != tree_.children_right[node]: # Not a leaf
            feat_idx = feature[node]
            feat_name = feature_names[feat_idx] if feature_names is not None else f"feature_{feat_idx}"
            thresh = float(threshold[node])
            
            # Left child: <= threshold
            left_cond = list(current_conditions)
            left_cond.append({"feature": feat_name, "operator": "<=", "threshold": thresh})
            recurse(tree_.children_left[node], left_cond)
            
            # Right child: > threshold
            right_cond = list(current_conditions)
            right_cond.append({"feature": feat_name, "operator": ">", "threshold": thresh})
            recurse(tree_.children_right[node], right_cond)
        else: # Leaf
            value = tree_.value[node][0]
            predicted_class_idx = np.argmax(value)
            predicted_class = class_names[predicted_class_idx] if class_names is not None else str(predicted_class_idx)
            samples = int(tree_.n_node_samples[node])
            
            rules.append({
                "rule_id": str(uuid.uuid4()),
                "conditions": current_conditions,
                "predicted_class": predicted_class,
                "samples": samples,
                "generators": ["tree_extraction"]
            })
            
    recurse(0, [])
    return rules
