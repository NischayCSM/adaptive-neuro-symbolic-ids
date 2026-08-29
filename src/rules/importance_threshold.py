import numpy as np
import uuid
from sklearn.ensemble import RandomForestClassifier

def generate_importance_rules(X, y, feature_names, class_names, top_k=5):
    """
    Trains a RandomForest to find the top K most important features.
    For each class, creates a rule capturing the mean +/- 1 standard deviation
    range for these top features.
    """
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_k:]
    
    tree_class_names = list(model.classes_)
    
    rules = []
    
    for cls_val in np.unique(y):
        # Data for this class
        X_cls = X[y == cls_val]
        if len(X_cls) < 10:
            continue
            
        cls_name = tree_class_names[list(model.classes_).index(cls_val)]
        
        conditions = []
        for feat_idx in top_indices:
            feat_name = feature_names[feat_idx]
            feat_vals = X_cls[:, feat_idx]
            
            mean_val = np.mean(feat_vals)
            std_val = np.std(feat_vals)
            
            # Skip if there's zero variance (constant feature for this class)
            if std_val < 1e-5:
                continue
                
            lower_bound = float(mean_val - std_val)
            upper_bound = float(mean_val + std_val)
            
            conditions.append({
                "feature": feat_name,
                "operator": ">=",
                "threshold": lower_bound
            })
            conditions.append({
                "feature": feat_name,
                "operator": "<=",
                "threshold": upper_bound
            })
            
        if conditions:
            rules.append({
                "rule_id": str(uuid.uuid4()),
                "conditions": conditions,
                "predicted_class": cls_name,
                "samples": len(X_cls),
                "generators": ["importance_threshold"]
            })
            
    return rules
