import numpy as np
import uuid
from sklearn.tree import DecisionTreeClassifier
def generate_shap_rules(X, y, feature_names, class_names, top_k=3, subsample_size=1000):
    """
    Trains a fast Decision Tree and uses SHAP values to extract high-impact features
    for each class, creating threshold-based rules from them.
    """
    try:
        import shap
    except Exception as e:
        print(f"SHAP rule generation unavailable: {e}. Skipping this generator.")
        return []
        
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    
    # Subsample for SHAP if dataset is too large
    if len(X) > subsample_size:
        indices = np.random.choice(len(X), subsample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    
    # shap_values could be a list (multiclass) or array
    if isinstance(shap_values, list):
        # List of length num_classes, each is (n_samples, n_features)
        shap_array = np.array(shap_values)
    elif len(shap_values.shape) == 3:
        # shape: (n_samples, n_features, n_classes)
        # transpose to (n_classes, n_samples, n_features)
        shap_array = np.transpose(shap_values, (2, 0, 1))
    else:
        # binary classification
        shap_array = np.array([-shap_values, shap_values])
        
    tree_class_names = list(model.classes_)
    rules = []
    
    for c_idx, cls_val in enumerate(np.unique(y)):
        # Mean absolute SHAP value for each feature for this class
        mean_abs_shap = np.mean(np.abs(shap_array[c_idx]), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:]
        
        # Data for this class
        X_cls = X[y == cls_val]
        if len(X_cls) < 10:
            continue
            
        cls_name = tree_class_names[list(model.classes_).index(cls_val)]
        
        conditions = []
        for feat_idx in top_indices:
            feat_name = feature_names[feat_idx]
            feat_vals = X_cls[:, feat_idx]
            median_val = np.median(feat_vals)
            
            # Simple heuristic: if the class values are mostly above overall median, use > threshold
            overall_median = np.median(X[:, feat_idx])
            if median_val > overall_median:
                conditions.append({
                    "feature": feat_name,
                    "operator": ">=",
                    "threshold": float(overall_median)
                })
            else:
                conditions.append({
                    "feature": feat_name,
                    "operator": "<=",
                    "threshold": float(overall_median)
                })
                
        if conditions:
            rules.append({
                "rule_id": str(uuid.uuid4()),
                "conditions": conditions,
                "predicted_class": cls_name,
                "samples": len(X_cls),
                "generators": ["shap_rules"]
            })
            
    return rules
