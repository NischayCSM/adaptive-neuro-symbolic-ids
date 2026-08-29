import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.rules.tree_extraction import extract_rules_from_tree

def generate_association_rules(X, y, feature_names, class_names, top_k_features=20):
    """
    Generates high-coverage association rules by training shallow Decision Stumps (depth=2)
    on individual high-variance features.
    """
    rules = []
    
    # Find high variance features to avoid useless constant features
    variances = np.var(X, axis=0)
    top_k = min(top_k_features, X.shape[1])
    top_feature_indices = np.argsort(variances)[-top_k:]
    
    for feat_idx in top_feature_indices:
        # Train a stump on just this one feature
        X_single = X[:, feat_idx].reshape(-1, 1)
        model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=50, random_state=42)
        model.fit(X_single, y)
        
        # We pass a temporary feature_names list that only contains this feature
        # to correctly map the index 0 back to the real feature name
        temp_feature_names = [feature_names[feat_idx]]
        tree_class_names = list(model.classes_)
        
        stump_rules = extract_rules_from_tree(model, temp_feature_names, tree_class_names)
        
        # Change generator name for tracking
        for r in stump_rules:
            r["generators"] = ["association_rules"]
            rules.append(r)
            
    return rules
