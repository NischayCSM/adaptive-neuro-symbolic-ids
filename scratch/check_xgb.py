import os
from pathlib import Path
import traceback

PROJECT_ROOT = Path(os.getcwd())
if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
    for parent in PROJECT_ROOT.parents:
        if parent.name == "adaptive-neuro-symbolic-ids":
            PROJECT_ROOT = parent
            break

import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models.baselines import XGBoostBaseline
    import xgboost as xgb
    
    xgb_path = PROJECT_ROOT / "results" / "models" / "xgboost.joblib"
    print(f"Loading {xgb_path}...")
    wrapper = XGBoostBaseline.load(xgb_path)
    model = wrapper.model
    print("Model loaded successfully.")
    
    print("Classes:", getattr(model, 'classes_', 'NO_CLASSES_ATTR'))
    print("Num Classes:", getattr(model, 'n_classes_', 'NO_N_CLASSES_ATTR'))
    
    # Try predicting something small just to check its output shape
    import numpy as np
    X_dummy = np.random.rand(2, 50)  # guessing some feature dim
    try:
        X_test = np.load(PROJECT_ROOT / "data" / "processed" / "X_test.npy")
        X_dummy = X_test[:2]
    except Exception as e:
        print("Could not load X_test:", e)
        
    proba = model.predict_proba(X_dummy)
    print("Proba shape:", proba.shape)
    
except Exception as e:
    traceback.print_exc()
