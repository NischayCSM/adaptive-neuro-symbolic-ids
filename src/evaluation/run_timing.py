"""
Environment report and timing:
- Python / PyTorch / xgboost / shap / scikit-learn versions, CPU, GPU, RAM
- XGBoost (10-class, settings from src/models/baselines.py) training time on X_train
  (trained fresh here; results/models/xgboost_multi.joblib is not overwritten)
- Inference time per 1,000 test flows: XGBoost alone, and XGBoost + KB fusion
  (SymbolicReasoner + NeuroSymbolicFuser, alpha=0.5, active rules cached) for the
  static KB and the continual KB. Each timing is repeated --repeats times.
"""

import sys
import json
import time
import platform
import argparse
import subprocess

import numpy as np

from src.models.baselines import XGBoostBaseline
from src.continual.cl_eval import find_project_root, symbolic_results, fuse_predictions


def powershell(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return r.stdout.strip()


def environment():
    import torch, xgboost, shap, sklearn
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda_available": torch.cuda.is_available(),
        "xgboost": xgboost.__version__, "shap": shap.__version__, "scikit-learn": sklearn.__version__,
        "numpy": np.__version__, "os": platform.platform(),
    }
    if sys.platform == "win32":
        env["cpu"] = powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        env["cpu_cores_logical"] = powershell("(Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum")
        env["gpu"] = powershell("(Get-CimInstance Win32_VideoController).Name -join '; '")
        env["ram_gb"] = round(int(powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")) / 2**30, 1)
    else:
        env["cpu"] = platform.processor()
    return env


def ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "runs": v.tolist()}


def main(repeats=3):
    root = find_project_root()
    processed = root / "data" / "processed"
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    idx = {c: i for i, c in enumerate(all_classes)}
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_train = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_train.npy"))])
    n = len(X_test)

    out = {"environment": environment(), "n_train": len(X_train), "n_test": n, "repeats": repeats}
    print(json.dumps(out["environment"], indent=2))

    train_s = []
    for _ in range(repeats):
        t = time.perf_counter()
        XGBoostBaseline().fit(X_train, y_train)
        train_s.append(time.perf_counter() - t)
    out["xgboost_train_seconds"] = ms(train_s)
    print("XGBoost train seconds:", out["xgboost_train_seconds"])

    xgb = XGBoostBaseline.load(root / "results" / "models" / "xgboost_multi.joblib").model
    xgb_ms = []
    for _ in range(repeats):
        t = time.perf_counter()
        probs = xgb.predict_proba(X_test)
        xgb_ms.append((time.perf_counter() - t) / n * 1000 * 1000)
    out["xgboost_inference_ms_per_1000_flows"] = ms(xgb_ms)
    print("XGBoost ms/1000 flows:", out["xgboost_inference_ms_per_1000_flows"])

    for kb_name in ("knowledge_base.db", "continual_kb.db"):
        fused_ms = []
        for _ in range(repeats):
            t = time.perf_counter()
            probs = xgb.predict_proba(X_test)
            sym = symbolic_results(processed / kb_name, pipe.feature_names, X_test)
            fuse_predictions(probs, xgb.classes_, sym, all_classes, idx, 0.5)
            fused_ms.append((time.perf_counter() - t) / n * 1000 * 1000)
        key = f"fused_{kb_name.replace('.db', '')}_inference_ms_per_1000_flows"
        out[key] = ms(fused_ms)
        print(key, out[key])

    with open(root / "results" / "environment_timing.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args().repeats)
