"""
10-class (attack_cat) static baselines on the same train/test split and features as
xgboost_multi.joblib, one run per seed (default 42, 1, 2, 3, 4; the seed is each model's
random_state): DecisionTree, RandomForest and XGBoost (settings from src/models/baselines.py),
NeuralIDS (10 outputs, full training set, recipe from baselines.py: Adam lr=1e-3,
batch 1024, 10 epochs), and that seed's XGBoost fused with the static KB
(data/processed/knowledge_base.db) at alpha = 0.5 and 0.9.

Metrics: accuracy, macro-F1, macro FPR = mean over the 10 classes of FP / (FP + TN).
"""

import json
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from src.device import DEVICE
from src.models.baselines import RandomForestBaseline, XGBoostBaseline
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import find_project_root, set_seed, predict_proba, symbolic_results, fuse_predictions
from src.evaluation.evaluate_milestone2 import calculate_fpr


def main(seeds=(42, 1, 2, 3, 4)):
    root = find_project_root()
    processed = root / "data" / "processed"
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    with open(processed / "label_classes.json") as f:
        all_classes = json.load(f)
    idx = {c: i for i, c in enumerate(all_classes)}
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_train = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_train.npy"))])
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    labels = list(range(len(all_classes)))

    sym = symbolic_results(processed / "knowledge_base.db", pipe.feature_names, X_test)
    rows = []

    def record(name, seed, y_pred, train_s):
        rows.append({
            "Model": name, "Seed": seed, "Accuracy": accuracy_score(y_test, y_pred),
            "Macro_F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Macro_FPR": float(np.mean(list(calculate_fpr(y_test, y_pred, labels).values()))),
            "Train_seconds": train_s,
        })
        print(rows[-1])

    for seed in seeds:
        t = time.perf_counter()
        dt = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)
        record("DecisionTree", seed, dt.predict(X_test), time.perf_counter() - t)

        t = time.perf_counter()
        rf = RandomForestBaseline(random_state=seed).fit(X_train, y_train)
        record("RandomForest", seed, rf.predict(X_test), time.perf_counter() - t)

        t = time.perf_counter()
        xgb = XGBoostBaseline(random_state=seed).fit(X_train, y_train).model
        train_s = time.perf_counter() - t
        probs = xgb.predict_proba(X_test)
        record("XGBoost", seed, xgb.classes_[probs.argmax(1)], train_s)
        for alpha in (0.5, 0.9):
            record(f"XGBoost + static KB (alpha={alpha})", seed,
                   fuse_predictions(probs, xgb.classes_, sym, all_classes, idx, alpha), float("nan"))

        set_seed(seed)
        t = time.perf_counter()
        model = NeuralIDS(input_dim=X_train.shape[1], output_dim=len(all_classes)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(DEVICE),
                                           torch.tensor(y_train, dtype=torch.long).to(DEVICE)),
            batch_size=1024, shuffle=True)
        model.train()
        for _ in range(10):
            for bx, by in loader:
                optimizer.zero_grad()
                criterion(model(bx), by).backward()
                optimizer.step()
        record("NeuralIDS", seed, predict_proba(model, X_test, DEVICE).argmax(1), time.perf_counter() - t)

    df = pd.DataFrame(rows)
    out = root / "results" / "multiclass_baselines.csv"
    df.to_csv(out, index=False)
    summary = df.groupby("Model", sort=False)[["Accuracy", "Macro_F1", "Macro_FPR"]].agg(["mean", "std"])
    summary.to_csv(root / "results" / "multiclass_baselines_summary.csv")
    print(df.to_string(index=False))
    print(summary.to_string())
    print(f"Saved to {out} (std = sample std over seeds {seeds})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    main(parser.parse_args().seeds)
