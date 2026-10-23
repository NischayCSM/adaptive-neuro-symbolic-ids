"""
Runs the naive sequential fine-tuning baseline (sequential_finetune.py, unchanged)
under fixed seeds and records the same metrics as replay_finetune.py.

Seeds are set globally before each call; the training code itself is untouched.
Seed 42 is run last so results/continual/continual_checkpoints/ holds the seed-42 model.
"""

import sys
import json
import shutil
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.device import DEVICE
from src.models.neural_ids import NeuralIDS
from src.continual.sequential_finetune import run_sequential_finetune
from src.continual.cl_eval import (
    find_project_root, set_seed, load_continual_data, cl_metrics,
    symbolic_results, fuse_predictions, predict_proba,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 42])
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    root = find_project_root()
    results_dir = root / "results" / "continual"
    kb_path = root / "data" / "processed" / "continual_kb.db"
    if not kb_path.exists():
        print(f"ERROR: {kb_path} not found. Run src/continual/continual_pipeline.py first.", file=sys.stderr)
        sys.exit(1)
    data = load_continual_data(root)
    num_classes = len(data["all_classes"])
    T = data["num_exps"]
    sym_cache = [symbolic_results(kb_path, data["feature_names"], X_k) for X_k, _ in data["test_splits"]]

    for seed in args.seeds:
        set_seed(seed)
        run_sequential_finetune()
        df = pd.read_csv(results_dir / "sequential_finetune.csv")
        df.to_csv(results_dir / f"naive_seed{seed}.csv", index=False)
        acc_matrix = df[[f"Exp{j}_Acc" for j in range(1, T + 1)]].values.tolist()
        f1_matrix = df[[f"Exp{j}_F1" for j in range(1, T + 1)]].values.tolist()

        model = NeuralIDS(input_dim=data["input_dim"], output_dim=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(results_dir / "continual_checkpoints" / f"neural_ids_exp_{T}.pth",
                                         map_location=DEVICE))
        ckpt_copy = results_dir / "continual_checkpoints" / f"neural_ids_exp_{T}_seed{seed}.pth"
        shutil.copy(results_dir / "continual_checkpoints" / f"neural_ids_exp_{T}.pth", ckpt_copy)

        fused_acc, fused_f1 = [], []
        for (X_k, y_k), sym in zip(data["test_splits"], sym_cache):
            probs = predict_proba(model, X_k, DEVICE)
            preds = fuse_predictions(probs, range(num_classes), sym, data["all_classes"],
                                     data["class_to_idx"], args.alpha)
            fused_acc.append(float(accuracy_score(y_k, preds)))
            fused_f1.append(float(f1_score(y_k, preds, average="macro", zero_division=0)))

        summary = {
            "seed": seed,
            "alpha": args.alpha,
            "acc_matrix": acc_matrix,
            "f1_matrix": f1_matrix,
            "neural": cl_metrics(acc_matrix),
            "kb_fused_final": {
                "kb_path": str(kb_path.relative_to(root)),
                "acc_per_exp": fused_acc,
                "f1_per_exp": fused_f1,
                "ACC": float(np.mean(fused_acc)),
                "retention": float(np.mean(fused_acc[:-1])),
            },
        }
        with open(results_dir / f"naive_summary_seed{seed}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[naive seed {seed}] {summary['neural']}  +KB retention={summary['kb_fused_final']['retention']:.4f}")


if __name__ == "__main__":
    main()
