"""
Effect of the continual KB versions on retention of the final replay model.

For buffer sizes 2000 / 5000 / 10000 and seeds 42, 1, 2, 3, 4, loads the replay checkpoint
saved after the last experience (results/continual/replay_checkpoints/) and evaluates it on
every experience's test set: neural only, and fused (NeuroSymbolicFuser) with each KB at
alpha = 0.5. Retention = mean accuracy on Exp 1..T-1. Then the fused evaluation is repeated
for alpha in {0.3, 0.5, 0.7} at buffer 10000 with the KB that has the highest mean retention
there at alpha = 0.5.

Fails if a checkpoint is missing (retrain it with src/continual/replay_finetune.py), and checks
that the neural accuracies match the per-seed replay summaries written at training time.

Writes results/continual/kb_retention.csv and results/continual/kb_retention_alpha.csv.
"""

import sys
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from src.device import DEVICE
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import (
    find_project_root, load_continual_data, predict_proba, symbolic_results, fuse_predictions,
)

SEEDS = [42, 1, 2, 3, 4]
BUFFERS = [2000, 5000, 10000]
KBS = {"v1": "continual_kb_v1.db", "v2": "continual_kb_v2.db", "v3": "continual_kb_v3.db"}
ALPHAS = [0.3, 0.5, 0.7]


def main():
    root = find_project_root()
    processed = root / "data" / "processed"
    out_dir = root / "results" / "continual"
    data = load_continual_data(root)
    T, classes, idx = data["num_exps"], data["all_classes"], data["class_to_idx"]
    tests = data["test_splits"]

    for name, f in KBS.items():
        if not (processed / f).exists():
            print(f"ERROR: {processed / f} not found.", file=sys.stderr)
            sys.exit(1)
    print("Running SymbolicReasoner for each KB on each experience's test set...")
    sym = {k: [symbolic_results(processed / f, data["feature_names"], X_k) for X_k, _ in tests]
           for k, f in KBS.items()}

    def tag(seed, b):
        return f"seed{seed}" if b == 2000 else f"seed{seed}_buf{b}"

    def load_probs(seed, b):
        ckpt = out_dir / "replay_checkpoints" / f"neural_ids_exp_{T}_{tag(seed, b)}.pth"
        if not ckpt.exists():
            print(f"ERROR: {ckpt} not found. Retrain with: python -m src.continual.replay_finetune "
                  f"--seed {seed} --buffer-size {b}", file=sys.stderr)
            sys.exit(1)
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=len(classes)).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return [predict_proba(model, X_k, DEVICE) for X_k, _ in tests]

    def accs(preds_per_exp):
        return [float(accuracy_score(y_k, p)) for (_, y_k), p in zip(tests, preds_per_exp)]

    rows, alpha_rows = [], []
    for b in BUFFERS:
        for seed in SEEDS:
            probs = load_probs(seed, b)
            neural = accs([p.argmax(1) for p in probs])
            with open(out_dir / f"replay_summary_{tag(seed, b)}.json") as f:
                stored = json.load(f)["acc_matrix"][-1]
            if not np.allclose(neural, stored, atol=1e-9):
                print(f"ERROR: neural accuracies {neural} differ from replay_summary_{tag(seed, b)}.json {stored}",
                      file=sys.stderr)
                sys.exit(1)
            row = {"buffer": b, "seed": seed}
            variants = {"neural": neural}
            for k in KBS:
                variants[k] = accs([fuse_predictions(p, range(len(classes)), s, classes, idx, 0.5)
                                    for p, s in zip(probs, sym[k])])
            for name, a in variants.items():
                for j in range(T):
                    row[f"{name}_exp{j+1}"] = a[j]
                row[f"{name}_retention"] = float(np.mean(a[:T - 1]))
            rows.append(row)
            print({k: round(v, 4) for k, v in row.items() if k.endswith("retention") or k.endswith(f"exp{T}")},
                  f"buffer={b} seed={seed}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "kb_retention.csv", index=False)

    b10 = df[df["buffer"] == 10000]
    best = max(KBS, key=lambda k: b10[f"{k}_retention"].mean())
    print(f"Best KB at buffer 10000, alpha 0.5 (mean retention): {best}")
    for seed in SEEDS:
        probs = load_probs(seed, 10000)
        for alpha in ALPHAS:
            a = accs([fuse_predictions(p, range(len(classes)), s, classes, idx, alpha)
                      for p, s in zip(probs, sym[best])])
            alpha_rows.append({"kb": best, "buffer": 10000, "seed": seed, "alpha": alpha,
                               **{f"exp{j+1}": a[j] for j in range(T)}, "retention": float(np.mean(a[:T - 1]))})
    pd.DataFrame(alpha_rows).to_csv(out_dir / "kb_retention_alpha.csv", index=False)
    print("Saved kb_retention.csv and kb_retention_alpha.csv")


if __name__ == "__main__":
    main()
