"""
EWC baseline (Kirkpatrick et al., 2017) on the same NeuralIDS and experience split.

Training matches sequential_finetune.py (Adam lr=1e-3, batch 64, 5 epochs, current
experience only) plus the penalty
    (lambda / 2) * sum_{past tasks t} sum_i F_t,i (theta_i - theta*_t,i)^2
F_t is the diagonal empirical Fisher (mean squared per-sample gradient of the
cross-entropy) on up to --fisher-samples training rows of task t, model in eval mode.

--save-checkpoint also saves the final model to
results/continual/ewc_checkpoints/neural_ids_exp_{T}_{tag}.pth and writes the CSV / summary as
ewc_{tag}_rerun.csv / ewc_summary_{tag}_rerun.json so the original run's files are kept.
"""

import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call, vmap, grad

from src.device import DEVICE, print_device_info
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import (
    find_project_root, set_seed, load_continual_data, evaluate_on_experiences,
    cl_metrics, matrix_rows,
)

DEFAULT_LAMBDA = 5000.0


def diagonal_fisher(model, X, y, device, n_samples):
    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    Xs = torch.tensor(X[idx], dtype=torch.float32).to(device)
    ys = torch.tensor(y[idx], dtype=torch.long).to(device)
    model.eval()
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_fn(p, x, t):
        return F.cross_entropy(functional_call(model, p, (x.unsqueeze(0),)), t.unsqueeze(0))

    grads = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, Xs, ys)
    return {k: (g ** 2).mean(0).detach().clone() for k, g in grads.items()}


def run_ewc_finetune(seed=42, ewc_lambda=DEFAULT_LAMBDA, fisher_samples=2000, save_checkpoint=False):
    root = find_project_root()
    results_dir = root / "results" / "continual"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"seed{seed}" if ewc_lambda == DEFAULT_LAMBDA else f"seed{seed}_lam{ewc_lambda:g}"
    out_tag = f"{tag}_rerun" if save_checkpoint else tag

    set_seed(seed)
    data = load_continual_data(root)
    print(f"Loaded data. seed={seed} lambda={ewc_lambda:g} fisher_samples={fisher_samples}")

    device = DEVICE
    print_device_info()
    model = NeuralIDS(input_dim=data["input_dim"], output_dim=len(data["all_classes"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    anchors = []  # list of (fisher, theta_star) per finished task
    acc_matrix, f1_matrix = [], []
    diagnostics = []

    for eid in range(1, data["num_exps"] + 1):
        print(f"\n--- Training on Experience {eid} (EWC) ---")
        X_exp, y_exp = data["train_splits"][eid - 1]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_exp, dtype=torch.float32).to(device),
            torch.tensor(y_exp, dtype=torch.long).to(device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(5):
            total_loss, total_pen = 0.0, 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                penalty = torch.zeros((), device=device)
                for fisher, theta_star in anchors:
                    for name, p in model.named_parameters():
                        penalty = penalty + (fisher[name] * (p - theta_star[name]) ** 2).sum()
                total = loss + (ewc_lambda / 2) * penalty
                total.backward()
                optimizer.step()
                total_loss += loss.item()
                total_pen += penalty.item()
            print(f"Epoch {epoch+1}/5 - CE: {total_loss/len(loader):.4f}  raw penalty: {total_pen/len(loader):.3e}"
                  f"  lambda/2*penalty: {ewc_lambda / 2 * total_pen/len(loader):.4g}")

        # Diagnostics: how far each layer moved from the previous anchor
        if anchors:
            prev = anchors[-1][1]
            drift = {k: float((p.detach() - prev[k]).norm()) for k, p in model.named_parameters()}
            print("  drift ||theta - theta*_prev|| per layer: "
                  + ", ".join(f"{k}={v:.4g}" for k, v in drift.items()))
            diagnostics.append({"exp": eid, "drift_from_prev_anchor": drift})

        fisher = diagonal_fisher(model, X_exp, y_exp, device, fisher_samples)
        theta_star = {k: v.detach().clone() for k, v in model.named_parameters()}
        anchors.append((fisher, theta_star))

        seen = sorted(set(np.unique(y_exp).tolist()))
        unseen = [c for c in range(len(data["all_classes"])) if c not in seen]
        fstats = {k: {"mean": float(v.mean()), "max": float(v.max())} for k, v in fisher.items()}
        for name in ("output_layer.weight", "output_layer.bias"):
            fstats[name]["mean_rows_this_exp_classes"] = float(fisher[name][seen].mean())
            fstats[name]["mean_rows_other_classes"] = float(fisher[name][unseen].mean())
        print(f"  Fisher after Exp{eid}:")
        for k, v in fstats.items():
            print("    " + k + ": " + ", ".join(f"{m}={x:.3e}" for m, x in v.items()))
        diagnostics.append({"exp": eid, "fisher": fstats})
        model.train()

        accs, f1s = evaluate_on_experiences(model, data["test_splits"], device)
        acc_matrix.append(accs)
        f1_matrix.append(f1s)
        print(f"After Exp{eid}: acc per exp = {[round(a, 4) for a in accs]}")

    pd.DataFrame(matrix_rows(acc_matrix, f1_matrix)).to_csv(results_dir / f"ewc_{out_tag}.csv", index=False)
    summary = {
        "seed": seed, "ewc_lambda": ewc_lambda, "fisher_samples": fisher_samples,
        "acc_matrix": acc_matrix, "f1_matrix": f1_matrix, "neural": cl_metrics(acc_matrix),
        "exp4_acc": acc_matrix[-1][-1], "diagnostics": diagnostics,
    }
    if save_checkpoint:
        ckpt_dir = results_dir / "ewc_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"neural_ids_exp_{data['num_exps']}_{tag}.pth"
        torch.save(model.state_dict(), ckpt_path)
        summary["checkpoint"] = str(ckpt_path.relative_to(root))
    with open(results_dir / f"ewc_summary_{out_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"EWC: {summary['neural']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ewc-lambda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--fisher-samples", type=int, default=2000)
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="save the final model; CSV / summary are then written with a _rerun suffix")
    args = parser.parse_args()
    run_ewc_finetune(args.seed, args.ewc_lambda, args.fisher_samples, args.save_checkpoint)
