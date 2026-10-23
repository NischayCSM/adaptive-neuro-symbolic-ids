"""
Joint-training upper bound: NeuralIDS trained once on the union of all experiences'
training data (Adam lr=1e-3, batch 64, 5 epochs - same as the continual runs),
evaluated on each experience's test set.

Options (defaults reproduce the original run and its output name joint_summary_seed{s}.json):
    --epochs N          number of epochs (default 5)
    --class-weighted    inverse-frequency class weights in CrossEntropyLoss (default off)
    --save-checkpoint   also save the final model (implied by a non-default --epochs or
                        --class-weighted)
Runs with any of these write results/continual/joint_summary_seed{s}_e{N}[_cw].json and
results/continual/joint_checkpoints/joint_seed{s}_e{N}[_cw].pth instead.
"""

import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.device import DEVICE, print_device_info
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import find_project_root, set_seed, load_continual_data, evaluate_on_experiences


DEFAULT_EPOCHS = 5


def run_joint(seed=42, epochs=DEFAULT_EPOCHS, class_weighted=False, save_checkpoint=False):
    root = find_project_root()
    results_dir = root / "results" / "continual"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Non-default settings (or --save-checkpoint) get their own files so they never overwrite the main run.
    named = epochs != DEFAULT_EPOCHS or class_weighted or save_checkpoint
    settings = (f"_e{epochs}" + ("_cw" if class_weighted else "")) if named else ""

    set_seed(seed)
    data = load_continual_data(root)
    print_device_info()

    X = np.vstack([X_k for X_k, _ in data["train_splits"]])
    y = np.concatenate([y_k for _, y_k in data["train_splits"]])
    print(f"Joint training on {len(X)} rows (union of {data['num_exps']} experiences), seed={seed}")

    model = NeuralIDS(input_dim=data["input_dim"], output_dim=len(data["all_classes"])).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    class_weights = None
    if class_weighted:
        # Inverse class frequency: w_c = N / (K * n_c)
        counts = np.bincount(y, minlength=len(data["all_classes"]))
        class_weights = (len(y) / (len(counts) * np.maximum(counts, 1))).astype(np.float32)
        print("Class weights: " + ", ".join(f"{c}={w:.3f}" for c, w in zip(data["all_classes"], class_weights)))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights).to(DEVICE) if class_weights is not None else None)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32).to(DEVICE),
                                       torch.tensor(y, dtype=torch.long).to(DEVICE)),
        batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    accs, f1s = evaluate_on_experiences(model, data["test_splits"], DEVICE)
    summary = {
        "seed": seed, "n_train": int(len(X)),
        "acc_per_exp": accs, "f1_per_exp": f1s,
        "mean_acc_exp1_to_T-1": float(np.mean(accs[:-1])),
        "ACC": float(np.mean(accs)),
    }
    if named:
        ckpt_dir = results_dir / "joint_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"joint_seed{seed}{settings}.pth"
        torch.save(model.state_dict(), ckpt_path)
        summary.update({"epochs": epochs, "class_weighted": class_weighted,
                        "class_weights": class_weights.tolist() if class_weights is not None else None,
                        "checkpoint": str(ckpt_path.relative_to(root))})
    with open(results_dir / f"joint_summary_seed{seed}{settings}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Joint: acc per exp={[round(a, 4) for a in accs]}  ACC={summary['ACC']:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--class-weighted", action="store_true",
                        help="inverse-frequency class weights in CrossEntropyLoss")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="save the final model (output files then carry the settings in their names)")
    args = parser.parse_args()
    run_joint(args.seed, args.epochs, args.class_weighted, args.save_checkpoint)
