import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from src.device import DEVICE, print_device_info
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.models.neural_ids import NeuralIDS
from src.continual.replay import ReplayBuffer
from src.continual.cl_eval import (
    find_project_root, set_seed, load_continual_data, evaluate_on_experiences,
    cl_metrics, matrix_rows, symbolic_results, fuse_predictions, predict_proba,
)

DEFAULT_BUFFER_SIZE = 2000
CLASS_WEIGHT_CLIP = (0.1, 50.0)


def mix_class_weights(y, num_classes):
    """Inverse-frequency weights on the actual training mix y: w_c = N / (C * n_c) over the C classes
    present (as in joint_train.py --class-weighted), i.e. normalised so the mean weight per training sample
    is 1, then clipped to CLASS_WEIGHT_CLIP. Classes absent from the mix get 1 (they have no targets, so
    their weight has no effect on the loss)."""
    counts = np.bincount(y, minlength=num_classes)
    present = counts > 0
    w = np.ones(num_classes)
    w[present] = len(y) / (present.sum() * counts[present])
    return np.clip(w, *CLASS_WEIGHT_CLIP).astype(np.float32), counts


def weight_align(model, old_classes, new_classes):
    """Weight Aligning (Zhao et al., CVPR 2020): rescale the output-layer weight rows of the new classes so
    their mean L2 norm equals that of the previously seen classes. Returns gamma = mean_norm(old) /
    mean_norm(new). Only the weights are rescaled, not the biases: the correction targets the weight-norm
    imbalance only, and the output biases are left exactly as trained."""
    W = model.output_layer.weight
    with torch.no_grad():
        norms = W.norm(dim=1)
        gamma = (norms[old_classes].mean() / norms[new_classes].mean()).item()
        W[new_classes] *= gamma
    return gamma


def run_replay_finetune(seed: int = 42, alpha: float = 0.5, buffer_size: int = DEFAULT_BUFFER_SIZE,
                        val_memory_per_class: int = 0, val_memory_max_fraction: float = None,
                        class_weighted: bool = False, weight_align_new: bool = False, processed_dir=None):
    """
    val_memory_per_class N > 0 (default 0 = off): before training on each experience, hold out N random
    training samples per class of that experience (fewer if the class has fewer; seeded), never train on
    them or put them in the replay buffer, and keep them in a validation memory that accumulates across
    experiences. Its indices are saved to replay_checkpoints/valmem_{tag}.npz; checkpoints, CSV and
    summary carry "_valmem{N}" in their names.
    val_memory_max_fraction f (default None = no cap): hold out at most floor(f * n) of a class's n training
    rows in the experience, so small classes keep rows for training (Worms: 44 -> 22 held out at f = 0.5).
    Names then also carry "_maxfrac{f}".
    class_weighted (default off): inverse-frequency class weights in CrossEntropyLoss, recomputed for each
    experience on its actual training mix (experience data + replay buffer); see mix_class_weights.
    weight_align_new (default off): Weight Aligning after training on each experience k > 1 (see
    weight_align); applied before the replay-buffer update and the checkpoint, so both use the aligned model.
    Names carry "_cw" and / or "_wa"; the weights and gammas are stored in the summary.
    processed_dir (default None = data/processed): read the preprocessed arrays from this directory instead;
    names then also carry "_{directory name}". If that directory has no continual_kb.db, the final
    KB-fused evaluation is skipped (kb_fused_final = null).
    """
    PROJECT_ROOT = find_project_root()
    PROCESSED_DIR = Path(processed_dir).resolve() if processed_dir else PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results" / "continual"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = RESULTS_DIR / "replay_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Non-default buffer sizes get their own files so they never overwrite the main run.
    tag = f"seed{seed}" if buffer_size == DEFAULT_BUFFER_SIZE else f"seed{seed}_buf{buffer_size}"
    if val_memory_per_class:
        tag = f"seed{seed}_buf{buffer_size}_valmem{val_memory_per_class}"
        if val_memory_max_fraction is not None:
            tag += f"_maxfrac{val_memory_max_fraction:g}"
    variant = ("_cw" if class_weighted else "") + ("_wa" if weight_align_new else "")
    tag += variant
    if processed_dir:
        tag += f"_{PROCESSED_DIR.name}"
    csv_out = RESULTS_DIR / f"replay_{tag}.csv"
    json_out = RESULTS_DIR / f"replay_summary_{tag}.json"
    kb_path = PROCESSED_DIR / "continual_kb.db"
    fuse_kb = kb_path.exists()
    if not fuse_kb and processed_dir:
        print(f"Note: {kb_path} not found; skipping the final KB-fused evaluation.")
    elif not fuse_kb:
        print(f"ERROR: {kb_path} not found. Run src/continual/continual_pipeline.py first.", file=sys.stderr)
        sys.exit(1)

    set_seed(seed)
    data = load_continual_data(PROJECT_ROOT, PROCESSED_DIR)
    print(f"Loaded data successfully. seed={seed} alpha={alpha} buffer_size={buffer_size}")

    num_exps = data["num_exps"]
    all_classes = data["all_classes"]
    num_classes = len(all_classes)

    device = DEVICE
    print_device_info()
    model = NeuralIDS(input_dim=data["input_dim"], output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    weights_log, gamma_log, seen = {}, {}, []

    replay_buffer = ReplayBuffer(max_size=buffer_size, fraction_hard=0.3)

    if val_memory_per_class:
        # Separate generator so the held-out choice does not consume the global NumPy random stream.
        val_rng = np.random.default_rng(seed)
        from src.preprocessing.pipeline import UNSW_NB15_Pipeline
        from src.continual.experience_builder import ExperienceBuilder
        y_cat_train = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib").label_encoder.inverse_transform(
            np.load(PROCESSED_DIR / "y_cat_train.npy"))
        builder = ExperienceBuilder(config_path=str(PROJECT_ROOT / "configs" / "continual.yaml"))
        val_mem = {"exp": [], "local_index": [], "global_index": [], "y": []}

    acc_matrix, f1_matrix, checkpoints = [], [], []

    for eid in range(1, num_exps + 1):
        print(f"\n--- Training on Experience {eid} (With Replay) ---")

        # Get data for current experience ONLY
        X_exp, y_exp = data["train_splits"][eid - 1]

        if val_memory_per_class:
            hold = np.zeros(len(y_exp), dtype=bool)
            for c in np.unique(y_exp):
                idx = np.flatnonzero(y_exp == c)
                n_hold = min(val_memory_per_class, len(idx))
                if val_memory_max_fraction is not None:
                    n_hold = min(n_hold, int(np.floor(val_memory_max_fraction * len(idx))))
                hold[val_rng.choice(idx, size=n_hold, replace=False)] = True
            local = np.flatnonzero(hold)
            # Row positions of this experience inside X_train.npy (same mask as ExperienceBuilder)
            exp_rows = np.flatnonzero(np.isin(y_cat_train, builder.get_classes_for_experience(eid)))
            val_mem["exp"].append(np.full(len(local), eid))
            val_mem["local_index"].append(local)
            val_mem["global_index"].append(exp_rows[local])
            val_mem["y"].append(y_exp[local])
            print(f"Validation memory: holding out {len(local)} samples ("
                  + ", ".join(f"{all_classes[c]}={int((y_exp[local] == c).sum())}" for c in np.unique(y_exp))
                  + "); never trained on")
            X_exp, y_exp = X_exp[~hold], y_exp[~hold]

        # Mix in Replay Buffer data if available
        X_replay, y_replay = replay_buffer.get_data()
        if len(X_replay) > 0:
            print(f"Mixing {len(X_replay)} replay samples into training batch.")
            X_train_batch = np.vstack([X_exp, X_replay])
            y_train_batch = np.concatenate([y_exp, y_replay])
        else:
            X_train_batch = X_exp
            y_train_batch = y_exp

        if class_weighted:
            w, counts = mix_class_weights(y_train_batch, num_classes)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
            weights_log[eid] = {all_classes[c]: float(w[c]) for c in range(num_classes) if counts[c] > 0}
            print("Class weights (training mix " + ", ".join(f"{all_classes[c]}={counts[c]}" for c in
                  range(num_classes) if counts[c] > 0) + "): "
                  + ", ".join(f"{c}={v:.3f}" for c, v in weights_log[eid].items()))

        X_tensor = torch.tensor(X_train_batch, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train_batch, dtype=torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        model.train()
        num_epochs = 5

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")

        new = [c for c in np.unique(y_exp).tolist() if c not in seen]
        if weight_align_new and eid > 1:
            gamma = weight_align(model, seen, new)
            gamma_log[eid] = gamma
            print(f"Weight aligning: gamma = {gamma:.4f} (old {[all_classes[c] for c in seen]}, "
                  f"new {[all_classes[c] for c in new]})")
        seen += new

        # Update Replay Buffer with current experience data (so it's available for next experiences)
        print(f"Updating Replay Buffer with Experience {eid} data...")
        replay_buffer.update(model, X_exp, y_exp, device)

        checkpoint_path = ckpt_dir / f"neural_ids_exp_{eid}_seed{seed}.pth"
        if buffer_size != DEFAULT_BUFFER_SIZE:
            checkpoint_path = ckpt_dir / f"neural_ids_exp_{eid}_seed{seed}_buf{buffer_size}.pth"
        if val_memory_per_class or variant or processed_dir:
            checkpoint_path = ckpt_dir / f"neural_ids_exp_{eid}_{tag}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoints.append(str(checkpoint_path.relative_to(PROJECT_ROOT)))
        print(f"Saved checkpoint to {checkpoint_path}")

        # Evaluate on every experience's test set
        print(f"Evaluating after Experience {eid}...")
        accs, f1s = evaluate_on_experiences(model, data["test_splits"], device)
        acc_matrix.append(accs)
        f1_matrix.append(f1s)

    df = pd.DataFrame(matrix_rows(acc_matrix, f1_matrix))
    df.to_csv(csv_out, index=False)
    print(f"\nReplay fine-tuning complete. Accuracy matrix saved to {csv_out}")
    print(df.to_string(index=False))

    metrics = cl_metrics(acc_matrix)
    print(f"Neural only: ACC={metrics['ACC']:.4f} BWT={metrics['BWT']:.4f} retention={metrics['retention']:.4f}")

    fused_acc, fused_f1 = [], []
    if fuse_kb:
        # Final model fused with the continual KB, on every experience
        print(f"\nEvaluating final model fused with {kb_path.name} (alpha={alpha})...")
        for eid, (X_k, y_k) in enumerate(data["test_splits"], start=1):
            probs = predict_proba(model, X_k, device)
            sym = symbolic_results(kb_path, data["feature_names"], X_k)
            preds = fuse_predictions(probs, range(num_classes), sym, all_classes, data["class_to_idx"], alpha)
            fused_acc.append(float(accuracy_score(y_k, preds)))
            fused_f1.append(float(f1_score(y_k, preds, average="macro", zero_division=0)))
            print(f"  Exp{eid}: fused acc={fused_acc[-1]:.4f} macro-F1={fused_f1[-1]:.4f}")

    summary = {
        "seed": seed,
        "alpha": alpha,
        "buffer_size": buffer_size,
        "classes": all_classes,
        "acc_matrix": acc_matrix,
        "f1_matrix": f1_matrix,
        "neural": metrics,
        "kb_fused_final": {
            "kb_path": str(kb_path.relative_to(PROJECT_ROOT)),
            "acc_per_exp": fused_acc,
            "f1_per_exp": fused_f1,
            "ACC": float(np.mean(fused_acc)),
            "retention": float(np.mean(fused_acc[:-1])),
        } if fuse_kb else None,
        "checkpoints": checkpoints,
    }
    if processed_dir:
        summary["processed_dir"] = str(PROCESSED_DIR.relative_to(PROJECT_ROOT))
    if class_weighted:
        summary["class_weights_per_exp"] = weights_log
    if weight_align_new:
        summary["weight_align_gamma_per_exp"] = gamma_log
    if val_memory_per_class:
        vm = {k: np.concatenate(v) for k, v in val_mem.items()}
        X_full = np.load(PROCESSED_DIR / "X_train.npy")
        X_held = np.concatenate([data["train_splits"][e - 1][0][vm["local_index"][vm["exp"] == e]]
                                 for e in range(1, num_exps + 1)])
        assert np.array_equal(X_full[vm["global_index"]], X_held), "validation memory indices do not match X_train"
        vm_path = ckpt_dir / f"valmem_{tag}.npz"
        np.savez(vm_path, **vm)
        counts = {all_classes[c]: int((vm["y"] == c).sum()) for c in range(num_classes)}
        n_buffer = len(replay_buffer.get_data()[1])
        if val_memory_max_fraction is not None:
            summary["val_memory_max_fraction"] = val_memory_max_fraction
        summary.update({"val_memory_per_class": val_memory_per_class, "val_memory_file":
                        str(vm_path.relative_to(PROJECT_ROOT)), "val_memory_counts": counts,
                        "val_memory_size": int(len(vm["y"])), "replay_buffer_size": int(n_buffer),
                        "total_memory": int(n_buffer + len(vm["y"]))})
        print(f"Validation memory: {len(vm['y'])} rows {counts}; replay buffer {n_buffer}; "
              f"total memory {n_buffer + len(vm['y'])}. Saved {vm_path}")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay-based continual fine-tuning of NeuralIDS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5, help="Neural weight in neuro-symbolic fusion")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    parser.add_argument("--val-memory-per-class", type=int, default=0,
                        help="hold out N training samples per class of each experience as a validation memory "
                             "(never trained on); default 0 = off")
    parser.add_argument("--val-memory-max-fraction", type=float, default=None,
                        help="with --val-memory-per-class: hold out at most this fraction of a class's rows "
                             "in the experience (default: no cap)")
    parser.add_argument("--class-weighted", action="store_true",
                        help="inverse-frequency class weights on each experience's training mix (default off)")
    parser.add_argument("--weight-align", action="store_true",
                        help="Weight Aligning of the new classes' output weights after each experience k > 1 "
                             "(default off)")
    parser.add_argument("--processed-dir", default=None,
                        help="directory with the preprocessed arrays (default: data/processed)")
    args = parser.parse_args()
    if args.val_memory_max_fraction is not None and not args.val_memory_per_class:
        parser.error("--val-memory-max-fraction needs --val-memory-per-class")
    run_replay_finetune(seed=args.seed, alpha=args.alpha, buffer_size=args.buffer_size,
                        val_memory_per_class=args.val_memory_per_class,
                        val_memory_max_fraction=args.val_memory_max_fraction,
                        class_weighted=args.class_weighted, weight_align_new=args.weight_align,
                        processed_dir=args.processed_dir)
