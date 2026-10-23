"""
One common metric for every continual method: the FINAL model (after the last experience)
evaluated on the FULL 10-class test set (processed X_test / y_cat_test, all classes together),
for seeds 42, 1, 2, 3, 4.

Methods (final checkpoints):
    naive fine-tuning     results/continual/continual_checkpoints/neural_ids_exp_4_seed{s}.pth
    replay 2000/5000/10000 results/continual/replay_checkpoints/neural_ids_exp_4_seed{s}[_buf{b}].pth
                          (+ fused with KB v1 / KB v2 at alpha = 0.5, same fusion as run_kb_retention.py)
    EWC (each lambda)     results/continual/ewc_checkpoints/neural_ids_exp_4_seed{s}[_lam{l}].pth
                          (ewc_finetune.py --save-checkpoint)
    joint training        results/continual/joint_checkpoints/joint_seed{s}_e{N}[_cw].pth
                          (joint_train.py --save-checkpoint / --epochs / --class-weighted)

Metrics: accuracy, macro-F1, weighted-F1, macro FPR (mean over the 10 classes of FP / (FP + TN),
calculate_fpr as in the static baseline scripts), per-class recall, and recency share = fraction of
all test predictions in the last experience's classes (vs. the true fraction of those classes).

Each checkpoint is also evaluated on the per-experience test splits and compared with the
accuracies stored when it was trained (Ref_MaxAbsDiff) and, for retrained EWC / joint models, with
the originally reported run (Orig_MaxAbsDiff).

Fails, listing them, if any checkpoint is missing.
Writes results/continual/final_model_eval.csv and results/continual/final_model_eval.md.
"""

import sys
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.device import DEVICE
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import (
    find_project_root, set_seed, load_continual_data, predict_proba, symbolic_results, fuse_predictions,
)
from src.evaluation.evaluate_milestone2 import calculate_fpr

SEEDS = [42, 1, 2, 3, 4]
BUFFERS = [2000, 5000, 10000]
EWC_LAMBDAS = [5000.0, 1e5, 1e6, 1e7]
JOINT_SETTINGS = {"_e5": "Joint training (5 epochs)", "_e20": "Joint training (20 epochs)",
                  "_e20_cw": "Joint training (20 epochs, class-weighted)"}
KBS = {"KB v1": "continual_kb_v1_seed42.db", "KB v2": "continual_kb_v2.db"}
ALPHA = 0.5
STATIC_REFS = ["XGBoost", "NeuralIDS"]


def neural_methods(out_dir, T):
    """[(method, kb or None, {seed: (checkpoint, ref summary + key, original summary + key or None)})]"""
    cc, rc = out_dir / "continual_checkpoints", out_dir / "replay_checkpoints"
    ec, jc = out_dir / "ewc_checkpoints", out_dir / "joint_checkpoints"
    last = lambda key: (lambda d: d[key][-1])  # last row of a stored accuracy matrix
    methods = [("Naive fine-tuning", {s: (cc / f"neural_ids_exp_{T}_seed{s}.pth",
                                          (out_dir / f"naive_summary_seed{s}.json", last("acc_matrix")), None)
                                      for s in SEEDS})]
    for b in BUFFERS:
        tag = (lambda s: f"seed{s}") if b == 2000 else (lambda s, b=b: f"seed{s}_buf{b}")
        methods.append((f"Replay (buffer={b})", {s: (rc / f"neural_ids_exp_{T}_{tag(s)}.pth",
                                                     (out_dir / f"replay_summary_{tag(s)}.json", last("acc_matrix")),
                                                     None) for s in SEEDS}))
    for lam in EWC_LAMBDAS:
        tag = (lambda s: f"seed{s}") if lam == 5000.0 else (lambda s, lam=lam: f"seed{s}_lam{lam:g}")
        methods.append((f"EWC (lambda={lam:g})", {s: (ec / f"neural_ids_exp_{T}_{tag(s)}.pth",
                                                      (out_dir / f"ewc_summary_{tag(s)}_rerun.json", last("acc_matrix")),
                                                      (out_dir / f"ewc_summary_{tag(s)}.json", last("acc_matrix")))
                                                  for s in SEEDS}))
    get = lambda d: d["acc_per_exp"]
    for settings, name in JOINT_SETTINGS.items():
        methods.append((name, {s: (jc / f"joint_seed{s}{settings}.pth",
                                   (out_dir / f"joint_summary_seed{s}{settings}.json", get),
                                   (out_dir / f"joint_summary_seed{s}.json", get) if settings == "_e5" else None)
                               for s in SEEDS}))
    return methods


def main():
    root = find_project_root()
    processed = root / "data" / "processed"
    out_dir = root / "results" / "continual"
    set_seed(42)
    data = load_continual_data(root)
    T, classes, idx = data["num_exps"], data["all_classes"], data["class_to_idx"]
    labels = list(range(len(classes)))

    # Full multi-class test set, decoded the same way as load_continual_data
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_test = np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])

    # Do the experience test splits partition the full test set?
    X_cat = np.vstack([X_k for X_k, _ in data["test_splits"]])
    y_cat = np.concatenate([y_k for _, y_k in data["test_splits"]])

    def sorted_rows(X, y):
        A = np.column_stack([X, y])
        return A[np.lexsort(A.T[::-1])]

    split_check = {
        "rows_full": int(len(X_test)), "rows_concat": int(len(X_cat)),
        "rows_per_exp": [int(len(y_k)) for _, y_k in data["test_splits"]],
        "classes_full": int(len(np.unique(y_test))), "classes_concat": int(len(np.unique(y_cat))),
        "class_counts_equal": bool(np.array_equal(np.bincount(y_test, minlength=len(classes)),
                                                  np.bincount(y_cat, minlength=len(classes)))),
        "same_rows_as_multiset": bool(np.array_equal(sorted_rows(X_test, y_test), sorted_rows(X_cat, y_cat),
                                                     equal_nan=True)),
    }
    print(f"Experience test splits vs full test set: {split_check}")

    recent = np.unique(data["test_splits"][-1][1])
    recent_names = [classes[c] for c in recent]
    true_recency = float(np.isin(y_test, recent).mean())
    print(f"Last experience's classes: {recent_names}; true share in test set = {true_recency:.4f}")

    methods = neural_methods(out_dir, T)
    missing = [str(ckpt.relative_to(root)) for _, runs in methods for ckpt, _, _ in runs.values()
               if not ckpt.exists()]
    missing += [str((processed / f).relative_to(root)) for f in KBS.values() if not (processed / f).exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    print("Running SymbolicReasoner for each KB on the full test set...")
    sym = {k: symbolic_results(processed / f, data["feature_names"], X_test) for k, f in KBS.items()}

    def metrics(y_pred):
        recalls = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
        return {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Macro_F1": float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
            "Weighted_F1": float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)),
            "Macro_FPR": float(np.mean(list(calculate_fpr(y_test, y_pred, labels).values()))),
            **{f"Recall_{c}": float(r) for c, r in zip(classes, recalls)},
            "Recency_Share": float(np.isin(y_pred, recent).mean()),
            "True_Recency_Share": true_recency,
        }

    def max_diff(model, ref):
        if ref is None or not ref[0].exists():
            return np.nan
        with open(ref[0]) as f:
            stored = ref[1](json.load(f))
        accs = [accuracy_score(y_k, predict_proba(model, X_k, DEVICE).argmax(1)) for X_k, y_k in data["test_splits"]]
        return float(np.max(np.abs(np.array(accs) - np.array(stored))))

    rows = []
    for method, runs in methods:
        for seed, (ckpt, ref, orig) in runs.items():
            model = NeuralIDS(input_dim=data["input_dim"], output_dim=len(classes)).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            probs = predict_proba(model, X_test, DEVICE)
            common = {"Seed": seed, "Checkpoint": str(ckpt.relative_to(root)),
                      "Ref_MaxAbsDiff": max_diff(model, ref), "Orig_MaxAbsDiff": max_diff(model, orig)}
            rows.append({"Method": method, **common, **metrics(probs.argmax(1))})
            print(f"{method} seed={seed}: acc={rows[-1]['Accuracy']:.4f} macroF1={rows[-1]['Macro_F1']:.4f} "
                  f"recency={rows[-1]['Recency_Share']:.4f} ref_diff={common['Ref_MaxAbsDiff']:.2e}")
            if method.startswith("Replay"):
                for k in KBS:
                    preds = fuse_predictions(probs, labels, sym[k], classes, idx, ALPHA)
                    rows.append({"Method": f"{method} + {k} (alpha={ALPHA})", **common, **metrics(preds)})
                    print(f"  + {k}: acc={rows[-1]['Accuracy']:.4f} macroF1={rows[-1]['Macro_F1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "final_model_eval.csv", index=False)
    write_markdown(df, root, out_dir / "final_model_eval.md", classes, recent_names, true_recency, split_check)
    print(f"Saved {out_dir / 'final_model_eval.csv'} and final_model_eval.md")


def write_markdown(df, root, path, classes, recent_names, true_recency, split_check):
    order = list(dict.fromkeys(df["Method"]))
    g = df.groupby("Method", sort=False)
    fmt = lambda col, m: f"{g[col].mean()[m]:.4f} ± {g[col].std(ddof=1)[m]:.4f}"

    static = pd.read_csv(root / "results" / "multiclass_baselines_summary.csv", header=[0, 1], index_col=0)
    static_n = pd.read_csv(root / "results" / "multiclass_baselines.csv").groupby("Model")["Seed"].nunique()

    L = ["# Final-model evaluation on the full 10-class test set", "",
         f"Final model after Exp 4, evaluated on all {split_check['rows_full']} test rows "
         f"(all {split_check['classes_full']} classes together). Seeds {SEEDS}; mean ± sample std (ddof=1). "
         f"Fused rows: NeuroSymbolicFuser, alpha = {ALPHA} (KB v1 = {KBS['KB v1']}, KB v2 = {KBS['KB v2']}).",
         "",
         f"Recency share = fraction of all test predictions in Exp 4's classes ({', '.join(recent_names)}). "
         f"True fraction of those classes in the test set: **{true_recency:.4f}**.",
         "",
         "Experience split check: the four experience test splits have "
         f"{' + '.join(map(str, split_check['rows_per_exp']))} = {split_check['rows_concat']} rows and "
         f"{split_check['classes_concat']} classes; full test set has {split_check['rows_full']} rows and "
         f"{split_check['classes_full']} classes; class counts equal: {split_check['class_counts_equal']}; "
         f"same rows (as a multiset): {split_check['same_rows_as_multiset']}.",
         "",
         "## Overall metrics", "",
         "| Method | n | Accuracy | Macro-F1 | Weighted-F1 | Macro-FPR | Recency share |",
         "|---|---|---|---|---|---|---|"]
    for m in order:
        L.append(f"| {m} | {g.size()[m]} | {fmt('Accuracy', m)} | {fmt('Macro_F1', m)} | {fmt('Weighted_F1', m)} "
                 f"| {fmt('Macro_FPR', m)} | {fmt('Recency_Share', m)} |")
    for m in STATIC_REFS:
        s = static.loc[m]
        cell = lambda col: f"{s[(col, 'mean')]:.4f} ± {s[(col, 'std')]:.4f}"
        L.append(f"| *{m} (static, full training set; NOT continual - reference only)* | {static_n[m]} "
                 f"| {cell('Accuracy')} | {cell('Macro_F1')} | n/a | {cell('Macro_FPR')} | n/a |")
    L += ["", "Static reference rows are copied from results/multiclass_baselines_summary.csv (same test set; "
              "weighted-F1 and recency share were not recorded there).", "",
          "## Per-class recall (mean over seeds)", "",
          "| Method | " + " | ".join(classes) + " |",
          "|---|" + "---|" * len(classes)]
    for m in order:
        L.append(f"| {m} | " + " | ".join(f"{g[f'Recall_{c}'].mean()[m]:.4f}" for c in classes) + " |")
    L += ["", "## Checkpoint check", "",
          "Max |accuracy difference| on the per-experience test splits between each loaded checkpoint and the "
          "accuracies stored when it was trained (Ref), and for retrained EWC / 5-epoch joint models, vs. the "
          "originally reported run (Orig). Maximum over seeds.", "",
          "| Method | Ref | Orig |", "|---|---|---|"]
    for m in order:
        if " + KB " in m:
            continue
        orig = g["Orig_MaxAbsDiff"].max()[m]
        L.append(f"| {m} | {g['Ref_MaxAbsDiff'].max()[m]:.2e} | {'n/a' if np.isnan(orig) else f'{orig:.2e}'} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
