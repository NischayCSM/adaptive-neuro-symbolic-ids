"""
Rerun of the KB-fused rows of results/continual/final_model_eval.csv with the deterministic NeuroSymbolicFuser
(exact score ties go to the lowest class index; previously Python set order, which depended on PYTHONHASHSEED).

For every replay checkpoint (buffers 2000 / 5000 / 10000, seeds 42, 1, 2, 3, 4): the neural-only row is recomputed
and must reproduce final_model_eval.csv (max |difference| <= 1e-9, else nothing is written); the fused rows
(KB v1 / KB v2, alpha 0.5, cl_eval.fuse_predictions) are recomputed and must equal the vectorised V0 argmax
(fusion_variants.v0_linear) on every row. All other rows are copied unchanged.

Writes results/continual/final_model_eval_v2.csv and final_model_eval_v2.md (same layout as final_model_eval.md,
plus a list of every value that changed by more than 0.0005); refuses to overwrite them. Re-launches itself with
PYTHONHASHSEED=0.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

TOL = 1e-9
REPORT_DELTA = 0.0005


def main():
    from src.continual.cl_eval import find_project_root
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_final_model_eval_v2"], cwd=root,
                                env=env).returncode)

    import torch
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from src.device import DEVICE
    from src.models.neural_ids import NeuralIDS
    from src.continual.cl_eval import set_seed, load_continual_data, predict_proba, symbolic_results, fuse_predictions
    from src.evaluation.evaluate_milestone2 import calculate_fpr
    from src.evaluation.run_final_model_eval import neural_methods, write_markdown, KBS, ALPHA
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    from src.reasoning import fusion_variants as fv

    processed, out_dir = root / "data" / "processed", root / "results" / "continual"
    out_csv, out_md = out_dir / "final_model_eval_v2.csv", out_dir / "final_model_eval_v2.md"
    for p in (out_csv, out_md):
        if p.exists():
            sys.exit(f"ERROR: {p.relative_to(root)} already exists; not overwriting")
    old = pd.read_csv(out_dir / "final_model_eval.csv")
    set_seed(42)
    data = load_continual_data(root)
    T, classes, idx = data["num_exps"], data["all_classes"], data["class_to_idx"]
    labels = list(range(len(classes)))
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_test = np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    recent = np.unique(data["test_splits"][-1][1])
    recent_names = [classes[c] for c in recent]
    true_recency = float(np.isin(y_test, recent).mean())
    X_cat = np.vstack([X_k for X_k, _ in data["test_splits"]])
    y_cat = np.concatenate([y_k for _, y_k in data["test_splits"]])
    srt = lambda X, y: (lambda A: A[np.lexsort(A.T[::-1])])(np.column_stack([X, y]))
    split_check = {
        "rows_full": int(len(X_test)), "rows_concat": int(len(X_cat)),
        "rows_per_exp": [int(len(y_k)) for _, y_k in data["test_splits"]],
        "classes_full": int(len(np.unique(y_test))), "classes_concat": int(len(np.unique(y_cat))),
        "class_counts_equal": bool(np.array_equal(np.bincount(y_test, minlength=len(classes)),
                                                  np.bincount(y_cat, minlength=len(classes)))),
        "same_rows_as_multiset": bool(np.array_equal(srt(X_test, y_test), srt(X_cat, y_cat), equal_nan=True)),
    }

    def metrics(y_pred):  # identical to run_final_model_eval.main's metrics()
        recalls = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
        return {"Accuracy": float(accuracy_score(y_test, y_pred)),
                "Macro_F1": float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
                "Weighted_F1": float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)),
                "Macro_FPR": float(np.mean(list(calculate_fpr(y_test, y_pred, labels).values()))),
                **{f"Recall_{c}": float(r) for c, r in zip(classes, recalls)},
                "Recency_Share": float(np.isin(y_pred, recent).mean()), "True_Recency_Share": true_recency}

    metric_cols = list(metrics(np.zeros(len(y_test), dtype=int)).keys())
    sym = {k: symbolic_results(processed / f, data["feature_names"], X_test) for k, f in KBS.items()}
    S = {k: fv.rule_evidence(processed / f, data["feature_names"], X_test, idx)[0] for k, f in KBS.items()}

    new = old.copy()
    neural_diff, v0_mismatch, n_tied = 0.0, 0, {}
    for method, runs in neural_methods(out_dir, T):
        if not method.startswith("Replay"):
            continue
        for seed, (ckpt, _, _) in runs.items():
            model = NeuralIDS(input_dim=data["input_dim"], output_dim=len(classes)).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            probs = predict_proba(model, X_test, DEVICE)
            ref = old[(old["Method"] == method) & (old["Seed"] == seed)].iloc[0]
            mine = metrics(probs.argmax(1))
            neural_diff = max(neural_diff, max(abs(mine[c] - float(ref[c])) for c in metric_cols))
            for k in KBS:
                preds = fuse_predictions(probs, labels, sym[k], classes, idx, ALPHA)
                v0_mismatch += int((preds != fv.v0_linear(probs, S[k], ALPHA)).sum())
                n_tied[(method, k, seed)] = int(fv.top_tied(fv.linear_scores(probs, S[k], ALPHA)).sum())
                name = f"{method} + {k} (alpha={ALPHA})"
                row = (new["Method"] == name) & (new["Seed"] == seed)
                assert row.sum() == 1, name
                for c, val in metrics(preds).items():
                    new.loc[row, c] = val
                print(f"{name} seed {seed}: tied rows {n_tied[(method, k, seed)]}, "
                      f"macro-F1 {float(old.loc[row, 'Macro_F1'].iloc[0]):.4f} -> {float(new.loc[row, 'Macro_F1'].iloc[0]):.4f}")
    print(f"Neural-only rows vs final_model_eval.csv: max |difference| = {neural_diff:.1e}; "
          f"fused predictions differing from the V0 argmax: {v0_mismatch}")
    if neural_diff > TOL or v0_mismatch:
        sys.exit("ERROR: neural rows not reproduced or fuser != V0 argmax; not writing")

    # ---- every value that changed by more than REPORT_DELTA: per seed and in the markdown's mean / std -----------
    fused = new["Method"].str.contains(" + KB ", regex=False)
    changes = []
    for i in new.index[fused]:
        for c in metric_cols:
            d = float(new.at[i, c]) - float(old.at[i, c])
            if abs(d) > REPORT_DELTA:
                changes.append(("seed", new.at[i, "Method"], int(new.at[i, "Seed"]), c, float(old.at[i, c]),
                                float(new.at[i, c]), d))
    go, gn = old[fused].groupby("Method"), new[fused].groupby("Method")
    for stat, f in (("mean", lambda g: g.mean(numeric_only=True)), ("std", lambda g: g.std(ddof=1, numeric_only=True))):
        a, b = f(go), f(gn)
        for m in a.index:
            for c in metric_cols:
                d = float(b.at[m, c]) - float(a.at[m, c])
                if abs(d) > REPORT_DELTA:
                    changes.append((stat, m, None, c, float(a.at[m, c]), float(b.at[m, c]), d))

    new.to_csv(out_csv, index=False)
    write_markdown(new, root, out_md, classes, recent_names, true_recency, split_check)
    text = out_md.read_text(encoding="utf-8").replace(
        "# Final-model evaluation on the full 10-class test set",
        "# Final-model evaluation on the full 10-class test set (v2: deterministic fuser)", 1)
    L = ["", "## v2: changes against final_model_eval.csv", "",
         "Fused rows recomputed with the deterministic NeuroSymbolicFuser (ties to the lowest class index; "
         "`results/fusion/tie_break_check.md`); every fused prediction equals the vectorised V0 argmax. Neural-only "
         f"rows reproduce final_model_eval.csv (max |difference| {neural_diff:.1e}); all non-fused rows are copied "
         "unchanged. Exact top-score ties per fused run (rows of 175341), seeds [42, 1, 2, 3, 4]:", "",
         "| Fused run | Tied rows per seed |", "|---|---|"]
    for (method, k) in dict.fromkeys((m, k) for m, k, _ in n_tied):
        L.append(f"| {method} + {k} | " + ", ".join(str(n_tied[(method, k, s)]) for s in (42, 1, 2, 3, 4)) + " |")
    L += ["", f"Values that changed by more than {REPORT_DELTA} (per seed, and in the mean / std shown above):", ""]
    if changes:
        L += ["| Level | Method | Seed | Metric | Old | New | Change |", "|---|---|---|---|---|---|---|"]
        L += [f"| {lv} | {m} | {'' if s is None else s} | {c} | {o:.4f} | {n:.4f} | {d:+.4f} |"
              for lv, m, s, c, o, n, d in changes]
    else:
        L.append("None.")
    out_md.write_text(text + "\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(changes)} values changed by more than {REPORT_DELTA}. Saved {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
