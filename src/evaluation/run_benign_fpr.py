"""
Operator-facing false-alarm and miss rates of the final settings.

    Normal->attack rate = share of true-Normal flows predicted as ANY attack class (benign false-alarm rate)
    attack->Normal rate = share of true-attack flows predicted as Normal (miss rate)
Macro-FPR (mean one-vs-rest FPR over the 10 classes) is reported next to them.

Settings, each on the evaluation split of results/fusion/split_indices.npz (140272 rows) and on the full test set:
    XGBoost alone              results/models/xgboost_multi.joblib (the one saved static model, n = 1)
    XGBoost + V2               neutral-absence fusion, alpha 0.5, KB continual_kb_v1_gain_clean.db
    R1 detector alone          replay 10000, class-weighted, validation memory 200 / max fraction 0.5; seeds 42, 1, 2, 3, 4
    R1 control                 detector-only stacking fit on the validation memory
    R1 V4                      rule-feature stacking fit on the validation memory, same KB
Cross-checks before writing: Normal->attack = 1 - Recall_Normal in results/fusion/fusion_final.csv (static, evaluation
split), results/fusion/detector_variants.csv (R1, evaluation split) and Benign_FPR in
results/final_test_phase2_metrics.csv (R1 detector / V4, both scopes); macro-FPR against the same files.

Writes results/fusion/benign_fpr.csv and results/fusion/benign_fpr.md; refuses to overwrite them. Re-launches itself
with PYTHONHASHSEED=0.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

TOL = 1e-9
SEEDS = [42, 1, 2, 3, 4]
KB_FILE = "continual_kb_v1_gain_clean.db"
CONTROL = "R1 control (stacking on log P only)"
SYSTEMS = ["XGBoost alone", "XGBoost + V2 (alpha 0.5)", "R1 detector alone", CONTROL, "R1 + V4 stacking"]
SCOPES = ["evaluation split", "full test set"]


def main():
    from src.continual.cl_eval import find_project_root
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_benign_fpr"], cwd=root, env=env).returncode)

    import torch
    from src.device import DEVICE
    from src.models.neural_ids import NeuralIDS
    from src.models.baselines import XGBoostBaseline
    from src.continual.cl_eval import set_seed, load_continual_data, predict_proba
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    from src.reasoning import fusion_variants as fv
    from src.evaluation.evaluate_milestone2 import calculate_fpr
    from src.evaluation.run_fusion_final import TAG

    processed, results = root / "data" / "processed", root / "results"
    out_csv, out_md = results / "fusion" / "benign_fpr.csv", results / "fusion" / "benign_fpr.md"
    for p in (out_csv, out_md):
        if p.exists():
            sys.exit(f"ERROR: {p.relative_to(root)} already exists; not overwriting")
    ckpt_dir = results / "continual" / "replay_checkpoints"
    tag = lambda s: f"seed{s}_{TAG}_cw"
    set_seed(42)

    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels, normal = len(classes), list(range(len(classes))), idx["Normal"]
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    ev = np.load(results / "fusion" / "split_indices.npz")["evaluation"]
    rows_of = {"evaluation split": ev, "full test set": np.arange(len(y_test))}
    S_te, F_te, M_te = fv.rule_evidence(processed / KB_FILE, feature_names, X_test, idx)
    zeros = lambda n: np.zeros((n, K))

    def rates(scope, system, seed, pred):
        r = rows_of[scope]
        y, p = y_test[r], pred[r]
        ben = y == normal
        return {"Scope": scope, "System": system, "Seed": seed, "Rows": len(y), "Normal_Rows": int(ben.sum()),
                "Attack_Rows": int((~ben).sum()),
                "Normal_to_Attack": float((p[ben] != normal).mean()),
                "Attack_to_Normal": float((p[~ben] == normal).mean()),
                "Macro_FPR": float(np.mean(list(calculate_fpr(y, p, labels).values())))}

    out = []
    xgb = XGBoostBaseline.load(results / "models" / "xgboost_multi.joblib").model
    assert list(xgb.classes_) == labels
    P_x = xgb.predict_proba(X_test)
    preds = {"XGBoost alone": P_x.argmax(1), "XGBoost + V2 (alpha 0.5)": fv.v2_neutral(P_x, S_te, F_te, 0.5)}
    for system, pred in preds.items():
        out += [rates(sc, system, 42, pred) for sc in SCOPES]
    for s in SEEDS:
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_dir / f"neural_ids_exp_4_{tag(s)}.pth", map_location=DEVICE))
        vm = np.load(ckpt_dir / f"valmem_{tag(s)}.npz")
        X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
        P_vm, P_te = predict_proba(model, X_vm, DEVICE), predict_proba(model, X_test, DEVICE)
        S_vm, F_vm, M_vm = fv.rule_evidence(processed / KB_FILE, feature_names, X_vm, idx)
        n_vm, n_te = len(y_vm), len(y_test)
        ctrl = fv.StackingFuser(detector_only=True).fit(P_vm, zeros(n_vm), zeros(n_vm), zeros(n_vm), y_vm)
        v4 = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm)
        preds = {"R1 detector alone": P_te.argmax(1),
                 CONTROL: ctrl.predict(P_te, zeros(n_te), zeros(n_te), zeros(n_te)),
                 "R1 + V4 stacking": v4.predict(P_te, S_te, F_te, M_te)}
        for system, pred in preds.items():
            out += [rates(sc, system, s, pred) for sc in SCOPES]
        print(f"seed {s}: " + ", ".join(f"{r['System']} ({r['Scope'][:4]}) N->A {r['Normal_to_Attack']:.4f}"
                                         for r in out[-6:]))
    df = pd.DataFrame(out)

    # ---- cross-checks ------------------------------------------------------------------------------------------
    get = lambda sc, sy, s: df[(df["Scope"] == sc) & (df["System"] == sy) & (df["Seed"] == s)].iloc[0]
    diffs = []
    ff = pd.read_csv(results / "fusion" / "fusion_final.csv")
    for system, method in [("XGBoost alone", "Detector alone"), ("XGBoost + V2 (alpha 0.5)", "V2 neutral-absence (alpha=0.5)")]:
        ref = ff[(ff["Detector"] == "XGBoost") & (ff["KB"] == "KB v1_gain_clean") & (ff["Method"] == method)].iloc[0]
        mine = get("evaluation split", system, 42)
        diffs += [abs(mine["Normal_to_Attack"] - (1 - ref["Recall_Normal"])), abs(mine["Macro_FPR"] - ref["Macro_FPR"])]
    dv = pd.read_csv(results / "fusion" / "detector_variants.csv")
    dv = dv[dv["Variant"] == "R1"]
    ph = pd.read_csv(results / "final_test_phase2_metrics.csv")
    for system, method in [("R1 detector alone", "Detector alone"), (CONTROL, "Control: stacking on log P only"),
                           ("R1 + V4 stacking", "V4 stacking")]:
        for s in SEEDS:
            ref = dv[(dv["Method"] == method) & (dv["Seed"] == s)].iloc[0]
            mine = get("evaluation split", system, s)
            diffs += [abs(mine["Normal_to_Attack"] - (1 - ref["Recall_Normal"])), abs(mine["Macro_FPR"] - ref["Macro_FPR"])]
            if system != CONTROL:
                for sc in SCOPES:
                    r2 = ph[(ph["Scope"] == sc) & (ph["System"] == system) & (ph["Seed"] == s)].iloc[0]
                    diffs.append(abs(get(sc, system, s)["Normal_to_Attack"] - r2["Benign_FPR"]))
    max_diff = max(diffs)
    print(f"Cross-check vs fusion_final.csv / detector_variants.csv / final_test_phase2_metrics.csv: "
          f"max |difference| = {max_diff:.1e} ({len(diffs)} values)")
    if max_diff > TOL:
        sys.exit("ERROR: recomputed rates do not reproduce the results files; not writing")
    df.to_csv(out_csv, index=False)
    write_markdown(df, max_diff, len(diffs), out_md)
    print(f"Saved {out_csv} and {out_md}")


def write_markdown(df, max_diff, n_checked, path):
    def cell(x):
        return f"{x.iloc[0]:.4f}" if len(x) == 1 else f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"
    first = df.iloc[0]
    L = ["# Benign false-alarm and miss rates of the final settings", "",
         "**Normal→attack** = share of true-Normal flows predicted as any attack class (the false-alarm rate an "
         "operator sees). **Attack→Normal** = share of true-attack flows predicted as Normal (missed attacks). "
         "**Macro-FPR** = mean over the 10 classes of the one-vs-rest FP / (FP + TN), the Phase I / II Test 1 metric. "
         "Continual rows: mean ± sample std over seeds [42, 1, 2, 3, 4]; static rows: the single saved model "
         "`xgboost_multi.joblib` (n = 1). KB `continual_kb_v1_gain_clean.db`; control and V4 fit on the validation "
         "memory only.", "",
         f"Rows: evaluation split {int(df[df['Scope'] == 'evaluation split']['Rows'].iloc[0])} "
         f"({int(df[df['Scope'] == 'evaluation split']['Normal_Rows'].iloc[0])} Normal), full test set "
         f"{int(df[df['Scope'] == 'full test set']['Rows'].iloc[0])} "
         f"({int(df[df['Scope'] == 'full test set']['Normal_Rows'].iloc[0])} Normal).", "",
         f"Cross-check: Normal→attack equals 1 − Recall_Normal of results/fusion/fusion_final.csv (static) and "
         f"results/fusion/detector_variants.csv (R1), and Benign_FPR of results/final_test_phase2_metrics.csv; "
         f"macro-FPR equals those files' Macro_FPR. Max |difference| over {n_checked} values: {max_diff:.1e}.", "",
         "| Setting | Scope | n | Normal→attack | Attack→Normal | Macro-FPR |", "|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        for scope in SCOPES:
            g = df[(df["System"] == system) & (df["Scope"] == scope)]
            L.append(f"| {system} | {scope} | {len(g)} | {cell(g['Normal_to_Attack'])} | {cell(g['Attack_to_Normal'])} "
                     f"| {cell(g['Macro_FPR'])} |")
    L += ["", "## Normal→attack per seed (evaluation split)", "",
          "| Setting | " + " | ".join(str(s) for s in SEEDS) + " |", "|---|" + "---|" * len(SEEDS)]
    for system in SYSTEMS[2:]:
        g = df[(df["System"] == system) & (df["Scope"] == "evaluation split")].set_index("Seed")
        L.append(f"| {system} | " + " | ".join(f"{g.loc[s, 'Normal_to_Attack']:.4f}" for s in SEEDS) + " |")
    ev = lambda sy: df[(df["System"] == sy) & (df["Scope"] == "evaluation split")]
    L += ["", "## Summary", "",
          f"- Static: Normal→attack {cell(ev('XGBoost alone')['Normal_to_Attack'])} (XGBoost alone), "
          f"{cell(ev('XGBoost + V2 (alpha 0.5)')['Normal_to_Attack'])} (+ V2); target < 0.05.",
          f"- Continual final configuration (R1 + V4): Normal→attack {cell(ev('R1 + V4 stacking')['Normal_to_Attack'])}, "
          f"against macro-FPR {cell(ev('R1 + V4 stacking')['Macro_FPR'])} for the same predictions.",
          f"- Continual, without rules: R1 detector alone {cell(ev('R1 detector alone')['Normal_to_Attack'])}, "
          f"control {cell(ev(CONTROL)['Normal_to_Attack'])}."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
