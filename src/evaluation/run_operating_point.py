"""
Operating-point analysis of the final continual configuration (attack vs Normal).

Models, seeds 42, 1, 2, 3, 4: R1 detector alone (softmax P), R1 control (detector-only stacking) and R1 + V4
(rule-feature stacking, KB continual_kb_v1_gain_clean.db); control and V4 are fit on the validation memory, exactly as
in run_benign_fpr.py. Static reference for the ROC: xgboost_multi.joblib (n = 1).

Alarm score s = 1 - P(Normal) (for control / V4: the stacker's predict_proba). A flow raises an alarm when s > t.
Threshold t is chosen on the VALIDATION MEMORY ONLY (never test data): with the validation-memory Normal scores sorted
in decreasing order, t is the (m+1)-th largest, m = floor(target x n_Normal), so at most a share `target` of the
memory's Normal flows score above t. Targets 0.05 and 0.01.
  threshold source "validation memory" = the scores of the deployed model on the memory (as specified). For control and
      V4 those rows are the stacker's own training rows (in-sample).
  threshold source "validation memory, out-of-fold" (control and V4 only) = 5-fold stratified cross-fitting inside the
      memory (seed 42): each memory row is scored by a stacker fit on the other four folds. Still no test data.
At each threshold, on the evaluation split of results/fusion/split_indices.npz: Normal->attack rate, attack->Normal miss
rate, detection rate per attack class (share of that class with s > t) and multi-class macro-F1 when flows with s <= t
are labelled Normal and the others get the model's top attack class. "default argmax" rows score the models' own argmax
predictions; they must reproduce results/fusion/benign_fpr.csv and the macro-F1 of results/fusion/detector_variants.csv.
ROC curve and ROC-AUC of attack vs Normal on the evaluation split for the three models and static XGBoost.

Writes results/fusion/operating_point.csv, results/fusion/operating_point.md and results/figures_phase2/F7_roc.png;
refuses to overwrite them. Re-launches itself with PYTHONHASHSEED=0.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

SEEDS = [42, 1, 2, 3, 4]
KB_FILE = "continual_kb_v1_gain_clean.db"
TARGETS = [0.05, 0.01]
N_FOLDS = 5
TOL = 1e-9
DET, CTRL, V4, XGB = "R1 detector alone", "R1 control (stacking on log P only)", "R1 + V4 stacking", "XGBoost alone (static)"
MODELS = [DET, CTRL, V4]
SRC_VM, SRC_OOF, SRC_ARGMAX = "validation memory", "validation memory, out-of-fold", "default argmax"


def threshold(normal_scores, target):
    """t such that at most floor(target * n) of the given Normal scores are > t."""
    s = np.sort(np.asarray(normal_scores))[::-1]
    m = int(np.floor(target * len(s)))
    return float(s[m]) if m < len(s) else float(-np.inf)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Operating-point analysis of the final continual configuration")
    parser.add_argument("--redo-figure", action="store_true",
                        help="recompute, check against the saved operating_point.csv, rewrite only F7_roc.png")
    parser.parse_args()
    from src.continual.cl_eval import find_project_root
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_operating_point", *sys.argv[1:]], cwd=root,
                                env=env).returncode)
    redo_figure = "--redo-figure" in sys.argv[1:]  # recompute, check against the saved CSV, rewrite only F7

    import torch
    from sklearn.metrics import f1_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold
    from src.device import DEVICE
    from src.models.neural_ids import NeuralIDS
    from src.models.baselines import XGBoostBaseline
    from src.continual.cl_eval import set_seed, load_continual_data, predict_proba
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    from src.reasoning import fusion_variants as fv
    from src.evaluation.evaluate_milestone2 import calculate_fpr
    from src.evaluation.run_fusion_final import TAG

    processed, results = root / "data" / "processed", root / "results"
    out_csv, out_md = results / "fusion" / "operating_point.csv", results / "fusion" / "operating_point.md"
    out_fig = results / "figures_phase2" / "F7_roc.png"
    for p in ((out_csv, out_md, out_fig) if not redo_figure else ()):
        if p.exists():
            sys.exit(f"ERROR: {p.relative_to(root)} already exists; not overwriting")
    ckpt_dir = results / "continual" / "replay_checkpoints"
    tag = lambda s: f"seed{s}_{TAG}_cw"
    set_seed(42)

    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels, normal = len(classes), list(range(len(classes))), idx["Normal"]
    attack_classes = [c for c in labels if c != normal]
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    ev = np.load(results / "fusion" / "split_indices.npz")["evaluation"]
    y_ev, X_ev = y_test[ev], X_test[ev]
    is_attack_ev = y_ev != normal
    S_ev, F_ev, M_ev = fv.rule_evidence(processed / KB_FILE, feature_names, X_ev, idx)
    zeros = lambda n: np.zeros((n, K))

    def stacker_proba(st, P, S, F, M):
        pr = st.model.predict_proba(st.features(P, S, F, M))
        assert list(st.model.classes_) == labels, st.model.classes_
        return pr

    def scored(model, seed, source, target, t, proba, pred=None):
        s = 1 - proba[:, normal]
        alarm = s > t if pred is None else pred != normal
        if pred is None:  # below the threshold -> Normal, above -> top attack class
            top_attack = np.array(attack_classes)[proba[:, attack_classes].argmax(1)]
            pred = np.where(alarm, top_attack, normal)
        row = {"Model": model, "Seed": seed, "Threshold_Source": source, "Target": target, "Threshold": t,
               "Normal_to_Attack": float(alarm[~is_attack_ev].mean()),
               "Attack_to_Normal": float((~alarm[is_attack_ev]).mean()),
               "Macro_F1": float(f1_score(y_ev, pred, labels=labels, average="macro", zero_division=0)),
               "Macro_FPR": float(np.mean(list(calculate_fpr(y_ev, pred, labels).values())))}
        row.update({f"DR_{classes[c]}": float(alarm[y_ev == c].mean()) for c in attack_classes})
        return row

    rows, roc, aucs = [], {}, []
    for seed in SEEDS:
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_dir / f"neural_ids_exp_4_{tag(seed)}.pth", map_location=DEVICE))
        vm = np.load(ckpt_dir / f"valmem_{tag(seed)}.npz")
        X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
        P_vm, P_ev = predict_proba(model, X_vm, DEVICE), predict_proba(model, X_ev, DEVICE)
        S_vm, F_vm, M_vm = fv.rule_evidence(processed / KB_FILE, feature_names, X_vm, idx)
        n_vm, n_ev = len(y_vm), len(y_ev)
        ev_inputs = {CTRL: (zeros(n_ev), zeros(n_ev), zeros(n_ev)), V4: (S_ev, F_ev, M_ev)}
        vm_inputs = {CTRL: (zeros(n_vm), zeros(n_vm), zeros(n_vm)), V4: (S_vm, F_vm, M_vm)}
        make = {CTRL: lambda: fv.StackingFuser(detector_only=True), V4: lambda: fv.StackingFuser()}

        proba_ev, proba_vm, proba_oof, preds = {DET: P_ev}, {DET: P_vm}, {}, {DET: P_ev.argmax(1)}
        for m in (CTRL, V4):
            st = make[m]().fit(P_vm, *vm_inputs[m], y_vm)
            proba_ev[m], proba_vm[m] = stacker_proba(st, P_ev, *ev_inputs[m]), stacker_proba(st, P_vm, *vm_inputs[m])
            preds[m] = st.predict(P_ev, *ev_inputs[m])
            oof = np.zeros((n_vm, K))
            for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(X_vm, y_vm):
                f = make[m]().fit(P_vm[tr], *(a[tr] for a in vm_inputs[m]), y_vm[tr])
                oof[te] = stacker_proba(f, P_vm[te], *(a[te] for a in vm_inputs[m]))
            proba_oof[m] = oof

        for m in MODELS:
            rows.append(scored(m, seed, SRC_ARGMAX, np.nan, np.nan, proba_ev[m], pred=preds[m]))
            vm_normal = y_vm == normal
            for target in TARGETS:
                t = threshold(1 - proba_vm[m][vm_normal, normal], target)
                rows.append(scored(m, seed, SRC_VM, target, t, proba_ev[m]))
                if m in proba_oof:
                    t_oof = threshold(1 - proba_oof[m][vm_normal, normal], target)
                    rows.append(scored(m, seed, SRC_OOF, target, t_oof, proba_ev[m]))
            score = 1 - proba_ev[m][:, normal]
            aucs.append({"Model": m, "Seed": seed, "ROC_AUC": float(roc_auc_score(is_attack_ev, score))})
            roc[(m, seed)] = roc_curve(is_attack_ev, score)
        print(f"seed {seed}: " + ", ".join(f"{r['Model'].split(' (')[0]} {r['Threshold_Source'][:10]} {r['Target']}: "
                                           f"N->A {r['Normal_to_Attack']:.4f}" for r in rows[-11:]))

    xgb = XGBoostBaseline.load(results / "models" / "xgboost_multi.joblib").model
    assert list(xgb.classes_) == labels
    P_x = xgb.predict_proba(X_ev)
    aucs.append({"Model": XGB, "Seed": 42, "ROC_AUC": float(roc_auc_score(is_attack_ev, 1 - P_x[:, normal]))})
    roc[(XGB, 42)] = roc_curve(is_attack_ev, 1 - P_x[:, normal])
    rows.append(scored(XGB, 42, SRC_ARGMAX, np.nan, np.nan, P_x, pred=P_x.argmax(1)))

    df = pd.DataFrame(rows).merge(pd.DataFrame(aucs), on=["Model", "Seed"], how="left")

    # ---- cross-checks of the default-argmax rows ----------------------------------------------------------------
    bf = pd.read_csv(results / "fusion" / "benign_fpr.csv")
    bf = bf[bf["Scope"] == "evaluation split"]
    dv = pd.read_csv(results / "fusion" / "detector_variants.csv")
    dv = dv[dv["Variant"] == "R1"]
    names = {DET: ("R1 detector alone", "Detector alone"), CTRL: (CTRL, "Control: stacking on log P only"),
             V4: ("R1 + V4 stacking", "V4 stacking"), XGB: ("XGBoost alone", None)}
    diffs = []
    for _, r in df[df["Threshold_Source"] == SRC_ARGMAX].iterrows():
        b = bf[(bf["System"] == names[r["Model"]][0]) & (bf["Seed"] == r["Seed"])].iloc[0]
        diffs += [abs(r[c] - b[c]) for c in ("Normal_to_Attack", "Attack_to_Normal", "Macro_FPR")]
        if names[r["Model"]][1]:
            d = dv[(dv["Method"] == names[r["Model"]][1]) & (dv["Seed"] == r["Seed"])].iloc[0]
            diffs.append(abs(r["Macro_F1"] - d["Macro_F1"]))
    max_diff = max(diffs)
    print(f"default-argmax rows vs benign_fpr.csv / detector_variants.csv: max |difference| = {max_diff:.1e} "
          f"({len(diffs)} values)")
    if max_diff > TOL:
        sys.exit("ERROR: default-argmax rows do not reproduce the results files; not writing")

    if redo_figure:
        saved = pd.read_csv(out_csv)
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        diff = float(np.nanmax(np.abs(saved[num].to_numpy(float) - df[num].to_numpy(float))))
        if diff > TOL:
            sys.exit(f"ERROR: recomputed results differ from {out_csv.name} by {diff:.1e}; figure not rewritten")
        make_figure(roc, df, out_fig)
        print(f"Recomputed results match {out_csv.name} (max |difference| {diff:.1e}); rewrote {out_fig}")
        return
    df.to_csv(out_csv, index=False)
    write_markdown(df, classes, attack_classes, max_diff, len(diffs), int((~is_attack_ev).sum()), len(y_ev), out_md)
    make_figure(roc, df, out_fig)
    print(f"Saved {out_csv}, {out_md} and {out_fig}")


def pm(x):
    return f"{x.iloc[0]:.4f}" if len(x) == 1 else f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"


def write_markdown(df, classes, attack_classes, max_diff, n_checked, n_normal, n_ev, path):
    att = [classes[c] for c in attack_classes]
    order = [(m, SRC_ARGMAX, None) for m in MODELS] + [
        (m, src, t) for t in TARGETS for m in MODELS for src in (SRC_VM, SRC_OOF) if not (m == DET and src == SRC_OOF)]
    sel = lambda m, src, t: df[(df["Model"] == m) & (df["Threshold_Source"] == src)
                               & ((df["Target"] == t) if t is not None else df["Target"].isna())]
    L = ["# Operating points of the final continual configuration (attack vs Normal)", "",
         "Alarm score s = 1 − P(Normal) from each model's output (stackers: predict_proba). A flow raises an alarm "
         "when s > t. The threshold t is chosen per seed on the **validation memory only** (200 Normal flows per seed, "
         "never test data): at most a share *target* of the memory's Normal flows score above t. "
         "**Threshold source** — *validation memory*: scores of the deployed model on the memory (as specified; for "
         "control and V4 these are the stacker's own training rows, i.e. in-sample); *validation memory, out-of-fold*: "
         f"{N_FOLDS}-fold stratified cross-fitting inside the memory (each memory row scored by a stacker fit on the "
         "other folds), control and V4 only; *default argmax*: the model's own predictions, no threshold. "
         f"All metrics on the evaluation split ({n_ev} rows, {n_normal} Normal); mean ± sample std over seeds "
         f"{SEEDS}. Multi-class macro-F1: flows with s ≤ t are labelled Normal, the others get the model's top "
         "attack class.", "",
         f"Default-argmax rows reproduce results/fusion/benign_fpr.csv (Normal→attack, attack→Normal, macro-FPR) and "
         f"results/fusion/detector_variants.csv (macro-F1): max |difference| over {n_checked} values {max_diff:.1e}.",
         "", "## ROC-AUC, attack vs Normal (evaluation split)", "", "| Model | n | ROC-AUC |", "|---|---|---|"]
    for m in MODELS + [XGB]:
        a = df[(df["Model"] == m)].drop_duplicates("Seed")["ROC_AUC"]
        L.append(f"| {m} | {len(a)} | {pm(a)} |")
    L += ["", "## Operating points", "",
          "| Model | Threshold source | Target | Threshold t | Normal→attack | Attack→Normal | Macro-F1 | Macro-FPR | "
          "Seeds with Normal→attack ≤ target |", "|---|---|---|---|---|---|---|---|---|"]
    for m, src, t in order:
        g = sel(m, src, t)
        within = "—" if t is None else f"{int((g['Normal_to_Attack'] <= t).sum())}/{len(g)}"
        L.append(f"| {m} | {src} | {'—' if t is None else t} | {'—' if t is None else pm(g['Threshold'])} | "
                 f"{pm(g['Normal_to_Attack'])} | {pm(g['Attack_to_Normal'])} | {pm(g['Macro_F1'])} | "
                 f"{pm(g['Macro_FPR'])} | {within} |")
    g = sel(XGB, SRC_ARGMAX, None)
    L.append(f"| *{XGB}* | {SRC_ARGMAX} | — | — | {pm(g['Normal_to_Attack'])} | {pm(g['Attack_to_Normal'])} | "
             f"{pm(g['Macro_F1'])} | {pm(g['Macro_FPR'])} | — |")
    L += ["", "## Detection rate per attack class (mean over seeds)", "",
          "| Model | Threshold source | Target | " + " | ".join(att) + " |", "|---|---|---|" + "---|" * len(att)]
    for m, src, t in order:
        g = sel(m, src, t)
        L.append(f"| {m} | {src} | {'—' if t is None else t} | " + " | ".join(f"{g[f'DR_{c}'].mean():.4f}" for c in att) + " |")
    L.append(f"| *{XGB}* | {SRC_ARGMAX} | — | " + " | ".join(f"{sel(XGB, SRC_ARGMAX, None)[f'DR_{c}'].mean():.4f}" for c in att) + " |")
    L += ["", "## Normal→attack per seed at the 5% target", "",
          "| Model | Threshold source | " + " | ".join(str(s) for s in SEEDS) + " |", "|---|---|" + "---|" * len(SEEDS)]
    for m, src, t in [o for o in order if o[2] == 0.05]:
        g = sel(m, src, t).set_index("Seed")
        L.append(f"| {m} | {src} | " + " | ".join(f"{g.loc[s, 'Normal_to_Attack']:.4f}" for s in SEEDS) + " |")
    L += ["", "## Summary", ""]
    for m, src in [(V4, SRC_VM), (V4, SRC_OOF), (DET, SRC_VM)]:
        for t in TARGETS:
            g = sel(m, src, t)
            L.append(f"- {m}, threshold from {src}, target {t}: realised Normal→attack {pm(g['Normal_to_Attack'])} "
                     f"({int((g['Normal_to_Attack'] <= t).sum())}/5 seeds within target; mean "
                     f"{'within' if g['Normal_to_Attack'].mean() <= t else 'above'} target), attack→Normal "
                     f"{pm(g['Attack_to_Normal'])}, macro-F1 {pm(g['Macro_F1'])}.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def make_figure(roc, df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.6})
    styles = {DET: ("#808080", "--", 1.3), CTRL: ("#808080", ":", 1.6), V4: ("#000000", "-", 1.6),
              XGB: ("#000000", "-.", 1.3)}
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for m in [XGB, DET, CTRL, V4]:
        col, ls, lw = styles[m]
        auc = df[df["Model"] == m].drop_duplicates("Seed")["ROC_AUC"]
        fpr, tpr, _ = roc[(m, 42)]
        lab = f"{m.replace(' (stacking on log P only)', '')}: AUC {pm(auc)}"
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, color=col, linestyle=ls, linewidth=lw, label=lab)
    for t, mk in zip(TARGETS, ("o", "s")):
        g = df[(df["Model"] == V4) & (df["Threshold_Source"] == SRC_VM) & (df["Target"] == t) & (df["Seed"] == 42)].iloc[0]
        ax.plot(max(g["Normal_to_Attack"], 1e-4), 1 - g["Attack_to_Normal"], marker=mk, markersize=7, color="#000000",
                markerfacecolor="#ffffff", linestyle="none",
                label=f"V4 at validation-chosen {int(t * 100)}% threshold (seed 42)")
    ax.axvline(0.05, color="#000000", linewidth=0.8, linestyle=(0, (1, 3)))
    ax.text(0.047, 0.42, "5% false-alarm target", fontsize=8, rotation=90, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Normal→attack rate (benign false-alarm rate; log scale)")
    ax.set_ylabel("Attack detection rate (1 − attack→Normal)")
    ax.set_title("ROC, attack vs Normal, evaluation split (curves: seed 42; AUC: mean ± SD)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
