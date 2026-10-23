"""
Stronger continual detectors: does a better detector close the gap, and do the rules still add value?

Detectors (replay buffer 10000, --val-memory-per-class 200 --val-memory-max-fraction 0.5, seeds 42, 1, 2, 3, 4):
    R0  the existing valmem runs of run_fusion_final.py (not retrained)
    R1  + --class-weighted   (inverse-frequency CE weights on each experience's training mix)
    R2  + --weight-align     (Weight Aligning of the new classes' output weights after each experience)
    R3  + --class-weighted --weight-align
Every variant's validation memory must hold exactly the rows of R0's (same seeded draw), so KB
v1_gain_clean (built with the union of R0's validation memories excluded) stays leakage-free for all.

On the evaluation split of results/fusion/split_indices.npz, per variant and seed:
    detector alone (accuracy, macro-F1, macro-FPR, per-class recall, recency share = fraction of predictions
    in the last experience's classes, as in run_final_model_eval.py), the detector-only stacking control and
    V4 stacking with KB v1_gain_clean (both fit on the validation memory only, as in run_fusion_final.py),
    rule contribution = V4 macro-F1 - control macro-F1, explanation / meaningful explanation rate.
R0 is recomputed and checked against results/fusion/fusion_final.csv. Reference rows (joint training 20
epochs class-weighted; static XGBoost) are copied from fusion_final.csv.

Writes results/fusion/detector_variants.csv and results/fusion/detector_variants.md. Re-launches itself with
PYTHONHASHSEED=0 (same environment as run_fusion_final.py).
"""

import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd
import torch

from src.device import DEVICE
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import find_project_root, set_seed, load_continual_data, predict_proba
from src.reasoning import fusion_variants as fv
from src.evaluation.run_fusion_variants import metrics, pm, SEEDS, JOINT_REF
from src.evaluation.run_fusion_legal import CONTROL
from src.evaluation.run_fusion_final import TAG, meaningful_fired

KB_NAME, KB_FILE = "KB v1_gain_clean", "continual_kb_v1_gain_clean.db"
VARIANTS = {"R0": ("", "replay (existing valmem runs)"), "R1": ("_cw", "+ class-weighted"),
            "R2": ("_wa", "+ weight align"), "R3": ("_cw_wa", "+ class-weighted + weight align")}
METHODS = ["Detector alone", CONTROL, "V4 stacking"]
STATIC_REF = "Static XGBoost (copied)"
JOINT_ROW = f"{JOINT_REF} (copied)"
R0_TOL = 1e-9


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_detector_variants", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    processed, out_dir = root / "data" / "processed", root / "results" / "fusion"
    res_dir = root / "results" / "continual"
    ckpt_dir = res_dir / "replay_checkpoints"
    set_seed(42)

    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels = len(classes), list(range(len(classes)))
    col = {n: i for i, n in enumerate(feature_names)}
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    ev = np.load(out_dir / "split_indices.npz")["evaluation"]  # the calibration split is NOT used here
    y_ev = y_test[ev]
    rows_ev = np.arange(len(ev))
    recent = np.unique(data["test_splits"][-1][1])
    true_recency = float(np.isin(y_ev, recent).mean())

    tag = lambda v, s: f"seed{s}_{TAG}{VARIANTS[v][0]}"
    needed = [processed / KB_FILE, out_dir / "fusion_final.csv", res_dir / f"kb_build_{KB_FILE[:-3]}.json"]
    for v in VARIANTS:
        needed += [ckpt_dir / f"neural_ids_exp_4_{tag(v, s)}.pth" for s in SEEDS]
        needed += [ckpt_dir / f"valmem_{tag(v, s)}.npz" for s in SEEDS]
        needed += [res_dir / f"replay_summary_{tag(v, s)}.json" for s in SEEDS]
    missing = [str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    # ---- validation memories must be identical to R0's; the KB excluded their union ------------------------
    vm_check = {}
    for s in SEEDS:
        ref = np.load(ckpt_dir / f"valmem_{tag('R0', s)}.npz")
        for v in list(VARIANTS)[1:]:
            z = np.load(ckpt_dir / f"valmem_{tag(v, s)}.npz")
            vm_check[(v, s)] = all(np.array_equal(z[k], ref[k]) for k in ("exp", "local_index", "global_index", "y"))
    print("Validation memory identical to R0:", vm_check)
    if not all(vm_check.values()):
        print("ERROR: validation memory differs from R0 for " + str([k for k, ok in vm_check.items() if not ok]),
              file=sys.stderr)
        sys.exit(1)
    union = np.unique(np.concatenate([np.load(ckpt_dir / f"valmem_{tag('R0', s)}.npz")["global_index"] for s in SEEDS]))
    with open(res_dir / f"kb_build_{KB_FILE[:-3]}.json") as fh:
        n_excl = json.load(fh).get("n_excluded_train_rows", 0)
    if n_excl != len(union):
        print(f"ERROR: {KB_FILE} excluded {n_excl} rows, expected the union {len(union)}", file=sys.stderr)
        sys.exit(1)

    S_te, F_te, M_te = fv.cached_rule_evidence(processed / KB_FILE, feature_names, X_test, idx,
                                               out_dir / "cache" / f"{KB_FILE[:-3]}_test.npz")
    S, F, M = S_te[ev], F_te[ev], M_te[ev]
    Fm = meaningful_fired(processed / KB_FILE, X_test, y_test, col, idx, K)[0][ev]

    # ---- runs ------------------------------------------------------------------------------------------
    rows, logs = [], {}
    for v, (suffix, desc) in VARIANTS.items():
        for s in SEEDS:
            model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_dir / f"neural_ids_exp_4_{tag(v, s)}.pth", map_location=DEVICE))
            vm = np.load(ckpt_dir / f"valmem_{tag(v, s)}.npz")
            with open(res_dir / f"replay_summary_{tag(v, s)}.json") as fh:
                summ = json.load(fh)
            logs[(v, s)] = {k: summ.get(k) for k in ("class_weights_per_exp", "weight_align_gamma_per_exp")}
            X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
            # full test set then [ev], exactly as run_fusion_final.py
            P_vm, Pe = predict_proba(model, X_vm, DEVICE), predict_proba(model, X_test, DEVICE)[ev]
            det_ev = Pe.argmax(1)
            S_vm, F_vm, M_vm = fv.rule_evidence(processed / KB_FILE, feature_names, X_vm, idx)
            zeros_ev, zeros_vm = np.zeros((len(ev), K)), np.zeros((len(y_vm), K))
            ctrl = fv.StackingFuser(detector_only=True).fit(P_vm, zeros_vm, zeros_vm, zeros_vm, y_vm)
            stacker = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm)
            for m, pred in [("Detector alone", det_ev), (CONTROL, ctrl.predict(Pe, zeros_ev, zeros_ev, zeros_ev)),
                            ("V4 stacking", stacker.predict(Pe, S, F, M))]:
                rows.append({"Source": "this step", "Variant": v, "Detector": f"{v} {desc}", "Seed": s, "KB": KB_NAME,
                             "Method": m, **metrics(y_ev, pred, det_ev, F, labels, classes),
                             "Meaningful_Explanation_Rate": float(Fm[rows_ev, pred].mean()),
                             "Recency_Share": float(np.isin(pred, recent).mean()),
                             "True_Recency_Share": true_recency})
            print(f"{v} seed {s}: " + ", ".join(f"{r['Method'].split(':')[0]}={r['Macro_F1']:.4f}" for r in rows[-3:]))

    df = pd.DataFrame(rows)

    # ---- R0 must reproduce fusion_final.csv --------------------------------------------------------------
    prev = pd.read_csv(out_dir / "fusion_final.csv", keep_default_na=False, na_values=[""])
    prev_cur = prev[(prev["Source"] == "this step") & (prev["KB"] == KB_NAME)].copy()
    prev_cur["Seed"] = prev_cur["Seed"].astype(int)
    prev_cur = prev_cur.set_index(["Method", "Seed"])
    r0 = df[df["Variant"] == "R0"].set_index(["Method", "Seed"])
    check_cols = ["Accuracy", "Macro_F1", "Macro_FPR", "Changed", "Explanation_Rate", "Meaningful_Explanation_Rate"]
    check_cols += [f"Recall_{c}" for c in classes]
    r0_diff = max(abs(float(r0.loc[k, c]) - float(prev_cur.loc[k, c])) for k in r0.index for c in check_cols)
    print(f"R0 vs fusion_final.csv: max |difference| = {r0_diff:.2e}")
    if r0_diff > R0_TOL:
        print("ERROR: recomputed R0 does not match fusion_final.csv", file=sys.stderr)
        sys.exit(1)

    # ---- reference rows copied from fusion_final.csv -------------------------------------------------------
    joint = prev[prev["Detector"] == JOINT_REF]
    xgb = prev[(prev["Detector"] == "XGBoost") & (prev["KB"] == "static KB") & (prev["Method"] == "Detector alone")]
    refs = []
    for name, g in [(JOINT_ROW, joint), (STATIC_REF, xgb)]:
        for _, r in g.iterrows():
            refs.append({**{k: r[k] for k in r.index if k in df.columns or k.startswith("Recall_")},
                         "Source": "copied from fusion_final.csv", "Variant": "ref", "Detector": name})
    df = pd.concat([df, pd.DataFrame(refs)], ignore_index=True)
    df.to_csv(out_dir / "detector_variants.csv", index=False)
    write_markdown(df, vm_check, len(union), r0_diff, logs, classes, recent, true_recency, len(ev),
                   out_dir / "detector_variants.md")
    print(f"Saved {out_dir / 'detector_variants.csv'} and detector_variants.md")


def write_markdown(df, vm_check, n_union, r0_diff, logs, classes, recent, true_recency, n_ev, path):
    cur = df[df["Source"] == "this step"]
    ref = df[df["Source"] != "this step"]
    get = lambda v, m: cur[(cur["Variant"] == v) & (cur["Method"] == m)].set_index("Seed").loc[SEEDS]
    recent_names = ", ".join(classes[c] for c in recent)
    L = ["# Stronger continual detectors: class weighting and weight aligning", "",
         f"Replay buffer 10000, `--val-memory-per-class 200 --val-memory-max-fraction 0.5`, seeds {SEEDS}. "
         "R0 = existing valmem runs (not retrained); R1 = `--class-weighted` (inverse-frequency CE weights "
         "N / (C n_c) on each experience's training mix, experience data + replay buffer, clipped to [0.1, 50]); "
         "R2 = `--weight-align` (Weight Aligning, Zhao et al. CVPR 2020: after each experience k > 1 the output "
         "weight rows of the new classes are scaled by gamma = mean norm(old) / mean norm(new); biases unchanged); "
         "R3 = both. Control and V4 use "
         f"`{KB_FILE}` and are fit on the validation memory only. Metrics on the evaluation split ({n_ev} rows).", "",
         f"- Validation memory identical to R0 (exp, local and global indices, labels) for all "
         f"{len(vm_check)} variant / seed pairs: **{all(vm_check.values())}**. The KB excluded the union of these "
         f"rows ({n_union}), so it is leakage-free for every variant.",
         f"- Recomputed R0 vs results/fusion/fusion_final.csv ({KB_NAME}: detector alone, control, V4; all metric "
         f"columns, all seeds): max |difference| = {r0_diff:.1e}.",
         f"- Recency share = fraction of predictions in the last experience's classes ({recent_names}); "
         f"true share on the evaluation split: **{true_recency:.4f}**.", ""]

    L += ["## Detector alone, control and V4 (mean ± sample std over 5 seeds)", "",
          "| Variant | Method | Accuracy | Macro-F1 | Macro-FPR | Recency share | Expl. | Meaningful expl. |",
          "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for m in METHODS:
            x = get(v, m)
            L.append(f"| {v} | {m} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | {pm(x['Macro_FPR'])} | "
                     f"{pm(x['Recency_Share'])} | {pm(x['Explanation_Rate'])} | {pm(x['Meaningful_Explanation_Rate'])} |")
    for name in (JOINT_ROW, STATIC_REF):
        x = ref[ref["Detector"] == name]
        f = (lambda c: pm(x[c])) if len(x) > 1 else (lambda c: f"{x[c].iloc[0]:.4f}")
        L.append(f"| *{name}* | Detector alone | {f('Accuracy')} | {f('Macro_F1')} | {f('Macro_FPR')} | n/a | n/a | n/a |")

    L += ["", "## Rule contribution: V4 minus control (macro-F1), per seed", "",
          "| Variant | " + " | ".join(str(s) for s in SEEDS) + " | mean ± std | seeds > 0 | V4 - detector alone |",
          "|---|" + "---|" * (len(SEEDS) + 3)]
    for v in VARIANTS:
        d = get(v, "V4 stacking")["Macro_F1"] - get(v, CONTROL)["Macro_F1"]
        dd = get(v, "V4 stacking")["Macro_F1"] - get(v, "Detector alone")["Macro_F1"]
        L.append(f"| {v} | " + " | ".join(f"{d[s]:+.4f}" for s in SEEDS) + f" | {d.mean():+.4f} ± {d.std(ddof=1):.4f} "
                 f"| {int((d > 0).sum())}/5 | {dd.mean():+.4f} ± {dd.std(ddof=1):.4f} |")

    L += ["", "## Macro-F1 per seed", "", "| Variant | Method | " + " | ".join(str(s) for s in SEEDS) + " |",
          "|---|---|" + "---|" * len(SEEDS)]
    for v in VARIANTS:
        for m in METHODS:
            x = get(v, m)["Macro_F1"]
            L.append(f"| {v} | {m} | " + " | ".join(f"{x[s]:.4f}" for s in SEEDS) + " |")

    L += ["", "## Per-class recall (mean over seeds)", "", "| Variant | Method | " + " | ".join(classes) + " |",
          "|---|---|" + "---|" * len(classes)]
    for v in VARIANTS:
        for m in METHODS:
            x = get(v, m)
            L.append(f"| {v} | {m} | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")
    for name in (JOINT_ROW, STATIC_REF):
        x = ref[ref["Detector"] == name]
        L.append(f"| *{name}* | Detector alone | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")

    L += ["", "## Logged class weights and gamma (seed 42)", ""]
    for v in VARIANTS:
        lg = logs[(v, 42)]
        if lg["class_weights_per_exp"]:
            L.append(f"- {v} class weights: " + "; ".join(
                f"Exp {e}: " + ", ".join(f"{c} {w:.3f}" for c, w in ws.items())
                for e, ws in lg["class_weights_per_exp"].items()))
        if lg["weight_align_gamma_per_exp"]:
            L.append(f"- {v} gamma: " + ", ".join(f"Exp {e}: {g:.4f}" for e, g in lg["weight_align_gamma_per_exp"].items()))

    # summary
    det = {v: get(v, "Detector alone")["Macro_F1"] for v in VARIANTS}
    v4 = {v: get(v, "V4 stacking")["Macro_F1"] for v in VARIANTS}
    joint = ref[ref["Detector"] == JOINT_ROW]["Macro_F1"]
    best_det = max(det, key=lambda v: det[v].mean())
    best_v4 = max(v4, key=lambda v: v4[v].mean())
    contrib = get(best_det, "V4 stacking")["Macro_F1"] - get(best_det, CONTROL)["Macro_F1"]
    rec = lambda v: get(v, "V4 stacking")[[f"Recall_{c}" for c in classes]].mean()
    gain = (rec(best_v4) - rec("R0")).sort_values(ascending=False)
    L += ["", "## Summary", "",
          f"- Best detector alone: {best_det}, macro-F1 {pm(det[best_det])} (R0 {pm(det['R0'])}; joint class-weighted "
          f"{pm(joint)}: {'reached' if det[best_det].mean() >= joint.mean() else 'not reached'}, above the joint mean "
          f"in {int((det[best_det] > joint.mean()).sum())}/5 seeds).",
          f"- Rule contribution (V4 - control) for {best_det}: {contrib.mean():+.4f} ± {contrib.std(ddof=1):.4f}, "
          f"positive in {int((contrib > 0).sum())}/5 seeds.",
          f"- Best V4: {best_v4}, macro-F1 {pm(v4[best_v4])} (R0 V4 {pm(v4['R0'])}). Largest V4 recall changes vs R0: "
          + ", ".join(f"{c[7:]} {d:+.4f}" for c, d in gain.items()) + "."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
