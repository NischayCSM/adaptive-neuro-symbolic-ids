"""
KS drift-threshold sensitivity of the final configuration.

1. Drift scores of Exp 2-4 for pipeline seeds 42, 1, 2, 3, 4 (read-only, no KB is built). The drift path of
   continual_pipeline.py is replayed exactly: np.random.seed(seed); reference = 5000-row subsample of Exp 1;
   at Exp 1 and every flagged experience the SHAP rule generator draws one 1000-row subsample from the
   global NumPy stream (the only other global-RNG consumer between drift checks) and the reference is
   replaced by a subsample of that experience. Done with and without the validation-memory exclusion of
   the final KB; the seed-42 replay must reproduce the drift scores logged by the existing KB builds.
2. KBs: continual_kb_v1_gain_clean.db (threshold 0.15, existing) and continual_kb_v1_gain_clean_ks{010,020,025}.db
   (continual_pipeline.py --gain-check --exclude-train-rows <R0 validation memories> --drift-threshold t).
   Flagged experiences and ACTIVE rules per class.
3. With the R1 detectors (replay 10000, --class-weighted, valmem 200, max fraction 0.5; seeds 42, 1, 2, 3, 4):
   V4 stacking and the detector-only control, both fit on the validation memory; macro-F1, recall of
   Fuzzers / Shellcode / Worms, rule contribution V4 - control per seed. The 0.15 rows must equal the R1 rows
   of results/fusion/detector_variants.csv.

Writes results/ablations/ks_threshold.csv and results/ablations/ks_threshold.md. Re-launches itself with
PYTHONHASHSEED=0.
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
from src.continual.drift_detection import DriftDetector
from src.continual.experience_builder import ExperienceBuilder
from src.reasoning import fusion_variants as fv
from src.evaluation.run_fusion_variants import metrics, pm, SEEDS
from src.evaluation.run_fusion_legal import CONTROL
from src.evaluation.run_fusion_final import TAG, meaningful_fired

THRESHOLDS = {0.10: "continual_kb_v1_gain_clean_ks010.db", 0.15: "continual_kb_v1_gain_clean.db",
              0.20: "continual_kb_v1_gain_clean_ks020.db", 0.25: "continual_kb_v1_gain_clean_ks025.db"}
DETECTOR = "R1 (replay 10000 + class-weighted, valmem 200, max fraction 0.5)"
R1_SUFFIX = "_cw"
SHAP_SUBSAMPLE = 1000  # generate_shap_rules' subsample_size
FOCUS = ["Fuzzers", "Shellcode", "Worms"]
METHODS = [CONTROL, "V4 stacking"]
LOGGED_NO_EXCLUSION = "continual_pipeline_metrics_continual_kb_v1_seed42.csv"  # the 0.521 / 0.485 / 0.159 build


def replay_drift(X_train, y_cat, builder, seed, threshold=0.15):
    """Drift scores of continual_pipeline.py's DriftDetector for pipeline seed `seed` (see module docstring)."""
    np.random.seed(seed)
    det = DriftDetector(threshold=threshold)
    det.update_reference(builder.build_experience(X_train, y_cat, exp_id=1, cumulative=False)[0])
    out = {}
    for eid in range(1, builder.get_num_experiences() + 1):
        X_exp, _ = builder.build_experience(X_train, y_cat, exp_id=eid, cumulative=False)
        flagged = True
        if eid > 1:
            r = det.detect(X_exp)
            out[eid] = r["drift_score"]
            flagged = r["drift_detected"]
        if flagged:
            if len(X_exp) > SHAP_SUBSAMPLE:
                np.random.choice(len(X_exp), SHAP_SUBSAMPLE, replace=False)
            det.update_reference(X_exp)
    return out


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_ks_threshold", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    processed, fusion_dir = root / "data" / "processed", root / "results" / "fusion"
    res_dir = root / "results" / "continual"
    ckpt_dir = res_dir / "replay_checkpoints"
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    builds = {t: res_dir / f"kb_build_{f[:-3]}.json" for t, f in THRESHOLDS.items()}
    pm_csv = {t: res_dir / f"continual_pipeline_metrics_{f[:-3]}.csv" for t, f in THRESHOLDS.items()}
    tag = lambda s: f"seed{s}_{TAG}{R1_SUFFIX}"
    needed = [processed / f for f in THRESHOLDS.values()] + list(builds.values()) + list(pm_csv.values())
    needed += [ckpt_dir / f"neural_ids_exp_4_{tag(s)}.pth" for s in SEEDS] + [ckpt_dir / f"valmem_{tag(s)}.npz" for s in SEEDS]
    needed += [fusion_dir / "detector_variants.csv", fusion_dir / "split_indices.npz", res_dir / LOGGED_NO_EXCLUSION]
    missing = [str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    # ---- 1. drift scores across pipeline seeds ---------------------------------------------------------
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_train = np.load(processed / "X_train.npy")
    y_cat = pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_train.npy"))
    builder = ExperienceBuilder(config_path=str(root / "configs" / "continual.yaml"))
    with open(builds[0.15]) as fh:
        excl_files = json.load(fh)["exclude_train_rows"]
    excl = np.unique(np.concatenate([np.load(root / f)["global_index"] for f in excl_files]))
    keep = np.ones(len(X_train), dtype=bool)
    keep[excl] = False
    drift = {}
    for setting, (X, y) in {"with exclusion (final KB)": (X_train[keep], y_cat[keep]),
                            "no exclusion": (X_train, y_cat)}.items():
        for s in SEEDS:
            drift[(setting, s)] = replay_drift(X, y, builder, s)
            print(f"drift {setting} seed {s}: {drift[(setting, s)]}")
    logged = {"with exclusion (final KB)": pd.read_csv(pm_csv[0.15]),
              "no exclusion": pd.read_csv(res_dir / LOGGED_NO_EXCLUSION)}
    replay_check = {}
    for setting, d in logged.items():
        # the older build stored the whole detector dict as a string in Drift_Detected
        scores = d.set_index("Experience")["Drift_Score" if "Drift_Score" in d else "Drift_Detected"]
        ref = {e: float(json.loads(v.replace("'", '"').replace("True", "true").replace("False", "false"))["drift_score"])
               if isinstance(v, str) and v.startswith("{") else float(v) for e, v in scores.items() if e > 1}
        replay_check[setting] = max(abs(drift[(setting, 42)][e] - ref[e]) for e in ref)
    print("Seed-42 replay vs logged drift scores (max |diff|):", replay_check)
    if max(replay_check.values()) > 1e-12:
        print("ERROR: the drift replay does not reproduce the logged seed-42 scores", file=sys.stderr)
        sys.exit(1)

    # ---- 2. KBs ------------------------------------------------------------------------------------------
    from src.rules.knowledge_base import KnowledgeBase
    kb_info = {}
    for t, f in THRESHOLDS.items():
        with open(builds[t]) as fh:
            log = json.load(fh)
        if log.get("drift_threshold", 0.15) != t or log.get("n_excluded_train_rows") != len(excl) or not log.get("gain_check"):
            print(f"ERROR: {f} was not built with threshold {t}, the exclusion and the gain check", file=sys.stderr)
            sys.exit(1)
        m = pd.read_csv(pm_csv[t])
        by_class = {}
        for r in KnowledgeBase(db_path=processed / f).get_active_rules():
            by_class[r["predicted_class"]] = by_class.get(r["predicted_class"], 0) + 1
        kb_info[t] = {"flagged": [int(e) for e in m.loc[m["Drift_Detected"], "Experience"]],
                      "scores": {int(e): float(v) for e, v in zip(m["Experience"], m["Drift_Score"]) if e > 1},
                      "accepted": {e["experience"]: e["accepted"] for e in log["experiences"]},
                      "active": sum(by_class.values()), "by_class": by_class}
        print(f"KB {t}: flagged {kb_info[t]['flagged']}, {kb_info[t]['active']} active rules {by_class}")

    # ---- 3. V4 / control with the R1 detectors ----------------------------------------------------------------
    set_seed(42)
    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels = len(classes), list(range(len(classes)))
    col = {n: i for i, n in enumerate(feature_names)}
    X_test = np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    ev = np.load(fusion_dir / "split_indices.npz")["evaluation"]
    y_ev, rows_ev = y_test[ev], np.arange(len(ev))
    evid = {t: [a[ev] for a in fv.cached_rule_evidence(processed / f, feature_names, X_test, idx,
                                                          out_dir / "cache" / f"{f[:-3]}_test.npz")]
            for t, f in THRESHOLDS.items()}
    mean_f = {t: meaningful_fired(processed / f, X_test, y_test, col, idx, K)[0][ev] for t, f in THRESHOLDS.items()}

    rows = []
    for s in SEEDS:
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_dir / f"neural_ids_exp_4_{tag(s)}.pth", map_location=DEVICE))
        vm = np.load(ckpt_dir / f"valmem_{tag(s)}.npz")
        X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
        P_vm, Pe = predict_proba(model, X_vm, DEVICE), predict_proba(model, X_test, DEVICE)[ev]
        det_ev = Pe.argmax(1)
        zeros_ev, zeros_vm = np.zeros((len(ev), K)), np.zeros((len(y_vm), K))
        ctrl_pred = fv.StackingFuser(detector_only=True).fit(P_vm, zeros_vm, zeros_vm, zeros_vm, y_vm).predict(
            Pe, zeros_ev, zeros_ev, zeros_ev)
        for t, f in THRESHOLDS.items():
            S, F, M = evid[t]
            S_vm, F_vm, M_vm = fv.rule_evidence(processed / f, feature_names, X_vm, idx)
            v4_pred = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm).predict(Pe, S, F, M)
            for m, pred in [(CONTROL, ctrl_pred), ("V4 stacking", v4_pred)]:
                rows.append({"Detector": DETECTOR, "Seed": s, "Drift_Threshold": t, "KB": f, "Method": m,
                             **metrics(y_ev, pred, det_ev, F, labels, classes),
                             "Meaningful_Explanation_Rate": float(mean_f[t][rows_ev, pred].mean())})
            print(f"seed {s} threshold {t}: control={rows[-2]['Macro_F1']:.4f} V4={rows[-1]['Macro_F1']:.4f}")
    df = pd.DataFrame(rows)

    # the 0.15 rows must equal R1 in detector_variants.csv
    dv = pd.read_csv(fusion_dir / "detector_variants.csv")
    dv = dv[(dv["Source"] == "this step") & (dv["Variant"] == "R1")].copy()
    dv["Seed"] = dv["Seed"].astype(int)
    dv = dv.set_index(["Method", "Seed"])
    cur = df[df["Drift_Threshold"] == 0.15].set_index(["Method", "Seed"])
    cols = ["Accuracy", "Macro_F1", "Macro_FPR", "Explanation_Rate", "Meaningful_Explanation_Rate"] + [f"Recall_{c}" for c in classes]
    r1_diff = max(abs(float(cur.loc[k, c]) - float(dv.loc[k, c])) for k in cur.index for c in cols)
    print(f"Threshold 0.15 vs detector_variants.csv R1: max |difference| = {r1_diff:.2e}")
    if r1_diff > 1e-9:
        print("ERROR: threshold-0.15 rows do not reproduce detector_variants.csv", file=sys.stderr)
        sys.exit(1)

    df.to_csv(out_dir / "ks_threshold.csv", index=False)
    write_markdown(df, drift, replay_check, kb_info, r1_diff, classes, len(excl), len(ev), out_dir / "ks_threshold.md")
    print(f"Saved {out_dir / 'ks_threshold.csv'} and ks_threshold.md")


def write_markdown(df, drift, replay_check, kb_info, r1_diff, classes, n_excl, n_ev, path):
    settings = list(dict.fromkeys(k[0] for k in drift))
    L = ["# KS drift-threshold sensitivity", "",
         f"Final configuration: {DETECTOR} detectors, KB v1-style + gain check + validation-memory exclusion "
         f"({n_excl} rows), V4 stacking and the detector-only control fit on the validation memory. Metrics on the "
         f"evaluation split ({n_ev} rows).", "",
         "## Drift scores across pipeline seeds (mean KS statistic; read-only replay)", "",
         "continual_pipeline.py's drift path replayed for each pipeline seed (np.random.seed(seed); one SHAP "
         "subsample per flagged experience is the only other global-RNG draw). The seed-42 replay reproduces the "
         "logged scores exactly (max |diff| " + ", ".join(f"{k}: {v:.1e}" for k, v in replay_check.items()) + ").", "",
         "| Setting | Experience | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean ± std | min | flagged at 0.15 / 0.20 / 0.25 |",
         "|---|---|" + "---|" * (len(SEEDS) + 4)]
    for st in settings:
        for e in (2, 3, 4):
            v = pd.Series([drift[(st, s)][e] for s in SEEDS])
            L.append(f"| {st} | Exp {e} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {pm(v)} | {v.min():.4f} | "
                     + " / ".join(f"{int((v > t).sum())}/5" for t in (0.15, 0.20, 0.25)) + " |")

    L += ["", "## KBs per threshold (pipeline seed 42)", "",
          "| Threshold | KB | Drift scores Exp 2 / 3 / 4 | Flagged | Accepted per exp | Active | " + " | ".join(classes) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(classes)]
    for t, f in THRESHOLDS.items():
        k = kb_info[t]
        L.append(f"| {t:.2f} | `{f}` | " + " / ".join(f"{k['scores'][e]:.4f}" for e in (2, 3, 4)) + f" | {k['flagged']} | "
                 + ", ".join(f"{e}: {a}" for e, a in k["accepted"].items()) + f" | {k['active']} | "
                 + " | ".join(str(k["by_class"].get(c, 0)) for c in classes) + " |")

    L += ["", f"## V4 and control with the R1 detectors (mean ± sample std over {len(SEEDS)} seeds)", "",
          f"Threshold 0.15 rows equal the R1 rows of results/fusion/detector_variants.csv (max |diff| {r1_diff:.1e}). "
          "The control does not use the KB, so it is identical across thresholds.", "",
          "| Threshold | Method | Accuracy | Macro-F1 | " + " | ".join(f"Recall {c}" for c in FOCUS)
          + " | Expl. | Meaningful expl. |", "|---|---|---|---|" + "---|" * (len(FOCUS) + 2)]
    for t in THRESHOLDS:
        for m in METHODS:
            x = df[(df["Drift_Threshold"] == t) & (df["Method"] == m)]
            L.append(f"| {t:.2f} | {m} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | "
                     + " | ".join(pm(x[f"Recall_{c}"]) for c in FOCUS)
                     + f" | {pm(x['Explanation_Rate'])} | {pm(x['Meaningful_Explanation_Rate'])} |")

    L += ["", "## Rule contribution: V4 - control (macro-F1), per seed", "",
          "| Threshold | " + " | ".join(str(s) for s in SEEDS) + " | mean ± std | seeds > 0 |", "|---|" + "---|" * (len(SEEDS) + 2)]
    contrib = {}
    for t in THRESHOLDS:
        g = df[df["Drift_Threshold"] == t].set_index(["Method", "Seed"])["Macro_F1"]
        d = pd.Series({s: g[("V4 stacking", s)] - g[(CONTROL, s)] for s in SEEDS})
        contrib[t] = d
        L.append(f"| {t:.2f} | " + " | ".join(f"{d[s]:+.4f}" for s in SEEDS) + f" | {d.mean():+.4f} ± {d.std(ddof=1):.4f} "
                 f"| {int((d > 0).sum())}/5 |")

    L += ["", "## Per-class recall of V4 (mean over seeds)", "", "| Threshold | " + " | ".join(classes) + " |",
          "|---|" + "---|" * len(classes)]
    for t in THRESHOLDS:
        x = df[(df["Drift_Threshold"] == t) & (df["Method"] == "V4 stacking")]
        L.append(f"| {t:.2f} | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")

    v4 = {t: df[(df["Drift_Threshold"] == t) & (df["Method"] == "V4 stacking")]["Macro_F1"] for t in THRESHOLDS}
    span = max(v.mean() for v in v4.values()) - min(v.mean() for v in v4.values())
    L += ["", "## Summary", "",
          f"- V4 macro-F1 across thresholds: " + ", ".join(f"{t:.2f}: {v.mean():.4f}" for t, v in v4.items())
          + f" (range {span:.4f}); rule contribution " + ", ".join(f"{t:.2f}: {d.mean():+.4f} ({int((d > 0).sum())}/5 > 0)"
                                                                for t, d in contrib.items()) + ".",
          "- Fuzzers rules: " + ", ".join(f"{t:.2f}: {kb_info[t]['by_class'].get('Fuzzers', 0)}" for t in THRESHOLDS) + "."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
