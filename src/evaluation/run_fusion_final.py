"""
Leakage-free continual fusion (the final setting), built on run_fusion_legal.py.

Detectors: replay buffer 10000 trained with --val-memory-per-class 200 --val-memory-max-fraction 0.5
(seeds 42, 1, 2, 3, 4). KBs built with the UNION of the five seeds' validation-memory rows excluded
from rule induction and validation (continual_pipeline.py --exclude-train-rows):
    KB v1_clean        v1-style, no extra rule check
    KB v1_gain_clean   v1-style + gain check
    KB v2_gain_clean   v2-style (mixed validation) + gain check
As in run_fusion_legal.py, alpha / tau and the V4 / control fits use ONLY the validation memory, and
metrics are on the evaluation split of results/fusion/split_indices.npz (the calibration split is unused).

Adds the "meaningful explanation rate": the fraction of rows where a fired rule with POSITIVE gain on
the test set ((test precision - base rate) / (1 - base rate) > 0) and test match fraction <= 0.5
supports the final predicted class. Reported next to the plain explanation rate, for the runs above
and for static XGBoost (detector alone / V0 / V2 at alpha 0.5) with the static KB and KB v1_gain_clean.

Reference rows are copied from results/fusion/fusion_legal.csv (not recomputed).
Writes results/fusion/fusion_final.csv and results/fusion/fusion_final.md. Re-launches itself with
PYTHONHASHSEED=0 (the sanity check calls the existing fuser).
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
from src.evaluation.run_fusion_variants import fast_macro_f1, metrics, pm, ALPHAS, TAUS, SEEDS, JOINT_REF
from src.evaluation import run_fusion_legal as legal
from src.evaluation.run_fusion_legal import BUFFER, VALMEM, CONTROL, METHODS
from src.evaluation.kb_rule_stats import rule_mask

MAX_FRACTION = 0.5
TAG = f"buf{BUFFER}_valmem{VALMEM}_maxfrac{MAX_FRACTION:g}"
KBS = {"KB v1_clean": "continual_kb_v1_clean.db", "KB v1_gain_clean": "continual_kb_v1_gain_clean.db",
       "KB v2_gain_clean": "continual_kb_v2_gain_clean.db"}
STATIC_KBS = {"static KB": "knowledge_base.db", "KB v1_gain_clean": "continual_kb_v1_gain_clean.db"}
MEANINGFUL_MAX_MATCH = 0.5
OLD_V4 = "V4 legal with old KB v1 (fusion_legal.csv)"


def meaningful_fired(kb_path, X, y, col, idx, K):
    """N x K: some ACTIVE rule for class k fires on the row AND has positive test gain AND test match <= 0.5."""
    from src.rules.knowledge_base import KnowledgeBase

    base = np.bincount(y, minlength=K) / len(y)
    out = np.zeros((len(X), K), dtype=bool)
    n_meaningful = 0
    for r in KnowledgeBase(db_path=kb_path).get_active_rules():
        m = rule_mask(r, X, col)
        k = idx[r["predicted_class"]]
        if not m.any():
            continue
        gain = ((y[m] == k).mean() - base[k]) / (1 - base[k])
        if gain > 0 and m.mean() <= MEANINGFUL_MAX_MATCH:
            out[:, k] |= m
            n_meaningful += 1
    return out, n_meaningful


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_fusion_final", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    processed, out_dir = root / "data" / "processed", root / "results" / "fusion"
    ckpt_dir = root / "results" / "continual" / "replay_checkpoints"
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

    builds = {kb: root / "results" / "continual" / f"kb_build_{f[:-3]}.json" for kb, f in KBS.items()}
    needed = [processed / f for f in {**KBS, **STATIC_KBS}.values()] + list(builds.values())
    needed += [ckpt_dir / f"neural_ids_exp_4_seed{s}_{TAG}.pth" for s in SEEDS]
    needed += [ckpt_dir / f"valmem_seed{s}_{TAG}.npz" for s in SEEDS]
    needed += [root / "results" / "continual" / f"replay_summary_seed{s}_{TAG}.json" for s in SEEDS]
    needed += [out_dir / "fusion_legal.csv", root / "results" / "models" / "xgboost_multi.joblib"]
    missing = [str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    # The KBs must have been built with exactly the union of these seeds' validation memories excluded.
    union = np.unique(np.concatenate([np.load(ckpt_dir / f"valmem_seed{s}_{TAG}.npz")["global_index"] for s in SEEDS]))
    excl = {}
    for kb, b in builds.items():
        with open(b) as fh:
            log = json.load(fh)
        excl[kb] = {"n": log.get("n_excluded_train_rows", 0), "by_class": log.get("excluded_by_class", {}),
                    "gain_check": log.get("gain_check", False), "mixed_validation": log["mixed_validation"]}
        if excl[kb]["n"] != len(union):
            print(f"ERROR: {KBS[kb]} excluded {excl[kb]['n']} rows, expected the union {len(union)}", file=sys.stderr)
            sys.exit(1)

    ev_test = {f: fv.cached_rule_evidence(processed / f, feature_names, X_test, idx,
                                          out_dir / "cache" / f"{f[:-3]}_test.npz")
               for f in set(KBS.values()) | set(STATIC_KBS.values())}
    meaningful, n_meaningful = {}, {}
    for f in set(KBS.values()) | set(STATIC_KBS.values()):
        Fm, n = meaningful_fired(processed / f, X_test, y_test, col, idx, K)
        meaningful[f], n_meaningful[f] = Fm[ev], n

    def neural(ckpt):
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model

    models = {s: neural(ckpt_dir / f"neural_ids_exp_4_seed{s}_{TAG}.pth") for s in SEEDS}
    P_test = {s: predict_proba(models[s], X_test, DEVICE) for s in SEEDS}

    # ---- sanity check (run_fusion_legal's, on these KBs) -----------------------------------------
    legal.KBS, legal.NEW_KBS = KBS, list(KBS)
    sanity = legal.sanity_check(P_test[SEEDS[0]], ev_test, ev, X_test, feature_names, classes, idx, processed)
    print("Sanity check:", sanity)
    if not sanity["passed"]:
        print("ERROR: vectorised evidence / V0 do not reproduce SymbolicReasoner / NeuroSymbolicFuser",
              file=sys.stderr)
        sys.exit(1)

    def row(src, setting, detector, seed, kb, f, m, pred, det_ev, a=np.nan, t=np.nan, fvm=np.nan):
        F = ev_test[f][1][ev]
        return {"Source": src, "Setting": setting, "Detector": detector, "Seed": seed, "KB": kb, "Method": m,
                "Alpha": a, "Tau": t, "ValMem_Macro_F1": fvm, **metrics(y_ev, pred, det_ev, F, labels, classes),
                "Meaningful_Explanation_Rate": float(meaningful[f][rows_ev, pred].mean())}

    # ---- continual runs -------------------------------------------------------------------------
    vm_info, rows = {}, []
    det_name = f"Replay (buffer={BUFFER}, valmem={VALMEM}, max fraction {MAX_FRACTION:g})"
    for s in SEEDS:
        vm = np.load(ckpt_dir / f"valmem_seed{s}_{TAG}.npz")
        with open(root / "results" / "continual" / f"replay_summary_seed{s}_{TAG}.json") as fh:
            summ = json.load(fh)
        X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
        vm_info[s] = {"counts": summ["val_memory_counts"], "size": summ["val_memory_size"],
                      "buffer": summ["replay_buffer_size"], "total": summ["total_memory"]}
        P_vm = predict_proba(models[s], X_vm, DEVICE)
        Pe = P_test[s][ev]
        det_ev = Pe.argmax(1)
        zeros_ev, zeros_vm = np.zeros((len(ev), K)), np.zeros((len(y_vm), K))
        ctrl = fv.StackingFuser(detector_only=True).fit(P_vm, zeros_vm, zeros_vm, zeros_vm, y_vm)
        ctrl_pred = ctrl.predict(Pe, zeros_ev, zeros_ev, zeros_ev)
        score = lambda p: fast_macro_f1(y_vm, p, K)
        for kb, f in KBS.items():
            S_vm, F_vm, M_vm = fv.rule_evidence(processed / f, feature_names, X_vm, idx)
            S, F, M = (a[ev] for a in ev_test[f])
            # first grid point wins ties, as in run_fusion_variants.py / run_fusion_legal.py
            f2, a2 = max(((score(fv.v2_neutral(P_vm, S_vm, F_vm, a)), a) for a in ALPHAS), key=lambda r: r[0])
            f3, (a3, t3) = max(((score(fv.v3_gated_neutral(P_vm, S_vm, F_vm, a, t)), (a, t))
                                for a in ALPHAS for t in TAUS), key=lambda r: r[0])
            stacker = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm)
            preds = [
                ("Detector alone", det_ev, np.nan, np.nan, score(P_vm.argmax(1))),
                ("V0 linear (alpha=0.5)", fv.v0_linear(Pe, S, 0.5), 0.5, np.nan, score(fv.v0_linear(P_vm, S_vm, 0.5))),
                ("V2 neutral-absence", fv.v2_neutral(Pe, S, F, a2), a2, np.nan, f2),
                ("V3 gated + neutral", fv.v3_gated_neutral(Pe, S, F, a3, t3), a3, t3, f3),
                ("V4 stacking", stacker.predict(Pe, S, F, M), np.nan, np.nan, score(stacker.predict(P_vm, S_vm, F_vm, M_vm))),
                (CONTROL, ctrl_pred, np.nan, np.nan, score(ctrl.predict(P_vm, zeros_vm, zeros_vm, zeros_vm))),
            ]
            for m, pred, a, t, fvm in preds:
                rows.append(row("this step", "Continual (valmem, leakage-free KB)", det_name, s, kb, f, m, pred,
                                det_ev, a, t, fvm))
            print(f"seed {s} {kb}: " + ", ".join(f"{m.split(' ')[0]}={r['Macro_F1']:.4f}"
                                                 for (m, *_), r in zip(preds, rows[-len(preds):])))

    # ---- static XGBoost --------------------------------------------------------------------------
    from src.models.baselines import XGBoostBaseline
    xgb = XGBoostBaseline.load(root / "results" / "models" / "xgboost_multi.joblib").model
    raw = xgb.predict_proba(X_test)
    P_xgb = np.zeros((len(X_test), K), dtype=raw.dtype)
    P_xgb[:, [int(c) for c in xgb.classes_]] = raw
    Pe = P_xgb[ev]
    det_ev = Pe.argmax(1)
    for kb, f in STATIC_KBS.items():
        S, F, M = (a[ev] for a in ev_test[f])
        for m, pred, a in [("Detector alone", det_ev, np.nan), ("V0 linear (alpha=0.5)", fv.v0_linear(Pe, S, 0.5), 0.5),
                           ("V2 neutral-absence (alpha=0.5)", fv.v2_neutral(Pe, S, F, 0.5), 0.5)]:
            rows.append(row("this step (static)", "Static", "XGBoost", "n/a", kb, f, m, pred, det_ev, a))

    # ---- reference rows copied from fusion_legal.csv ---------------------------------------------
    prev = pd.read_csv(out_dir / "fusion_legal.csv", keep_default_na=False, na_values=[""])
    joint = prev[prev["Detector"] == JOINT_REF]
    old_v4 = prev[(prev["Source"] == "this step") & (prev["KB"] == "KB v1") & (prev["Method"] == "V4 stacking")]
    for src, g in [("copied from fusion_legal.csv: " + JOINT_REF, joint), ("copied from fusion_legal.csv: " + OLD_V4, old_v4)]:
        for _, r in g.iterrows():
            rows.append({**{k: r[k] for k in r.index}, "Source": src})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fusion_final.csv", index=False)
    write_markdown(df, vm_info, sanity, excl, len(union), n_meaningful, classes, len(ev), out_dir / "fusion_final.md")
    print(f"Saved {out_dir / 'fusion_final.csv'} and fusion_final.md")


def write_markdown(df, vm_info, sanity, excl, n_union, n_meaningful, classes, n_ev, path):
    cur = df[df["Source"] == "this step"]
    det = cur["Detector"].iloc[0]
    L = ["# Leakage-free continual fusion (final)", "",
         f"Detectors: replay buffer {BUFFER} trained with `--val-memory-per-class {VALMEM} --val-memory-max-fraction "
         f"{MAX_FRACTION:g}`, seeds {SEEDS}: before each experience, min({VALMEM}, floor({MAX_FRACTION:g} x n)) training "
         "rows of each of its classes are held out (never trained on, never in the replay buffer) and kept in a "
         "validation memory. KBs: built by continual_pipeline.py (seed 42) with the union of the five seeds' "
         f"validation-memory rows ({n_union} rows) excluded from rule induction and validation, so neither the "
         "detector nor the rules have seen those rows. **alpha / tau and the V4 / control fits use only the "
         "validation memory**; the test calibration split is not used. All metrics are on the evaluation split "
         f"({n_ev} rows, results/fusion/split_indices.npz).", "",
         "Meaningful expl. = fraction of rows where a fired rule with positive test gain "
         "((test precision - base rate) / (1 - base rate) > 0) and test match fraction <= "
         f"{MEANINGFUL_MAX_MATCH:g} supports the final predicted class (the plain Expl. counts any fired rule). Rules "
         "that qualify: " + ", ".join(f"`{f}` {n}" for f, n in sorted(n_meaningful.items())) + ".", "",
         f"Sanity check on these KBs ({sanity['rows']} evaluation rows, alpha in {ALPHAS}, PYTHONHASHSEED=0): "
         f"vectorised rule evidence identical to SymbolicReasoner: {sanity['evidence_equal']}; V0 matches "
         f"NeuroSymbolicFuser on {sanity['v0_matching']}/{sanity['v0_checked']} predictions "
         f"({sanity['v0_tied_mismatches']} exact-tie differences, {sanity['v0_other_mismatches']} other).", ""]

    L += ["## Validation memory and excluded rows", "",
          "| Seed | " + " | ".join(classes) + " | Validation memory | Replay buffer | Total memory |",
          "|---|" + "---|" * (len(classes) + 3)]
    for s, v in vm_info.items():
        L.append(f"| {s} | " + " | ".join(str(v["counts"][c]) for c in classes)
                 + f" | {v['size']} | {v['buffer']} | {v['total']} |")
    by_class = next(iter(excl.values()))["by_class"]
    L += ["", f"Union of the five validation memories, excluded from every KB build: {n_union} rows ("
          + ", ".join(f"{c} {by_class.get(c, 0)}" for c in classes) + ").", ""]

    L += ["## alpha / tau chosen on validation-memory macro-F1", "", f"Per seed, in order {SEEDS}.", "",
          "| KB | V2 alpha | V3 alpha / tau |", "|---|---|---|"]
    for kb in KBS:
        g = cur[cur["KB"] == kb].set_index(["Method", "Seed"])
        L.append(f"| {kb} | " + ", ".join(f"{g.loc[('V2 neutral-absence', s), 'Alpha']:g}" for s in SEEDS) + " | "
                 + ", ".join(f"{g.loc[('V3 gated + neutral', s), 'Alpha']:g} / {g.loc[('V3 gated + neutral', s), 'Tau']:g}"
                             for s in SEEDS) + " |")

    det_alone = cur[cur["Method"] == "Detector alone"].set_index(["KB", "Seed"])
    for kb in KBS:
        g = cur[cur["KB"] == kb]
        L += ["", f"## {det} + {kb} (mean ± sample std over {len(SEEDS)} seeds)", "",
              "| Method | Accuracy | Macro-F1 | Macro-FPR | Changed | Fixed | Broken | Expl. | Meaningful expl. | "
              f"ΔMacro-F1 vs detector alone, per seed {SEEDS} |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            x = g[g["Method"] == m].set_index("Seed")
            if m == "Detector alone":
                d = "—"
            else:
                diffs = [x.loc[s, "Macro_F1"] - det_alone.loc[(kb, s), "Macro_F1"] for s in SEEDS]
                d = f"{np.mean(diffs):+.4f} ({', '.join(f'{v:+.4f}' for v in diffs)}; {sum(v > 0 for v in diffs)}/5 > 0)"
            L.append(f"| {m} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | {pm(x['Macro_FPR'])} | "
                     f"{x['Changed'].mean():.0f} | {x['Fixed'].mean():.0f} | {x['Broken'].mean():.0f} | "
                     f"{pm(x['Explanation_Rate'])} | {pm(x['Meaningful_Explanation_Rate'])} | {d} |")

    for kb in KBS:
        g = cur[cur["KB"] == kb]
        L += ["", f"### Per-class recall: {kb} (mean over seeds)", "",
              "| Method | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for m in METHODS:
            x = g[g["Method"] == m]
            L.append(f"| {m} | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")

    # V4 vs references, per seed
    ref = df[df["Source"].str.startswith("copied")]
    joint = ref[ref["Detector"] == JOINT_REF].copy()
    old_v4 = ref[ref["Source"].str.endswith(OLD_V4)].copy()
    for g in (joint, old_v4):
        g["Seed"] = g["Seed"].astype(int)
    joint, old_v4 = joint.set_index("Seed")["Macro_F1"], old_v4.set_index("Seed")["Macro_F1"]
    L += ["", "## V4 per seed vs the copied references (macro-F1, evaluation split)", "",
          "| Seed | " + " | ".join(f"V4 + {kb}" for kb in KBS) + f" | {JOINT_REF} (copied) | {OLD_V4} (copied) |",
          "|---|" + "---|" * (len(KBS) + 2)]
    v4 = {kb: cur[(cur["KB"] == kb) & (cur["Method"] == "V4 stacking")].set_index("Seed")["Macro_F1"] for kb in KBS}
    for s in SEEDS:
        L.append(f"| {s} | " + " | ".join(f"{v4[kb][s]:.4f}" for kb in KBS) + f" | {joint[s]:.4f} | {old_v4[s]:.4f} |")
    L.append("| mean ± std | " + " | ".join(pm(v4[kb]) for kb in KBS) + f" | {pm(joint)} | {pm(old_v4)} |")
    for kb in KBS:
        L.append("")
        L.append(f"- V4 + {kb}: above the joint mean ({joint.mean():.4f}) in {int((v4[kb] > joint.mean()).sum())}/5 seeds, "
                 f"above the same-seed joint run in {sum(v4[kb][s] > joint[s] for s in SEEDS)}/5, above the best joint "
                 f"seed ({joint.max():.4f}) in {int((v4[kb] > joint.max()).sum())}/5; vs old-KB V4 per seed: "
                 + ", ".join(f"{v4[kb][s] - old_v4[s]:+.4f}" for s in SEEDS) + ".")

    st = df[df["Setting"] == "Static"]
    L += ["", "## Static XGBoost: plain vs meaningful explanation rate (evaluation split)", "",
          "| KB | Method | Accuracy | Macro-F1 | Changed | Expl. | Meaningful expl. |", "|---|---|---|---|---|---|---|"]
    for _, r in st.iterrows():
        L.append(f"| {r['KB']} | {r['Method']} | {r['Accuracy']:.4f} | {r['Macro_F1']:.4f} | {r['Changed']} | "
                 f"{r['Explanation_Rate']:.4f} | {r['Meaningful_Explanation_Rate']:.4f} |")

    means = cur.groupby(["KB", "Method"], sort=False)[["Macro_F1", "Recall_Fuzzers", "Explanation_Rate",
                                                       "Meaningful_Explanation_Rate"]].mean()
    hand = means[means.index.get_level_values(1).isin(["V2 neutral-absence", "V3 gated + neutral"])]
    best_hand = hand["Macro_F1"].idxmax()
    best = means["Macro_F1"].idxmax()
    L += ["", "## Summary", "",
          f"- Best configuration: {best[0]} + {best[1]}, macro-F1 {means.loc[best, 'Macro_F1']:.4f}, explanation rate "
          f"{means.loc[best, 'Explanation_Rate']:.4f}, meaningful explanation rate "
          f"{means.loc[best, 'Meaningful_Explanation_Rate']:.4f}. {JOINT_REF} (copied): {joint.mean():.4f}; "
          f"{OLD_V4}: {old_v4.mean():.4f}.",
          f"- Best V2 / V3 (no stacking): {best_hand[0]} + {best_hand[1]}, macro-F1 {hand.loc[best_hand, 'Macro_F1']:.4f}, "
          f"Fuzzers recall {hand.loc[best_hand, 'Recall_Fuzzers']:.4f}, meaningful explanation rate "
          f"{hand.loc[best_hand, 'Meaningful_Explanation_Rate']:.4f}."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
