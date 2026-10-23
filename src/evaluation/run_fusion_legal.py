"""
Continual-legal fusion: V2 / V3 tuning, V4 stacking and the detector-only control are fitted ONLY on
the validation memory that replay_finetune.py --val-memory-per-class 200 held out of each experience's
training data (never trained on). The test calibration split of the previous step is not used.

Detectors: final replay checkpoints (buffer 10000, valmem 200, seeds 42, 1, 2, 3, 4).
KBs: KB v1, KB v2 (old) and KB v1_lift, KB v2_lift (lift-checked rule validation).
Evaluation: the 80% evaluation split of results/fusion/split_indices.npz.

Methods: detector alone, V0 (alpha=0.5), V2 neutral-absence (alpha tuned on validation memory),
V3 gated + neutral (alpha / tau tuned on validation memory), V4 stacking (fit on validation memory),
and the detector-only stacking control (fit on validation memory).

Also: explanation rate of the static XGBoost settings of the previous step (detector alone, V0, V2),
from the cached rule evidence; and reference rows copied from results/fusion/fusion_variants.csv.

Sanity check (run first; stops on failure): for the two lift-checked KBs, the vectorised rule evidence
must equal SymbolicReasoner's matches on 2,000 evaluation rows, and V0 must equal
cl_eval.fuse_predictions (NeuroSymbolicFuser) except on exact float32 top-score ties. The script
re-launches itself with PYTHONHASHSEED=0.

Writes results/fusion/fusion_legal.csv and results/fusion/fusion_legal.md.
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
from src.continual.cl_eval import (
    find_project_root, set_seed, load_continual_data, predict_proba, symbolic_results, fuse_predictions,
)
from src.reasoning import fusion_variants as fv
from src.evaluation.run_fusion_variants import fast_macro_f1, metrics, pm, ALPHAS, TAUS, SEEDS, JOINT_REF

BUFFER = 10000
VALMEM = 200
KBS = {"KB v1": "continual_kb_v1_seed42.db", "KB v2": "continual_kb_v2.db",
       "KB v1_lift": "continual_kb_v1_lift.db", "KB v2_lift": "continual_kb_v2_lift.db"}
NEW_KBS = ["KB v1_lift", "KB v2_lift"]
STATIC_KBS = {"static KB": "knowledge_base.db", "KB v1": "continual_kb_v1_seed42.db"}
CONTROL = "Control: stacking on log P only"
METHODS = ["Detector alone", "V0 linear (alpha=0.5)", "V2 neutral-absence", "V3 gated + neutral", "V4 stacking",
           CONTROL]
N_SANITY = 2000
TAG = f"buf{BUFFER}_valmem{VALMEM}"


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_fusion_legal", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    processed, out_dir = root / "data" / "processed", root / "results" / "fusion"
    ckpt_dir = root / "results" / "continual" / "replay_checkpoints"
    set_seed(42)

    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels = len(classes), list(range(len(classes)))
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    ev = np.load(out_dir / "split_indices.npz")["evaluation"]  # the calibration split is NOT used here
    y_ev = y_test[ev]

    needed = [processed / f for f in KBS.values()] + [processed / f for f in STATIC_KBS.values()]
    needed += [ckpt_dir / f"neural_ids_exp_4_seed{s}_{TAG}.pth" for s in SEEDS]
    needed += [ckpt_dir / f"valmem_seed{s}_{TAG}.npz" for s in SEEDS]
    needed += [root / "results" / "continual" / f"replay_summary_seed{s}_{TAG}.json" for s in SEEDS]
    needed += [out_dir / "fusion_variants.csv", root / "results" / "models" / "xgboost_multi.joblib"]
    missing = [str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    ev_test = {f: fv.cached_rule_evidence(processed / f, feature_names, X_test, idx,
                                          out_dir / "cache" / f"{f[:-3]}_test.npz")
               for f in set(KBS.values()) | set(STATIC_KBS.values())}

    def neural(ckpt):
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model

    models = {s: neural(ckpt_dir / f"neural_ids_exp_4_seed{s}_{TAG}.pth") for s in SEEDS}
    P_test = {s: predict_proba(models[s], X_test, DEVICE) for s in SEEDS}

    # ---- sanity check on the new KBs ------------------------------------------------------------
    sanity = sanity_check(P_test[SEEDS[0]], ev_test, ev, X_test, feature_names, classes, idx, processed)
    print("Sanity check:", sanity)
    if not sanity["passed"]:
        print("ERROR: vectorised evidence / V0 do not reproduce SymbolicReasoner / NeuroSymbolicFuser on the "
              "lift-checked KBs", file=sys.stderr)
        sys.exit(1)

    # ---- validation memory per seed -------------------------------------------------------------
    vm_info, rows = {}, []
    for s in SEEDS:
        vm = np.load(ckpt_dir / f"valmem_seed{s}_{TAG}.npz")
        with open(root / "results" / "continual" / f"replay_summary_seed{s}_{TAG}.json") as f:
            summ = json.load(f)
        X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
        vm_info[s] = {"counts": summ["val_memory_counts"], "size": summ["val_memory_size"],
                      "buffer": summ["replay_buffer_size"], "total": summ["total_memory"]}
        P_vm = predict_proba(models[s], X_vm, DEVICE)
        Pe = P_test[s][ev]
        det_ev = Pe.argmax(1)
        zeros_ev = np.zeros((len(ev), K))

        ctrl = fv.StackingFuser(detector_only=True).fit(P_vm, *(np.zeros((len(y_vm), K)),) * 3, y_vm)
        ctrl_pred = ctrl.predict(Pe, zeros_ev, zeros_ev, zeros_ev)
        for kb_name, f in KBS.items():
            S_vm, F_vm, M_vm = fv.rule_evidence(processed / f, feature_names, X_vm, idx)
            S, F, M = (a[ev] for a in ev_test[f])
            score = lambda p: fast_macro_f1(y_vm, p, K)
            # first grid point wins ties, as in run_fusion_variants.py
            f2, a2 = max(((score(fv.v2_neutral(P_vm, S_vm, F_vm, a)), a) for a in ALPHAS), key=lambda r: r[0])
            f3, (a3, t3) = max(((score(fv.v3_gated_neutral(P_vm, S_vm, F_vm, a, t)), (a, t))
                                for a in ALPHAS for t in TAUS), key=lambda r: r[0])
            stacker = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm)
            preds = {
                "Detector alone": (det_ev, np.nan, np.nan, score(P_vm.argmax(1))),
                "V0 linear (alpha=0.5)": (fv.v0_linear(Pe, S, 0.5), 0.5, np.nan, score(fv.v0_linear(P_vm, S_vm, 0.5))),
                "V2 neutral-absence": (fv.v2_neutral(Pe, S, F, a2), a2, np.nan, f2),
                "V3 gated + neutral": (fv.v3_gated_neutral(Pe, S, F, a3, t3), a3, t3, f3),
                "V4 stacking": (stacker.predict(Pe, S, F, M), np.nan, np.nan, score(stacker.predict(P_vm, S_vm, F_vm, M_vm))),
                CONTROL: (ctrl_pred, np.nan, np.nan, score(ctrl.predict(P_vm, *(np.zeros((len(y_vm), K)),) * 3))),
            }
            for m, (pred, a, t, fvm) in preds.items():
                rows.append({"Source": "this step", "Setting": "Continual (valmem)", "Detector":
                             f"Replay (buffer={BUFFER}, valmem={VALMEM})", "Seed": s, "KB": kb_name, "Method": m,
                             "Alpha": a, "Tau": t, "ValMem_Macro_F1": fvm,
                             **metrics(y_ev, pred, det_ev, F, labels, classes)})
            print(f"seed {s} {kb_name}: " + ", ".join(f"{m.split(' ')[0]}={r['Macro_F1']:.4f}"
                                                       for m, r in zip(preds, rows[-len(preds):])))

    # ---- static XGBoost explanation rates (previous step's settings, cached evidence) ------------
    prev = pd.read_csv(out_dir / "fusion_variants.csv", keep_default_na=False, na_values=[""])
    from src.models.baselines import XGBoostBaseline
    xgb = XGBoostBaseline.load(root / "results" / "models" / "xgboost_multi.joblib").model
    raw = xgb.predict_proba(X_test)
    P_xgb = np.zeros((len(X_test), K), dtype=raw.dtype)
    P_xgb[:, [int(c) for c in xgb.classes_]] = raw
    Pe = P_xgb[ev]
    det_ev = Pe.argmax(1)
    for kb_name, f in STATIC_KBS.items():
        S, F, M = (a[ev] for a in ev_test[f])
        a_prev = float(prev[(prev["Setting"] == "Static") & (prev["KB"] == kb_name)
                            & (prev["Method"] == "V2 neutral-absence")]["Alpha"].iloc[0])
        for m, pred, a in [("Detector alone", det_ev, np.nan), ("V0 linear (alpha=0.5)", fv.v0_linear(Pe, S, 0.5), 0.5),
                           ("V2 neutral-absence (alpha=0.5)", fv.v2_neutral(Pe, S, F, 0.5), 0.5),
                           (f"V2 neutral-absence (previous step's alpha={a_prev:g})", fv.v2_neutral(Pe, S, F, a_prev), a_prev)]:
            rows.append({"Source": "this step (static, cached evidence)", "Setting": "Static", "Detector": "XGBoost",
                         "Seed": "n/a", "KB": kb_name, "Method": m, "Alpha": a, "Tau": np.nan,
                         **metrics(y_ev, pred, det_ev, F, labels, classes)})

    # ---- reference rows copied from the previous step (not recomputed) --------------------------
    for (det, kb, m), g in prev[((prev["Setting"] == "Reference") & (prev["Method"] == "Detector alone"))
                               | ((prev["Detector"] == "Replay (buffer=10000)") & prev["KB"].isin(["KB v1", "KB v2"])
                                  & prev["Method"].isin(["Detector alone", "V0 linear (alpha=0.5)",
                                                         "V2 neutral-absence", "V3 gated + neutral", "V4 stacking"]))
                               ].groupby(["Detector", "KB", "Method"], sort=False):
        src = ("previous step, copied" if m in ("Detector alone", "V0 linear (alpha=0.5)") else
               "previous step, copied; tuned / fitted on the TEST calibration split (all 10 classes)")
        for _, r in g.iterrows():
            rows.append({"Source": src, **{k: r[k] for k in r.index if k in prev.columns}})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fusion_legal.csv", index=False)
    write_markdown(df, vm_info, sanity, classes, len(ev), out_dir / "fusion_legal.md")
    print(f"Saved {out_dir / 'fusion_legal.csv'} and fusion_legal.md")


def sanity_check(P, ev_test, ev, X_test, feature_names, classes, idx, processed):
    rng = np.random.default_rng(42)
    rows = np.sort(rng.choice(ev, N_SANITY, replace=False))
    K = len(classes)
    out = {"rows": N_SANITY, "evidence_equal": True, "v0_checked": 0, "v0_matching": 0, "v0_tied_mismatches": 0,
           "v0_other_mismatches": 0}
    for kb_name in NEW_KBS:
        f = KBS[kb_name]
        sym = symbolic_results(processed / f, feature_names, X_test[rows])
        S_ref, M_ref = np.zeros((N_SANITY, K)), np.zeros((N_SANITY, K), dtype=np.int32)
        for i, r in enumerate(sym):
            for rule in r.get("all_matches", []):
                k = idx[rule["predicted_class"]]
                M_ref[i, k] += 1
                S_ref[i, k] = max(S_ref[i, k], rule.get("precision", 0.0))
        S, F, M = (a[rows] for a in ev_test[f])
        out["evidence_equal"] &= bool(np.array_equal(S, S_ref) and np.array_equal(M, M_ref)
                                      and np.array_equal(F, M_ref > 0))
        for alpha in ALPHAS:
            ref = fuse_predictions(P[rows], range(K), sym, classes, idx, alpha)
            scores = fv.linear_scores(P[rows], S, alpha)
            mine = scores.argmax(1)
            diff = np.flatnonzero(ref != mine)
            tied = [i for i in diff if scores[i, ref[i]] == scores[i].max() and fv.top_tied(scores[i:i + 1])[0]]
            out["v0_checked"] += N_SANITY
            out["v0_matching"] += N_SANITY - len(diff)
            out["v0_tied_mismatches"] += len(tied)
            out["v0_other_mismatches"] += len(diff) - len(tied)
    out["passed"] = out["evidence_equal"] and out["v0_other_mismatches"] == 0
    return out


def write_markdown(df, vm_info, sanity, classes, n_ev, path):
    cur = df[df["Source"] == "this step"]
    det = cur["Detector"].iloc[0]
    L = ["# Continual-legal fusion (validation memory)", "",
         f"Detectors: replay buffer {BUFFER} trained with `--val-memory-per-class {VALMEM}`, seeds {SEEDS}. "
         f"Before each experience, up to {VALMEM} training rows per class of that experience were held out (never "
         "trained on, never put in the replay buffer) and kept in a validation memory. **Every choice of alpha / tau "
         "and every fit of V4 and the control uses only that validation memory** (final model's probabilities plus "
         "rule features on the held-out rows). The test calibration split is not used. All metrics are on the "
         f"previous step's evaluation split ({n_ev} rows, results/fusion/split_indices.npz).", "",
         "Caveat: the KBs were built by continual_pipeline.py from the full training experiences, so the rules "
         "were induced and validated on data that includes the validation-memory rows. The detector never saw "
         "those rows, but the rule features on them may be slightly optimistic.", "",
         f"Sanity check on the lift-checked KBs ({sanity['rows']} evaluation rows, alpha in {ALPHAS}, PYTHONHASHSEED=0): "
         f"vectorised rule evidence identical to SymbolicReasoner: {sanity['evidence_equal']}; V0 matches "
         f"NeuroSymbolicFuser on {sanity['v0_matching']}/{sanity['v0_checked']} predictions "
         f"({sanity['v0_tied_mismatches']} exact-tie differences, {sanity['v0_other_mismatches']} other).", ""]

    # validation memory
    L += ["## Validation memory", "", "| Seed | " + " | ".join(classes) + " | Validation memory | Replay buffer | Total memory |",
          "|---|" + "---|" * (len(classes) + 3)]
    for s, v in vm_info.items():
        L.append(f"| {s} | " + " | ".join(str(v["counts"][c]) for c in classes)
                 + f" | {v['size']} | {v['buffer']} | {v['total']} |")

    # chosen parameters
    L += ["", "## alpha / tau chosen on validation-memory macro-F1", "",
          f"Per seed, in order {SEEDS}.", "", "| KB | V2 alpha | V3 alpha / tau |", "|---|---|---|"]
    for kb in KBS:
        g = cur[cur["KB"] == kb].set_index(["Method", "Seed"])
        L.append(f"| {kb} | " + ", ".join(f"{g.loc[('V2 neutral-absence', s), 'Alpha']:g}" for s in SEEDS) + " | "
                 + ", ".join(f"{g.loc[('V3 gated + neutral', s), 'Alpha']:g} / {g.loc[('V3 gated + neutral', s), 'Tau']:g}"
                             for s in SEEDS) + " |")

    # main tables
    det_alone = cur[cur["Method"] == "Detector alone"].set_index(["KB", "Seed"])
    for kb in KBS:
        g = cur[cur["KB"] == kb]
        L += ["", f"## {det} + {kb} (mean ± sample std over {len(SEEDS)} seeds)", "",
              "| Method | Accuracy | Macro-F1 | Macro-FPR | Fixed | Broken | Expl. | "
              f"ΔMacro-F1 vs detector alone, per seed {SEEDS} |", "|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            x = g[g["Method"] == m].set_index("Seed")
            if m == "Detector alone":
                d = "—"
            else:
                diffs = [x.loc[s, "Macro_F1"] - det_alone.loc[(kb, s), "Macro_F1"] for s in SEEDS]
                d = f"{np.mean(diffs):+.4f} ({', '.join(f'{v:+.4f}' for v in diffs)}; {sum(v > 0 for v in diffs)}/5 > 0)"
            L.append(f"| {m} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | {pm(x['Macro_FPR'])} | "
                     f"{x['Fixed'].mean():.0f} | {x['Broken'].mean():.0f} | {pm(x['Explanation_Rate'])} | {d} |")

    # per-class recall
    for kb in KBS:
        g = cur[cur["KB"] == kb]
        L += ["", f"### Per-class recall: {kb} (mean over seeds)", "",
              "| Method | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for m in METHODS:
            x = g[g["Method"] == m]
            L.append(f"| {m} | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")

    # references
    ref = df[df["Source"].str.startswith("previous step")]
    L += ["", "## Reference rows (copied from results/fusion/fusion_variants.csv, not recomputed)", "",
          "| Source | Detector | KB | Method | Macro-F1 | Accuracy | Fuzzers recall | Expl. |", "|---|---|---|---|---|---|---|---|"]
    for (src, d, kb, m), g in ref.groupby(["Source", "Detector", "KB", "Method"], sort=False):
        L.append(f"| {src} | {d} | {kb} | {m} | {pm(g['Macro_F1'])} | {pm(g['Accuracy'])} | "
                 f"{g['Recall_Fuzzers'].mean():.4f} | {pm(g['Explanation_Rate'])} |")

    # static explanation rate
    st = df[df["Setting"] == "Static"]
    L += ["", "## Static XGBoost: explanation rate (evaluation split, cached rule evidence)", "",
          "| KB | Method | Accuracy | Macro-F1 | Changed | Explanation rate |", "|---|---|---|---|---|---|"]
    for _, r in st.iterrows():
        L.append(f"| {r['KB']} | {r['Method']} | {r['Accuracy']:.4f} | {r['Macro_F1']:.4f} | {r['Changed']} | "
                 f"{r['Explanation_Rate']:.4f} |")

    # summary
    means = cur.groupby(["KB", "Method"], sort=False)["Macro_F1"].mean()
    joint = ref[(ref["Detector"] == JOINT_REF)]["Macro_F1"]
    best = means.idxmax()
    v4 = {kb: means.loc[(kb, "V4 stacking")] for kb in KBS}
    rec = cur.groupby(["KB", "Method"], sort=False)[["Recall_Fuzzers", "Recall_Dos", "Recall_Reconnaissance"]].mean()
    L += ["", "## Summary", "",
          f"- Best legal configuration: {best[0]} + {best[1]}, mean macro-F1 {means.max():.4f}; V4 by KB: "
          + ", ".join(f"{kb} {v:.4f}" for kb, v in v4.items())
          + f"; {JOINT_REF}: {joint.mean():.4f} (copied). Detector alone: "
          f"{means.loc[(list(KBS)[0], 'Detector alone')]:.4f}.",
          "- Fuzzers / DoS / Recon recall (mean over seeds): " + "; ".join(
              f"{kb} {m.split(' ')[0]} {rec.loc[(kb, m), 'Recall_Fuzzers']:.3f} / {rec.loc[(kb, m), 'Recall_Dos']:.3f} / "
              f"{rec.loc[(kb, m), 'Recall_Reconnaissance']:.3f}"
              for kb in KBS for m in ["V0 linear (alpha=0.5)", "V2 neutral-absence"]) + ".",
          "- Static XGBoost explanation rate: " + "; ".join(
              f"{r['KB']} {r['Method']} {r['Explanation_Rate']:.4f}" for _, r in st.iterrows()) + "."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
