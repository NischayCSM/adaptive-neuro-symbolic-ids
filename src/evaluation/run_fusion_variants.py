"""
Fusion variants (src/reasoning/fusion_variants.py) vs. the detector alone, with an honest tuning
protocol on the 10-class test set.

Protocol: the test set is split ONCE into a stratified 20% calibration split and an 80% evaluation
split (seed 42; indices saved to results/fusion/split_indices.npz and reused). alpha in
{0.3, 0.5, 0.7, 0.9} and tau in {0.5, ..., 0.9} are chosen on calibration macro-F1 and V4 is fitted
on calibration only; every reported metric is on the evaluation split.

Settings (no detector is retrained):
    static      XGBoost (results/models/xgboost_multi.joblib) with the static KB
                (knowledge_base.db) and with continual_kb_v1_seed42.db
    continual   final replay checkpoints (buffer 2000 / 5000 / 10000, seeds 42, 1, 2, 3, 4) with
                KB v1 / v2 / v3 (continual_kb_v1_seed42.db, continual_kb_v2.db, continual_kb_v3.db)
    reference   class-weighted joint training (20 epochs) final checkpoints, detector alone

Sanity check (run first; the script stops if it fails): on 5,000 random test rows, the vectorised
rule evidence must equal SymbolicReasoner's matches, and V0 must give the same predictions as
cl_eval.fuse_predictions (NeuroSymbolicFuser) for every detector x KB x alpha. A differing prediction
is accepted only if it is an exact tie: in the fuser's float32 arithmetic the top fused score is
shared by two or more classes and both the fuser's class and V0's class are among them (the fuser
breaks such ties by Python set order, V0 by lowest class index). Any other difference stops the run.
The script re-launches itself with PYTHONHASHSEED=0 so the fuser's set order is fixed.

Rule evidence is cached under results/fusion/cache/. Writes results/fusion/fusion_variants.csv,
results/fusion/sanity_check.csv, results/fusion/sanity_check_mismatches.csv and
results/fusion/fusion_variants.md.
"""

import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from src.device import DEVICE
from src.models.neural_ids import NeuralIDS
from src.continual.cl_eval import (
    find_project_root, set_seed, load_continual_data, predict_proba, symbolic_results, fuse_predictions,
)
from src.evaluation.evaluate_milestone2 import calculate_fpr
from src.reasoning import fusion_variants as fv

SEEDS = [42, 1, 2, 3, 4]
BUFFERS = [2000, 5000, 10000]
ALPHAS = [0.3, 0.5, 0.7, 0.9]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
CAL_FRACTION = 0.2
SPLIT_SEED = 42
N_SANITY = 5000
N_BOOT = 1000
STATIC_KBS = {"static KB": "knowledge_base.db", "KB v1": "continual_kb_v1_seed42.db"}
CONTINUAL_KBS = {"KB v1": "continual_kb_v1_seed42.db", "KB v2": "continual_kb_v2.db", "KB v3": "continual_kb_v3.db"}
METHODS = ["Detector alone", "V0 linear (alpha=0.5)", "V0 linear (tuned)", "V1 gated", "V2 neutral-absence",
           "V3 gated + neutral", "V4 stacking"]
FUSION_METHODS = METHODS[1:]
JOINT_REF = "Joint training (20 epochs, class-weighted)"
CONTROL = "Control: stacking on log P only"  # V4 without rule features; KB "none"


def fast_macro_f1(y, p, K):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)))


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        # NeuroSymbolicFuser breaks exact ties by set order; fix the hash seed for the sanity check.
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_fusion_variants", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    processed = root / "data" / "processed"
    out_dir = root / "results" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels = len(classes), list(range(len(classes)))
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_test = np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    N = len(y_test)

    # ---- one fixed calibration / evaluation split -------------------------------------------
    split_path = out_dir / "split_indices.npz"
    if split_path.exists():
        z = np.load(split_path)
        cal, ev = z["calibration"], z["evaluation"]
        if len(cal) + len(ev) != N or int(z["seed"]) != SPLIT_SEED:
            print(f"ERROR: {split_path} does not match the current test set / seed.", file=sys.stderr)
            sys.exit(1)
    else:
        ev, cal = train_test_split(np.arange(N), test_size=CAL_FRACTION, stratify=y_test, random_state=SPLIT_SEED)
        cal, ev = np.sort(cal), np.sort(ev)
        np.savez(split_path, calibration=cal, evaluation=ev, seed=SPLIT_SEED)
    print(f"Calibration {len(cal)} rows, evaluation {len(ev)} rows")

    # ---- rule evidence per KB (cached) --------------------------------------------------------
    kb_files = {**STATIC_KBS, **CONTINUAL_KBS}
    missing = [f for f in set(kb_files.values()) if not (processed / f).exists()]
    evidence = {}
    for f in sorted(set(kb_files.values())):
        if f in missing:
            continue
        evidence[f] = fv.cached_rule_evidence(processed / f, feature_names, X_test, idx,
                                              out_dir / "cache" / f"{f[:-3]}_test.npz")

    # ---- detectors ----------------------------------------------------------------------------
    xgb_path = root / "results" / "models" / "xgboost_multi.joblib"
    rc, jc = root / "results" / "continual" / "replay_checkpoints", root / "results" / "continual" / "joint_checkpoints"
    replay_ckpts = {(b, s): rc / (f"neural_ids_exp_4_seed{s}.pth" if b == 2000 else f"neural_ids_exp_4_seed{s}_buf{b}.pth")
                    for b in BUFFERS for s in SEEDS}
    joint_ckpts = {s: jc / f"joint_seed{s}_e20_cw.pth" for s in SEEDS}
    missing += [str(p.relative_to(root)) for p in [xgb_path, *replay_ckpts.values(), *joint_ckpts.values()]
                if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    from src.models.baselines import XGBoostBaseline
    xgb = XGBoostBaseline.load(xgb_path).model
    xgb_raw = xgb.predict_proba(X_test)
    P_xgb = np.zeros((N, K), dtype=xgb_raw.dtype)
    P_xgb[:, [int(c) for c in xgb.classes_]] = xgb_raw

    def neural_probs(ckpt):
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return predict_proba(model, X_test, DEVICE)

    # (setting, detector, seed, {kb name: file}, P, raw probs, raw prob classes)
    runs = [("Static", "XGBoost", "n/a", STATIC_KBS, P_xgb, xgb_raw, xgb.classes_)]
    for b in BUFFERS:
        for s in SEEDS:
            P = neural_probs(replay_ckpts[(b, s)])
            runs.append(("Continual", f"Replay (buffer={b})", s, CONTINUAL_KBS, P, P, range(K)))

    # ---- sanity check: vectorised evidence + V0 == existing fusion path ------------------------
    sanity_rows, mismatch_rows = sanity_check(runs, evidence, X_test, feature_names, classes, idx, processed)
    pd.DataFrame(sanity_rows).to_csv(out_dir / "sanity_check.csv", index=False)
    pd.DataFrame(mismatch_rows, columns=["Setting", "Detector", "Seed", "KB", "Alpha", "Test_Row", "Fuser_Class",
                                         "V0_Class", "Top_Score", "Fuser_Confidence", "Tied_Classes", "Tied"]
                 ).to_csv(out_dir / "sanity_check_mismatches.csv", index=False)
    total = sum(r["Rows"] for r in sanity_rows)
    matched = sum(r["Matching"] for r in sanity_rows)
    not_tied = sum(r["Mismatching_Not_Tied"] for r in sanity_rows)
    print(f"V0 sanity check: {matched}/{total} predictions match ({matched / total:.6f}); "
          f"{total - matched - not_tied} differing rows are exact ties, {not_tied} are not; "
          f"{sum(r['Tied_Rows'] for r in sanity_rows)} sampled rows have a tied top score")
    if not_tied or not all(r["Evidence_Equal"] for r in sanity_rows):
        print("ERROR: V0 does not reproduce NeuroSymbolicFuser (non-tie mismatch or evidence mismatch); see "
              "results/fusion/sanity_check.csv and sanity_check_mismatches.csv", file=sys.stderr)
        sys.exit(1)

    # ---- tune on calibration, evaluate on evaluation -----------------------------------------
    y_cal, y_ev = y_test[cal], y_test[ev]
    rows, boot_preds = [], {}
    for setting, detector, seed, kbs, P, _, _ in runs:
        det_ev = P[ev].argmax(1)
        for kb_name, f in kbs.items():
            S, F, M = evidence[f]
            preds, params = tune_and_predict(P, S, F, M, cal, ev, y_cal, K)
            for method in METHODS:
                rows.append({"Setting": setting, "Detector": detector, "Seed": seed, "KB": kb_name, "Method": method,
                             **params[method], **metrics(y_ev, preds[method], det_ev, F[ev], labels, classes)})
            if setting == "Static":
                boot_preds[kb_name] = (det_ev, preds, params)
            print(f"{setting} {detector} seed={seed} {kb_name}: " +
                  ", ".join(f"{m.split(' ')[0]}={r['Macro_F1']:.4f}" for m, r in zip(METHODS, rows[-len(METHODS):])))
        rows.append(control_row(setting, detector, seed, P, cal, ev, y_cal, y_ev, labels, classes))
    for s in SEEDS:
        P = neural_probs(joint_ckpts[s])
        det_ev = P[ev].argmax(1)
        rows.append({"Setting": "Reference", "Detector": JOINT_REF, "Seed": s, "KB": "none",
                     "Method": "Detector alone", "Alpha": np.nan, "Tau": np.nan, "Cal_Macro_F1": np.nan,
                     "Top_Score_Ties": int(fv.top_tied(P[ev]).sum()),
                     **metrics(y_ev, det_ev, det_ev, np.zeros((len(ev), K), dtype=bool), labels, classes)})
        rows.append(control_row("Reference", JOINT_REF, s, P, cal, ev, y_cal, y_ev, labels, classes))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fusion_variants.csv", index=False)

    # KB v2: which classes' rules fire on Fuzzers evaluation rows, and which classes' rules never fire.
    F2 = evidence[CONTINUAL_KBS["KB v2"]][1]
    kb2 = {"fuzz_fire": dict(zip(classes, F2[ev[y_ev == idx["Fuzzers"]]].mean(0))),
           "never_fires": [c for c, k in zip(classes, F2.any(0)) if not k]}

    boot = static_bootstrap(boot_preds, y_ev, K)
    write_markdown(df, boot, sanity_rows, mismatch_rows, kb2, out_dir / "fusion_variants.md", classes,
                   len(cal), len(ev))
    print(f"Saved {out_dir / 'fusion_variants.csv'} and fusion_variants.md")


def sanity_check(runs, evidence, X_test, feature_names, classes, idx, processed):
    from src.reasoning.fusion import NeuroSymbolicFuser

    rng = np.random.default_rng(SPLIT_SEED)
    rows_idx = np.sort(rng.choice(len(X_test), N_SANITY, replace=False))
    K = len(classes)
    sym_cache, ev_equal = {}, {}
    out, mismatches = [], []
    for setting, detector, seed, kbs, P, raw, raw_classes in runs:
        for kb_name, f in kbs.items():
            if f not in sym_cache:
                sym = symbolic_results(processed / f, feature_names, X_test[rows_idx])
                S_ref, M_ref = np.zeros((N_SANITY, K)), np.zeros((N_SANITY, K), dtype=np.int32)
                for i, r in enumerate(sym):
                    for rule in r.get("all_matches", []):
                        k = idx[rule["predicted_class"]]
                        M_ref[i, k] += 1
                        S_ref[i, k] = max(S_ref[i, k], rule.get("precision", 0.0))
                S, F, M = evidence[f]
                sym_cache[f] = sym
                ev_equal[f] = bool(np.array_equal(S[rows_idx], S_ref) and np.array_equal(M[rows_idx], M_ref)
                                   and np.array_equal(F[rows_idx], M_ref > 0))
            S = evidence[f][0][rows_idx]
            for alpha in ALPHAS:
                ref = fuse_predictions(raw[rows_idx], raw_classes, sym_cache[f], classes, idx, alpha)
                scores = fv.linear_scores(P[rows_idx], S, alpha)  # the fuser's float32 arithmetic
                mine = scores.argmax(1)
                top = scores.max(1)
                n_tied = 0
                for i in np.flatnonzero(ref != mine):
                    # The fuser's own score for its chosen class must equal the top score as well.
                    n_probs = {classes[int(c)]: raw[rows_idx[i]][j] for j, c in enumerate(raw_classes)}
                    conf = NeuroSymbolicFuser(alpha=alpha).fuse(n_probs, sym_cache[f][i])["final_confidence"]
                    tied_cls = np.flatnonzero(scores[i] == top[i])
                    tied = bool(len(tied_cls) >= 2 and ref[i] in tied_cls and mine[i] in tied_cls and conf == top[i])
                    n_tied += tied
                    mismatches.append([setting, detector, seed, kb_name, alpha, int(rows_idx[i]), classes[ref[i]],
                                       classes[mine[i]], float(top[i]), float(conf),
                                       ";".join(classes[c] for c in tied_cls), tied])
                n_diff = int((ref != mine).sum())
                out.append({"Setting": setting, "Detector": detector, "Seed": seed, "KB": kb_name, "Alpha": alpha,
                            "Rows": N_SANITY, "Matching": N_SANITY - n_diff, "Tied_Rows": int(fv.top_tied(scores).sum()),
                            "Tied_Test_Rows": ";".join(str(r) for r in rows_idx[fv.top_tied(scores)]),
                            "Mismatching": n_diff, "Mismatching_Tied": n_tied, "Mismatching_Not_Tied": n_diff - n_tied,
                            "All_Mismatches_Tied": n_diff == n_tied, "Evidence_Equal": ev_equal[f],
                            "Passed": n_diff == n_tied and ev_equal[f]})
    return out, mismatches


def tune_and_predict(P, S, F, M, cal, ev, y_cal, K):
    Pc, Sc, Fc = P[cal], S[cal], F[cal]
    score = lambda p: fast_macro_f1(y_cal, p, K)

    def best(fn, grid):
        results = [(score(fn(*g)), g) for g in grid]
        return max(results, key=lambda r: r[0])  # first grid point wins ties

    grid_a = [(a,) for a in ALPHAS]
    grid_at = [(a, t) for a in ALPHAS for t in TAUS]
    f0, (a0,) = best(lambda a: fv.v0_linear(Pc, Sc, a), grid_a)
    f1, (a1, t1) = best(lambda a, t: fv.v1_gated(Pc, Sc, a, t), grid_at)
    f2, (a2,) = best(lambda a: fv.v2_neutral(Pc, Sc, Fc, a), grid_a)
    f3, (a3, t3) = best(lambda a, t: fv.v3_gated_neutral(Pc, Sc, Fc, a, t), grid_at)
    stacker = fv.StackingFuser().fit(Pc, Sc, Fc, M[cal], y_cal)

    Pe, Se, Fe = P[ev], S[ev], F[ev]
    nan = np.nan
    # Eval rows whose top score (of the scores the method takes argmax over) is an exact tie.
    tP = fv.top_tied(Pe)
    ties = {
        "Detector alone": tP,
        "V0 linear (alpha=0.5)": fv.top_tied(fv.linear_scores(Pe, Se, 0.5)),
        "V0 linear (tuned)": fv.top_tied(fv.linear_scores(Pe, Se, a0)),
        "V1 gated": np.where(Pe.max(1) >= t1, tP, fv.top_tied(fv.linear_scores(Pe, Se, a1))),
        "V2 neutral-absence": fv.top_tied(fv.neutral_scores(Pe, Se, Fe, a2)),
        "V3 gated + neutral": np.where(Pe.max(1) >= t3, tP, fv.top_tied(fv.neutral_scores(Pe, Se, Fe, a3))),
        "V4 stacking": fv.top_tied(stacker.decision_function(Pe, Se, Fe, M[ev])),
    }
    preds = {
        "Detector alone": Pe.argmax(1),
        "V0 linear (alpha=0.5)": fv.v0_linear(Pe, Se, 0.5),
        "V0 linear (tuned)": fv.v0_linear(Pe, Se, a0),
        "V1 gated": fv.v1_gated(Pe, Se, a1, t1),
        "V2 neutral-absence": fv.v2_neutral(Pe, Se, Fe, a2),
        "V3 gated + neutral": fv.v3_gated_neutral(Pe, Se, Fe, a3, t3),
        "V4 stacking": stacker.predict(Pe, Se, Fe, M[ev]),
    }
    params = {
        "Detector alone": {"Alpha": nan, "Tau": nan, "Cal_Macro_F1": score(Pc.argmax(1))},
        "V0 linear (alpha=0.5)": {"Alpha": 0.5, "Tau": nan, "Cal_Macro_F1": score(fv.v0_linear(Pc, Sc, 0.5))},
        "V0 linear (tuned)": {"Alpha": a0, "Tau": nan, "Cal_Macro_F1": f0},
        "V1 gated": {"Alpha": a1, "Tau": t1, "Cal_Macro_F1": f1},
        "V2 neutral-absence": {"Alpha": a2, "Tau": nan, "Cal_Macro_F1": f2},
        "V3 gated + neutral": {"Alpha": a3, "Tau": t3, "Cal_Macro_F1": f3},
        "V4 stacking": {"Alpha": nan, "Tau": nan, "Cal_Macro_F1": score(stacker.predict(Pc, Sc, Fc, M[cal]))},
    }
    for m in params:
        params[m]["Top_Score_Ties"] = int(ties[m].sum())
    return preds, params


def control_row(setting, detector, seed, P, cal, ev, y_cal, y_ev, labels, classes):
    """V4's stacker on the detector's log-probabilities only, fitted on calibration (no rule evidence)."""
    K = len(classes)
    none = np.zeros((len(P), K))
    ctrl = fv.StackingFuser(detector_only=True).fit(P[cal], none[cal], none[cal], none[cal], y_cal)
    pred = ctrl.predict(P[ev], none[ev], none[ev], none[ev])
    return {"Setting": setting, "Detector": detector, "Seed": seed, "KB": "none", "Method": CONTROL,
            "Alpha": np.nan, "Tau": np.nan,
            "Cal_Macro_F1": fast_macro_f1(y_cal, ctrl.predict(P[cal], none[cal], none[cal], none[cal]), K),
            "Top_Score_Ties": int(fv.top_tied(ctrl.decision_function(P[ev], none[ev], none[ev], none[ev])).sum()),
            **metrics(y_ev, pred, P[ev].argmax(1), np.zeros((len(ev), K), dtype=bool), labels, classes)}


def metrics(y, pred, det, F, labels, classes):
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    changed = pred != det
    macro_f1 = float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))
    assert abs(macro_f1 - fast_macro_f1(y, pred, len(labels))) < 1e-12
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "Macro_F1": macro_f1,
        "Macro_FPR": float(np.mean(list(calculate_fpr(y, pred, labels).values()))),
        **{f"Recall_{c}": float(r) for c, r in zip(classes, recalls)},
        "Changed": int(changed.sum()),
        "Fixed": int((changed & (det != y) & (pred == y)).sum()),
        "Broken": int((changed & (det == y) & (pred != y)).sum()),
        "Explanation_Rate": float(F[np.arange(len(pred)), pred].mean()),
    }


def static_bootstrap(boot_preds, y, K):
    """Paired bootstrap (eval rows resampled) of macro-F1(method) - macro-F1(XGBoost alone)."""
    rng = np.random.default_rng(SPLIT_SEED)
    out = {}
    diffs = {kb: {m: [] for m in FUSION_METHODS} for kb in boot_preds}
    for _ in range(N_BOOT):
        b = rng.integers(0, len(y), len(y))
        yb = y[b]
        for kb, (det, preds, _) in boot_preds.items():
            base = fast_macro_f1(yb, det[b], K)
            for m in FUSION_METHODS:
                diffs[kb][m].append(fast_macro_f1(yb, preds[m][b], K) - base)
    for kb, (det, preds, params) in boot_preds.items():
        for m in FUSION_METHODS:
            d = np.array(diffs[kb][m])
            out[(kb, m)] = {"point": fast_macro_f1(y, preds[m], K) - fast_macro_f1(y, det, K),
                            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                            "cal": params[m]["Cal_Macro_F1"]}
    return out


def pm(series):
    return f"{series.mean():.4f} ± {series.std(ddof=1):.4f}"


def write_markdown(df, boot, sanity_rows, mismatch_rows, kb2, path, classes, n_cal, n_ev):
    ctrl, df = df[df["Method"] == CONTROL], df[df["Method"] != CONTROL]
    tied_checks = [f"{r['Detector']} seed {r['Seed']} + {r['KB']}, alpha {r['Alpha']:g}: test row(s) "
                   f"{r['Tied_Test_Rows'].replace(';', ', ')}" for r in sanity_rows if r["Tied_Rows"]]
    total = sum(r["Rows"] for r in sanity_rows)
    matched = sum(r["Matching"] for r in sanity_rows)
    n_tied_mis = sum(r["Mismatching_Tied"] for r in sanity_rows)
    mis = "; ".join(f"{m[1]} seed {m[2]} + {m[3]}, alpha {m[4]:g}, test row {m[5]}: fuser {m[6]}, V0 {m[7]}, "
                    f"tied classes {m[10].replace(';', ' / ')} at {m[8]:.7g}" for m in mismatch_rows)
    L = ["# Fusion variants vs. detector alone", "",
         f"Test set split once (stratified, seed {SPLIT_SEED}): calibration {n_cal} rows (20%), evaluation "
         f"{n_ev} rows (80%); indices in results/fusion/split_indices.npz. alpha in {ALPHAS} and tau in {TAUS} "
         "are chosen on calibration macro-F1, V4 is fitted on calibration only, and **every number below is on "
         "the evaluation split**. V0 (alpha=0.5) is the current fusion, untuned.", "",
         "**Assumption:** the tuned variants and V4 use labelled calibration data from all 10 classes. In the "
         "continual setting that is a post-hoc calibration assumption (a deployed continual model would not have "
         "labelled data from future classes), so the tuned continual rows are optimistic relative to a strictly "
         "online protocol.", "",
         f"V0 sanity check (5,000 random test rows, every detector x KB x alpha in {ALPHAS}, "
         f"PYTHONHASHSEED=0): {matched}/{total} predictions identical to cl_eval.fuse_predictions / "
         f"NeuroSymbolicFuser ({matched / total:.4%}); "
         + (f"the {total - matched} differing prediction(s) are all exact float32 top-score ties "
            f"({n_tied_mis}), which the fuser breaks by Python set order and V0 by lowest class index ({mis})"
            if total > matched else
            "no prediction differs") + f". Sampled checks with a tied V0 top score: "
         f"{sum(r['Tied_Rows'] for r in sanity_rows)} of {total}"
         + (f" ({'; '.join(tied_checks)}); on such rows the fuser's choice depends on PYTHONHASHSEED"
            if tied_checks else "") + ". Vectorised rule evidence identical to "
         f"SymbolicReasoner on all KBs: {all(r['Evidence_Equal'] for r in sanity_rows)}.", "",
         "Columns: Changed / Fixed / Broken = eval rows whose prediction differs from the detector alone / "
         "of those, detector wrong -> fused right / detector right -> fused wrong. Expl. = explanation rate, "
         "fraction of rows where at least one fired rule supports the final predicted class.", ""]

    # chosen parameters
    L += ["## Chosen alpha / tau (on calibration macro-F1)", "", "### Static (XGBoost)", "",
          "| KB | V0 alpha | V1 alpha / tau | V2 alpha | V3 alpha / tau |", "|---|---|---|---|---|"]
    st = df[df["Setting"] == "Static"]
    for kb in STATIC_KBS:
        r = st[st["KB"] == kb].set_index("Method")
        L.append(f"| {kb} | {r.loc['V0 linear (tuned)', 'Alpha']:g} | {r.loc['V1 gated', 'Alpha']:g} / "
                 f"{r.loc['V1 gated', 'Tau']:g} | {r.loc['V2 neutral-absence', 'Alpha']:g} | "
                 f"{r.loc['V3 gated + neutral', 'Alpha']:g} / {r.loc['V3 gated + neutral', 'Tau']:g} |")
    L += ["", f"### Continual (replay), per seed in order {SEEDS}", "",
          "| Detector | KB | V0 alpha | V1 alpha/tau | V2 alpha | V3 alpha/tau |", "|---|---|---|---|---|---|"]
    co = df[df["Setting"] == "Continual"]
    for (det, kb), g in co.groupby(["Detector", "KB"], sort=False):
        g = g.set_index(["Method", "Seed"])
        a = lambda m: ", ".join(f"{g.loc[(m, s), 'Alpha']:g}" for s in SEEDS)
        at = lambda m: ", ".join(f"{g.loc[(m, s), 'Alpha']:g}/{g.loc[(m, s), 'Tau']:g}" for s in SEEDS)
        L.append(f"| {det} | {kb} | {a('V0 linear (tuned)')} | {at('V1 gated')} | {a('V2 neutral-absence')} "
                 f"| {at('V3 gated + neutral')} |")

    # static main table + bootstrap
    for kb in STATIC_KBS:
        r = st[st["KB"] == kb].set_index("Method")
        L += ["", f"## Static: XGBoost + {kb}", "",
              "| Method | Accuracy | Macro-F1 | Macro-FPR | Changed | Fixed | Broken | Expl. | ΔMacro-F1 vs XGBoost [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            x = r.loc[m]
            ci = "—" if m == "Detector alone" else (
                f"{boot[(kb, m)]['point']:+.4f} [{boot[(kb, m)]['lo']:+.4f}, {boot[(kb, m)]['hi']:+.4f}]")
            L.append(f"| {m} | {x['Accuracy']:.4f} | {x['Macro_F1']:.4f} | {x['Macro_FPR']:.4f} | {x['Changed']} "
                     f"| {x['Fixed']} | {x['Broken']} | {x['Explanation_Rate']:.4f} | {ci} |")
        best = max(FUSION_METHODS[1:], key=lambda m: boot[(kb, m)]["cal"])
        b = boot[(kb, best)]
        L += ["", f"Best variant chosen on calibration macro-F1: **{best}**; eval macro-F1 difference vs XGBoost "
                  f"alone {b['point']:+.4f}, 95% paired-bootstrap CI [{b['lo']:+.4f}, {b['hi']:+.4f}] "
                  f"({N_BOOT} resamples of the evaluation rows)."]

    # continual main tables
    det_alone = co[co["Method"] == "Detector alone"].set_index(["Detector", "KB", "Seed"])
    ref = df[df["Setting"] == "Reference"]
    for det in [f"Replay (buffer={b})" for b in BUFFERS]:
        L += ["", f"## Continual: {det} (mean ± sample std over {len(SEEDS)} seeds)", "",
              "| KB | Method | Accuracy | Macro-F1 | Macro-FPR | Changed | Fixed | Broken | Expl. | "
              f"ΔMacro-F1 vs replay alone, per seed {SEEDS} |", "|---|---|---|---|---|---|---|---|---|---|"]
        for kb in CONTINUAL_KBS:
            g = co[(co["Detector"] == det) & (co["KB"] == kb)]
            for m in METHODS:
                x = g[g["Method"] == m].set_index("Seed")
                if m == "Detector alone":
                    d = "—"
                else:
                    diffs = [x.loc[s, "Macro_F1"] - det_alone.loc[(det, kb, s), "Macro_F1"] for s in SEEDS]
                    d = f"{np.mean(diffs):+.4f} ({', '.join(f'{v:+.4f}' for v in diffs)}; {sum(v > 0 for v in diffs)}/5 > 0)"
                L.append(f"| {kb} | {m} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | {pm(x['Macro_FPR'])} | "
                         f"{x['Changed'].mean():.0f} | {x['Fixed'].mean():.0f} | {x['Broken'].mean():.0f} | "
                         f"{pm(x['Explanation_Rate'])} | {d} |")
    L += ["", f"Reference ({JOINT_REF}, detector alone, same evaluation split): accuracy {pm(ref['Accuracy'])}, "
              f"macro-F1 {pm(ref['Macro_F1'])}, macro-FPR {pm(ref['Macro_FPR'])}."]

    # per-class recall, buffer 10k + KB v2
    g = co[(co["Detector"] == "Replay (buffer=10000)") & (co["KB"] == "KB v2")]
    L += ["", "## Per-class recall: Replay (buffer=10000) + KB v2 (mean over seeds)", "",
          "| Method | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for m in METHODS:
        x = g[g["Method"] == m]
        L.append(f"| {m} | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")
    L.append(f"| *{JOINT_REF} (reference)* | " + " | ".join(f"{ref[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")

    # flip analysis
    L += ["", "## Flip analysis vs detector alone (evaluation split)", "",
          "Static: rows; continual: mean over seeds. Net = Fixed - Broken.", "",
          "| Setting | KB | Method | Changed | Fixed | Broken | Net |", "|---|---|---|---|---|---|---|"]
    for (setting, det, kb), g in df[df["Setting"] != "Reference"].groupby(["Setting", "Detector", "KB"], sort=False):
        for m in FUSION_METHODS:
            x = g[g["Method"] == m]
            fx, br = x["Fixed"].mean(), x["Broken"].mean()
            L.append(f"| {det} | {kb} | {m} | {x['Changed'].mean():.0f} | {fx:.0f} | {br:.0f} | {fx - br:+.0f} |")

    # control: V4's stacker without rule features
    kb_cols = list(STATIC_KBS) + [k for k in CONTINUAL_KBS if k not in STATIC_KBS]
    L += ["", "## Control: how much of V4's gain comes from the rule features?", "",
          f"'{CONTROL}' fits the same balanced logistic regression on the calibration split using only "
          "log(P + 1e-9), i.e. a recalibration of the detector with no rule evidence. Macro-F1 on the evaluation "
          "split (continual and reference: mean ± sample std over seeds).", "",
          "| Detector | Detector alone | " + CONTROL + " | " + " | ".join(f"V4 + {k}" for k in kb_cols) + " |",
          "|---|---|---|" + "---|" * len(kb_cols)]
    fmt = lambda s: "—" if s.empty else (f"{s.iloc[0]:.4f}" if len(s) == 1 else pm(s))
    for det, g in pd.concat([df, ctrl]).groupby("Detector", sort=False):
        f1 = lambda m, kb=None: g[(g["Method"] == m) & ((g["KB"] == kb) if kb else True)]["Macro_F1"]
        da = g[g["Method"] == "Detector alone"].drop_duplicates("Seed")["Macro_F1"]
        L.append(f"| {det} | {fmt(da)} | {fmt(f1(CONTROL))} | "
                 + " | ".join(fmt(f1("V4 stacking", kb)) for kb in kb_cols) + " |")

    # top-score ties on the evaluation split
    L += ["", "## Exact top-score ties on the evaluation split", "",
          f"Evaluation rows (of {n_ev}) where the top score a method takes argmax over is shared exactly by two or "
          "more classes; argmax then takes the lowest class index. Static: rows; continual: mean (max) over seeds.",
          "", "| Setting | KB | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for (setting, det, kb), g in df[df["Setting"] != "Reference"].groupby(["Setting", "Detector", "KB"], sort=False):
        t = g.groupby("Method")["Top_Score_Ties"]
        mean, mx = t.mean(), t.max()
        cell = (lambda m: f"{mean[m]:.0f}") if setting == "Static" else (lambda m: f"{mean[m]:.1f} ({mx[m]})")
        L.append(f"| {det} | {kb} | " + " | ".join(cell(m) for m in METHODS) + " |")
    ref_t = df[df["Setting"] == "Reference"]["Top_Score_Ties"]
    L.append(f"| *{JOINT_REF}* | none | {ref_t.mean():.1f} ({ref_t.max()}) |" + " — |" * (len(METHODS) - 1))

    L += ["", "## Summary", ""] + summary(df, ctrl, boot, classes, kb2, n_ev)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def summary(df, ctrl, boot, classes, kb2, n_ev):
    """Plain-English findings; every number and comparison word is computed from the results."""
    st, co = df[df["Setting"] == "Static"], df[df["Setting"] == "Continual"]
    ref = df[df["Setting"] == "Reference"]
    hand = FUSION_METHODS[1:5]  # tuned V0, V1, V2, V3
    means = co.groupby(["Detector", "KB", "Method"], sort=False)["Macro_F1"].mean()
    out = []

    # static
    ci = lambda kb, m: f"{boot[(kb, m)]['point']:+.4f} [{boot[(kb, m)]['lo']:+.4f}, {boot[(kb, m)]['hi']:+.4f}]"
    verdict = lambda kb, m: ("worse" if boot[(kb, m)]["hi"] < 0 else
                             "better" if boot[(kb, m)]["lo"] > 0 else "not distinguishable")
    sig = [(kb, m) for kb in STATIC_KBS for m in FUSION_METHODS if boot[(kb, m)]["lo"] > 0]
    sh = st[st["Method"].isin(hand)]
    v0 = FUSION_METHODS[0]
    out += [
        ("**Static: no variant beats XGBoost alone.** No fusion variant has a macro-F1 difference whose 95% CI "
         "lies above 0. " if not sig else "**Static:** variants with a macro-F1 CI above 0: "
         + ", ".join(f"{m} + {kb}" for kb, m in sig) + ". ")
        + f"Tuning moves the fusion towards XGBoost alone: the tuned variants V0-V3 change at most "
        f"{sh['Changed'].max()} of {n_ev} evaluation rows ({sh['Changed'].max() / n_ev:.1%}) and end within "
        f"{max(abs(boot[(kb, m)]['point']) for kb in STATIC_KBS for m in hand):.4f} macro-F1 of it, so the rules "
        "add almost nothing XGBoost does not already know. The current fusion (V0, alpha=0.5): "
        + ", ".join(f"{kb} {ci(kb, v0)} ({verdict(kb, v0)})" for kb in STATIC_KBS)
        + ". V4 stacking, the best variant on calibration, changes about "
        f"{st[st['Method'] == 'V4 stacking']['Changed'].mean() / n_ev:.0%} of predictions: "
        + ", ".join(f"{kb} {ci(kb, 'V4 stacking')} ({verdict(kb, 'V4 stacking')})" for kb in STATIC_KBS) + ".", ""]

    # continual: V4
    pairs = [(f"Replay (buffer={b})", kb) for b in BUFFERS for kb in CONTINUAL_KBS]
    v4_wins = sum(means.loc[p].idxmax() == "V4 stacking" for p in pairs)
    cm = ctrl[ctrl["Setting"] == "Continual"].groupby("Detector")["Macro_F1"].mean()
    dets = [f"Replay (buffer={b})" for b in BUFFERS]
    share = [(means.loc[(d, "KB v1", "V4 stacking")] - cm[d])
             / (means.loc[(d, "KB v1", "V4 stacking")] - means.loc[(d, "KB v1", "Detector alone")]) for d in dets]
    d10 = "Replay (buffer=10000)"
    g = co[(co["Detector"] == d10) & (co["KB"] == "KB v2")].groupby("Method")
    rec = g[[f"Recall_{c}" for c in classes]].mean()
    acc, expl = g["Accuracy"].mean(), g["Explanation_Rate"].mean()
    lost = [c for c in classes if rec.loc["Detector alone", f"Recall_{c}"] < 0.05]
    no_rules = [c for c in lost if c in kb2["never_fires"]]
    out += [
        f"**Continual: V4 stacking has the highest mean macro-F1** in {v4_wins} of {len(pairs)} replay x KB "
        "settings. To see whether that comes from the rules or from simply recalibrating the detector on labelled "
        "data, the control stacks on the detector's log-probabilities alone: it reaches "
        + ", ".join(f"{cm[d]:.4f}" for d in dets) + " for buffers " + "/".join(str(b) for b in BUFFERS)
        + ", against " + ", ".join(f"{means.loc[(d, 'KB v1', 'V4 stacking')]:.4f}" for d in dets)
        + " for V4 with KB v1. So the rule features account for "
        + "/".join(f"{s:.0%}" for s in share) + " of V4's gain over replay alone. The logistic regression "
        "learns, from labelled calibration rows of all 10 classes, which combinations of rule firings and detector "
        "scores go with each class. That brings back classes the replay detector no longer predicts (10k + KB v2 "
        "recall, detector -> V4: "
        + ", ".join(f"{c} {rec.loc['Detector alone', f'Recall_{c}']:.2f} -> {rec.loc['V4 stacking', f'Recall_{c}']:.2f}"
                    for c in lost) + ")"
        + (f", including {', '.join(no_rules[:-1]) + ' and ' if len(no_rules) > 1 else ''}{no_rules[-1]}, for which "
           "no KB v2 rule fires on any test row" if no_rules else "")
        + ". Caveats: it relies on the post-hoc calibration assumption above; at 10k + KB v2 it changes accuracy "
        f"from {acc['Detector alone']:.4f} to {acc['V4 stacking']:.4f} and DoS recall from "
        f"{rec.loc['Detector alone', 'Recall_Dos']:.2f} to {rec.loc['V4 stacking', 'Recall_Dos']:.2f}; and its "
        f"explanation rate is {expl['V4 stacking']:.2f} against {expl['V0 linear (tuned)']:.2f} for tuned V0, "
        "so more of its predictions have no supporting rule.", ""]

    # continual: hand-designed variants and Fuzzers
    m10 = means.loc[(d10, "KB v2")]
    best_hand = max(hand, key=lambda m: m10[m])
    fired = sorted(((v, c) for c, v in kb2["fuzz_fire"].items() if v > 0.05 and c != "Fuzzers"), reverse=True)
    r = lambda m, c: rec.loc[m, f"Recall_{c}"]
    out += [
        "**Among the hand-designed variants (V0-V3), tuning alpha does most of the work.** At 10k + KB v2 the "
        f"current V0 (alpha=0.5) changes macro-F1 by {m10[v0] - m10['Detector alone']:+.4f} against replay alone, "
        f"tuned V0 by {m10['V0 linear (tuned)'] - m10['Detector alone']:+.4f}, and gating / neutral-absence add "
        f"{max(m10[m] for m in hand[1:]) - m10['V0 linear (tuned)']:+.4f} at most on top. The best of them "
        f"({best_hand}) has DoS / Reconnaissance / Fuzzers recall {r(best_hand, 'Dos'):.3f} / "
        f"{r(best_hand, 'Reconnaissance'):.3f} / {r(best_hand, 'Fuzzers'):.3f}, against {r(v0, 'Dos'):.3f} / "
        f"{r(v0, 'Reconnaissance'):.3f} / {r(v0, 'Fuzzers'):.3f} for V0 alpha=0.5 and "
        f"{r('Detector alone', 'Dos'):.3f} / {r('Detector alone', 'Reconnaissance'):.3f} / "
        f"{r('Detector alone', 'Fuzzers'):.3f} for replay alone. So it keeps most of the DoS / Recon gain and "
        "recovers most, but not all, of Fuzzers. "
        + ("KB v2 has no rule that fires for Fuzzers, so neutral-absence already scores Fuzzers by the detector "
           "alone on every row. " if "Fuzzers" in kb2["never_fires"] else "")
        + "The remaining loss comes from rules for other classes firing on Fuzzers traffic: on Fuzzers evaluation "
        "rows, KB v2 rules fire for " + ", ".join(f"{c} ({v:.0%})" for v, c in fired)
        + ", and their high stored precision outvotes the detector.", ""]

    # joint training reference
    best = means.idxmax()
    bs = co[(co["Detector"] == best[0]) & (co["KB"] == best[1]) & (co["Method"] == best[2])]["Macro_F1"]
    rf = ref[ref["Method"] == "Detector alone"]["Macro_F1"]
    rc = ctrl[ctrl["Setting"] == "Reference"]["Macro_F1"]
    cmp = lambda a, b: "above" if a > b else "below"
    out += [
        f"**Against joint training:** the best continual configuration ({best[0]} + {best[1]} + {best[2]}) has "
        f"mean macro-F1 {bs.mean():.4f} (seeds {bs.min():.4f}-{bs.max():.4f}), {cmp(bs.mean(), rf.mean())} "
        f"{JOINT_REF} at {rf.mean():.4f} (seeds {rf.min():.4f}-{rf.max():.4f}) on the same evaluation split. The "
        "comparison favours V4, which is fitted on labelled data from the test distribution. With the same "
        f"rule-free calibration step, joint training reaches {rc.mean():.4f} ± {rc.std(ddof=1):.4f}, which is "
        f"{cmp(rc.mean(), bs.mean())} the best continual configuration's {bs.mean():.4f}.", ""]

    # compact per-setting facts
    out += ["Per setting:", ""]
    for kb in STATIC_KBS:
        worse = [m for m in FUSION_METHODS if boot[(kb, m)]["hi"] < 0]
        out.append(f"- Static, XGBoost + {kb}: macro-F1 CI above 0: "
                   f"{', '.join(m for k, m in sig if k == kb) or 'none'}; CI below 0: {', '.join(worse) or 'none'}.")
    for det, kb in pairs:
        m = means.loc[(det, kb)]
        out.append(f"- {det} + {kb}: highest mean macro-F1 is **{m.idxmax()}** ({m.max():.4f}; replay alone "
                   f"{m['Detector alone']:.4f}, current V0 alpha=0.5 {m[v0]:.4f}).")
    return out


if __name__ == "__main__":
    main()
