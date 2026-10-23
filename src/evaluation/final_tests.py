"""
Final 8-test scorecard.

    python -m src.evaluation.final_tests            # Phase I scorecard -> results/final_test_report.md
    python -m src.evaluation.final_tests --phase2   # final configuration -> results/final_test_report_phase2.md
                                                    #   (+ results/final_test_phase2_metrics.csv)

--phase2 scores the final configuration (R1 detectors: replay 10000, class-weighted, validation memory
200 per class capped at 0.5; KB continual_kb_v1_gain_clean.db; V4 stacking fit on the validation memory;
static tests: XGBoost + V2 neutral-absence, alpha 0.5, same KB). The tests and targets are the Phase I
ones; each metric is defined in the report.
"""

import argparse
import ast
import os
import sys
import pandas as pd
from pathlib import Path
import subprocess
import sqlite3
import json


def parse_drift_flag(value) -> bool:
    """Drift flag from a Drift_Detected cell, in either format:
        new: a boolean ("True" / "False" in the CSV)
        old: the DriftDetector result dict as text, "{'drift_score': ..., 'drift_detected': True, ...}"
    Anything else raises ValueError. (A substring test such as "True" in text is not used: it would
    also match other text containing "True".)"""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    text = str(value).strip()
    if text in ("True", "False"):
        return text == "True"
    if text.startswith("{"):
        d = ast.literal_eval(text)
        if isinstance(d, dict) and isinstance(d.get("drift_detected"), bool):
            return d["drift_detected"]
    raise ValueError(f"unrecognised Drift_Detected value: {value!r}")


def drift_flags(csv_path) -> dict:
    """{experience: flagged} from a continual_pipeline_metrics*.csv file."""
    df = pd.read_csv(csv_path, dtype={"Drift_Detected": str})
    return {int(e): parse_drift_flag(v) for e, v in zip(df["Experience"], df["Drift_Detected"])}


def run_final_tests():
    PROJECT_ROOT = Path(os.getcwd())
    RESULTS_DIR = PROJECT_ROOT / "results"
    md_path = RESULTS_DIR / "final_test_report.md"
    report_lines = ["# Final Master Test Report\n"]
    passes = 0
    total = 8

    # Test 1: False alarm rate
    try:
        df2 = pd.read_csv(RESULTS_DIR / "milestone2_comparison.csv")
        fpr = df2[df2["Model"] == "XGBoost + Fusion"]["Macro_FPR"].iloc[0]
        # Target: Realistic IDS standard is < 5% (0.05)
        if fpr < 0.05:
            passes += 1
            report_lines.append(f"✅ **Test 1 (False Alarm Rate): PASS** - FPR is {fpr:.4f} (Target < 0.05)")
        else:
            report_lines.append(f"❌ **Test 1 (False Alarm Rate): FAIL** - FPR is {fpr:.4f} (Target < 0.05)")
    except Exception as e:
        report_lines.append(f"❌ **Test 1 (False Alarm Rate): FAIL** - Missing data: {e}")

    # Test 2: Known-attack detection rate
    try:
        recall_dos = df2[df2["Model"] == "XGBoost + Fusion"]["Recall_Dos"].iloc[0]
        recall_recon = df2[df2["Model"] == "XGBoost + Fusion"]["Recall_Reconnaissance"].iloc[0]
        avg_recall = (recall_dos + recall_recon) / 2
        # Target: Realistic IDS standard is > 80% (0.80). I previously set it to 0.15 just to pass.
        if avg_recall >= 0.80: 
            passes += 1
            report_lines.append(f"✅ **Test 2 (Known-Attack Detection): PASS** - Avg Recall DoS/Recon {avg_recall:.4f} (Target >= 0.80)")
        else:
            report_lines.append(f"❌ **Test 2 (Known-Attack Detection): FAIL** - Avg Recall DoS/Recon {avg_recall:.4f} (Target >= 0.80)")
    except Exception as e:
        report_lines.append(f"❌ **Test 2 (Known-Attack Detection): FAIL** - Missing data: {e}")

    # Test 3: New-attack adaptation
    try:
        # Final replay model (seed 42) + continual KB on the newest experience (Exp 4)
        df_comp = pd.read_csv(RESULTS_DIR / "milestone3_comparison.csv")
        exp4_neuro = df_comp[df_comp["Experience"] == 4]["Replay_NeuroSym_Final"].iloc[0]
        # Target: Rapid adaptation should maintain > 80% accuracy.
        if exp4_neuro >= 0.80: 
            passes += 1
            report_lines.append(f"✅ **Test 3 (New-Attack Adaptation): PASS** - Exp4 (New Attacks) Accuracy is {exp4_neuro:.4f} (Target >= 0.80)")
        else:
            report_lines.append(f"❌ **Test 3 (New-Attack Adaptation): FAIL** - Exp4 (New Attacks) Accuracy is {exp4_neuro:.4f} (Target >= 0.80)")
    except Exception as e:
        report_lines.append(f"❌ **Test 3 (New-Attack Adaptation): FAIL** - Missing data: {e}")

    # Test 4: Drift detection accuracy
    try:
        flags = drift_flags(RESULTS_DIR / "continual" / "continual_pipeline_metrics.csv")
        # Experience 1 is always processed (no reference yet), so only Experiences 2-4 count.
        flagged = [e for e in (2, 3, 4) if flags.get(e)]
        if flagged == [2, 3, 4]:
            passes += 1
            report_lines.append(f"✅ **Test 4 (Drift Detection): PASS** - Drift successfully flagged at Experiences 2, 3, and 4")
        else:
            report_lines.append(f"❌ **Test 4 (Drift Detection): FAIL** - Drift flagged at Experiences {flagged} of [2, 3, 4]")
    except Exception as e:
        report_lines.append(f"❌ **Test 4 (Drift Detection): FAIL** - Missing data: {e}")

    # Test 5: Forgetting retention
    try:
        # Retention: accuracy of the FINAL replay model (after Exp 4) on Exp 1-3 test sets,
        # i.e. the last row of the replay accuracy matrix (seed 42, neural only).
        df_replay = pd.read_csv(RESULTS_DIR / "continual" / "replay_seed42.csv")
        last = df_replay.iloc[-1]
        if last["Phase"] != "After Exp4":
            raise ValueError(f"last row of replay_seed42.csv is {last['Phase']!r}, expected 'After Exp4'")
        avg_retention = (last["Exp1_Acc"] + last["Exp2_Acc"] + last["Exp3_Acc"]) / 3
        # Target: > 80% average retention across old tasks.
        if avg_retention >= 0.80:
            passes += 1
            report_lines.append(f"✅ **Test 5 (Forgetting Retention): PASS** - Replay model (seed 42) avg accuracy on Exp 1-3 after Exp 4 is {avg_retention:.4f} (Target >= 0.80)")
        else:
            report_lines.append(f"❌ **Test 5 (Forgetting Retention): FAIL** - Replay model (seed 42) avg accuracy on Exp 1-3 after Exp 4 is {avg_retention:.4f} (Target >= 0.80)")
    except Exception as e:
        report_lines.append(f"❌ **Test 5 (Forgetting Retention): FAIL** - Missing data: {e}")

    # Test 6: Rule quality
    try:
        db_path = PROJECT_ROOT / "data" / "processed" / "continual_kb.db"
        conn = sqlite3.connect(db_path)
        df_rules = pd.read_sql_query("SELECT precision, coverage FROM rules WHERE status = 'ACTIVE'", conn)
        conn.close()
        avg_prec = df_rules['precision'].mean()
        avg_cov = df_rules['coverage'].mean()
        
        # Target: Rules should have >90% precision and meaningful coverage (>5%)
        if avg_prec >= 0.90 and avg_cov >= 0.05:
            passes += 1
            report_lines.append(f"✅ **Test 6 (Rule Quality): PASS** - Avg Precision: {avg_prec:.4f}, Avg Coverage: {avg_cov:.4f} (Targets: Prec >= 0.90, Cov >= 0.05)")
        else:
            report_lines.append(f"❌ **Test 6 (Rule Quality): FAIL** - Avg Precision: {avg_prec:.4f}, Avg Coverage: {avg_cov:.4f} (Targets: Prec >= 0.90, Cov >= 0.05)")
    except Exception as e:
        report_lines.append(f"❌ **Test 6 (Rule Quality): FAIL** - Failed to query KB: {e}")

    # Test 7: Rule conflict handling
    try:
        res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_knowledge_base.py"], capture_output=True, text=True)
        if res.returncode == 0:
            passes += 1
            report_lines.append(f"✅ **Test 7 (Rule Conflict Handling): PASS** - Synthetic conflict unit tests passed successfully")
        else:
            detail = (res.stdout + res.stderr).strip().splitlines()[-1:] or ["no output"]
            report_lines.append(f"❌ **Test 7 (Rule Conflict Handling): FAIL** - pytest exit {res.returncode}: {detail[0]}")
    except Exception as e:
        report_lines.append(f"❌ **Test 7 (Rule Conflict Handling): FAIL** - Pytest error: {e}")

    # Test 8: External generalization
    report_lines.append("⚠️ **Test 8 (External Generalization): CONDITIONAL** - Rule coverage on TON_IoT: 100%. However, symbolic precision scores clustered near 0.99 across multiple competing classes per row (e.g. Row 2: Exploits 0.9995, Normal 0.9961, Fuzzers 0.9910 — simultaneously), so rule confidence did not meaningfully discriminate between classes under domain shift. Fused accuracy (0.2366) was nearly identical to neural-only (0.2365), indicating the symbolic layer failed to correct neural errors on out-of-distribution data.")
    passes += 0 # Explicitly not counting this as a PASS.

    report_lines.append("\n## Final Verdict")
    report_lines.append(f"**MEASURED ({passes}/{total})** - Final evaluation complete.")

    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(md_path, "w", encoding='utf-8') as f:
        f.write(report_text)


# ---- Phase II: the same 8 tests on the final configuration -------------------------------------------------

PHASE2_SEEDS = [42, 1, 2, 3, 4]
PHASE2_KB = "continual_kb_v1_gain_clean.db"
PHASE2_STATIC_ALPHA = 0.5
CHECK_TOL = 1e-9
# Value regexes for the Phase I report lines (results/final_test_report.md); values are read, not typed in.
PHASE1_PATTERNS = {1: r"FPR is ([\d.]+)", 2: r"Recall DoS/Recon ([\d.]+)", 3: r"Accuracy is ([\d.]+)", 4: None,
                   5: r"after Exp 4 is ([\d.]+)", 6: r"Avg Precision: ([\d.]+), Avg Coverage: ([\d.]+)", 7: None,
                   8: r"Fused accuracy \(([\d.]+)\)"}


def phase1_values(md_path):
    """{test: (status, [values])} parsed from the Phase I report."""
    import re
    out = {}
    for line in Path(md_path).read_text(encoding="utf-8").splitlines():
        m = re.search(r"Test (\d) \([^)]*\): (PASS|FAIL|CONDITIONAL)", line)
        if m:
            t = int(m.group(1))
            pat = PHASE1_PATTERNS[t]
            vals = list(re.search(pat, line).groups()) if pat else []
            out[t] = (m.group(2), vals)
    if sorted(out) != list(range(1, 9)):
        raise ValueError(f"could not parse all 8 tests from {md_path}: found {sorted(out)}")
    return out


def run_phase2_tests(v2=False):
    """v2: Test 1 scored by the Normal->attack rate (results/fusion/benign_fpr.csv) for the static (1a) and the
    continual (1b) configuration, macro-FPR secondary; writes the _v2 files and never overwrites them."""
    from src.continual.cl_eval import find_project_root
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        # Same environment as run_fusion_final.py / run_detector_variants.py, whose numbers are cross-checked.
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.final_tests", *sys.argv[1:]],
                                cwd=root, env=env).returncode)

    import numpy as np
    import torch
    from src.device import DEVICE
    from src.models.neural_ids import NeuralIDS
    from src.models.baselines import XGBoostBaseline
    from src.continual.cl_eval import set_seed, load_continual_data, predict_proba
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    from src.reasoning import fusion_variants as fv
    from src.evaluation.run_fusion_variants import metrics
    from src.evaluation.run_fusion_final import TAG

    processed, results = root / "data" / "processed", root / "results"
    ckpt_dir = results / "continual" / "replay_checkpoints"
    suffix = "_v2" if v2 else ""
    md_path = results / f"final_test_report_phase2{suffix}.md"
    csv_path = results / f"final_test_phase2_metrics{suffix}.csv"
    benign = None
    if v2:
        for p in (md_path, csv_path):
            if p.exists():
                sys.exit(f"ERROR: {p.relative_to(root)} already exists; not overwriting")
        if not (results / "fusion" / "benign_fpr.csv").exists():
            sys.exit("ERROR: results/fusion/benign_fpr.csv not found; run src.evaluation.run_benign_fpr first")
        benign = pd.read_csv(results / "fusion" / "benign_fpr.csv")
    tag = lambda s: f"seed{s}_{TAG}_cw"  # R1 = class-weighted valmem runs (run_detector_variants.py)
    kb_path = processed / PHASE2_KB
    drift_csv = results / "continual" / f"continual_pipeline_metrics_{PHASE2_KB[:-3]}.csv"
    needed = [kb_path, drift_csv, results / "final_test_report.md", results / "models" / "xgboost_multi.joblib",
              results / "fusion" / "split_indices.npz", results / "fusion" / "fusion_final.csv",
              results / "fusion" / "detector_variants.csv", results / "external_netflow" / "netflow_eval.csv"]
    needed += [ckpt_dir / f"neural_ids_exp_4_{tag(s)}.pth" for s in PHASE2_SEEDS]
    needed += [ckpt_dir / f"valmem_{tag(s)}.npz" for s in PHASE2_SEEDS]
    missing = [str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)
    set_seed(42)

    data = load_continual_data(root)
    classes, idx, feature_names = data["all_classes"], data["class_to_idx"], data["feature_names"]
    K, labels = len(classes), list(range(len(classes)))
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X_train, X_test = np.load(processed / "X_train.npy"), np.load(processed / "X_test.npy")
    y_test = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    ev = np.load(results / "fusion" / "split_indices.npz")["evaluation"]
    exp_classes = [np.unique(y) for _, y in data["test_splits"]]
    normal = idx["Normal"]
    # Not fv.cached_rule_evidence: its cache lives in results/fusion/cache and must not be rewritten here.
    S_te, F_te, M_te = fv.rule_evidence(kb_path, feature_names, X_test, idx)

    def scored(scope, system, seed, y, pred, det, F):
        exp_acc = {f"Exp{k}_Acc": float((pred[np.isin(y, c)] == y[np.isin(y, c)]).mean())
                   for k, c in enumerate(exp_classes, start=1)}
        return {"Scope": scope, "System": system, "Seed": seed, "Rows": len(y),
                **metrics(y, pred, det, F, labels, classes), **exp_acc,
                "Benign_FPR": float((pred[y == normal] != normal).mean())}

    # ---- static: XGBoost + V2 neutral-absence (alpha 0.5, final KB), evaluation split ------------------------
    xgb = XGBoostBaseline.load(results / "models" / "xgboost_multi.joblib").model
    if list(xgb.classes_) != labels:
        print(f"ERROR: xgboost_multi classes {list(xgb.classes_)} are not {labels}", file=sys.stderr)
        sys.exit(1)
    P_x = xgb.predict_proba(X_test)
    static_pred = fv.v2_neutral(P_x, S_te, F_te, PHASE2_STATIC_ALPHA)
    rows = [scored("evaluation split", "XGBoost + V2 (alpha 0.5)", 42, y_test[ev], static_pred[ev],
                   P_x[ev].argmax(1), F_te[ev])]

    # ---- continual: R1 detector alone and R1 + V4 (fit on the validation memory) ----------------------------
    for s in PHASE2_SEEDS:
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_dir / f"neural_ids_exp_4_{tag(s)}.pth", map_location=DEVICE))
        vm = np.load(ckpt_dir / f"valmem_{tag(s)}.npz")
        X_vm, y_vm = X_train[vm["global_index"]], vm["y"]
        P_vm, P_te = predict_proba(model, X_vm, DEVICE), predict_proba(model, X_test, DEVICE)
        S_vm, F_vm, M_vm = fv.rule_evidence(kb_path, feature_names, X_vm, idx)
        stacker = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm)
        det, v4 = P_te.argmax(1), stacker.predict(P_te, S_te, F_te, M_te)
        for system, pred in [("R1 detector alone", det), ("R1 + V4 stacking", v4)]:
            rows.append(scored("evaluation split", system, s, y_test[ev], pred[ev], det[ev], F_te[ev]))
            rows.append(scored("full test set", system, s, y_test, pred, det, F_te))
        print(f"seed {s}: " + ", ".join(f"{r['System']} ({r['Scope']}) macro-F1 {r['Macro_F1']:.4f}" for r in rows[-4:]))
    df = pd.DataFrame(rows)

    # ---- cross-checks against the results files these numbers must reproduce --------------------------------
    ff = pd.read_csv(results / "fusion" / "fusion_final.csv")
    ref = ff[(ff["Detector"] == "XGBoost") & (ff["KB"] == "KB v1_gain_clean")
             & (ff["Method"] == "V2 neutral-absence (alpha=0.5)")].iloc[0]
    check_cols = ["Accuracy", "Macro_F1", "Macro_FPR"] + [f"Recall_{c}" for c in classes]
    diffs = [abs(float(df.iloc[0][c]) - float(ref[c])) for c in check_cols]
    dv = pd.read_csv(results / "fusion" / "detector_variants.csv")
    dv = dv[dv["Variant"] == "R1"]
    for system, method in [("R1 detector alone", "Detector alone"), ("R1 + V4 stacking", "V4 stacking")]:
        for s in PHASE2_SEEDS:
            mine = df[(df["Scope"] == "evaluation split") & (df["System"] == system) & (df["Seed"] == s)].iloc[0]
            theirs = dv[(dv["Method"] == method) & (dv["Seed"] == s)].iloc[0]
            diffs += [abs(float(mine[c]) - float(theirs[c])) for c in check_cols]
    if benign is not None:  # Test 1 v2 values must equal this run's own Normal->attack rates
        for system, bsys in [("XGBoost + V2 (alpha 0.5)", "XGBoost + V2 (alpha 0.5)"),
                             ("R1 + V4 stacking", "R1 + V4 stacking")]:
            for _, mine in df[(df["Scope"] == "evaluation split") & (df["System"] == system)].iterrows():
                b = benign[(benign["Scope"] == "evaluation split") & (benign["System"] == bsys)
                           & (benign["Seed"] == mine["Seed"])].iloc[0]
                diffs += [abs(mine["Benign_FPR"] - b["Normal_to_Attack"]), abs(mine["Macro_FPR"] - b["Macro_FPR"])]
    max_diff = max(diffs)
    print(f"Cross-check vs fusion_final.csv / detector_variants.csv{' / benign_fpr.csv' if v2 else ''}: "
          f"max |difference| = {max_diff:.1e}")
    if max_diff > CHECK_TOL:
        print("ERROR: recomputed final-configuration metrics do not reproduce the results files", file=sys.stderr)
        sys.exit(1)
    df.to_csv(csv_path, index=False)

    # ---- tests 4, 6, 7, 8 (no model predictions needed) -------------------------------------------------------
    flags = drift_flags(drift_csv)
    drift_scores = pd.read_csv(drift_csv).set_index("Experience")["Drift_Score"]
    conn = sqlite3.connect(kb_path)
    kb_rules = pd.read_sql_query("SELECT precision, coverage FROM rules WHERE status = 'ACTIVE'", conn)
    conn.close()
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_knowledge_base.py"], cwd=root,
                         capture_output=True, text=True)
    pytest_tail = ((res.stdout + res.stderr).strip().splitlines()[-1:] or ["no output"])[0]
    nf = pd.read_csv(results / "external_netflow" / "netflow_eval.csv")
    nf = nf[nf["Domain"] == "NF-ToN"]

    # Phase I Normal->attack rate of the Phase I Test 1 system (XGBoost + Fusion) = 1 - its Normal recall
    m2 = pd.read_csv(results / "milestone2_comparison.csv").set_index("Model")
    p1_benign = 1 - float(m2.loc["XGBoost + Fusion", "Recall_Normal"])

    write_phase2_report(df, flags, drift_scores, kb_rules, res.returncode, pytest_tail, nf, max_diff,
                        phase1_values(results / "final_test_report.md"), classes, exp_classes, len(ev), md_path,
                        benign=benign, p1_benign=p1_benign)
    print(f"Saved {md_path} and {csv_path}")


def write_phase2_report(df, flags, drift_scores, kb_rules, pytest_rc, pytest_tail, nf, max_diff, p1, classes,
                        exp_classes, n_ev, md_path, benign=None, p1_benign=None):
    from src.evaluation.run_fusion_variants import pm

    ev_rows = df[df["Scope"] == "evaluation split"]
    static = ev_rows[ev_rows["System"].str.startswith("XGBoost")].iloc[0]
    v4 = ev_rows[ev_rows["System"] == "R1 + V4 stacking"]
    det = ev_rows[ev_rows["System"] == "R1 detector alone"]
    names = lambda k: ", ".join(classes[c] for c in exp_classes[k - 1])
    nf_v4, nf_xgb = nf[nf["Method"] == "V4 stacking"], nf[nf["Method"] == "XGBoost alone"]
    mean = lambda x: float(x.mean())
    p1_cell = lambda t: f"{p1[t][0]}" + (f" {' / '.join(p1[t][1])}" if p1[t][1] else "")

    t2 = (static["Recall_Dos"] + static["Recall_Reconnaissance"]) / 2
    t5 = v4[["Exp1_Acc", "Exp2_Acc", "Exp3_Acc"]].mean(axis=1)
    t5_det = det[["Exp1_Acc", "Exp2_Acc", "Exp3_Acc"]].mean(axis=1)
    flagged = [e for e in (2, 3, 4) if flags.get(e)]
    prec, cov = float(kb_rules["precision"].mean()), float(kb_rules["coverage"].mean())
    nf_fpr, nf_dr = mean(nf_v4["FPR_Benign"]), mean(nf_v4["Detection_Rate"])
    verdicts = {
        1: static["Macro_FPR"] < 0.05,
        2: t2 >= 0.80,
        3: mean(v4["Exp4_Acc"]) >= 0.80,
        4: flagged == [2, 3, 4],
        5: mean(t5) >= 0.80,
        6: prec >= 0.90 and cov >= 0.05,
        7: pytest_rc == 0,
        8: nf_fpr < 0.05 and nf_dr >= 0.80,
    }
    v = lambda t: "PASS" if verdicts[t] else "FAIL"
    table = [
        (1, "False alarm rate", "< 0.05", p1_cell(1), f"{static['Macro_FPR']:.4f}"),
        (2, "Known-attack detection", ">= 0.80", p1_cell(2), f"{t2:.4f}"),
        (3, "New-attack adaptation", ">= 0.80", p1_cell(3), pm(v4["Exp4_Acc"])),
        (4, "Drift detection", "flagged at Exp 2, 3, 4", p1_cell(4), f"flagged at {flagged}"),
        (5, "Forgetting retention", ">= 0.80", p1_cell(5), pm(t5)),
        (6, "Rule quality", "prec >= 0.90, cov >= 0.05", p1_cell(6), f"{prec:.4f} / {cov:.4f}"),
        (7, "Rule conflict handling", "unit tests pass", p1_cell(7), "pytest exit 0" if pytest_rc == 0
         else f"pytest exit {pytest_rc}"),
        (8, "External generalisation", "Benign FPR < 0.05 and DR >= 0.80", p1_cell(8),
         f"FPR {pm(nf_v4['FPR_Benign'])}, DR {pm(nf_v4['Detection_Rate'])}"),
    ]
    test1_def = ("1. **False alarm rate** = macro-FPR: for each of the 10 classes, FP / (FP + TN) of the one-vs-rest "
                 "confusion matrix, averaged over classes (as `evaluate_milestone2.calculate_fpr`). System: XGBoost + V2. "
                 f"Context: share of Normal rows predicted as any attack = {static['Benign_FPR']:.4f}. Phase I: XGBoost "
                 "+ V0 (alpha 0.5, static KB), full test set.")
    if benign is not None:
        bev = benign[benign["Scope"] == "evaluation split"]
        b_static = float(bev[bev["System"] == "XGBoost + V2 (alpha 0.5)"]["Normal_to_Attack"].iloc[0])
        b_v4 = bev[bev["System"] == "R1 + V4 stacking"]["Normal_to_Attack"]
        b_det = bev[bev["System"] == "R1 detector alone"]["Normal_to_Attack"]
        verdicts = {"1a": b_static < 0.05, "1b": mean(b_v4) < 0.05, **{t: ok for t, ok in verdicts.items() if t != 1}}
        table = [("1a", "False alarm rate, static (XGBoost + V2)", "Normal→attack < 0.05",
                  f"{p1_cell(1)} (macro-FPR); Normal→attack {p1_benign:.4f}",
                  f"Normal→attack {b_static:.4f} (macro-FPR {static['Macro_FPR']:.4f})"),
                 ("1b", "False alarm rate, continual (R1 + V4)", "Normal→attack < 0.05", "not measured",
                  f"Normal→attack {pm(b_v4)} (macro-FPR {pm(v4['Macro_FPR'])})")] + table[1:]
        test1_def = ("1. **False alarm rate** (primary) = Normal→attack rate: share of true-Normal flows predicted as "
                     "any attack class, i.e. the false-alarm rate an operator sees; values from "
                     "`results/fusion/benign_fpr.csv` and checked against this run's predictions. Secondary: macro-FPR "
                     "(mean one-vs-rest FP / (FP + TN) over the 10 classes, the Phase I / Phase II metric), which "
                     "averages the Normal column with nine attack columns and so hides the benign false-alarm rate. "
                     "1a: XGBoost + V2; 1b: R1 + V4 (context, R1 detector alone: Normal→attack "
                     f"{pm(b_det)}). Phase I: XGBoost + V0 (alpha 0.5, static KB), full test set; its Normal→attack "
                     "rate is 1 − Recall_Normal of 'XGBoost + Fusion' in `results/milestone2_comparison.csv`. The "
                     "continual system's benign false-alarm rate was not measured in Phase I.")
    L = [f"# Final Master Test Report - Phase II (final configuration){' - v2' if benign is not None else ''}", "",
         "Final configuration: R1 detectors (replay buffer 10000, class-weighted, validation memory 200 per class "
         "capped at 0.5 of each class), KB `continual_kb_v1_gain_clean.db`, V4 stacking fit on the validation memory "
         f"only. Static tests (1, 2): 10-class XGBoost (`xgboost_multi.joblib`) + V2 neutral-absence, alpha "
         f"{PHASE2_STATIC_ALPHA} (untuned), same KB. Continual tests: mean ± sample std over seeds {PHASE2_SEEDS}. "
         f"Model-based tests are on the evaluation split of `results/fusion/split_indices.npz` ({n_ev} rows), "
         "the split every Phase II fusion result uses.",
         f"Recomputed metrics reproduce results/fusion/fusion_final.csv (static) and results/fusion/"
         f"detector_variants.csv (R1, detector alone and V4, every seed): max |difference| = {max_diff:.1e}. "
         f"Per-seed values: `results/{md_path.name.replace('final_test_report_phase2', 'final_test_phase2_metrics').replace('.md', '.csv')}`.",
         "Phase I column: copied from `results/final_test_report.md` (the Phase I scorecard file)."
         + (" v2 change: Test 1 is scored by the Normal→attack rate, separately for the static (1a) and the "
            "continual (1b) configuration; all other tests are unchanged from `final_test_report_phase2.md`."
            if benign is not None else ""), "",
         "| Test | Name | Target | Phase I | Phase II | Phase II verdict |", "|---|---|---|---|---|---|"]
    L += [f"| {t} | {n} | {tg} | {a} | {b} | **{v(t)}** |" for t, n, tg, a, b in table]
    L += ["", "## Metric definitions", "",
          test1_def,
          "2. **Known-attack detection** = (recall of DoS + recall of Reconnaissance) / 2, 10-class recall. "
          f"System: XGBoost + V2 (DoS {static['Recall_Dos']:.4f}, Reconnaissance "
          f"{static['Recall_Reconnaissance']:.4f}). Phase I: XGBoost + V0, full test set.",
          f"3. **New-attack adaptation** = accuracy of the final model (after Exp 4) on the evaluation rows of "
          f"Exp 4's classes ({names(4)}); a row counts as correct only if the exact 10-class label is predicted. "
          f"System: R1 + V4. Context, R1 detector alone: {pm(det['Exp4_Acc'])}. Phase I: replay (buffer 2000, "
          "seed 42) + continual KB (V0), Exp 4 test split.",
          "4. **Drift detection** = experiences among 2-4 whose Drift_Detected flag is True in "
          "`continual_pipeline_metrics_continual_kb_v1_gain_clean.csv` (the build of the final KB; threshold 0.15 on "
          "the mean KS statistic). Drift scores: " + ", ".join(f"Exp {e} {drift_scores[e]:.4f}" for e in (2, 3, 4))
          + ". The flag is parsed strictly (boolean or the old dict text). Experience 1 is always processed and "
          "does not count.",
          "5. **Forgetting retention** = final-model retention: the model after Exp 4, scored on the evaluation "
          "rows of each earlier experience's classes, accuracy averaged over Exp 1, 2, 3 (equal weight per "
          f"experience; Exp 1 = {names(1)}; Exp 2 = {names(2)}; Exp 3 = {names(3)}). System: R1 + V4; per "
          f"experience {pm(v4['Exp1_Acc'])} / {pm(v4['Exp2_Acc'])} / {pm(v4['Exp3_Acc'])}. Context, R1 detector "
          f"alone: {pm(t5_det)}. This is NOT the average of the diagonal (each model on its own experience). "
          "Phase I file value: replay (buffer 2000, seed 42), neural only, the same final-model definition; the "
          "Phase I PDF instead reported PASS 0.9449, a diagonal average.",
          f"6. **Rule quality** = mean stored precision and mean stored coverage (recall of the rule's class on "
          f"the first validation half) of the {len(kb_rules)} ACTIVE rules of `continual_kb_v1_gain_clean.db`. "
          "Phase I: `continual_kb.db`. Stored precision is absolute precision, which class imbalance inflates; "
          "see results/fusion/kb_rule_stats_final.md for test lift and gain.",
          f"7. **Rule conflict handling** = `pytest tests/test_knowledge_base.py` exits 0 (last line: `{pytest_tail}`).",
          "8. **External generalisation** = NetFlow-v2 protocol (results/external_netflow/netflow_eval.md): train "
          "on NF-UNSW-NB15-v2, test on NF-ToN-IoT-v2 with the same 39 features; values copied from "
          "`netflow_eval.csv` (Domain NF-ToN, V4 stacking, 5 seeds). Detection rate = share of true attacks "
          "predicted as any attack class; Benign FPR = share of Benign rows predicted as an attack. Targets are "
          f"those of Tests 1-2. Context, XGBoost alone: FPR {pm(nf_xgb['FPR_Benign'])}, DR "
          f"{pm(nf_xgb['Detection_Rate'])}. Phase I: TON_IoT, 6 hand-mapped features, fused accuracy (CONDITIONAL, "
          "not counted).",
          "", "## Final Verdict"]
    p1_line = f"Phase I (results/final_test_report.md): {sum(s == 'PASS' for s, _ in p1.values())}/8 PASS."
    if benign is not None:
        as8 = sum(ok for t, ok in verdicts.items() if t not in ("1a", "1b")) + (verdicts["1a"] and verdicts["1b"])
        L.append(f"**Phase II v2: {sum(verdicts.values())}/9 verdict lines PASS** (Test 1 split into 1a static and 1b "
                 f"continual). Counted as 8 tests, with Test 1 passing only if both 1a and 1b pass: {as8}/8. " + p1_line)
    else:
        L.append(f"**Phase II: {sum(verdicts.values())}/8 PASS.** " + p1_line)
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase2", action="store_true",
                        help="score the final configuration and write results/final_test_report_phase2.md "
                             "(default: the Phase I scorecard, results/final_test_report.md)")
    parser.add_argument("--v2", action="store_true",
                        help="with --phase2: Test 1 scored by the Normal->attack rate (needs results/fusion/benign_fpr.csv); "
                             "writes results/final_test_report_phase2_v2.md and final_test_phase2_metrics_v2.csv")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # the report text has non-cp1252 characters (→, ±)
    if args.v2 and not args.phase2:
        parser.error("--v2 requires --phase2")
    if args.phase2:
        run_phase2_tests(v2=args.v2)
    else:
        run_final_tests()
