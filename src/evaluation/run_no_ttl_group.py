"""
TTL-group ablation: run_no_ttl.py with the whole TTL group (sttl, dttl, ct_state_ttl) dropped.

Data: data/processed (default, 42 features) vs data/processed_no_ttl_group
(python -m src.preprocessing.run_pipeline --drop-features sttl dttl ct_state_ttl --output-dir data/processed_no_ttl_group).

Static: 10-class XGBoost with the settings of results/models/xgboost_multi.joblib (XGBoostBaseline defaults,
random_state = seed), seeds 42, 1, 2, 3, 4, with and without the TTL group. Same metrics as run_no_ttl.py.

Continual: the final configuration rerun on data/processed_no_ttl_group: R1 detectors (replay_finetune.py
--buffer-size 10000 --class-weighted --val-memory-per-class 200 --val-memory-max-fraction 0.5
--processed-dir data/processed_no_ttl_group), KB continual_kb_v1_gain_clean_nottlgroup.db (continual_pipeline.py
--gain-check --kb-name continual_kb_v1_gain_clean_nottlgroup.db --processed-dir data/processed_no_ttl_group
--exclude-train-rows <the no-ttl-group validation memories>), V4 stacking and the detector-only control fit on
the validation memory. The split is rebuilt into results/ablations/split_indices_no_ttl_group.npz. With-ttl R1
rows are copied from results/fusion/detector_variants.csv. Also lists the active rules of the original final KB
(data/processed/continual_kb_v1_gain_clean.db) that use any TTL-group feature.

Writes results/ablations/no_ttl_group.csv, no_ttl_group_importances.csv and no_ttl_group.md.
Re-launches itself with PYTHONHASHSEED=0.
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
from src.models.baselines import XGBoostBaseline
from src.continual.cl_eval import find_project_root, set_seed, load_continual_data, predict_proba
from src.reasoning import fusion_variants as fv
from src.evaluation.evaluate_milestone2 import calculate_fpr
from src.evaluation.run_fusion_variants import metrics, pm, SEEDS, CAL_FRACTION, SPLIT_SEED
from src.evaluation.run_fusion_legal import CONTROL
from src.evaluation.run_fusion_final import meaningful_fired
from src.evaluation.run_no_ttl import load, R1_TAG, METHODS, TOP, KB_TTL

FEATURES = ["sttl", "dttl", "ct_state_ttl"]
WITH, WITHOUT = "with TTL group", "without TTL group"
DATASETS = {WITH: "processed", WITHOUT: "processed_no_ttl_group"}
KB_NOTTL = "continual_kb_v1_gain_clean_nottlgroup.db"
OUT = "no_ttl_group"


def fmt_rule(r):
    return " AND ".join(f"{c['feature']} {c['operator']} {c['threshold']:.4g}" for c in r["conditions"]) \
        + f" -> {r['predicted_class']}"


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_no_ttl_group", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    fusion_dir, res_dir = root / "results" / "fusion", root / "results" / "continual"
    ckpt_dir = res_dir / "replay_checkpoints"
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    nottl = root / "data" / DATASETS[WITHOUT]
    suffix = f"_{nottl.name}"
    for f in (f"{OUT}.csv", f"{OUT}_importances.csv", f"{OUT}.md", f"split_indices_{OUT}.npz"):
        if (out_dir / f).exists():
            print(f"ERROR: results/ablations/{f} already exists; not overwriting", file=sys.stderr)
            sys.exit(1)

    needed = [root / "data" / d / f for d in DATASETS.values() for f in ("pipeline.joblib", "X_train.npy", "X_test.npy")]
    needed += [root / "results" / "models" / "xgboost_multi.joblib", fusion_dir / "split_indices.npz",
               fusion_dir / "detector_variants.csv", nottl / KB_NOTTL, res_dir / f"kb_build_{KB_NOTTL[:-3]}.json",
               res_dir / f"continual_pipeline_metrics_{KB_NOTTL[:-3]}.csv", root / "data" / "processed" / KB_TTL]
    needed += [ckpt_dir / f"neural_ids_exp_4_{R1_TAG(s, suffix)}.pth" for s in SEEDS]
    needed += [ckpt_dir / f"valmem_{R1_TAG(s, suffix)}.npz" for s in SEEDS]
    needed += [ckpt_dir / f"valmem_{R1_TAG(s)}.npz" for s in SEEDS]
    missing = [str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    sets = {name: load(root, d) for name, d in DATASETS.items()}
    p_ttl, f_ttl, classes, Xtr_ttl, ytr, Xte_ttl, yte = sets[WITH]
    p_no, f_no, classes_no, Xtr_no, ytr_no, Xte_no, yte_no = sets[WITHOUT]
    j = [f_ttl.index(f) for f in FEATURES]
    data_check = {"features_dropped": f_no == [f for f in f_ttl if f not in FEATURES], "classes_equal": classes == classes_no,
                  "labels_equal": bool(np.array_equal(ytr, ytr_no) and np.array_equal(yte, yte_no)),
                  "X_equal_without_columns": bool(np.array_equal(np.delete(Xtr_ttl, j, 1), Xtr_no)
                                                  and np.array_equal(np.delete(Xte_ttl, j, 1), Xte_no))}
    print("Data check:", data_check)
    if not all(data_check.values()):
        print(f"ERROR: {nottl.name} is not data/processed without {FEATURES}", file=sys.stderr)
        sys.exit(1)
    K, labels = len(classes), list(range(len(classes)))

    # new split file built the same way as run_fusion_variants.py
    ev_old = np.load(fusion_dir / "split_indices.npz")["evaluation"]
    ev, cal = train_test_split(np.arange(len(yte_no)), test_size=CAL_FRACTION, stratify=yte_no, random_state=SPLIT_SEED)
    cal, ev = np.sort(cal), np.sort(ev)
    np.savez(out_dir / f"split_indices_{OUT}.npz", calibration=cal, evaluation=ev, seed=SPLIT_SEED)
    split_equal = bool(np.array_equal(ev, ev_old))
    print(f"Rebuilt split equals results/fusion/split_indices.npz: {split_equal}")
    y_ev = yte[ev]

    def basic(y, pred):
        rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
        return {"Accuracy": float(accuracy_score(y, pred)),
                "Macro_F1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
                "Macro_FPR": float(np.mean(list(calculate_fpr(y, pred, labels).values()))),
                **{f"Recall_{c}": float(r) for c, r in zip(classes, rec)}}

    # ---- static XGBoost ------------------------------------------------------------------------------------
    rows, importances = [], []
    saved = XGBoostBaseline.load(root / "results" / "models" / "xgboost_multi.joblib").model
    saved_pred = saved.classes_[saved.predict_proba(Xte_ttl).argmax(1)]
    xgb_check = None
    for name, (_, feats, _, Xtr, ytr_, Xte, _) in sets.items():
        for s in SEEDS:
            m = XGBoostBaseline(random_state=s).fit(Xtr, ytr_).model
            pred = m.classes_[m.predict_proba(Xte).argmax(1)]
            if name == WITH and s == 42:
                xgb_check = int((pred != saved_pred).sum())
                print(f"seed-42 XGBoost {WITH} vs xgboost_multi.joblib: {xgb_check} differing predictions")
            gain = m.get_booster().get_score(importance_type="gain")
            g = pd.Series({feats[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v for k, v in gain.items()})
            g = (g / g.sum()).sort_values(ascending=False)
            for rank, (feat, v) in enumerate(g.head(TOP).items(), start=1):
                importances.append({"Data": name, "Seed": s, "Rank": rank, "Feature": feat, "Gain_Share": float(v)})
            rows.append({"Part": "static", "Data": name, "Model": "XGBoost", "Seed": s, "Method": "Detector alone",
                         **basic(y_ev, pred[ev]), "Macro_F1_Full_Test": float(f1_score(yte, pred, labels=labels,
                                                                                       average="macro", zero_division=0))})
            print(f"static {name} seed {s}: macro-F1 {rows[-1]['Macro_F1']:.4f} (full test {rows[-1]['Macro_F1_Full_Test']:.4f})")
    if xgb_check != 0:
        print("ERROR: retrained seed-42 XGBoost does not reproduce xgboost_multi.joblib", file=sys.stderr)
        sys.exit(1)

    # ---- continual, without the TTL group ---------------------------------------------------------------------
    vm_equal = all(all(np.array_equal(np.load(ckpt_dir / f"valmem_{R1_TAG(s, suffix)}.npz")[k],
                                      np.load(ckpt_dir / f"valmem_{R1_TAG(s)}.npz")[k])
                       for k in ("exp", "local_index", "global_index", "y")) for s in SEEDS)
    union = np.unique(np.concatenate([np.load(ckpt_dir / f"valmem_{R1_TAG(s, suffix)}.npz")["global_index"] for s in SEEDS]))
    with open(res_dir / f"kb_build_{KB_NOTTL[:-3]}.json") as fh:
        log = json.load(fh)
    kb_ok = (log.get("n_excluded_train_rows") == len(union) and log.get("gain_check")
             and log.get("processed_dir") == str(nottl.relative_to(root)) and not log["mixed_validation"])
    print(f"No-ttl-group validation memories identical to the with-ttl R1 ones: {vm_equal}; KB build ok: {kb_ok}")
    if not (vm_equal and kb_ok):
        print("ERROR: validation memory or KB build of the no-ttl-group run is not as expected", file=sys.stderr)
        sys.exit(1)
    from src.rules.knowledge_base import KnowledgeBase
    kb_stats = {}
    for name, kb in [(WITH, root / "data" / "processed" / KB_TTL), (WITHOUT, nottl / KB_NOTTL)]:
        rules = KnowledgeBase(db_path=kb).get_active_rules()
        by_class = {}
        for r in rules:
            by_class[r["predicted_class"]] = by_class.get(r["predicted_class"], 0) + 1
        ttl_rules = [r for r in rules if any(c["feature"] in FEATURES for c in r["conditions"])]
        kb_stats[name] = {"active": len(rules), "by_class": by_class, "using_ttl": len(ttl_rules), "ttl_rules": ttl_rules,
                          "per_feature": {f: sum(any(c["feature"] == f for c in r["conditions"]) for r in rules)
                                          for f in FEATURES}}
    print(f"Active rules using {FEATURES}: " + ", ".join(f"{n} {k['using_ttl']}" for n, k in kb_stats.items()))
    pmet = pd.read_csv(res_dir / f"continual_pipeline_metrics_{KB_NOTTL[:-3]}.csv")
    kb_stats[WITHOUT]["flagged"] = [int(e) for e in pmet.loc[pmet["Drift_Detected"], "Experience"]]
    kb_stats[WITHOUT]["scores"] = [float(v) for v in pmet["Drift_Score"].dropna()]

    set_seed(42)
    data = load_continual_data(root, nottl)
    idx = data["class_to_idx"]
    col = {n: i for i, n in enumerate(f_no)}
    recent = np.unique(data["test_splits"][-1][1])
    S, F, M = (a[ev] for a in fv.cached_rule_evidence(nottl / KB_NOTTL, f_no, Xte_no, idx,
                                                      out_dir / "cache" / f"{KB_NOTTL[:-3]}_test.npz"))
    Fm = meaningful_fired(nottl / KB_NOTTL, Xte_no, yte_no, col, idx, K)[0][ev]
    rows_ev = np.arange(len(ev))
    for s in SEEDS:
        model = NeuralIDS(input_dim=data["input_dim"], output_dim=K).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_dir / f"neural_ids_exp_4_{R1_TAG(s, suffix)}.pth", map_location=DEVICE))
        vm = np.load(ckpt_dir / f"valmem_{R1_TAG(s, suffix)}.npz")
        X_vm, y_vm = Xtr_no[vm["global_index"]], vm["y"]
        P_vm, Pe = predict_proba(model, X_vm, DEVICE), predict_proba(model, Xte_no, DEVICE)[ev]
        det_ev = Pe.argmax(1)
        S_vm, F_vm, M_vm = fv.rule_evidence(nottl / KB_NOTTL, f_no, X_vm, idx)
        zeros_ev, zeros_vm = np.zeros((len(ev), K)), np.zeros((len(y_vm), K))
        ctrl = fv.StackingFuser(detector_only=True).fit(P_vm, zeros_vm, zeros_vm, zeros_vm, y_vm)
        stacker = fv.StackingFuser().fit(P_vm, S_vm, F_vm, M_vm, y_vm)
        for m, pred in [("Detector alone", det_ev), (CONTROL, ctrl.predict(Pe, zeros_ev, zeros_ev, zeros_ev)),
                        ("V4 stacking", stacker.predict(Pe, S, F, M))]:
            rows.append({"Part": "continual", "Data": WITHOUT, "Model": "R1", "Seed": s, "KB": KB_NOTTL,
                         "Method": m, **metrics(y_ev, pred, det_ev, F, labels, classes),
                         "Meaningful_Explanation_Rate": float(Fm[rows_ev, pred].mean()),
                         "Recency_Share": float(np.isin(pred, recent).mean())})
        print(f"continual no-ttl-group seed {s}: " + ", ".join(f"{r['Method'].split(':')[0]}={r['Macro_F1']:.4f}" for r in rows[-3:]))

    dv = pd.read_csv(fusion_dir / "detector_variants.csv")
    for _, r in dv[(dv["Source"] == "this step") & (dv["Variant"] == "R1")].iterrows():
        rows.append({**{k: r[k] for k in r.index if k not in ("Source", "Variant", "Detector", "True_Recency_Share")},
                     "Part": "continual (copied from detector_variants.csv)", "Data": WITH, "Model": "R1",
                     "Seed": int(r["Seed"]), "KB": KB_TTL})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{OUT}.csv", index=False)
    imp = pd.DataFrame(importances)
    imp.to_csv(out_dir / f"{OUT}_importances.csv", index=False)
    write_markdown(df, imp, data_check, split_equal, xgb_check, vm_equal, len(union), kb_stats, classes, len(ev),
                   len(f_no), out_dir / f"{OUT}.md")
    print(f"Saved {out_dir / f'{OUT}.csv'}, {OUT}_importances.csv and {OUT}.md")


def write_markdown(df, imp, data_check, split_equal, xgb_check, vm_equal, n_union, kb_stats, classes, n_ev, n_feat, path):
    names = list(DATASETS)
    feats = ", ".join(FEATURES)
    L = ["# TTL-group ablation (sttl, dttl, ct_state_ttl)", "",
         f"`data/{DATASETS[WITHOUT]}/` = `python -m src.preprocessing.run_pipeline --drop-features {' '.join(FEATURES)} "
         f"--output-dir data/{DATASETS[WITHOUT]}` ({n_feat} features). Checks: "
         + ", ".join(f"{k} {v}" for k, v in data_check.items())
         + f"; rebuilt split (stratified 80/20, seed 42) equals results/fusion/split_indices.npz: {split_equal}. "
         f"Metrics on the evaluation split ({n_ev} rows), mean ± sample std over seeds {SEEDS}.", "",
         "## Static 10-class XGBoost", "",
         "XGBoostBaseline defaults (the settings of results/models/xgboost_multi.joblib), random_state = seed. "
         f"Seed-42 model {WITH} vs xgboost_multi.joblib: {xgb_check} differing test predictions.", "",
         "| Data | Accuracy | Macro-F1 | Macro-F1 (full test set) | Macro-FPR |", "|---|---|---|---|---|"]
    st = df[df["Part"] == "static"]
    for n in names:
        x = st[st["Data"] == n]
        L.append(f"| {n} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | {pm(x['Macro_F1_Full_Test'])} | {pm(x['Macro_FPR'])} |")
    d = st[st["Data"] == names[1]].set_index("Seed")["Macro_F1"] - st[st["Data"] == names[0]].set_index("Seed")["Macro_F1"]
    L += ["", f"Δ macro-F1 (without - with), per seed {SEEDS}: " + ", ".join(f"{d[s]:+.4f}" for s in SEEDS)
          + f"; mean {d.mean():+.4f} ± {d.std(ddof=1):.4f}.", "",
          "### Per-class recall (mean ± std)", "", "| Data | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for n in names:
        x = st[st["Data"] == n]
        L.append(f"| {n} | " + " | ".join(pm(x[f"Recall_{c}"]) for c in classes) + " |")

    L += ["", "### Top-10 features by gain share (each model)", ""]
    for n in names:
        for s in SEEDS:
            x = imp[(imp["Data"] == n) & (imp["Seed"] == s)]
            L.append(f"- {n}, seed {s}: " + ", ".join(f"{r.Feature} {r.Gain_Share:.3f}" for r in x.itertuples()))

    ct = df[df["Part"].str.startswith("continual")]
    L += ["", "## Continual final configuration (R1 detectors + KB v1_gain_clean, V4 / control fit on the validation memory)", "",
          f"No-ttl-group validation memories identical to the with-ttl R1 ones: {vm_equal}; the no-ttl-group KB excluded "
          f"their union ({n_union} rows). With-ttl rows copied from results/fusion/detector_variants.csv.", "",
          f"| KB | Active rules | Rules using any of {feats} | " + " | ".join(classes) + " |",
          "|---|---|---|" + "---|" * len(classes)]
    for n in names:
        k = kb_stats[n]
        L.append(f"| {n} | {k['active']} | {k['using_ttl']} | " + " | ".join(str(k["by_class"].get(c, 0)) for c in classes) + " |")
    k0 = kb_stats[names[0]]
    L += ["", f"Active rules of the original final KB ({KB_TTL}) using a TTL-group feature: {k0['using_ttl']} of "
          f"{k0['active']} (per feature: " + ", ".join(f"{f} {v}" for f, v in k0["per_feature"].items())
          + "). Thresholds are in the preprocessed feature space."]
    if k0["ttl_rules"]:
        L += [""] + [f"- `{r['rule_id'][:8]}` {fmt_rule(r)} (precision {r['precision']:.3f}, coverage {r['coverage']:.3f})"
                     for r in sorted(k0["ttl_rules"], key=lambda r: (r["predicted_class"], fmt_rule(r)))]
    k = kb_stats[names[1]]
    L += ["", f"No-ttl-group KB build: flagged experiences {k['flagged']}, drift scores Exp 2-4 "
          + ", ".join(f"{v:.4f}" for v in k["scores"]) + ".", "",
          "| Data | Method | Accuracy | Macro-F1 | Macro-FPR | Recency share | Expl. | Meaningful expl. |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        for m in METHODS:
            x = ct[(ct["Data"] == n) & (ct["Method"] == m)]
            L.append(f"| {n} | {m} | {pm(x['Accuracy'])} | {pm(x['Macro_F1'])} | {pm(x['Macro_FPR'])} | "
                     f"{pm(x['Recency_Share'])} | {pm(x['Explanation_Rate'])} | {pm(x['Meaningful_Explanation_Rate'])} |")

    L += ["", "### Rule contribution: V4 - control (macro-F1), per seed", "",
          "| Data | " + " | ".join(str(s) for s in SEEDS) + " | mean ± std | seeds > 0 |", "|---|" + "---|" * (len(SEEDS) + 2)]
    contrib = {}
    for n in names:
        g = ct[ct["Data"] == n].set_index(["Method", "Seed"])["Macro_F1"]
        c = pd.Series({s: g[("V4 stacking", s)] - g[(CONTROL, s)] for s in SEEDS})
        contrib[n] = c
        L.append(f"| {n} | " + " | ".join(f"{c[s]:+.4f}" for s in SEEDS) + f" | {c.mean():+.4f} ± {c.std(ddof=1):.4f} | "
                 f"{int((c > 0).sum())}/5 |")

    L += ["", "### Per-class recall (mean over seeds)", "", "| Data | Method | " + " | ".join(classes) + " |",
          "|---|---|" + "---|" * len(classes)]
    for n in names:
        for m in METHODS:
            x = ct[(ct["Data"] == n) & (ct["Method"] == m)]
            L.append(f"| {n} | {m} | " + " | ".join(f"{x[f'Recall_{c}'].mean():.4f}" for c in classes) + " |")

    v4 = {n: ct[(ct["Data"] == n) & (ct["Method"] == "V4 stacking")]["Macro_F1"] for n in names}
    det = {n: ct[(ct["Data"] == n) & (ct["Method"] == "Detector alone")]["Macro_F1"] for n in names}
    L += ["", "## Summary", "",
          f"- Static XGBoost macro-F1: {pm(st[st['Data'] == names[0]]['Macro_F1'])} with, "
          f"{pm(st[st['Data'] == names[1]]['Macro_F1'])} without the TTL group (Δ {d.mean():+.4f}).",
          f"- Continual R1 detector alone: {pm(det[names[0]])} with, {pm(det[names[1]])} without; V4: {pm(v4[names[0]])} with, "
          f"{pm(v4[names[1]])} without.",
          f"- Rule contribution (V4 - control): {contrib[names[0]].mean():+.4f} with ({int((contrib[names[0]] > 0).sum())}/5 > 0), "
          f"{contrib[names[1]].mean():+.4f} without ({int((contrib[names[1]] > 0).sum())}/5 > 0).",
          f"- Original final KB: {k0['using_ttl']} of {k0['active']} active rules use a TTL-group feature; "
          f"the rebuilt KB: {kb_stats[names[1]]['using_ttl']} of {kb_stats[names[1]]['active']}."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
