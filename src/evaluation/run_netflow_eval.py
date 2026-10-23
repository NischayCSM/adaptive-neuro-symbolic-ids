"""
NetFlow-v2 cross-dataset evaluation: models trained on NF-UNSW-NB15-v2, tested in-domain and on NF-ToN-IoT-v2.

Inputs: data/processed_netflow/ (netflow_prepare.py) and the KB data/processed_netflow/kb_netflow_gain.db
(netflow_kb.py). Everything is trained on NF-UNSW train only; fusion is fitted on NF-UNSW validation only.

Models, per seed 42, 1, 2, 3, 4:
  XGBoost alone    10-class XGBoostBaseline defaults (the main project's settings), random_state = seed
  Control          StackingFuser(detector_only=True) on log P, fitted on NF-UNSW validation
  V2               neutral-absence fusion, alpha 0.5 (fusion_variants.v2_neutral; untuned)
  V4               StackingFuser on [log P, S, F, log1p M], fitted on NF-UNSW validation
The KB is seed-independent; only XGBoost and the stackers change with the seed.

Evaluation (C1) NF-UNSW test; (C2a) binary detection on all NF-ToN sample rows; (C2b) 4-class accuracy / macro-F1 on
the NF-ToN rows of mapped classes; (C2c) rule behaviour under shift; (C3) the old TON_IoT result, side by side.

Writes results/external_netflow/netflow_eval.csv (one row per seed x method x domain), netflow_rules.csv (one row
per active rule) and netflow_eval.md. Re-launches itself with PYTHONHASHSEED=0.
"""

import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.continual.cl_eval import find_project_root
from src.models.baselines import XGBoostBaseline
from src.reasoning import fusion_variants as fv
from src.evaluation.evaluate_milestone2 import calculate_fpr
from src.evaluation.run_fusion_variants import pm, SEEDS
from src.evaluation.run_fusion_legal import CONTROL
from src.evaluation.kb_rule_stats import rule_mask, fmt_conditions
from src.rules.knowledge_base import KnowledgeBase

ALPHA = 0.5
XGB, V2, V4 = "XGBoost alone", "V2 neutral-absence (alpha 0.5)", "V4 stacking"
METHODS = [XGB, CONTROL, V2, V4]
SMALL_TEST = 100            # classes with fewer NF-UNSW test rows are marked † everywhere
MEANINGFUL_MAX_MATCH = 0.5  # as in run_fusion_final.meaningful_fired
MIN_RULE_SUPPORT = 20       # NF-ToN mapped-row firings a rule needs to be ranked for degradation
OLD = {"Accuracy": 0.2366, "Macro_F1": 0.0957}  # results/external_validation.md (XGBoost, "Neural Only")
SENS_DROP = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]  # corrupted byte-rate fields (data_card.md)
SENS_SEED = 42
SENS_METRICS = ["Detection_Rate", "FPR_Benign", "DR_Seen_Attacks", "DR_Unseen", "Mapped_Accuracy", "Mapped_Macro_F1"]


def meaningful(masks, rule_k, y, K):
    """N x K: a rule for class k fires AND has positive gain on (X, y) AND matches <= 0.5 of the rows
    (run_fusion_final.meaningful_fired on precomputed masks; y = -1 rows count as wrong for every rule)."""
    base = np.array([(y == k).mean() for k in range(K)])
    out = np.zeros((masks.shape[1], K), dtype=bool)
    for m, k in zip(masks, rule_k):
        if m.any() and base[k] < 1:
            gain = ((y[m] == k).mean() - base[k]) / (1 - base[k])
            if gain > 0 and m.mean() <= MEANINGFUL_MAX_MATCH:
                out[:, k] |= m
    return out


def rules_alone(S, F):
    """Class of the most precise fired rule (lowest index on ties); -1 where no rule fires."""
    return np.where(F.any(1), S.argmax(1), -1)


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_netflow_eval", *sys.argv[1:]],
                                cwd=root, env=env).returncode)
    data_dir, out = root / "data" / "processed_netflow", root / "results" / "external_netflow"
    kb_path = data_dir / "kb_netflow_gain.db"
    for f in ("netflow_eval.csv", "netflow_rules.csv", "netflow_sensitivity.csv", "netflow_eval.md"):
        if (out / f).exists():
            sys.exit(f"ERROR: results/external_netflow/{f} already exists; not overwriting")
    if not kb_path.exists():
        sys.exit("ERROR: data/processed_netflow/kb_netflow_gain.db not found; run src.evaluation.netflow_kb first")
    rd = lambda n: json.loads((data_dir / n).read_text())
    features, classes, ton_classes, mapping = rd("feature_names.json"), rd("label_classes.json"), \
        rd("ton_classes.json"), rd("label_mapping.json")
    counts, kb_log = json.loads((out / "data_counts.json").read_text()), json.loads((out / "kb_build_netflow.json").read_text())
    X = {k: np.load(data_dir / f"X_{k}.npy") for k in ("train", "val", "test", "ton")}
    y_tr, y_va, y_te = (np.load(data_dir / f"y_{k}.npy") for k in ("train", "val", "test"))
    y_ton, y_ton_label = np.load(data_dir / "y_ton.npy"), np.load(data_dir / "y_ton_label.npy")
    K, labels = len(classes), list(range(len(classes)))
    cidx = {c: i for i, c in enumerate(classes)}
    ben = cidx[next(v for k, v in mapping.items() if k.lower() == "benign")]
    ton_name = np.array(ton_classes)[y_ton]
    mapped = np.isin(ton_name, list(mapping))
    unseen_classes = [c for c in ton_classes if c not in mapping]
    y_u = np.full(len(y_ton), -1)  # NF-ToN labels in NF-UNSW class space; -1 = unseen attack
    y_u[mapped] = [cidx[mapping[c]] for c in ton_name[mapped]]
    mapped_k = sorted(cidx[v] for v in mapping.values())
    attack = y_ton_label == 1
    assert not attack[y_u == ben].any() and attack[y_u != ben].all(), "NF-ToN Label inconsistent with mapping"
    small = [c for c in classes if (y_te == cidx[c]).sum() < SMALL_TEST]
    mark = lambda c: f"{c} †" if c in small else c
    big = [cidx[c] for c in classes if c not in small]
    raw_ton = counts["ton_raw_counts"]

    # ---- rule evidence (seed-independent) -------------------------------------------------------------------
    rules = KnowledgeBase(db_path=kb_path).get_active_rules()
    rule_k = np.array([cidx[r["predicted_class"]] for r in rules])
    col = {f: i for i, f in enumerate(features)}
    ev, masks = {}, {}
    for k in ("val", "test", "ton"):
        ev[k] = fv.cached_rule_evidence(kb_path, features, X[k], cidx, out / "cache" / f"evidence_{k}.npz")
        masks[k] = np.array([rule_mask(r, X[k], col) for r in rules]).reshape(len(rules), len(X[k]))
        Fk = np.zeros_like(ev[k][1])
        for m, c in zip(masks[k], rule_k):
            Fk[:, c] |= m
        assert np.array_equal(Fk, ev[k][1]), f"rule masks disagree with rule_evidence on {k}"
    Fm = {"test": meaningful(masks["test"], rule_k, y_te, K), "ton": meaningful(masks["ton"], rule_k, y_u, K)}
    print(f"{len(rules)} active rules; any rule fires on {ev['test'][1].any(1).mean():.3f} of NF-UNSW test, "
          f"{ev['ton'][1].any(1).mean():.3f} of NF-ToN")

    # ---- models and metrics per seed ------------------------------------------------------------------------
    rows, unseen_pred = [], {m: np.zeros((len(ton_classes), K)) for m in (XGB, V4)}
    for s in SEEDS:
        model = XGBoostBaseline(random_state=s).fit(X["train"], y_tr).model
        assert list(model.classes_) == labels
        P = {k: model.predict_proba(X[k]) for k in ("val", "test", "ton")}
        zv = np.zeros((len(y_va), K))
        ctrl = fv.StackingFuser(detector_only=True).fit(P["val"], zv, zv, zv, y_va)
        v4 = fv.StackingFuser().fit(P["val"], *ev["val"], y_va)
        for dom, y in (("NF-UNSW test", y_te), ("NF-ToN", y_u)):
            k = "test" if dom == "NF-UNSW test" else "ton"
            S, F, M = ev[k]
            z = np.zeros((len(y), K))
            preds = {XGB: P[k].argmax(1), CONTROL: ctrl.predict(P[k], z, z, z), V2: fv.v2_neutral(P[k], S, F, ALPHA),
                     V4: v4.predict(P[k], S, F, M)}
            ar = np.arange(len(y))
            for meth, pred in preds.items():
                r = {"Seed": s, "Method": meth, "Domain": dom, "Explanation_Rate": float(F[ar, pred].mean()),
                     "Meaningful_Explanation_Rate": float(Fm[k][ar, pred].mean())}
                if k == "test":
                    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
                    r.update({"Accuracy": accuracy_score(y, pred),
                              "Macro_F1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
                              "Macro_F1_Without_Small": f1_score(y, pred, labels=big, average="macro", zero_division=0),
                              "Macro_FPR": float(np.mean(list(calculate_fpr(y, pred, labels).values()))),
                              **{f"Recall_{c}": float(v) for c, v in zip(classes, rec)}})
                else:
                    det = pred != ben
                    dr_c = {c: float(det[ton_name == c].mean()) for c in ton_classes}
                    att = [c for c in ton_classes if c.lower() != "benign"]
                    w = lambda cs: sum(raw_ton[c] * dr_c[c] for c in cs) / sum(raw_ton[c] for c in cs)
                    rec = recall_score(y[mapped], pred[mapped], labels=mapped_k, average=None, zero_division=0)
                    r.update({"Detection_Rate": float(det[attack].mean()), "FPR_Benign": float(det[~attack].mean()),
                              "DR_Seen_Attacks": float(det[mapped & attack].mean()), "DR_Unseen": float(det[~mapped].mean()),
                              "DR_Weighted": w(att), "DR_Unseen_Weighted": w(unseen_classes),
                              **{f"DR_{c}": v for c, v in dr_c.items()},
                              "Mapped_Accuracy": accuracy_score(y[mapped], pred[mapped]),
                              "Mapped_Macro_F1": f1_score(y[mapped], pred[mapped], labels=mapped_k, average="macro",
                                                          zero_division=0),
                              **{f"Mapped_Recall_{classes[c]}": float(v) for c, v in zip(mapped_k, rec)},
                              "All_Rows_Accuracy": float((pred == y).mean())})
                    if meth in unseen_pred:
                        for i, c in enumerate(ton_classes):
                            unseen_pred[meth][i] += np.bincount(pred[ton_name == c], minlength=K) / (ton_name == c).sum() / len(SEEDS)
                    for c in mapped_k:  # precision of predicting class c on mapped rows / on NF-UNSW test
                        sel = pred[mapped] == c
                        r[f"Mapped_Precision_{classes[c]}"] = float((y[mapped][sel] == c).mean()) if sel.any() else np.nan
                rows.append(r)
            if k == "test":
                for c in mapped_k:
                    sel = preds[XGB] == c
                    rows[-4][f"Test_Precision_{classes[c]}"] = float((y_te[sel] == c).mean()) if sel.any() else np.nan
            print(f"seed {s} {dom}: " + ", ".join(
                f"{r['Method'].split(' ')[0]} {r.get('Macro_F1', r.get('Mapped_Macro_F1')):.4f}" for r in rows[-4:]), flush=True)
    df = pd.DataFrame(rows)

    # ---- sensitivity: XGBoost seed 42 without the corrupted byte-rate features ----------------------------
    # Preprocessing is per feature, so dropping the columns from the processed arrays equals preprocessing without them.
    keep = [i for i, f in enumerate(features) if f not in SENS_DROP]
    assert len(keep) == len(features) - len(SENS_DROP)
    m_d = XGBoostBaseline(random_state=SENS_SEED).fit(X["train"][:, keep], y_tr).model
    sens = []
    for name, p_te, p_ton in (
            ("All 39 features (main result)", df[(df["Seed"] == SENS_SEED) & (df["Method"] == XGB)], None),
            (f"{len(keep)} features, {' and '.join(SENS_DROP)} dropped", m_d.predict(X["test"][:, keep]),
             m_d.predict(X["ton"][:, keep]))):
        if p_ton is None:
            r_t, r_o = p_te[p_te["Domain"] == "NF-UNSW test"].iloc[0], p_te[p_te["Domain"] == "NF-ToN"].iloc[0]
            sens.append({"Features": name, **{k: float(r_o[k]) for k in SENS_METRICS},
                         "NF-UNSW_Test_Macro_F1": float(r_t["Macro_F1"])})
            continue
        det = p_ton != ben
        sens.append({"Features": name, "Detection_Rate": float(det[attack].mean()), "FPR_Benign": float(det[~attack].mean()),
                     "DR_Seen_Attacks": float(det[mapped & attack].mean()), "DR_Unseen": float(det[~mapped].mean()),
                     "Mapped_Accuracy": accuracy_score(y_u[mapped], p_ton[mapped]),
                     "Mapped_Macro_F1": f1_score(y_u[mapped], p_ton[mapped], labels=mapped_k, average="macro", zero_division=0),
                     "NF-UNSW_Test_Macro_F1": f1_score(y_te, p_te, labels=labels, average="macro", zero_division=0)})
    sdf = pd.DataFrame(sens)
    print("sensitivity (XGBoost seed 42):\n" + sdf.to_string(index=False), flush=True)

    # ---- per-rule behaviour -------------------------------------------------------------------------------
    rrows = []
    for r, k, mv, mt, mo in zip(rules, rule_k, masks["val"], masks["test"], masks["ton"]):
        mm = mo & mapped
        rrows.append({"Rule_ID": r["rule_id"], "Class": classes[k], "Has_NF_ToN_Counterpart": classes[k] in mapping.values(),
                      "N_Conditions": len(r["conditions"]), "Conditions": fmt_conditions(r),
                      "Stored_Precision_Val1": r["precision"], "Val_Support": int(mv.sum()),
                      "Val_Precision": float((y_va[mv] == k).mean()) if mv.any() else np.nan,
                      "Test_Support": int(mt.sum()), "Test_Precision": float((y_te[mt] == k).mean()) if mt.any() else np.nan,
                      "ToN_Support": int(mo.sum()), "ToN_Mapped_Support": int(mm.sum()),
                      "ToN_Mapped_Precision": float((y_u[mm] == k).mean()) if mm.any() else np.nan,
                      "ToN_Attack_Level_Precision": float(((~attack) if k == ben else attack)[mo].mean()) if mo.any() else np.nan})
    rdf = pd.DataFrame(rrows)
    rdf["Delta_Precision"] = rdf["ToN_Mapped_Precision"] - rdf["Val_Precision"]

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "netflow_eval.csv", index=False)
    rdf.to_csv(out / "netflow_rules.csv", index=False)
    sdf.to_csv(out / "netflow_sensitivity.csv", index=False)
    ctx = dict(sdf=sdf,classes=classes, ton_classes=ton_classes, mapping=mapping, unseen=unseen_classes, small=small, mark=mark,
               kb_log=kb_log, counts=counts, ev=ev, masks=masks, rule_k=rule_k, y_te=y_te, y_u=y_u, y_va=y_va,
               mapped=mapped, ton_name=ton_name, ben=ben, mapped_k=mapped_k, unseen_pred=unseen_pred,
               n_test=len(y_te), n_ton=len(y_u), n_rules=len(rules))
    write_markdown(df, rdf, ctx, out / "netflow_eval.md")
    print("Saved results/external_netflow/netflow_eval.csv, netflow_rules.csv and netflow_eval.md")


def write_markdown(df, rdf, c, path):
    classes, mark, ton_classes, mapping = c["classes"], c["mark"], c["ton_classes"], c["mapping"]
    t, o = df[df["Domain"] == "NF-UNSW test"], df[df["Domain"] == "NF-ToN"]
    g = lambda d, m, col: d[d["Method"] == m][col]
    small_note = (f"† = fewer than {SMALL_TEST} NF-UNSW test rows ({', '.join(c['small'])}: "
                  f"{', '.join(str(int((c['y_te'] == classes.index(s)).sum())) for s in c['small'])}); shown for "
                  "completeness, no conclusion rests on these classes alone.") if c["small"] else ""
    unseen_mark = lambda n: f"{n} *" if n in c["unseen"] else n
    L = ["# NetFlow-v2 cross-dataset evaluation (NF-UNSW-NB15-v2 → NF-ToN-IoT-v2)", "",
         "Trained on NF-UNSW train only; Control and V4 fitted on NF-UNSW validation only; V2 untuned (alpha 0.5). "
         f"Mean ± sample std over seeds {SEEDS} (XGBoost and stacker seeds; the KB is fixed). Data, sampling, features "
         "and label mapping: `data_card.md`.", "",
         "> The 50k-per-class cap rebalances both datasets (Benign ≈ 96% of raw NF-UNSW-NB15-v2, ≈ 34% of the sample), "
         "so accuracy, FPR and pooled rates below are comparable **within this experiment only**, not with "
         "prevalence-weighted results in the literature. Per-class rates do not depend on the cap.", ""]
    if small_note:
        L += [small_note, ""]

    # ---- KB
    kl = c["kb_log"]
    L += ["## Knowledge base (`data/processed_netflow/kb_netflow_gain.db`)", "",
          f"Four generators on NF-UNSW train ({kl['n_train']:,} rows) → {kl['candidates']} candidates "
          f"({', '.join(f'{k} {v}' for k, v in kl['candidates_per_generator'].items())}) → {kl['merged']} after merging → "
          f"validator with gain check (min_gain {kl['min_gain']}, max_match_fraction {kl['max_match_fraction']}) on the "
          f"NF-UNSW validation halves ({kl['val1_size']:,} / {kl['val2_size']:,} rows, min support "
          f"{kl['effective_min_support']}): **{kl['accepted']} rules accepted** ({kl['gain_rejected']} rejected by the gain "
          f"check, {kl['conflicts']} conflicts). Accepted per generator: "
          + ", ".join(f"{k} {v}" for k, v in kl["accepted_per_generator"].items()) + ".", "",
          "| " + " | ".join(mark(x) for x in classes) + " | Total |", "|" + "---:|" * (len(classes) + 1),
          "| " + " | ".join(str(kl["active_per_class"][x]) for x in classes) + f" | {kl['accepted']} |", ""]

    # ---- C1
    L += [f"## C1. In-domain: NF-UNSW test ({c['n_test']:,} rows)", "",
          "| Method | Accuracy | Macro-F1 | Macro-FPR | Expl. | Meaningful expl. |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | {pm(g(t, m, 'Accuracy'))} | {pm(g(t, m, 'Macro_F1'))} | {pm(g(t, m, 'Macro_FPR'))} | "
                 f"{pm(g(t, m, 'Explanation_Rate'))} | {pm(g(t, m, 'Meaningful_Explanation_Rate'))} |")
    L += ["", "Per-class recall (mean ± std):", "", "| Method | " + " | ".join(mark(x) for x in classes) + " |",
          "|---|" + "---|" * len(classes)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(pm(g(t, m, f"Recall_{x}")) for x in classes) + " |")
    L += ["", "Macro-F1 differences per seed (NF-UNSW test):", "",
          "| Difference | " + " | ".join(str(s) for s in SEEDS) + " | mean ± std | seeds > 0 |", "|---|" + "---|" * (len(SEEDS) + 2)]
    contrib = {}
    for a, b in ((V4, CONTROL), (V4, XGB), (V2, XGB), (CONTROL, XGB)):
        d = g(t, a, "Macro_F1").to_numpy() - g(t, b, "Macro_F1").to_numpy()
        contrib[(a, b)] = d
        L.append(f"| {a.split(' (')[0]} − {b.split(':')[0]} | " + " | ".join(f"{v:+.4f}" for v in d)
                 + f" | {d.mean():+.4f} ± {d.std(ddof=1):.4f} | {(d > 0).sum()}/{len(d)} |")
    if c["small"]:
        d = g(t, V4, "Macro_F1_Without_Small").to_numpy() - g(t, CONTROL, "Macro_F1_Without_Small").to_numpy()
        L += ["", f"Macro-F1 over the classes without † (V4 {pm(g(t, V4, 'Macro_F1_Without_Small'))}, control "
              f"{pm(g(t, CONTROL, 'Macro_F1_Without_Small'))}): V4 − control per seed "
              + ", ".join(f"{v:+.4f}" for v in d) + f"; mean {d.mean():+.4f} ± {d.std(ddof=1):.4f}, "
              f"{(d > 0).sum()}/{len(d)} seeds > 0."]
    L.append("")

    # ---- C2a
    L += [f"## C2a. Binary detection on NF-ToN (all {c['n_ton']:,} sampled rows, unseen classes included)", "",
          "\"Detected\" = predicted any NF-UNSW attack class (not Benign). Detection rate = recall over true attacks; "
          "FPR = share of Benign rows predicted as an attack. *Weighted* rates re-weight the per-class rates by the raw "
          "NF-ToN class frequencies (undoing the cap).", "",
          "| Method | Detection rate | FPR (Benign) | DR seen attacks (dos, backdoor, scanning) | DR unseen attacks | "
          "DR weighted | DR unseen weighted |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | {pm(g(o, m, 'Detection_Rate'))} | {pm(g(o, m, 'FPR_Benign'))} | {pm(g(o, m, 'DR_Seen_Attacks'))} | "
                 f"{pm(g(o, m, 'DR_Unseen'))} | {pm(g(o, m, 'DR_Weighted'))} | {pm(g(o, m, 'DR_Unseen_Weighted'))} |")
    L += ["", "Share predicted as an attack, per NF-ToN class (mean over seeds; for Benign this is the FPR; * = unseen class):", "",
          "| Method | " + " | ".join(unseen_mark(x) for x in ton_classes) + " |", "|---|" + "---|" * len(ton_classes)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(f"{g(o, m, f'DR_{x}').mean():.4f}" for x in ton_classes) + " |")
    L += ["", f"NF-ToN rows per class in the sample: " + ", ".join(f"{x} {c['counts']['ton_sample_counts'][x]:,}" for x in ton_classes) + ".", ""]
    for m in (XGB, V4):
        L += [f"Predicted NF-UNSW class per NF-ToN class, {m} (share of rows, mean over seeds):", "",
              "| NF-ToN class | " + " | ".join(mark(x) for x in classes) + " |", "|---|" + "---|" * len(classes)]
        for i, x in enumerate(ton_classes):
            L.append(f"| {unseen_mark(x)} | " + " | ".join(f"{v:.3f}" for v in c["unseen_pred"][m][i]) + " |")
        L.append("")
    ovf = c["counts"]["float32_overflow_replaced"]["per_split"]["NF-ToN"]
    L += [f"### Sensitivity: corrupted byte-rate features (XGBoost alone, seed {SENS_SEED})", "",
          f"Main result: {' and '.join(f'`{f}`' for f in SENS_DROP)} kept, with the {ovf['values_replaced']} NF-ToN values "
          f"({ovf['rows_replaced']} rows) whose standardised value would overflow float32 median-imputed (`data_card.md`). "
          "Alternative: both features dropped (XGBoost retrained on the remaining features, same seed).", "",
          "| Features | Detection rate | FPR (Benign) | DR seen attacks | DR unseen attacks | Mapped accuracy | "
          "Mapped macro-F1 | NF-UNSW test macro-F1 |", "|---|---|---|---|---|---|---|---|"]
    for r in c["sdf"].itertuples(index=False):
        L.append(f"| {r[0]} | " + " | ".join(f"{v:.4f}" for v in r[1:]) + " |")
    d = c["sdf"].iloc[1, 1:] - c["sdf"].iloc[0, 1:]
    L += ["", "Dropped − kept: " + ", ".join(f"{k.replace('_', ' ')} {v:+.4f}" for k, v in d.items()) + ".", ""]

    # ---- C2b
    mk = [classes[k] for k in c["mapped_k"]]
    L += [f"## C2b. Multi-class on mapped NF-ToN classes only ({int(c['mapped'].sum()):,} rows: "
          + ", ".join(f"{a} → {b}" for a, b in mapping.items()) + ")", "",
          "Predictions may be any of the 10 NF-UNSW classes; a prediction outside the four mapped classes is an error. "
          "Macro-F1 over the four mapped classes.", "",
          "| Method | Accuracy | Macro-F1 (4 classes) | " + " | ".join(f"Recall {x}" for x in mk) + " |",
          "|---|---|---|" + "---|" * len(mk)]
    for m in METHODS:
        L.append(f"| {m} | {pm(g(o, m, 'Mapped_Accuracy'))} | {pm(g(o, m, 'Mapped_Macro_F1'))} | "
                 + " | ".join(pm(g(o, m, f"Mapped_Recall_{x}")) for x in mk) + " |")
    L.append("")

    # ---- C2c
    S_t, F_t, _ = c["ev"]["test"]
    S_o, F_o, _ = c["ev"]["ton"]
    L += ["## C2c. Rule behaviour under shift", "",
          f"Any rule fires on **{F_t.any(1).mean():.4f}** of NF-UNSW test rows and **{F_o.any(1).mean():.4f}** of NF-ToN rows. "
          "Per NF-ToN class: " + ", ".join(f"{x} {F_o[c['ton_name'] == x].any(1).mean():.3f}" for x in ton_classes) + ".", "",
          "Explanation rate = share of predictions with a fired rule for the predicted class; meaningful = that rule also "
          f"has positive gain and matches ≤ {MEANINGFUL_MAX_MATCH} of the rows of the evaluated set (as in "
          "run_fusion_final.py; on NF-ToN, rows of unseen classes count as wrong for every rule).", "",
          "| Method | Expl. NF-UNSW test | Expl. NF-ToN | Meaningful NF-UNSW test | Meaningful NF-ToN |", "|---|---|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | {pm(g(t, m, 'Explanation_Rate'))} | {pm(g(o, m, 'Explanation_Rate'))} | "
                 f"{pm(g(t, m, 'Meaningful_Explanation_Rate'))} | {pm(g(o, m, 'Meaningful_Explanation_Rate'))} |")

    fired = rdf[rdf["ToN_Mapped_Support"] > 0]
    agg = lambda d, sup, prec: (d[sup] * d[prec]).sum() / d[sup].sum() if d[sup].sum() else np.nan
    cp = rdf[rdf["Has_NF_ToN_Counterpart"]]
    L += ["", f"Rule precision: validation = full NF-UNSW validation split; NF-ToN = NF-ToN rows of the four mapped classes "
          "(a rule for a class with no NF-ToN counterpart is always wrong there). Support-weighted means over rules.", "",
          "| Rules | Count | Fire on NF-ToN (any row) | Fire on mapped rows | Precision val | Precision NF-UNSW test | "
          "Precision NF-ToN mapped | Attack-level precision NF-ToN (all rows) |", "|---|---|---|---|---|---|---|---|"]
    for name, d in (("All rules", rdf), ("Rules for mapped classes (Benign, DoS, Backdoor, Recon)", cp),
                    ("Rules for classes without counterpart", rdf[~rdf["Has_NF_ToN_Counterpart"]])):
        L.append(f"| {name} | {len(d)} | {(d['ToN_Support'] > 0).sum()} | {(d['ToN_Mapped_Support'] > 0).sum()} | "
                 f"{agg(d, 'Val_Support', 'Val_Precision'):.4f} | {agg(d, 'Test_Support', 'Test_Precision'):.4f} | "
                 f"{agg(d, 'ToN_Mapped_Support', 'ToN_Mapped_Precision'):.4f} | "
                 f"{agg(d, 'ToN_Support', 'ToN_Attack_Level_Precision'):.4f} |")

    # rules vs XGBoost, per mapped class
    L += ["", "Rules vs XGBoost, per mapped class: precision of \"predict class k\" on NF-UNSW (rules: validation, "
          "XGBoost: test, mean over seeds) and on NF-ToN mapped rows. Rules = support-weighted over that class's rules.", "",
          "| Class | Rules: n | Rules prec. NF-UNSW val | Rules prec. NF-ToN | Rules Δ | XGBoost prec. NF-UNSW test | "
          "XGBoost prec. NF-ToN | XGBoost Δ |", "|---|---|---|---|---|---|---|---|"]
    for k in c["mapped_k"]:
        x = classes[k]
        d = rdf[rdf["Class"] == x]
        rv, ro = agg(d, "Val_Support", "Val_Precision"), agg(d, "ToN_Mapped_Support", "ToN_Mapped_Precision")
        xv = t[t["Method"] == XGB][f"Test_Precision_{x}"].mean()
        xo = o[o["Method"] == XGB][f"Mapped_Precision_{x}"].mean()
        L.append(f"| {mark(x)} | {len(d)} | {rv:.4f} | {ro:.4f} | {ro - rv:+.4f} | {xv:.4f} | {xo:.4f} | {xo - xv:+.4f} |")

    # rules-alone classifier vs XGBoost on the same rows
    ra_t, ra_o = rules_alone(S_t, F_t), rules_alone(S_o, F_o)
    cov_t, cov_o = ra_t >= 0, (ra_o >= 0) & c["mapped"]
    xa_t = t[t["Method"] == XGB]["Accuracy"].mean()
    xa_o = o[o["Method"] == XGB]["Mapped_Accuracy"].mean()
    L += ["", "Rules alone (class of the most precise fired rule; abstains when none fires) vs XGBoost:", "",
          "| Domain | Rules coverage | Rules accuracy on covered rows | XGBoost accuracy (all rows of the domain) |",
          "|---|---|---|---|",
          f"| NF-UNSW test | {cov_t.mean():.4f} | {(ra_t[cov_t] == c['y_te'][cov_t]).mean():.4f} | {xa_t:.4f} |",
          f"| NF-ToN mapped rows | {cov_o[c['mapped']].mean():.4f} | {(ra_o[cov_o] == c['y_u'][cov_o]).mean():.4f} | {xa_o:.4f} |"]

    deg = rdf[rdf["ToN_Mapped_Support"] >= MIN_RULE_SUPPORT].sort_values(["Delta_Precision", "ToN_Mapped_Support"],
                                                                        ascending=[True, False]).head(10)
    L += ["", f"### 10 rules that degrade most (validation precision → NF-ToN mapped-row precision; rules with ≥ "
          f"{MIN_RULE_SUPPORT} mapped-row firings: {(rdf['ToN_Mapped_Support'] >= MIN_RULE_SUPPORT).sum()} of {len(rdf)})", "",
          "| Rule | Class | Conditions (standardised features) | Val prec. (support) | NF-ToN prec. (mapped support) | Δ | "
          "NF-ToN attack-level prec. |", "|---|---|---|---|---|---|---|"]
    for r in deg.itertuples():
        L.append(f"| `{r.Rule_ID[:8]}` | {mark(r.Class)} | {r.Conditions} | {r.Val_Precision:.3f} ({r.Val_Support}) | "
                 f"{r.ToN_Mapped_Precision:.3f} ({r.ToN_Mapped_Support}) | {r.Delta_Precision:+.3f} | "
                 f"{r.ToN_Attack_Level_Precision:.3f} |")

    # ---- C3
    x_acc, x_f1, x_all = (o[o["Method"] == XGB][k] for k in ("Mapped_Accuracy", "Mapped_Macro_F1", "All_Rows_Accuracy"))
    L += ["", "## C3. Comparison with the old TON_IoT result (different, confounded protocol)", "",
          "| Protocol | Train → test | Features | Test rows | Classes scored | Accuracy | Macro-F1 |", "|---|---|---|---|---|---|---|",
          f"| Old (`results/external_validation.md`), XGBoost | UNSW-NB15 (original) → TON_IoT network | 6 hand-mapped "
          f"columns, the rest zero-filled / `<UNKNOWN>` | 25,000 | all TON_IoT labels, unseen ones included | "
          f"{OLD['Accuracy']:.4f} | {OLD['Macro_F1']:.4f} |",
          f"| New, XGBoost, all NF-ToN rows | NF-UNSW-NB15-v2 → NF-ToN-IoT-v2 | the same 39 NetFlow features | "
          f"{c['n_ton']:,} (capped sample) | 10 NF-UNSW classes; unseen-class rows always wrong | {pm(x_all)} | n/a |",
          f"| New, XGBoost, mapped classes only | NF-UNSW-NB15-v2 → NF-ToN-IoT-v2 | the same 39 NetFlow features | "
          f"{int(c['mapped'].sum()):,} | Benign, DoS, Backdoor, Reconnaissance | {pm(x_acc)} | {pm(x_f1)} |", "",
          "The old number is **not directly comparable**: different datasets (original UNSW-NB15 / TON_IoT vs their "
          "NetFlow-v2 versions), a feature mismatch handled by hand-mapping and zero-filling, a different class mix and "
          "scoring over unseen labels. The new protocol removes the feature-mismatch confound but adds the 50k-per-class "
          "rebalancing.", ""]

    # ---- summary
    v4c = contrib[(V4, CONTROL)]
    L += ["## Summary", "",
          f"- In-domain macro-F1: XGBoost {pm(g(t, XGB, 'Macro_F1'))}, control {pm(g(t, CONTROL, 'Macro_F1'))}, "
          f"V2 {pm(g(t, V2, 'Macro_F1'))}, V4 {pm(g(t, V4, 'Macro_F1'))}; V4 − control {v4c.mean():+.4f} ± "
          f"{v4c.std(ddof=1):.4f} ({(v4c > 0).sum()}/{len(v4c)} seeds > 0).",
          f"- NF-ToN binary: XGBoost detection rate {pm(g(o, XGB, 'Detection_Rate'))} at Benign FPR "
          f"{pm(g(o, XGB, 'FPR_Benign'))}; unseen attacks detected as some attack {pm(g(o, XGB, 'DR_Unseen'))} "
          f"(V4 {pm(g(o, V4, 'DR_Unseen'))}).",
          f"- NF-ToN mapped classes: XGBoost accuracy {pm(x_acc)}, macro-F1 {pm(x_f1)}; V4 accuracy "
          f"{pm(g(o, V4, 'Mapped_Accuracy'))}, macro-F1 {pm(g(o, V4, 'Mapped_Macro_F1'))}.",
          f"- Rules fire on {F_o.any(1).mean():.1%} of NF-ToN rows (NF-UNSW test {F_t.any(1).mean():.1%}); support-weighted "
          f"rule precision {agg(rdf, 'Val_Support', 'Val_Precision'):.3f} on NF-UNSW validation vs "
          f"{agg(rdf, 'ToN_Mapped_Support', 'ToN_Mapped_Precision'):.3f} on NF-ToN mapped rows."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
