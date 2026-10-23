"""
Rule statistics for the continual KBs (old v1 / v2 / v3 and the lift-checked v1 / v2).

Per KB:
    active rules per class after each experience (from results/continual/kb_build_<kb>.json:
        rules added up to that experience minus rules retired up to it; checked against the KB's
        final ACTIVE rules)
    on the full test set, over the final ACTIVE rules (same matching as SymbolicReasoner):
        match fraction = fraction of test rows a rule fires on (mean / median)
        number of rules firing on > 50% of test rows
        lift = test precision / test base rate of the rule's class (mean over rules that fire)
    the 5 broadest rules (highest test match fraction)

Writes results/fusion/kb_rule_stats.csv (one row per KB) and results/fusion/kb_rule_stats.md.

Options (defaults reproduce the run above): --kbs NAME=FILE ... (other KBs), --out-name NAME (output
results/fusion/NAME.csv / .md), --gain (also report test gain = (precision - base rate) / (1 - base rate)
per rule: mean over rules that fire, and a column in the broadest-rules tables).
"""

import argparse

import json

import numpy as np
import pandas as pd

from src.continual.cl_eval import find_project_root, load_continual_data
from src.reasoning.reasoner import evaluate_condition
from src.rules.knowledge_base import KnowledgeBase

KBS = {"KB v1": "continual_kb_v1_seed42.db", "KB v2": "continual_kb_v2.db", "KB v3": "continual_kb_v3.db",
       "KB v1_lift": "continual_kb_v1_lift.db", "KB v2_lift": "continual_kb_v2_lift.db"}
N_BROADEST = 5


def rule_mask(rule, X, col):
    mask = np.ones(len(X), dtype=bool)
    for cond in rule["conditions"]:
        if cond["feature"] not in col:
            return np.zeros(len(X), dtype=bool)
        mask &= np.asarray(evaluate_condition(X[:, col[cond["feature"]]], cond["operator"], cond["threshold"]),
                           dtype=bool)
    return mask


def fmt_conditions(rule):
    return " AND ".join(f"{c['feature']} {c['operator']} {c['threshold']:.4g}" for c in rule["conditions"])


def per_experience_counts(build, kb, classes):
    """Active rules per class after each experience, from the build log."""
    all_rules = {r["rule_id"]: r["predicted_class"] for r in kb_all_rules(kb)}
    added_at = build["rule_experience"]
    out = []
    active = {c: 0 for c in classes}
    for e in build["experiences"]:
        eid = e["experience"]
        for rid, ex in added_at.items():
            if ex == eid:
                active[all_rules[rid]] += 1
        for c, n in e.get("retired_by_class", {}).items():
            active[c] -= n
        assert sum(active.values()) == e["active_after"], (eid, active, e["active_after"])
        out.append(dict(active))
    return out


def kb_all_rules(kb):
    import sqlite3
    with sqlite3.connect(kb.db_path) as conn:
        return [{"rule_id": rid, "predicted_class": pc} for rid, pc in conn.execute("SELECT rule_id, predicted_class FROM rules")]


def main(kbs=KBS, out_name="kb_rule_stats", with_gain=False):
    root = find_project_root()
    processed, out_dir = root / "data" / "processed", root / "results" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_continual_data(root)
    classes, idx = data["all_classes"], data["class_to_idx"]
    col = {n: i for i, n in enumerate(data["feature_names"])}
    from src.preprocessing.pipeline import UNSW_NB15_Pipeline
    pipe = UNSW_NB15_Pipeline.load(processed / "pipeline.joblib")
    X = np.load(processed / "X_test.npy")
    y = np.array([idx[c] for c in pipe.label_encoder.inverse_transform(np.load(processed / "y_cat_test.npy"))])
    base = np.bincount(y, minlength=len(classes)) / len(y)

    rows, sections = [], []
    for name, f in kbs.items():
        kb = KnowledgeBase(db_path=processed / f)
        rules = kb.get_active_rules()
        with open(root / "results" / "continual" / f"kb_build_{f[:-3]}.json") as fh:
            build = json.load(fh)
        per_exp = per_experience_counts(build, kb, classes)
        final = {c: 0 for c in classes}
        for r in rules:
            final[r["predicted_class"]] += 1
        assert per_exp[-1] == final, (name, per_exp[-1], final)

        stats = []
        for r in rules:
            m = rule_mask(r, X, col)
            k = idx[r["predicted_class"]]
            n = int(m.sum())
            prec = float((y[m] == k).mean()) if n else np.nan
            stats.append({"class": r["predicted_class"], "match_fraction": n / len(X), "support": n,
                          "stored_precision": float(r.get("precision", 0.0)), "test_precision": prec,
                          "lift": prec / base[k] if n else np.nan,
                          "gain": (prec - base[k]) / (1 - base[k]) if n else np.nan,
                          "conditions": fmt_conditions(r)})
        st = pd.DataFrame(stats)
        mf = st["match_fraction"] if len(st) else pd.Series(dtype=float)
        lifts = st["lift"].dropna() if len(st) else pd.Series(dtype=float)
        row = {"KB": name, "File": f, "Active_Rules": len(rules),
               **{f"Exp{e}_{c}": per_exp[e - 1][c] for e in range(1, len(per_exp) + 1) for c in classes},
               "Mean_Match_Fraction": float(mf.mean()) if len(mf) else np.nan,
               "Median_Match_Fraction": float(mf.median()) if len(mf) else np.nan,
               "Rules_Over_50pct": int((mf > 0.5).sum()),
               "Mean_Test_Lift": float(lifts.mean()) if len(lifts) else np.nan,
               **({"Mean_Test_Gain": float(st["gain"].dropna().mean()) if len(st) else np.nan,
                   "Rules_Positive_Gain_Under_50pct": int(((st["gain"] > 0) & (mf <= 0.5)).sum()) if len(st) else 0}
                  if with_gain else {}),
               "Rules_Never_Firing": int((st["support"] == 0).sum()) if len(st) else 0}
        rows.append(row)

        # markdown section for this KB
        present = [c for c in classes if any(p[c] for p in per_exp)]
        sec = [f"### {name} (`{f}`): {len(rules)} active rules", "",
               "| After experience | " + " | ".join(present) + " | Total |", "|---|" + "---|" * (len(present) + 1)]
        for e, p in enumerate(per_exp, start=1):
            sec.append(f"| {e} | " + " | ".join(str(p[c]) for c in present) + f" | {sum(p.values())} |")
        sec += ["", f"Broadest {N_BROADEST} rules (test set):", "",
                "| Class | Match fraction | Stored precision | Test precision | Test lift | "
                + ("Test gain | " if with_gain else "") + "Conditions |",
                "|---|---|---|---|---|" + ("---|" if with_gain else "") + "---|"]
        for _, r in (st.sort_values("match_fraction", ascending=False).head(N_BROADEST).iterrows() if len(st) else []):
            sec.append(f"| {r['class']} | {r['match_fraction']:.3f} | {r['stored_precision']:.3f} | "
                       f"{r['test_precision']:.3f} | {r['lift']:.2f} | " + (f"{r['gain']:+.3f} | " if with_gain else "")
                       + f"`{r['conditions']}` |")
        sections += sec + [""]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{out_name}.csv", index=False)
    L = ["# Continual KB rule statistics", "",
         f"Test set: {len(X)} rows. Statistics are over each KB's final ACTIVE rules, using the same matching as "
         "SymbolicReasoner. Lift = test precision / test base rate of the rule's class (mean over rules that fire "
         "on at least one test row). Test base rates: " + ", ".join(f"{c} {b:.3f}" for c, b in zip(classes, base)) + ".",
         "", "| KB | Active rules | Mean match fraction | Median match fraction | Rules firing on > 50% | "
             "Mean test lift | " + ("Mean test gain | Rules with gain > 0 and match <= 50% | " if with_gain else "")
             + "Rules never firing |", "|---|---|---|---|---|---|" + ("---|---|" if with_gain else "") + "---|"]
    for r in rows:
        L.append(f"| {r['KB']} | {r['Active_Rules']} | {r['Mean_Match_Fraction']:.3f} | "
                 f"{r['Median_Match_Fraction']:.3f} | {r['Rules_Over_50pct']} | {r['Mean_Test_Lift']:.2f} | "
                 + (f"{r['Mean_Test_Gain']:+.3f} | {r['Rules_Positive_Gain_Under_50pct']} | " if with_gain else "")
                 + f"{r['Rules_Never_Firing']} |")
    L += ["", "## Active rules per class after each experience, and broadest rules", ""] + sections
    (out_dir / f"{out_name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kbs", nargs="+", metavar="NAME=FILE", help="KBs to report (default: the five above)")
    parser.add_argument("--out-name", default="kb_rule_stats")
    parser.add_argument("--gain", action="store_true", help="also report test gain")
    args = parser.parse_args()
    main(dict(k.split("=", 1) for k in args.kbs) if args.kbs else KBS, args.out_name, args.gain)
