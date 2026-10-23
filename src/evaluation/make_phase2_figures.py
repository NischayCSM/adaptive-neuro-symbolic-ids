"""
Phase II figures (PNG, 300 dpi, grayscale-safe: identity by gray level + hatching + labels, never hue).

    python -m src.evaluation.make_phase2_figures [--force]

Reads only results files (and, for F4, the ACTIVE Fuzzers rule count of each KS-threshold KB):
    F1  results/continual/final_model_eval.csv, results/final_test_phase2_metrics.csv (R1, full test set)
    F2  results/fusion/detector_variants.csv
    F3  results/fusion/kb_rule_stats.csv, results/fusion/kb_rule_stats_final.csv
    F4  results/ablations/ks_threshold.csv, data/processed/continual_kb_v1_gain_clean*.db,
        results/continual/continual_pipeline_metrics_continual_kb_v1_gain_clean.csv (Exp 4 drift score)
    F5  results/fusion/detector_variants.csv (R1)
    F6  results/external_netflow/netflow_indomain_binary.csv, results/external_netflow/netflow_eval.csv
Writes results/figures_phase2/F1_*.png ... F6_*.png; refuses to overwrite an existing figure unless --force.
"""

import argparse
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.continual.cl_eval import find_project_root

DPI = 300
DARK, MID, LIGHT, WHITE, INK = "#333333", "#808080", "#c8c8c8", "#ffffff", "#000000"
SEEDS = [42, 1, 2, 3, 4]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e0e0e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "hatch.linewidth": 0.6,
                     "savefig.facecolor": WHITE})


def bar_kw(fill, hatch=None):
    return dict(facecolor=fill, edgecolor=INK, linewidth=0.6, hatch=hatch,
                error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2.5, capthick=0.8))


def f1_final_model(root):
    fm = pd.read_csv(root / "results/continual/final_model_eval.csv")
    ph2 = pd.read_csv(root / "results/final_test_phase2_metrics.csv")
    ewc = fm[fm["Method"].str.startswith("EWC")].groupby("Method")["Macro_F1"].mean()
    best_ewc = ewc.idxmax()
    rows = [("Naive fine-tuning", "Naive fine-tuning", "cont"),
            (best_ewc, f"EWC, best λ = {best_ewc.split('=')[1].rstrip(')')}", "cont"),
            ("Replay (buffer=2000)", "Replay, buffer 2k", "cont"),
            ("Replay (buffer=5000)", "Replay, buffer 5k", "cont"),
            ("Replay (buffer=10000)", "Replay, buffer 10k", "cont"),
            ("R1", "R1: replay 10k + class weights\n(+ validation memory)", "cont"),
            ("Joint training (5 epochs)", "Joint, 5 epochs", "joint"),
            ("Joint training (20 epochs)", "Joint, 20 epochs", "joint"),
            ("Joint training (20 epochs, class-weighted)", "Joint, 20 epochs, class-weighted", "joint")]
    vals = []
    for key, label, kind in rows:
        if key == "R1":
            x = ph2[(ph2["Scope"] == "full test set") & (ph2["System"] == "R1 detector alone")]["Macro_F1"]
        else:
            x = fm[fm["Method"] == key]["Macro_F1"]
        assert len(x) == 5, (key, len(x))
        vals.append((label, kind, x.mean(), x.std(ddof=1)))
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    y = np.arange(len(vals))[::-1]
    for yi, (label, kind, m, s) in zip(y, vals):
        ax.barh(yi, m, xerr=s, height=0.62, **bar_kw(MID if kind == "cont" else WHITE, None if kind == "cont" else "///"))
        ax.text(m + s + 0.008, yi, f"{m:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, [v[0] for v in vals])
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Macro-F1 of the final model on the full test set (175,341 rows; dimensionless, 0–1)")
    ax.set_title("Final-model macro-F1 after all four experiences (mean ± SD, 5 seeds)", loc="left")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(**{k: v for k, v in bar_kw(MID).items() if k != "error_kw"}, label="continual"),
                       Patch(**{k: v for k, v in bar_kw(WHITE, "///").items() if k != "error_kw"},
                             label="joint training (upper bound)")], loc="upper right", frameon=False, fontsize=8)
    return fig


def f2_rule_contribution(root):
    dv = pd.read_csv(root / "results/fusion/detector_variants.csv")
    labels = {"R0": "R0\nreplay", "R1": "R1\nclass-weighted", "R2": "R2\nweight-aligned", "R3": "R3\nboth"}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, v in enumerate(labels):
        g = dv[dv["Variant"] == v].set_index(["Method", "Seed"])["Macro_F1"]
        d = np.array([g[("V4 stacking", s)] - g[("Control: stacking on log P only", s)] for s in SEEDS])
        jitter = np.linspace(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d, s=28, marker="o", facecolor=WHITE, edgecolor=INK, linewidth=0.9, zorder=3,
                   label="one seed" if i == 0 else None)
        ax.plot([i - 0.22, i + 0.22], [d.mean()] * 2, color=INK, linewidth=2, zorder=4,
                label="mean over 5 seeds" if i == 0 else None)
        ax.text(i + 0.26, d.mean(), f"{d.mean():+.3f}", va="center", fontsize=8)
    ax.axhline(0, color=INK, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(labels)), labels.values())
    ax.set_xlim(-0.5, len(labels) - 0.3)
    ax.set_ylim(-0.005, 0.075)
    ax.set_ylabel("Rule contribution: Δ macro-F1\n(V4 stacking − detector-only control)")
    ax.set_title("Rule features add macro-F1 for every detector and seed", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def f3_kb_size(root):
    a = pd.read_csv(root / "results/fusion/kb_rule_stats.csv")
    b = pd.read_csv(root / "results/fusion/kb_rule_stats_final.csv")
    kb = pd.concat([a, b], ignore_index=True).set_index("KB")
    order = ["KB v1", "KB v2", "KB v3", "KB v1_lift", "KB v2_lift", "KB v1_clean", "KB v1_gain_clean"]
    act, broad = kb.loc[order, "Active_Rules"].to_numpy(), kb.loc[order, "Rules_Over_50pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    x, w = np.arange(len(order)), 0.38
    ax.bar(x - w / 2, act, w, **bar_kw(MID), label="active rules")
    ax.bar(x + w / 2, broad, w, **bar_kw(WHITE, "///"), label="broad rules (fire on > 50% of test rows)")
    for xi, n in zip(x - w / 2, act):
        ax.text(xi, n + 3, str(n), ha="center", fontsize=8)
    for xi, n in zip(x + w / 2, broad):
        ax.text(xi, n + 3, str(n), ha="center", fontsize=8)
    ax.set_xticks(x, [o.replace("KB ", "") for o in order], rotation=20, ha="right")
    ax.set_xlabel("Continual knowledge base (rule-validation variant)")
    ax.set_ylabel("Number of rules")
    ax.set_ylim(0, max(act) * 1.15)
    ax.set_title("Knowledge-base size and broad-rule count by validation criterion", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    return fig


def f4_ks_threshold(root):
    ks = pd.read_csv(root / "results/ablations/ks_threshold.csv")
    v4 = ks[ks["Method"] == "V4 stacking"].groupby("Drift_Threshold")["Macro_F1"]
    ctrl = ks[ks["Method"].str.startswith("Control")].groupby("Drift_Threshold")["Macro_F1"].mean()
    thr = np.array(sorted(ks["Drift_Threshold"].unique()))
    kbs = ks.drop_duplicates("Drift_Threshold").set_index("Drift_Threshold")["KB"]
    fuzz = []
    for t in thr:
        conn = sqlite3.connect(root / "data/processed" / kbs[t])
        fuzz.append(conn.execute("SELECT COUNT(*) FROM rules WHERE status = 'ACTIVE' AND predicted_class = 'Fuzzers'")
                    .fetchone()[0])
        conn.close()
    drift = pd.read_csv(root / "results/continual/continual_pipeline_metrics_continual_kb_v1_gain_clean.csv")
    exp4 = float(drift.set_index("Experience").loc[4, "Drift_Score"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True, gridspec_kw={"hspace": 0.18})
    a1.errorbar(thr, v4.mean().loc[thr], yerr=v4.std(ddof=1).loc[thr], color=INK, marker="o", markersize=5,
                markerfacecolor=WHITE, linewidth=1.5, capsize=3, label="V4 stacking (mean ± SD, 5 seeds)")
    a1.plot(thr, ctrl.loc[thr], color=MID, linestyle="--", linewidth=1.2, label="detector-only control (no rules)")
    a1.set_ylabel("Macro-F1 (evaluation split)")
    a1.set_ylim(0.36, 0.45)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a2.bar(thr, fuzz, width=0.025, **bar_kw(MID))
    for t, n in zip(thr, fuzz):
        a2.text(t, n + 0.6, str(n), ha="center", fontsize=8)
    a2.set_ylabel("Active Fuzzers rules")
    a2.set_ylim(0, max(fuzz) * 1.35 + 1)
    for ax in (a1, a2):
        ax.axvline(exp4, color=INK, linewidth=0.8, linestyle=":")
    a2.text(exp4 + 0.003, max(fuzz) * 1.2, f"Exp 4 drift score {exp4:.3f}", fontsize=8)
    a2.set_xticks(thr, [f"{t:.2f}" for t in thr])
    a2.set_xlabel("KS drift threshold (mean two-sample KS statistic; dimensionless)")
    a1.set_title("A threshold above the Exp 4 drift score skips the Fuzzers rules", loc="left")
    a2.grid(axis="x", visible=False)
    return fig


def f5_per_class_recall(root):
    dv = pd.read_csv(root / "results/fusion/detector_variants.csv")
    r1 = dv[dv["Variant"] == "R1"]
    classes = [c[len("Recall_"):] for c in dv.columns if c.startswith("Recall_")]
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    y, h = np.arange(len(classes))[::-1], 0.38
    for off, (m, fill, hatch, lab) in zip((h / 2, -h / 2), [("Detector alone", MID, None, "R1 detector alone"),
                                                         ("V4 stacking", WHITE, "///", "R1 + V4 stacking")]):
        g = r1[r1["Method"] == m]
        ax.barh(y + off, [g[f"Recall_{c}"].mean() for c in classes], h,
                xerr=[g[f"Recall_{c}"].std(ddof=1) for c in classes], **bar_kw(fill, hatch), label=lab)
    ax.set_yticks(y, ["DoS" if c == "Dos" else c for c in classes])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall per class (evaluation split, 140,272 rows; mean ± SD, 5 seeds)")
    ax.set_title("Per-class recall of the final configuration: detector alone vs V4", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", visible=False)
    return fig


def f6_netflow(root):
    ind = pd.read_csv(root / "results/external_netflow/netflow_indomain_binary.csv")
    nf = pd.read_csv(root / "results/external_netflow/netflow_eval.csv")
    cross = nf[nf["Domain"] == "NF-ToN"]
    methods = [("XGBoost alone", MID, None), ("V4 stacking", WHITE, "///")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True)
    for ax, (col, title) in zip(axes, [("Detection_Rate", "Detection rate (attacks flagged)"),
                                       ("FPR_Benign", "Benign false-positive rate")]):
        x, w = np.arange(2), 0.36
        for off, (m, fill, hatch) in zip((-w / 2, w / 2), methods):
            vals = [d[d["Method"] == m][col] for d in (ind, cross)]
            means, sds = [v.mean() for v in vals], [v.std(ddof=1) for v in vals]
            ax.bar(x + off, means, w, yerr=sds, **bar_kw(fill, hatch), label=m)
            for xi, mv in zip(x + off, means):
                ax.text(xi, mv + 0.025, f"{mv:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(x, ["in-domain\n(NF-UNSW test)", "cross-dataset\n(NF-ToN)"])
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of rows (0–1; mean ± SD, 5 seeds)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("NetFlow-v2: near-perfect in-domain detection does not transfer across datasets",
                 x=0.01, ha="left", fontsize=10)
    return fig


FIGURES = [("F1_final_model_macro_f1.png", f1_final_model), ("F2_rule_contribution.png", f2_rule_contribution),
           ("F3_kb_size_broad_rules.png", f3_kb_size), ("F4_ks_threshold.png", f4_ks_threshold),
           ("F5_per_class_recall.png", f5_per_class_recall), ("F6_netflow_transfer.png", f6_netflow)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing figures")
    args = parser.parse_args()
    root = find_project_root()
    out = root / "results" / "figures_phase2"
    existing = [n for n, _ in FIGURES if (out / n).exists()]
    if existing and not args.force:
        sys.exit(f"ERROR: {existing} already exist in {out}; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES:
        fig = make(root)
        fig.savefig(out / name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out / name}")


if __name__ == "__main__":
    main()
