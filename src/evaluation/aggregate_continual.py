"""
Aggregates per-seed continual-learning summaries into mean ± std tables
(std = sample std, ddof=1). Writes results/continual/aggregate.md.
"""

import json
import glob
import re

import numpy as np

from src.continual.cl_eval import find_project_root

SEEDS = [42, 1, 2, 3, 4]


def load(pattern_fmt, seeds=SEEDS):
    root = find_project_root() / "results" / "continual"
    out = []
    for s in seeds:
        path = root / pattern_fmt.format(seed=s)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path) as f:
            out.append(json.load(f))
    return out


def ms(values):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def matrix_mean(runs):
    return np.mean([r["acc_matrix"] for r in runs], axis=0)


def main():
    root = find_project_root() / "results" / "continual"
    lines = []

    naive = load("naive_summary_seed{seed}.json")
    replay = {b: load("replay_summary_seed{seed}.json" if b == 2000 else f"replay_summary_seed{{seed}}_buf{b}.json")
              for b in (2000, 5000, 10000)}
    ewc_files = sorted({re.sub(r"seed\d+", "seed{seed}", p.split("\\")[-1].split("/")[-1])
                        for p in glob.glob(str(root / "ewc_summary_seed*.json"))})
    ewc = {}
    for fmt in ewc_files:
        runs = load(fmt)
        ewc[runs[0]["ewc_lambda"]] = runs

    T = len(naive[0]["acc_matrix"])
    lines.append(f"Seeds: {SEEDS} (n={len(SEEDS)}); mean ± sample std\n")

    lines.append("## Accuracy matrix (mean over seeds): row = after training Exp k, col = test on Exp j\n")
    lines.append("| After | " + " | ".join(f"Exp{j} replay | Exp{j} naive" for j in range(1, T + 1)) + " |")
    lines.append("|---" * (2 * T + 1) + "|")
    rm, nm = matrix_mean(replay[2000]), matrix_mean(naive)
    for k in range(T):
        lines.append(f"| Exp{k+1} | " + " | ".join(f"{rm[k][j]:.4f} | {nm[k][j]:.4f}" for j in range(T)) + " |")

    lines.append("\n## ACC / BWT / retention (retention = mean acc on Exp 1..T-1 after Exp T)\n")
    lines.append("| Method | ACC | BWT | Retention | Acc on last exp |")
    lines.append("|---|---|---|---|---|")

    def row(name, runs):
        lines.append(f"| {name} | {ms([r['neural']['ACC'] for r in runs])} | "
                     f"{ms([r['neural']['BWT'] for r in runs])} | {ms([r['neural']['retention'] for r in runs])} | "
                     f"{ms([r['acc_matrix'][-1][-1] for r in runs])} |")

    def kb_row(name, runs):
        lines.append(f"| {name} | {ms([r['kb_fused_final']['ACC'] for r in runs])} | n/a (final model only) | "
                     f"{ms([r['kb_fused_final']['retention'] for r in runs])} | {ms([r['kb_fused_final']['acc_per_exp'][-1] for r in runs])} |")

    joint = load("joint_summary_seed{seed}.json")
    lines.append(f"| Joint training (all experiences at once) | {ms([r['ACC'] for r in joint])} | n/a | "
                 f"{ms([r['mean_acc_exp1_to_T-1'] for r in joint])} | {ms([r['acc_per_exp'][-1] for r in joint])} |")
    row("Naive fine-tuning", naive)
    kb_row("Naive fine-tuning + continual KB (alpha=0.5)", naive)
    for lam in sorted(ewc):
        row(f"EWC (lambda={lam:g})", ewc[lam])
    for b in (2000, 5000, 10000):
        row(f"Replay (buffer={b})", replay[b])
        kb_row(f"Replay (buffer={b}) + continual KB (alpha=0.5)", replay[b])

    lines.append("\n## Retention by buffer size\n")
    lines.append("| Buffer | Retention (neural) | Retention (+KB) | Per-exp after Exp T (neural, mean) |")
    lines.append("|---|---|---|---|")
    for b in (2000, 5000, 10000):
        last = matrix_mean(replay[b])[-1]
        lines.append(f"| {b} | {ms([r['neural']['retention'] for r in replay[b]])} | "
                     f"{ms([r['kb_fused_final']['retention'] for r in replay[b]])} | "
                     + ", ".join(f"Exp{j+1}={last[j]:.4f}" for j in range(T)) + " |")

    text = "\n".join(lines)
    print(text)
    with open(root / "aggregate.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
