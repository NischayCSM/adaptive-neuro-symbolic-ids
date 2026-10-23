"""
Summarises results/continual/kb_retention.csv and kb_retention_alpha.csv (from
run_kb_retention.py) as mean ± sample std (ddof=1) over seeds. Gain = fused retention -
neural-only retention, per seed. Writes results/continual/kb_retention.md.
"""

import numpy as np
import pandas as pd

from src.continual.cl_eval import find_project_root

KBS = ["v1", "v2", "v3"]


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def main():
    root = find_project_root() / "results" / "continual"
    df = pd.read_csv(root / "kb_retention.csv")
    alpha = pd.read_csv(root / "kb_retention_alpha.csv")
    T = max(int(c.split("exp")[1]) for c in df.columns if c.startswith("neural_exp"))
    lines = [f"Seeds: {sorted(int(s) for s in df['seed'].unique())}; mean ± sample std; alpha = 0.5\n",
             "## Retention (mean acc on Exp 1..T-1 after Exp T)\n",
             "| Buffer | Neural only | + v1 | + v2 | + v3 |", "|---|---|---|---|---|"]
    for b, g in df.groupby("buffer"):
        lines.append(f"| {b} | " + " | ".join(ms(g[f"{k}_retention"]) for k in ["neural"] + KBS) + " |")

    lines += [f"\n## Accuracy on Exp {T}\n", "| Buffer | Neural only | + v1 | + v2 | + v3 |", "|---|---|---|---|---|"]
    for b, g in df.groupby("buffer"):
        lines.append(f"| {b} | " + " | ".join(ms(g[f"{k}_exp{T}"]) for k in ["neural"] + KBS) + " |")

    lines += ["\n## ACC (mean acc on all experiences after Exp T)\n", "| Buffer | Neural only | + v1 | + v2 | + v3 |",
              "|---|---|---|---|---|"]
    for b, g in df.groupby("buffer"):
        lines.append(f"| {b} | " + " | ".join(ms(g[[f"{k}_exp{j}" for j in range(1, T + 1)]].mean(axis=1))
                                           for k in ["neural"] + KBS) + " |")

    lines += ["\n## Retention gain over neural only (per seed, then mean ± std)\n",
              "| Buffer | Gain v1 | Gain v2 | Gain v3 |", "|---|---|---|---|"]
    for b, g in df.groupby("buffer"):
        lines.append(f"| {b} | " + " | ".join(ms(g[f"{k}_retention"] - g["neural_retention"]) for k in KBS) + " |")
    lines.append("| Runs with gain > 0 (of %d) | " % len(df)
                 + " | ".join(str(int((df[f"{k}_retention"] > df["neural_retention"]).sum())) for k in KBS) + " |")

    lines += ["\n## Per-seed gains\n", "| Buffer | Seed | Neural | Gain v1 | Gain v2 | Gain v3 |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {int(r.buffer)} | {int(r.seed)} | {r.neural_retention:.4f} | "
                     + " | ".join(f"{r[f'{k}_retention'] - r.neural_retention:+.4f}" for k in KBS) + " |")

    kb = alpha["kb"].iloc[0]
    neural10 = df[df["buffer"] == 10000].set_index("seed")["neural_retention"]
    lines += [f"\n## Alpha sweep, buffer 10000, KB {kb}\n",
              f"| alpha | Retention | Gain over neural | Acc Exp {T} | ACC |", "|---|---|---|---|---|"]
    for a, g in alpha.groupby("alpha"):
        gain = g.set_index("seed")["retention"] - neural10
        lines.append(f"| {a} | {ms(g['retention'])} | {ms(gain)} | {ms(g[f'exp{T}'])} | {ms(g[[f'exp{j}' for j in range(1, T + 1)]].mean(axis=1))} |")

    text = "\n".join(lines)
    print(text)
    with open(root / "kb_retention.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
