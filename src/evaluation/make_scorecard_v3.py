"""
Phase II scorecard v3 = results/final_test_report_phase2_v2.md plus one row: Test 1b at the validation-chosen 5%
operating point (R1 + V4, threshold on 1 - P(Normal) chosen on the validation memory; results/fusion/operating_point.csv).
All v2 rows are copied unchanged. Writes results/final_test_report_phase2_v3.md; refuses to overwrite it.

    python -m src.evaluation.make_scorecard_v3
"""

import sys

import pandas as pd

from src.continual.cl_eval import find_project_root

TARGET = 0.05


def main():
    root = find_project_root()
    res = root / "results"
    src_md, out_md = res / "final_test_report_phase2_v2.md", res / "final_test_report_phase2_v3.md"
    if out_md.exists():
        sys.exit(f"ERROR: {out_md.relative_to(root)} already exists; not overwriting")
    op = pd.read_csv(res / "fusion" / "operating_point.csv")
    g = op[(op["Model"] == "R1 + V4 stacking") & (op["Threshold_Source"] == "validation memory") & (op["Target"] == TARGET)]
    if len(g) != 5:
        sys.exit(f"ERROR: expected 5 seeds of R1 + V4 at target {TARGET}, found {len(g)}")
    pm = lambda x: f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"
    nta = g["Normal_to_Attack"]
    within = int((nta <= TARGET).sum())
    passed = nta.mean() < TARGET
    lines = src_md.read_text(encoding="utf-8").splitlines()

    row = (f"| 1b-op | False alarm rate, continual (R1 + V4) at the validation-chosen 5% operating point | "
           f"Normal→attack < 0.05 | not measured | Normal→attack {pm(nta)} ({within}/5 seeds ≤ 0.05; attack→Normal "
           f"{pm(g['Attack_to_Normal'])}, macro-F1 {pm(g['Macro_F1'])}) | **{'PASS' if passed else 'FAIL'}** |")
    i1b = next(i for i, l in enumerate(lines) if l.startswith("| 1b |"))
    lines.insert(i1b + 1, row)
    lines[0] = lines[0].replace(" - v2", " - v3")
    lines = [l.replace("Per-seed values: `results/final_test_phase2_metrics_v2.csv`.",
                       "Per-seed values: `results/final_test_phase2_metrics_v2.csv` and `results/fusion/operating_point.csv`.")
             for l in lines]
    i_def = next(i for i, l in enumerate(lines) if l.startswith("1. **False alarm rate**"))
    lines.insert(i_def + 1,
                 "   **1b-op (v3)**: the same R1 + V4 models with an alarm threshold t on s = 1 − P(Normal) "
                 "(stacker predict_proba). Per seed, t is chosen on the validation memory only (200 Normal flows, never "
                 "test data) so that at most 5% of the memory's Normal flows have s > t. Flows with s ≤ t are labelled "
                 "Normal, the others get the model's top attack class. Metrics on the evaluation split. The threshold is "
                 "set on the stacker's own training rows (in-sample); an out-of-fold variant and the 1% target are in "
                 "`results/fusion/operating_point.md`. Test 1b (default argmax) is kept unchanged.")

    # verdict: count every verdict line, and the 8-test view with Test 1 judged on 1a and the continual operating point
    verdict_rows = [l for l in lines if l.startswith("| ") and l.rstrip().endswith(("**PASS** |", "**FAIL** |"))]
    n_pass = sum(l.rstrip().endswith("**PASS** |") for l in verdict_rows)
    ok = {l.split("|")[1].strip(): l.rstrip().endswith("**PASS** |") for l in verdict_rows}
    as8 = sum(v for k, v in ok.items() if k not in ("1a", "1b", "1b-op")) + (ok["1a"] and ok["1b-op"])
    import re
    iv = next(i for i, l in enumerate(lines) if l.startswith("**Phase II v2:"))
    v2_lines, v2_as8 = re.search(r"Phase II v2: (\d+/\d+) verdict lines PASS.*?: (\d+/8)\.", lines[iv]).groups()
    phase1 = re.search(r"(Phase I \(results/final_test_report\.md\): \d+/8 PASS\.)", lines[iv]).group(1)
    lines[iv] = (f"**Phase II v3: {n_pass}/{len(verdict_rows)} verdict lines PASS** (Test 1 split into 1a static, 1b "
                 f"continual at the default argmax, and 1b-op continual at the validation-chosen 5% operating point). "
                 f"Counted as 8 tests, with Test 1 passing if 1a and 1b-op pass: {as8}/8. v2 (without 1b-op): "
                 f"{v2_lines} verdict lines PASS; {v2_as8} as 8 tests (Test 1 passing only if 1a and 1b pass). {phase1}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:22]))
    print(lines[iv])
    print(f"Saved {out_md}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()
