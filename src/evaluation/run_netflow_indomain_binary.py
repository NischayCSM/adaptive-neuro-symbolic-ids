"""
In-domain binary detection rate and Benign FPR on the NF-UNSW-NB15-v2 test split.

run_netflow_eval.py records binary rates only for NF-ToN; for NF-UNSW test it records 10-class metrics.
This script retrains the same models (XGBoostBaseline(random_state=seed) on NF-UNSW train; V4 stacking fit
on NF-UNSW validation with kb_netflow_gain.db; seeds 42, 1, 2, 3, 4), checks that accuracy, macro-F1 and
per-class recall on NF-UNSW test reproduce results/external_netflow/netflow_eval.csv, and only then writes

    results/external_netflow/netflow_indomain_binary.csv

with the binary rates, defined as in run_netflow_eval.py: detection rate = share of attack rows predicted
as any attack class; FPR = share of Benign rows predicted as an attack. Re-launches itself with
PYTHONHASHSEED=0 (same environment as run_netflow_eval.py). Refuses to overwrite its output.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.continual.cl_eval import find_project_root
from src.models.baselines import XGBoostBaseline
from src.reasoning import fusion_variants as fv
from src.evaluation.run_netflow_eval import SEEDS, XGB, V4

TOL = 1e-9


def main():
    root = find_project_root()
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        sys.exit(subprocess.run([sys.executable, "-m", "src.evaluation.run_netflow_indomain_binary"],
                                cwd=root, env=env).returncode)
    data_dir, out = root / "data" / "processed_netflow", root / "results" / "external_netflow"
    out_csv = out / "netflow_indomain_binary.csv"
    if out_csv.exists():
        sys.exit(f"ERROR: {out_csv.relative_to(root)} already exists; not overwriting")
    kb_path = data_dir / "kb_netflow_gain.db"
    features = json.loads((data_dir / "feature_names.json").read_text())
    classes = json.loads((data_dir / "label_classes.json").read_text())
    cidx = {c: i for i, c in enumerate(classes)}
    labels = list(range(len(classes)))
    ben = cidx["Benign"]
    X = {k: np.load(data_dir / f"X_{k}.npy") for k in ("train", "val", "test")}
    y_tr, y_va, y_te = (np.load(data_dir / f"y_{k}.npy") for k in ("train", "val", "test"))
    # Not fv.cached_rule_evidence: its cache in results/external_netflow/cache must not be rewritten here.
    ev = {k: fv.rule_evidence(kb_path, features, X[k], cidx) for k in ("val", "test")}
    ref = pd.read_csv(out / "netflow_eval.csv")
    ref = ref[ref["Domain"] == "NF-UNSW test"].set_index(["Method", "Seed"])
    check = ["Accuracy", "Macro_F1"] + [f"Recall_{c}" for c in classes]

    rows, max_diff = [], 0.0
    attack = y_te != ben
    for s in SEEDS:
        model = XGBoostBaseline(random_state=s).fit(X["train"], y_tr).model
        assert list(model.classes_) == labels
        P = {k: model.predict_proba(X[k]) for k in ("val", "test")}
        v4 = fv.StackingFuser().fit(P["val"], *ev["val"], y_va)
        for meth, pred in ((XGB, P["test"].argmax(1)), (V4, v4.predict(P["test"], *ev["test"]))):
            rec = recall_score(y_te, pred, labels=labels, average=None, zero_division=0)
            mine = {"Accuracy": accuracy_score(y_te, pred),
                    "Macro_F1": f1_score(y_te, pred, labels=labels, average="macro", zero_division=0),
                    **{f"Recall_{c}": float(v) for c, v in zip(classes, rec)}}
            max_diff = max(max_diff, max(abs(mine[c] - float(ref.loc[(meth, s), c])) for c in check))
            det = pred != ben
            rows.append({"Seed": s, "Method": meth, "Domain": "NF-UNSW test",
                         "Detection_Rate": float(det[attack].mean()), "FPR_Benign": float(det[~attack].mean())})
        print(f"seed {s}: " + ", ".join(f"{r['Method']} DR {r['Detection_Rate']:.4f} FPR {r['FPR_Benign']:.4f}"
                                         for r in rows[-2:]))
    print(f"Reproduction of netflow_eval.csv (NF-UNSW test, {len(check)} metrics): max |difference| = {max_diff:.1e}")
    if max_diff > TOL:
        sys.exit("ERROR: retrained models do not reproduce netflow_eval.csv; not writing")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
