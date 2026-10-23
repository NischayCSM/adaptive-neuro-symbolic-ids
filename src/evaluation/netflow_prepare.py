"""
NetFlow-v2 cross-dataset data preparation (NF-UNSW-NB15-v2 -> NF-ToN-IoT-v2).

Reads the two raw CSVs from --source-dir (default C:\\datasets\\netflow, outside the project) in chunks; nothing raw
is copied into the project.
  1. Pass 1 (columns Label, Attack only): rows, columns, class counts; checks Label == (Attack != Benign).
  2. Stratified sample, seed 42: at most CAP (50,000) rows per class, all rows of smaller classes.
  3. Pass 2: reads the sampled rows only. Features = the shared NetFlow columns minus the flow identifiers
     (IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT) and the labels (Label, Attack).
  4. NF-UNSW sample split 70/15/15 into train / validation / test (stratified, seed 42); NF-ToN is test-only.
  5. Preprocessing as in the main project's UNSW_NB15_Pipeline: inf -> NaN, median imputation, StandardScaler,
     all fitted on NF-UNSW train only. Every NetFlow column is numeric (PROTOCOL, L7_PROTO, ICMP_TYPE, ... are
     numeric codes), so no categorical encoding is applied. Stored as float32. Extension of inf -> NaN: a raw value
     whose standardised value would overflow float32 (|z| > float32 max; corrupted byte-rate values up to ~1e261 in
     NF-ToN's *_SECOND_BYTES) is also set to NaN and imputed with the NF-UNSW-train median. Every replaced value is
     counted per dataset / split / feature and recorded in the data card.

Writes data/processed_netflow/ (arrays, preprocessor, class lists, label mapping) and results/external_netflow/
(sample_indices.npz, data_counts.json, data_card.md). Refuses to overwrite existing outputs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.continual.cl_eval import find_project_root

SEED = 42
CAP = 50_000
CHUNK_LABELS, CHUNK_ROWS = 2_000_000, 500_000
ID_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
LABEL_COLUMNS = ["Label", "Attack"]
# NF-ToN class -> NF-UNSW class (matched case-insensitively against the actual spellings)
MAPPING = {"benign": "benign", "dos": "dos", "backdoor": "backdoor", "scanning": "reconnaissance"}
SMALL_TEST = 100  # classes with fewer test rows are marked in every table
F32_MAX = float(np.finfo(np.float32).max)
RATE_FEATURES = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]  # always reported in the overflow record


def find_csv(src, *keys):
    hits = [p for p in src.glob("*.csv") if all(k.lower() in p.name.lower() for k in keys)]
    if len(hits) != 1:
        sys.exit(f"ERROR: expected one CSV in {src} matching {keys}, found {[p.name for p in hits]}")
    return hits[0]


def pass_labels(path):
    """Per-row class codes (int16) and class names, plus label-consistency counts."""
    names, codes, mismatch, n = {}, [], 0, 0
    t = time.time()
    for ch in pd.read_csv(path, usecols=LABEL_COLUMNS, chunksize=CHUNK_LABELS, dtype={"Attack": str}):
        for a in ch["Attack"].unique():
            names.setdefault(a, len(names))
        codes.append(ch["Attack"].map(names).to_numpy(np.int16))
        mismatch += int(((ch["Label"] != 0) != (ch["Attack"].str.lower() != "benign")).sum())
        n += len(ch)
        print(f"  {path.name}: {n:,} rows read ({time.time() - t:.0f}s)", flush=True)
    inv = {v: k for k, v in names.items()}
    return np.concatenate(codes), inv, mismatch


def sample(codes, inv, rng):
    picked = []
    for c in sorted(inv, key=inv.get):  # class-name order, so the draw does not depend on file order
        pos = np.flatnonzero(codes == c)
        picked.append(np.sort(rng.choice(pos, CAP, replace=False)) if len(pos) > CAP else pos)
    return np.sort(np.concatenate(picked))


def pass_rows(path, rows, features):
    """Read only the selected data rows (0-based, header excluded)."""
    out, off = [], 0
    t = time.time()
    dtypes = {f: np.float64 for f in features}
    for ch in pd.read_csv(path, usecols=features + LABEL_COLUMNS, chunksize=CHUNK_ROWS, dtype={**dtypes, "Attack": str}):
        lo, hi = np.searchsorted(rows, [off, off + len(ch)])
        if hi > lo:
            out.append(ch.iloc[rows[lo:hi] - off])
        off += len(ch)
        if off % (10 * CHUNK_ROWS) == 0:
            print(f"  {path.name}: {off:,} rows scanned ({time.time() - t:.0f}s)", flush=True)
    df = pd.concat(out)
    assert len(df) == len(rows)
    return df[features + LABEL_COLUMNS].reset_index(drop=True)


def counts_table(names, cols):
    """Markdown table: one row per class, one column per (title, Counter-like dict)."""
    L = ["| Class | " + " | ".join(t for t, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for c in names:
        L.append(f"| {c} | " + " | ".join(f"{d.get(c, 0):,}" for _, d in cols) + " |")
    L.append("| **Total** | " + " | ".join(f"**{sum(d.values()):,}**" for _, d in cols) + " |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source-dir", type=Path, default=Path(r"C:\datasets\netflow"))
    args = ap.parse_args()
    root = find_project_root()
    data_dir, res_dir = root / "data" / "processed_netflow", root / "results" / "external_netflow"
    for d in (data_dir, res_dir):
        if d.exists() and any(d.iterdir()):
            sys.exit(f"ERROR: {d.relative_to(root)} already exists and is not empty; not overwriting")
    unsw_path, ton_path = find_csv(args.source_dir, "unsw", "v2"), find_csv(args.source_dir, "ton", "iot", "v2")
    header = {p: pd.read_csv(p, nrows=0).columns.tolist() for p in (unsw_path, ton_path)}
    if header[unsw_path] != header[ton_path]:
        sys.exit("ERROR: the two CSVs have different columns")
    columns = header[unsw_path]
    missing = [c for c in ID_COLUMNS + LABEL_COLUMNS if c not in columns]
    if missing:
        sys.exit(f"ERROR: expected columns not found: {missing}")
    features = [c for c in columns if c not in ID_COLUMNS + LABEL_COLUMNS]
    ttl = [f for f in features if "TTL" in f.upper()]
    print(f"{len(columns)} columns; {len(features)} features; TTL-related: {ttl}")

    # ---- pass 1: labels -----------------------------------------------------------------------------------
    info = {}
    for key, path in (("unsw", unsw_path), ("ton", ton_path)):
        t = time.time()
        codes, inv, mismatch = pass_labels(path)
        info[key] = {"path": path, "codes": codes, "inv": inv, "mismatch": mismatch, "secs": time.time() - t,
                     "counts": {inv[c]: int(n) for c, n in zip(*np.unique(codes, return_counts=True))}}
        print(f"{path.name}: {len(codes):,} rows, Label/Attack mismatches {mismatch}, {info[key]['counts']}")
        if mismatch:
            sys.exit(f"ERROR: {mismatch} rows of {path.name} have Label inconsistent with Attack")

    unsw_lower = {c.lower(): c for c in info["unsw"]["counts"]}
    ton_lower = {c.lower(): c for c in info["ton"]["counts"]}
    bad = [k for k in MAPPING if k not in ton_lower] + [v for v in MAPPING.values() if v not in unsw_lower]
    if bad:
        sys.exit(f"ERROR: label(s) for the mapping not found: {bad}")
    mapping = {ton_lower[k]: unsw_lower[v] for k, v in MAPPING.items()}
    print("Label mapping (NF-ToN -> NF-UNSW):", mapping)

    # ---- sampling and pass 2 --------------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    raw = {}
    for key in ("unsw", "ton"):
        d = info[key]
        d["rows"] = sample(d["codes"], d["inv"], rng)
        t = time.time()
        raw[key] = pass_rows(d["path"], d["rows"], features)
        d["secs2"] = time.time() - t
        expect = np.array([d["inv"][c] for c in d["codes"][d["rows"]]])
        assert np.array_equal(raw[key]["Attack"].to_numpy(), expect), "pass-2 labels do not match pass 1"
        d["sample_counts"] = raw[key]["Attack"].value_counts().to_dict()
        print(f"{key}: sampled {len(d['rows']):,} rows ({d['secs2']:.0f}s)")
        del d["codes"]

    classes = sorted(info["unsw"]["counts"])
    ton_classes = sorted(info["ton"]["counts"])
    cidx, tidx = {c: i for i, c in enumerate(classes)}, {c: i for i, c in enumerate(ton_classes)}
    y_all = raw["unsw"]["Attack"].map(cidx).to_numpy(np.int64)
    pos = np.arange(len(y_all))
    tr, rest = train_test_split(pos, test_size=0.30, stratify=y_all, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y_all[rest], random_state=SEED)
    tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    assert len(np.intersect1d(tr, va)) == len(np.intersect1d(tr, te)) == len(np.intersect1d(va, te)) == 0
    assert len(tr) + len(va) + len(te) == len(pos)

    # ---- preprocessing (fit on NF-UNSW train only) ---------------------------------------------------------
    def numeric(df):
        a = df[features].to_numpy(np.float64)
        return np.where(np.isfinite(a), a, np.nan)

    A_unsw, A_ton = numeric(raw["unsw"]), numeric(raw["ton"])
    nonfinite = {"unsw": int(np.isnan(A_unsw).sum()), "ton": int(np.isnan(A_ton).sum())}  # NaN or inf in the CSV
    A = {"train": A_unsw[tr], "val": A_unsw[va], "test": A_unsw[te], "ton": A_ton}
    A_orig = {k: a.copy() for k, a in A.items()}  # before the overflow replacement, for the record

    def fit(a):
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(a)
        return imp, StandardScaler().fit(imp.transform(a))

    def overflow(a, imp, sc):  # raw values whose standardised value does not fit in float32
        z = sc.transform(imp.transform(a))
        return ~np.isfinite(z) | (np.abs(z) > F32_MAX)

    imputer, scaler = fit(A["train"])
    for _ in range(5):  # an overflow in train would change the fit itself: replace and refit until none remain
        m = overflow(A["train"], imputer, scaler)
        if not m.any():
            break
        A["train"][m] = np.nan
        imputer, scaler = fit(A["train"])
    else:
        sys.exit("ERROR: float32 overflow in NF-UNSW train persists after refitting")
    replaced = {"train": np.isnan(A["train"]) & ~np.isnan(A_orig["train"])}
    for k in ("val", "test", "ton"):
        replaced[k] = overflow(A[k], imputer, scaler)
        A[k][replaced[k]] = np.nan
    tf = lambda a: scaler.transform(imputer.transform(a)).astype(np.float32)
    X = {k: tf(A[k]) for k in ("train", "val", "test", "ton")}
    for k, a in X.items():
        if not np.isfinite(a).all():
            sys.exit(f"ERROR: non-finite values in processed {k} after float32 cast")
    y = {"train": y_all[tr], "val": y_all[va], "test": y_all[te]}
    y_ton = raw["ton"]["Attack"].map(tidx).to_numpy(np.int64)
    zero_var = [f for f, v in zip(features, scaler.var_) if v == 0]
    ovf = overflow_summary(A_orig, replaced, features, imputer, scaler, raw["ton"]["Attack"].to_numpy())
    print("float32-overflow replacements:", {k: (d["values_replaced"], d["rows_replaced"])
                                             for k, d in ovf["per_split"].items()}, "features:", ovf["affected_features"])

    # ---- save -----------------------------------------------------------------------------------------------
    data_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    for k, a in X.items():
        np.save(data_dir / f"X_{k}.npy", a)
    for k, a in y.items():
        np.save(data_dir / f"y_{k}.npy", a)
    np.save(data_dir / "y_ton.npy", y_ton)
    np.save(data_dir / "y_ton_label.npy", raw["ton"]["Label"].to_numpy(np.int8))
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_names": features}, data_dir / "preprocessor.joblib")
    for name, obj in (("feature_names.json", features), ("label_classes.json", classes),
                      ("ton_classes.json", ton_classes), ("label_mapping.json", mapping)):
        (data_dir / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    rows_u = info["unsw"]["rows"]
    np.savez(res_dir / "sample_indices.npz", unsw_train_rows=rows_u[tr], unsw_val_rows=rows_u[va],
             unsw_test_rows=rows_u[te], ton_rows=info["ton"]["rows"], seed=SEED, cap=CAP)

    split_counts = {k: {classes[c]: int(n) for c, n in zip(*np.unique(v, return_counts=True))} for k, v in y.items()}
    counts = {"files": {k: str(info[k]["path"]) for k in info}, "columns": len(columns), "features": features,
              "ttl_features": ttl, "seed": SEED, "cap": CAP, "mapping": mapping, "nonfinite_raw_in_sample": nonfinite,
              "zero_variance_in_train": zero_var, "float32_overflow_replaced": ovf,
              **{f"{k}_raw_counts": info[k]["counts"] for k in info},
              **{f"{k}_sample_counts": {c: int(n) for c, n in info[k]["sample_counts"].items()} for k in info},
              "unsw_split_counts": split_counts,
              "read_seconds": {k: [round(info[k]["secs"]), round(info[k]["secs2"])] for k in info}}
    (res_dir / "data_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    write_card(res_dir / "data_card.md", info, columns, features, ttl, classes, ton_classes, mapping,
               split_counts, nonfinite, zero_var, X, ovf)
    print(f"Saved {data_dir.relative_to(root)} and {res_dir.relative_to(root)}")


def overflow_summary(A_orig, replaced, features, imputer, scaler, ton_attack):
    """Per split and feature: values / rows replaced, max raw value; NF-ToN replaced rows per class."""
    split_name = {"train": "NF-UNSW train", "val": "NF-UNSW validation", "test": "NF-UNSW test", "ton": "NF-ToN"}
    affected = sorted({features[j] for m in replaced.values() for j in np.flatnonzero(m.any(axis=0))},
                      key=features.index)
    report = list(dict.fromkeys(RATE_FEATURES + affected))
    per_split = {}
    for k in ("train", "val", "test", "ton"):
        m = replaced[k]
        per_split[split_name[k]] = {
            "rows": int(len(m)), "values_replaced": int(m.sum()), "rows_replaced": int(m.any(axis=1).sum()),
            "per_feature": {f: {"values_replaced": int(m[:, features.index(f)].sum()),
                                "max_raw": float(np.nanmax(A_orig[k][:, features.index(f)])),
                                "median_raw": float(np.nanmedian(A_orig[k][:, features.index(f)]))}
                            for f in report}}
    cls, n = np.unique(ton_attack[replaced["ton"].any(axis=1)], return_counts=True)
    return {"rule": "raw value set to NaN (then median-imputed) when |standardised value| > float32 max",
            "float32_max": F32_MAX, "affected_features": affected, "reported_features": report,
            "train_scale": {f: {"mean": float(scaler.mean_[features.index(f)]),
                                "std": float(scaler.scale_[features.index(f)]),
                                "imputation_median": float(imputer.statistics_[features.index(f)])} for f in report},
            "per_split": per_split, "ton_rows_replaced_per_class": {str(c): int(x) for c, x in zip(cls, n)}}


def overflow_card(ovf):
    """Data-card section for the float32-overflow replacement."""
    g = lambda x: f"{x:.3g}"
    feats = ovf["reported_features"]
    L = ["### Corrupted byte-rate values (float32 overflow)", "",
         f"Rule: {ovf['rule']} ({ovf['float32_max']:.3g}). This extends the main pipeline's inf -> NaN -> median "
         "handling to values that would become inf when stored as float32. Imputation uses the NF-UNSW-train median. "
         f"Affected features: {', '.join(f'`{f}`' for f in ovf['affected_features']) or 'none'}.", "",
         "| Split | Rows | Values replaced | Rows replaced | "
         + " | ".join(f"`{f}` replaced | `{f}` max raw" for f in feats) + " |",
         "|---|---:|---:|---:|" + "---:|---:|" * len(feats)]
    for k, d in ovf["per_split"].items():
        L.append(f"| {k} | {d['rows']:,} | {d['values_replaced']:,} | {d['rows_replaced']:,} | "
                 + " | ".join(f"{d['per_feature'][f]['values_replaced']:,} | {g(d['per_feature'][f]['max_raw'])}"
                              for f in feats) + " |")
    L += ["", "NF-UNSW-train statistics of these features (raw units): " + "; ".join(
        f"`{f}` mean {g(s['mean'])}, std {g(s['std'])}, median = imputation value {g(s['imputation_median'])}"
        for f, s in ovf["train_scale"].items()) + ".", "",
          "NF-ToN rows replaced, by class: "
          + (", ".join(f"{c} {n}" for c, n in ovf["ton_rows_replaced_per_class"].items()) or "none") + ".", "",
          "These look like **corrupted byte-rate fields**: bytes per second computed over a near-zero flow duration, "
          "giving physically impossible rates (orders of magnitude beyond any link speed) while the medians are in "
          "the hundreds of bytes/s. A few such values already inflate the NF-UNSW-train std. Extreme values that are "
          "still representable in float32 are left unchanged: trees and rules compare against thresholds, and the "
          "stackers see probabilities, not features. Part C reports a sensitivity check with both features dropped.",
          ""]
    return L


def write_card(path, info, columns, features, ttl, classes, ton_classes, mapping, split, nonfinite, zero_var, X,
               ovf):
    u, t = info["unsw"], info["ton"]
    n_u, n_t = sum(u["counts"].values()), sum(t["counts"].values())
    ben_u = next(c for c in classes if c.lower() == "benign")
    small = [c for c in classes if split["test"].get(c, 0) < SMALL_TEST]
    mark = lambda c: f"{c} †" if c in small else c
    L = ["# NetFlow-v2 cross-dataset data card", "",
         f"Source (read in place, in chunks; nothing raw copied into the project): `{u['path']}` and `{t['path']}`.", "",
         "## Files", "",
         "| Dataset | Rows | Columns | Classes | Sampled rows |", "|---|---:|---:|---:|---:|",
         f"| NF-UNSW-NB15-v2 | {n_u:,} | {len(columns)} | {len(u['counts'])} | {sum(u['sample_counts'].values()):,} |",
         f"| NF-ToN-IoT-v2 | {n_t:,} | {len(columns)} | {len(t['counts'])} | {sum(t['sample_counts'].values()):,} |", "",
         "Both files have identical headers. `Label` agrees with `Attack != Benign` on every row of both files.", "",
         "## Sampling and rebalancing", "",
         f"Stratified sample, seed {SEED}: at most {CAP:,} rows per class, all rows of smaller classes. NF-UNSW sample "
         f"split 70/15/15 into train / validation / test (stratified, seed {SEED}); NF-ToN is test-only. Row indices "
         "(0-based data rows of the CSVs, header excluded) are in `results/external_netflow/sample_indices.npz`.", "",
         f"> **The cap rebalances the data.** Benign is {u['counts'][ben_u] / n_u:.1%} of raw NF-UNSW-NB15-v2 but "
         f"{u['sample_counts'][ben_u] / sum(u['sample_counts'].values()):.1%} of the sample; in NF-ToN-IoT-v2 Benign is "
         f"{t['counts'][next(c for c in ton_classes if c.lower() == 'benign')] / n_t:.1%} of the raw file and "
         f"{t['sample_counts'][next(c for c in ton_classes if c.lower() == 'benign')] / sum(t['sample_counts'].values()):.1%} "
         "of the sample. Accuracy, FPR and any pooled rate in this experiment are therefore comparable **within this "
         "experiment only**, not with prevalence-weighted results reported in the literature on the full files.", "",
         "### NF-UNSW-NB15-v2: raw, sample and per-split class counts", ""]
    n_s = sum(u["sample_counts"].values())
    L += ["| Class | Raw | Raw share | Sample | Sample share | Train | Validation | Test |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in classes:
        L.append(f"| {mark(c)} | {u['counts'][c]:,} | {u['counts'][c] / n_u:.2%} | {u['sample_counts'][c]:,} | "
                 f"{u['sample_counts'][c] / n_s:.2%} | {split['train'].get(c, 0):,} | {split['val'].get(c, 0):,} | "
                 f"{split['test'].get(c, 0):,} |")
    L.append(f"| **Total** | **{n_u:,}** | 100% | **{n_s:,}** | 100% | **{sum(split['train'].values()):,}** | "
             f"**{sum(split['val'].values()):,}** | **{sum(split['test'].values()):,}** |")
    L += ["", f"† fewer than {SMALL_TEST} test rows ({', '.join(f'{c}: {split['test'][c]}' for c in small) or 'none'}); "
          "marked in every results table, and no conclusion rests on these classes alone.", "",
          "### NF-ToN-IoT-v2: raw and sample class counts (test-only)", ""]
    L += counts_table(ton_classes, [("Raw", t["counts"]), ("Sample", t["sample_counts"])])
    L += ["", "## Features", "",
          f"{len(features)} features = the {len(columns)} shared NetFlow-v2 columns minus the flow identifiers "
          f"({', '.join(ID_COLUMNS)}) and the labels ({', '.join(LABEL_COLUMNS)}):", "",
          ", ".join(f"`{f}`" for f in features), "",
          f"TTL-related features (kept): {', '.join(f'`{f}`' for f in ttl)}. `MIN_TTL` / `MAX_TTL` are IP TTLs (the "
          "NetFlow counterpart of UNSW-NB15's sttl / dttl); `DNS_TTL_ANSWER` is the TTL of a DNS answer record, "
          "TTL-related by name only.", "",
          "## Preprocessing", "",
          "As in the main project's `UNSW_NB15_Pipeline`: inf -> NaN, median imputation, StandardScaler, all fitted on "
          "NF-UNSW **train only** and applied unchanged to validation, test and NF-ToN. All NetFlow columns are numeric; "
          "code-valued columns (PROTOCOL, L7_PROTO, TCP flag bitmasks, ICMP_TYPE, DNS_QUERY_TYPE, FTP_COMMAND_RET_CODE) "
          "are kept as numbers, so no categorical encoding is applied. Arrays stored as float32 in "
          f"`data/processed_netflow/` (train {X['train'].shape}, validation {X['val'].shape}, test {X['test'].shape}, "
          f"NF-ToN {X['ton'].shape}).", "",
          f"Non-finite raw values in the sample: NF-UNSW {nonfinite['unsw']}, NF-ToN {nonfinite['ton']}. "
          f"Zero-variance features in NF-UNSW train: {', '.join(zero_var) or 'none'}.", ""]
    L += overflow_card(ovf)
    L += ["## Label mapping for the cross-dataset test", "",
          "| NF-ToN-IoT-v2 | NF-UNSW-NB15-v2 |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in mapping.items()]
    unseen = [c for c in ton_classes if c not in mapping]
    L += ["", f"Every other NF-ToN class is an **unseen attack** (no NF-UNSW counterpart): {', '.join(unseen)}. "
          f"NF-UNSW classes with no NF-ToN counterpart: {', '.join(c for c in classes if c not in mapping.values())}."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
