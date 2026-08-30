# Adaptive Neuro-Symbolic Intrusion Detection System

A machine learning pipeline that automatically detects network attacks, learns human-readable symbolic rules from traffic patterns, and continuously adapts to new attack types without forgetting previously learned behaviours.

**Core research contribution:** Neuro-symbolic fusion does not improve raw accuracy over a strong XGBoost baseline, but provides 189 automatically-induced human-readable rules and dramatically reduces catastrophic forgetting (~90–94% knowledge retention vs ~4% for naive sequential retraining).

**Dataset:** [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) (primary) · [TON_IoT](https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset) (external validation)

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Project Structure](#project-structure)
5. [Running the Pipeline](#running-the-pipeline)
6. [Running Individual Stages](#running-individual-stages)
7. [Understanding the Output](#understanding-the-output)
8. [Key Results](#key-results)
9. [Troubleshooting](#troubleshooting)

---

## Requirements

### System
- Windows 10/11, macOS, or Linux
- Python **3.10–3.12** (Python 3.14 has known compatibility issues with `llvmlite`/`numba` — avoid it)
- At least **8 GB RAM** (16 GB recommended)
- At least **5 GB free disk space** (dataset + models + results)

### GPU (optional but strongly recommended)
- NVIDIA GPU with CUDA support — the pipeline auto-detects and uses it if available
- CUDA 11.8 or 12.x matching your driver
- Without a GPU the pipeline still runs on CPU — expect neural training to take 3–5× longer

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/NischayCSM/adaptive-neuro-symbolic-ids.git
cd adaptive-neuro-symbolic-ids
```

### Step 2 — Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 3 — Install PyTorch

Install PyTorch **before** the other requirements, matching your system:

**If you have an NVIDIA GPU (recommended):**

First check your CUDA version:
```bash
nvidia-smi
```
Then install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch
```

Verify GPU is accessible (should print `True` if you have an NVIDIA GPU with CUDA):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 4 — Install all other dependencies

```bash
pip install -r requirements.txt
```

> **Note on SHAP:** The `shap` package depends on `numba`/`llvmlite`. On some Windows systems with Application Control policies (managed/school machines), these DLLs may be blocked. If this happens, the pipeline degrades gracefully — it skips SHAP-based rule generation and continues with the other three methods (tree extraction, association rules, importance thresholding). You will see a warning like `SHAP rule generation unavailable... Skipping this generator.` This is expected and not an error.

### Step 5 — Install the project as a package

```bash
pip install -e .
```

This makes the `src/` modules importable from anywhere in the project, which is required for the pipeline to work.

---

## Dataset Setup

The pipeline requires two CSV files from the UNSW-NB15 dataset. These must be downloaded manually (Kaggle requires a free account).

### Primary dataset — UNSW-NB15 (required)

1. Go to: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2. Sign in or create a free Kaggle account
3. Click the **Download** button (down-arrow icon near the top right)
4. Extract the zip file
5. Find these two files and copy them into `data/raw/`:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

Your directory should look like:
```
data/
└── raw/
    ├── UNSW_NB15_training-set.csv   ← 82,332 rows
    └── UNSW_NB15_testing-set.csv    ← 175,341 rows
```

### External validation dataset — TON_IoT (optional)

Required only if you want to run the external generalization evaluation (Stage 5, external validation step). If you skip this, the pipeline aborts at that step but all other results are still produced.

1. Go to: https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset
2. Download and extract
3. Find the network flow CSV (often named `train_test_network.csv`)
4. Rename it to `TON_IoT_Network.csv` and place it in `data/external/`:

```
data/
└── external/
    └── TON_IoT_Network.csv
```

---

## Project Structure

```
adaptive-neuro-symbolic-ids/
│
├── data/
│   ├── raw/              ← Place downloaded CSVs here
│   ├── processed/        ← Auto-generated: scaled arrays, label encoder, pipeline
│   ├── batches/          ← Auto-generated: sequential experience splits
│   └── external/         ← Place TON_IoT_Network.csv here (optional)
│
├── src/
│   ├── preprocessing/    ← Data loading, cleaning, feature engineering
│   ├── models/           ← XGBoost, RandomForest, MLP baselines + NeuralIDS (PyTorch)
│   ├── rules/            ← Rule generators, validator, knowledge base, static extraction
│   ├── reasoning/        ← Symbolic reasoner and neuro-symbolic fuser
│   ├── continual/        ← Experience builder, drift detection, continual pipeline
│   └── evaluation/       ← All evaluation and reporting scripts
│
├── results/              ← Auto-generated: all metrics, plots, reports
├── experiments/          ← Experiment logs
├── configs/              ← Threshold and hyperparameter config files (YAML)
│
├── run.bat               ← Windows: run the full pipeline with one command
├── run.sh                ← macOS/Linux: run the full pipeline with one command
├── src/run_all.py        ← Orchestrator — controls stage order and dependencies
└── requirements.txt
```

---

## Running the Pipeline

### Full pipeline (recommended for first run)

**Windows:**
```bash
.\run.bat
```

**macOS / Linux:**
```bash
chmod +x run.sh
./run.sh
```

This runs all 6 stages in order and takes approximately **15–30 minutes** depending on your hardware (GPU significantly speeds up the neural training stages). A full run on a machine with an NVIDIA RTX GPU takes around 18–20 minutes.

When complete, all results are in `results/`. The console prints a summary at the end:
```
==================================================
ORCHESTRATOR SUMMARY
==================================================
Total Runtime: XXXX seconds
Stages Executed: preprocessing, baselines, milestone2, milestone3, milestone4
All outputs and reports are located in: results/
==================================================
```

### What each stage does

| Stage | Name | What runs | Key outputs |
|-------|------|-----------|-------------|
| 1–2 | Preprocessing | Loads UNSW-NB15, encodes features, scales, saves arrays | `data/processed/` |
| 3 | Baselines | Trains RandomForest, XGBoost, MLP, NeuralIDS | `results/models/`, `results/neural_model/` |
| 4 | Milestone 2 | Static rule extraction, XGBoost + fusion evaluation | `results/milestone2_comparison.csv` |
| 5 | Milestone 3 | Continual pipeline (4 experiences), forgetting experiments, ablation, external validation | `results/continual/`, `results/milestone3_*.csv` |
| 6 | Milestone 4 | CLI demo (5 batches), final 8-test master report | `results/demo_run.jsonl`, `results/final_test_report.md` |

---

## Running Individual Stages

If the full pipeline already ran and you only want to re-run one stage (e.g. after changing a config value):

```bash
# Windows
.\run.bat --stage milestone3

# macOS/Linux
./run.sh --stage milestone3
```

Valid stage names: `preprocessing`, `baselines`, `milestone2`, `milestone3`, `milestone4`

You can also run individual scripts directly. From the project root:

```bash
# Re-run only the ablation study
python -m src.evaluation.run_ablation_study

# Re-run only the forgetting experiment
python -m src.evaluation.run_milestone3_forgetting

# Re-run only the CLI demo
python -m src.evaluation.run_demo --batches 10 --speed 0.5

# Re-run the final test report
python -m src.evaluation.final_tests
```

> Always use `python -m src.module.name` (module syntax) rather than `python src/module/name.py` (file syntax) to ensure imports resolve correctly.

---

## Understanding the Output

After a full run, `results/` contains:

```
results/
├── models/                        ← Trained sklearn/XGBoost model files
├── neural_model/
│   └── neural_ids_weights.pth     ← Trained PyTorch NeuralIDS weights
│
├── milestone2_comparison.csv      ← XGBoost Only vs XGBoost + Fusion (static)
├── milestone2_summary.md          ← Human-readable summary of Milestone 2
│
├── continual/
│   ├── sequential_finetune.csv    ← Per-experience forgetting table (naive baseline)
│   └── continual_pipeline_metrics.csv ← Rule counts and drift flags per experience
│
├── milestone3_comparison.csv      ← 4-column experience table (static/continual × neural/neuro-symbolic)
├── milestone3_forgetting.csv      ← Same as sequential_finetune.csv (copied for convenience)
├── milestone3_forgetting.png      ← Forgetting curve plot
├── milestone3_ablation.csv        ← 3-point alpha ablation (0.0, 0.5, 1.0)
├── milestone3_ablation_sweep.csv  ← Full 7-point sweep (0.0 through 1.0)
├── milestone3_interpretation.md   ← Auto-generated written interpretation
│
├── external_validation.md         ← TON_IoT generalization results and analysis
│
├── demo_run.jsonl                 ← Per-batch CLI demo log (JSON lines)
├── demo_summary.md                ← Human-readable demo summary
│
└── final_test_report.md           ← Master 8-test scorecard with PASS/FAIL/CONDITIONAL
```

### The most important files to read first

1. **`final_test_report.md`** — The overall verdict across 8 practical tests
2. **`milestone3_ablation_sweep.csv`** — The definitive alpha sweep result
3. **`milestone3_comparison.csv`** — Side-by-side comparison of all 4 system variants across all 4 experiences
4. **`external_validation.md`** — What happens when the trained system sees a completely different network environment

---

## Key Results

These are the verified results from the final run on UNSW-NB15:

| Metric | Value |
|--------|-------|
| Rules automatically induced | **189 active rules** (precision ≥ 0.85) |
| Knowledge retention — adaptive system | **~90–94%** on old experiences after learning new attacks |
| Knowledge retention — naive baseline | **~4%** (catastrophic forgetting) |
| XGBoost Only accuracy (10-class) | **76.29%** / Macro-F1 0.525 |
| XGBoost + Fusion, best alpha=0.9 | **76.30%** / Macro-F1 0.524 |
| Scorecard | **6/8 PASS**, 1 FAIL (explained), 1 CONDITIONAL |

**Honest summary:** Neuro-symbolic fusion does not improve raw accuracy over the XGBoost baseline at any blend ratio (the maximum gap across a 7-point alpha sweep is 0.01 percentage points — noise). Its real contribution is the 189 automatically-induced, human-readable detection rules (providing interpretability) and the dramatic reduction in catastrophic forgetting when learning new attack types continuously.

### Alpha sweep (full result)

| Alpha | Config | Accuracy | Macro-F1 |
|-------|--------|----------|----------|
| 0.0 | Symbolic Only | 28.3% | 0.145 |
| 0.5 | Neuro-Symbolic | 75.87% | 0.447 |
| 0.6 | Neuro-Symbolic | 76.08% | 0.493 |
| 0.7 | Neuro-Symbolic | 76.19% | 0.510 |
| 0.8 | Neuro-Symbolic | 76.26% | 0.517 |
| 0.9 | Neuro-Symbolic | 76.30% | 0.524 |
| 1.0 | Neural Only | 76.29% | 0.525 |

---

## Troubleshooting

### "Dataset not found" on first run
You need to manually download the CSVs from Kaggle — see [Dataset Setup](#dataset-setup). The pipeline cannot download them automatically because Kaggle requires authentication.

### `ModuleNotFoundError: No module named 'src'`
You're running a script as a file path instead of a module. Use:
```bash
python -m src.evaluation.run_ablation_study   # correct
python src/evaluation/run_ablation_study.py   # incorrect
```
Also make sure you ran `pip install -e .` in Step 5 of installation.

### SHAP blocked by Application Control (Windows)
If you see `SHAP rule generation unavailable: DLL load failed`, your system's security policy is blocking `llvmlite.dll`. The pipeline continues without SHAP — this only removes one of four rule generators. No action needed; results are still valid.

### `PermissionError (WinError 32)` on `continual_kb.db`
This is a Windows file-lock issue, most commonly caused by OneDrive syncing the file while the pipeline is also writing to it. To fix:
1. Pause OneDrive sync temporarily (right-click the OneDrive icon in the system tray → Pause syncing)
2. Close any SQLite browser or DB viewer you have open
3. Re-run the pipeline

If you move the project folder outside of OneDrive (e.g. `C:\Projects\` instead of `C:\Users\...\OneDrive\...`) this problem will not occur.

### `torch.cuda.is_available()` returns `False`
Your PyTorch install is the CPU-only version. Re-install with the CUDA build matching your driver (see Step 3 of Installation). Check your driver version with `nvidia-smi` first.

### XGBoost `use_label_encoder` warning
```
UserWarning: Parameters: { "use_label_encoder" } are not used.
```
This is a harmless deprecation warning from newer XGBoost versions. It does not affect results.

### Pipeline aborts at external validation
If you have not placed `TON_IoT_Network.csv` in `data/external/`, the external validation step correctly aborts with a clear error. All Milestone 2 and 3 results are still fully produced — this only affects the final validation section. Place the file and re-run `.\run.bat --stage milestone3` to complete it.

### Results look suspicious / all identical numbers
The pipeline has hard-fail checks: if a required input file is missing, scripts exit with a non-zero code and a clear error message rather than substituting fake/mock data. If you see an `ERROR:` line in the console, fix that step before interpreting downstream results.

---

## Configuration

Key thresholds and hyperparameters are in `configs/`:

| File | Controls |
|------|---------|
| `configs/rules.yaml` | Rule scoring weights (precision 0.4, coverage 0.3, stability 0.2, conflict 0.2), acceptance threshold (0.6), precision floor (0.85), coverage cap (0.5 for scoring) |
| `configs/continual.yaml` | Experience-to-class mapping, drift detection threshold, replay buffer size |

Changes here take effect on the next run without modifying any Python code.

---

## Citation

If you use this codebase or its results in your work, please cite:

```
Adaptive Neuro-Symbolic Intrusion Detection System:
Bridging Interpretability and Continual Learning
UNSW-NB15 Dataset · TON_IoT External Validation · 2026
https://github.com/NischayCSM/adaptive-neuro-symbolic-ids
```