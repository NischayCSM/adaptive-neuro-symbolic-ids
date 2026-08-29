# Adaptive Neuro-Symbolic Intrusion Detection System (NSIDS)

**Adaptive Neuro-Symbolic Intrusion Detection with Automatic Rule Induction and Continual Learning**

This project combines neural network–based anomaly detection with symbolic reasoning to build an intrusion detection system (IDS) that:

- Learns complex traffic patterns from raw network data (neural component).
- Automatically induces human-interpretable detection rules from learned representations (symbolic component).
- Continuously adapts to emerging attack types through continual / incremental learning.

## Folder Layout

```
adaptive-neuro-symbolic-ids/
├── data/
│   ├── raw/              # Original, unprocessed datasets
│   ├── processed/        # Cleaned & feature-engineered datasets
│   └── batches/          # Incremental data batches for continual learning
├── notebooks/            # Exploratory / analysis Jupyter notebooks
├── src/                  # Main source package ("nsids")
│   ├── preprocessing/    # Data loading, cleaning, feature engineering
│   ├── models/           # Neural and ensemble model definitions
│   ├── rules/            # Symbolic rule representations & induction
│   ├── reasoning/        # Neuro-symbolic integration & inference
│   ├── continual/        # Continual / incremental learning strategies
│   └── evaluation/       # Metrics, reporting, and visualization
├── experiments/          # Experiment scripts & orchestration
├── results/              # Saved outputs, figures, and logs
├── configs/              # YAML / JSON configuration files
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Package metadata (importable as "nsids")
└── README.md             # This file
```

## Running the Demo

The final project demo is a command-line script that replays test batches through the full adaptive neuro-symbolic pipeline. It simulates real-time network stream monitoring without requiring any web server or GUI framework.

```bash
python src/evaluation/run_demo.py --batches 20 --speed 0.5
```
This logs the telemetry (traffic volume, latest detections, and Knowledge Base rules added/retired) to the console, and writes the structured results to `results/demo_run.jsonl` and `results/demo_summary.md`.

## Quick Start

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

## License

TBD
