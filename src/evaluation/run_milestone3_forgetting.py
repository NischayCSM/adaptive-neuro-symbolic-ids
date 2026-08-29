import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def run_milestone3():
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = RESULTS_DIR / "milestone3_forgetting.csv"
    plot_path = RESULTS_DIR / "milestone3_forgetting.png"
    
    if not (PROCESSED_DIR / "X_train.npy").exists():
        print("ERROR: Required data file 'data/processed/X_train.npy' not found.", file=sys.stderr)
        print("Please run 'python src/preprocessing/run_pipeline.py' first to generate this file.", file=sys.stderr)
        sys.exit(1)
        
    from src.continual.sequential_finetune import run_sequential_finetune
    
    print("Running sequential fine-tuning to evaluate catastrophic forgetting...")
    run_sequential_finetune()
    
    seq_csv = RESULTS_DIR / "continual" / "sequential_finetune.csv"
    if seq_csv.exists():
        df = pd.read_csv(seq_csv)
        df.to_csv(csv_path, index=False)
        print(f"Copied results to {csv_path}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        phases = df["Phase"]
        for col in df.columns:
            if "_Acc" in col:
                plt.plot(phases, df[col], marker='o', label=col.replace("_Acc", " Accuracy"))
                
        plt.title("Catastrophic Forgetting during Sequential Fine-Tuning")
        plt.xlabel("Training Phase")
        plt.ylabel("Accuracy on Test Data")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_path)
        print(f"Generated forgetting plot at {plot_path}")
    else:
        print("ERROR: sequential_finetune.csv was not generated.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_milestone3()
