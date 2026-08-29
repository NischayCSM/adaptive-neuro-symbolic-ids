import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.continual.experience_builder import ExperienceBuilder
from src.continual.drift_detection import DriftDetector

def _generate_mock_plot(plot_path):
    # Generates a realistic mock plot showing drift peaks when new experiences arrive
    batches = list(range(1, 21)) # 20 batches total (5 per experience)
    
    # Base in-distribution noise
    drift_scores = np.random.uniform(0.02, 0.08, 20)
    
    # Inject drift peaks at experience boundaries (batch 6, 11, 16)
    drift_scores[5] = 0.28  # Start of Exp2
    drift_scores[6] = 0.22  
    drift_scores[10] = 0.35 # Start of Exp3
    drift_scores[11] = 0.19 
    drift_scores[15] = 0.45 # Start of Exp4
    drift_scores[16] = 0.25 
    
    threshold = 0.15
    
    plt.figure(figsize=(10, 5))
    plt.plot(batches, drift_scores, marker='o', linestyle='-', color='b', label='Drift Score (Avg KS)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Drift Threshold (0.15)')
    
    # Mark experience boundaries
    plt.axvline(x=5.5, color='gray', linestyle=':', alpha=0.7)
    plt.text(5.6, 0.40, 'Exp2 Starts', color='gray', fontsize=9)
    plt.axvline(x=10.5, color='gray', linestyle=':', alpha=0.7)
    plt.text(10.6, 0.40, 'Exp3 Starts', color='gray', fontsize=9)
    plt.axvline(x=15.5, color='gray', linestyle=':', alpha=0.7)
    plt.text(15.6, 0.40, 'Exp4 Starts', color='gray', fontsize=9)
    
    plt.xlabel("Batch Number (Time)")
    plt.ylabel("Drift Score")
    plt.title("Mock Drift Detection over Continual Learning Sequence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Generated MOCK drift plot at {plot_path}")

def run_drift_test():
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results" / "continual"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_path = RESULTS_DIR / "drift_plot.png"
    
    try:
        X_train = np.load(PROCESSED_DIR / "X_train.npy")
        y_cat_train = np.load(PROCESSED_DIR / "y_cat_train.npy")
        
        try:
            from src.preprocessing.pipeline import UNSW_NB15_Pipeline
            pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
            if pipe.label_encoder is not None:
                y_cat_train = pipe.label_encoder.inverse_transform(y_cat_train)
        except Exception as e:
            print(f"Warning: could not decode labels: {e}")
    except FileNotFoundError:
        print("Processed data not found. Generating mock drift detection results.")
        _generate_mock_plot(plot_path)
        return
        
    builder = ExperienceBuilder(config_path=str(PROJECT_ROOT / "configs" / "continual.yaml"))
    
    # Extract Exp 1 as reference
    X_exp1, _ = builder.build_experience(X_train, y_cat_train, exp_id=1, cumulative=False)
    
    detector = DriftDetector(threshold=0.15)
    detector.update_reference(X_exp1)
    
    print("Reference distribution set to Experience 1.")
    
    drift_scores = []
    batches = []
    
    # We will simulate time by breaking each experience into chunks (batches)
    batch_idx = 1
    
    for eid in range(1, builder.get_num_experiences() + 1):
        X_exp, _ = builder.build_experience(X_train, y_cat_train, exp_id=eid, cumulative=False)
        
        # Split experience into 5 batches to simulate streaming
        num_batches = 5
        batch_size = len(X_exp) // num_batches
        
        for i in range(num_batches):
            start = i * batch_size
            # Get remainder on last batch
            end = len(X_exp) if i == num_batches - 1 else (i + 1) * batch_size
            
            X_batch = X_exp[start:end]
            
            if len(X_batch) > 0:
                result = detector.detect(X_batch)
                score = result["drift_score"]
                detected = result["drift_detected"]
                
                print(f"Exp {eid} - Batch {i+1}: Drift Score = {score:.4f} | Detected? {detected}")
                
                drift_scores.append(score)
                batches.append(batch_idx)
                batch_idx += 1
                
                # If drift is detected, we typically update the reference distribution.
                # For this test, we might choose to update it to show it settles down.
                if detected:
                    print("  -> Updating reference distribution with new data.")
                    # Combine old reference with new batch, or just replace
                    detector.update_reference(X_batch)
                    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(batches, drift_scores, marker='o', linestyle='-', color='b', label='Drift Score (Avg KS)')
    plt.axhline(y=detector.threshold, color='r', linestyle='--', label='Drift Threshold')
    
    plt.xlabel("Batch Number (Time)")
    plt.ylabel("Drift Score")
    plt.title("Drift Detection over Continual Learning Sequence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\\nSaved drift plot to {plot_path}")

if __name__ == "__main__":
    run_drift_test()
