import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.device import DEVICE, print_device_info
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from src.models.neural_ids import NeuralIDS
from src.continual.experience_builder import ExperienceBuilder

def run_sequential_finetune():
    PROJECT_ROOT = Path(os.getcwd())
    if PROJECT_ROOT.name != "adaptive-neuro-symbolic-ids":
        for parent in PROJECT_ROOT.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                PROJECT_ROOT = parent
                break
                
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results" / "continual"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_out = RESULTS_DIR / "sequential_finetune.csv"
    
    ckpt_dir = RESULTS_DIR / "continual_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    
    try:
        X_train = np.load(PROCESSED_DIR / "X_train.npy")
        y_cat_train = np.load(PROCESSED_DIR / "y_cat_train.npy")
        X_test = np.load(PROCESSED_DIR / "X_test.npy")
        y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy")
        
        try:
            from src.preprocessing.pipeline import UNSW_NB15_Pipeline
            pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")
            if pipe.label_encoder is not None:
                y_cat_train = pipe.label_encoder.inverse_transform(y_cat_train)
                y_cat_test = pipe.label_encoder.inverse_transform(y_cat_test)
        except Exception as e:
            print(f"Warning: could not decode labels: {e}")

    except FileNotFoundError as e:
        print(f"ERROR: processed data not found ({e}). Run 'python src/preprocessing/run_pipeline.py' first.", file=sys.stderr)
        sys.exit(1)
        
    print("Loaded data successfully.")
    
    builder = ExperienceBuilder(config_path=str(PROJECT_ROOT / "configs" / "continual.yaml"))
    num_exps = builder.get_num_experiences()
    
    # Load canonical class list (single source of truth)
    import json
    classes_path = PROCESSED_DIR / "label_classes.json"
    if not classes_path.exists():
        print("ERROR: label_classes.json not found. Re-run preprocessing.", file=sys.stderr)
        sys.exit(1)
    with open(classes_path) as f:
        all_classes = json.load(f)
    num_classes = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    y_train_idx = np.array([class_to_idx[c] for c in y_cat_train])
    y_test_idx = np.array([class_to_idx[c] for c in y_cat_test])
    
    device = DEVICE
    print_device_info()
    model = NeuralIDS(input_dim=X_train.shape[1], output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    
    # Sequential training loop
    for eid in range(1, num_exps + 1):
        print(f"\\n--- Training on Experience {eid} ---")
        
        # Get data for current experience ONLY
        X_exp, y_exp_cat = builder.build_experience(X_train, y_cat_train, exp_id=eid, cumulative=False)
        y_exp = np.array([class_to_idx[c] for c in y_exp_cat])
        
        X_tensor = torch.tensor(X_exp, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_exp, dtype=torch.long).to(device)
        
        # Create simple DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        model.train()
        num_epochs = 5
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")
            
        # Save checkpoint
        checkpoint_path = ckpt_dir / f"neural_ids_exp_{eid}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
            
        # Evaluation step: Evaluate on ALL test sets seen so far
        print(f"Evaluating after Experience {eid}...")
        model.eval()
        
        row_res = {"Phase": f"After Exp{eid}"}
        
        for eval_eid in range(1, num_exps + 1):
            X_test_k, y_test_k_cat = builder.build_experience(X_test, y_cat_test, exp_id=eval_eid, cumulative=False)
            
            if len(X_test_k) > 0:
                y_test_k = np.array([class_to_idx[c] for c in y_test_k_cat])
                
                with torch.no_grad():
                    logits = model(torch.tensor(X_test_k, dtype=torch.float32).to(device))
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    
                acc = accuracy_score(y_test_k, preds)
                f1 = f1_score(y_test_k, preds, average='macro', zero_division=0)
            else:
                acc = 0.0
                f1 = 0.0
                
            row_res[f"Exp{eval_eid}_Acc"] = acc
            row_res[f"Exp{eval_eid}_F1"] = f1
                
        results.append(row_res)
        
    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    print(f"\\nSequential fine-tuning complete. Results saved to {csv_out}")
    print(df.to_string(index=False))
    
if __name__ == "__main__":
    run_sequential_finetune()
