import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, expected_output_file=None, dry_run=False):
    if dry_run:
        print(f"  [DRY-RUN] Would execute: {' '.join(cmd)}")
        return True
    else:
        print(f"  Executing: {' '.join(cmd)}")
        
        initial_mtime = None
        if expected_output_file:
            path = Path(expected_output_file)
            if path.exists():
                initial_mtime = os.path.getmtime(path)
                
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            print(f"ERROR: Command failed with exit code {result.returncode}: {' '.join(cmd)}", file=sys.stderr)
            return False
            
        if expected_output_file:
            if not path.exists():
                print(f"ERROR: Expected output file {path} was not created.", file=sys.stderr)
                return False
            new_mtime = os.path.getmtime(path)
            if initial_mtime is not None and new_mtime <= initial_mtime:
                print(f"ERROR: Expected output file {path} was not modified (mtime unchanged).", file=sys.stderr)
                return False
                
        return True

def run_preprocessing(project_root, dry_run=False):
    print("\n" + "="*50)
    print("STAGE 1 & 2: Preprocessing")
    print("="*50)
    
    # 1. Download Data
    raw_dir = project_root / "data" / "raw"
    if (raw_dir / "UNSW_NB15_training-set.csv").exists():
        print("  [Stage 1] data/raw already populated. Skipping download_data.py.")
    else:
        cmd1 = [sys.executable, str(project_root / "src" / "preprocessing" / "download_data.py")]
        if not run_command(cmd1, dry_run):
            return False
            
    # 2. Run Pipeline
    if not (raw_dir / "UNSW_NB15_training-set.csv").exists() and not dry_run:
        print("ERROR: Missing required input 'data/raw/UNSW_NB15_training-set.csv' for run_pipeline.py.", file=sys.stderr)
        print("Please ensure download_data.py succeeded.", file=sys.stderr)
        return False
        
    cmd2 = [sys.executable, str(project_root / "src" / "preprocessing" / "run_pipeline.py")]
    if not run_command(cmd2, dry_run):
        return False
        
    print("[Stage 1 & 2 Complete] Preprocessing finished. Datasets saved to data/processed/")
    return True

def run_baselines(project_root, dry_run=False):
    print("\n" + "="*50)
    print("STAGE 3: Milestone 1 - Baselines")
    print("="*50)
    
    processed_dir = project_root / "data" / "processed"
    if not (processed_dir / "X_train.npy").exists() and not dry_run:
        print("ERROR: Missing required input 'data/processed/X_train.npy' for baseline training.", file=sys.stderr)
        print("Please run the 'preprocessing' stage first.", file=sys.stderr)
        return False
        
    cmd = [sys.executable, str(project_root / "src" / "models" / "baselines.py")]
    if not run_command(cmd, dry_run):
        return False
        
    print("[Stage 3 Complete] Baselines evaluated.")
    return True

def run_milestone2(project_root, dry_run=False):
    print("\n" + "="*50)
    print("STAGE 4: Milestone 2 - Static Rule Extraction & Fusion")
    print("="*50)
    
    processed_dir = project_root / "data" / "processed"
    if not (processed_dir / "X_test.npy").exists() and not dry_run:
        print("ERROR: Missing required input 'data/processed/X_test.npy' for Milestone 2.", file=sys.stderr)
        print("Please run the 'preprocessing' stage first.", file=sys.stderr)
        return False
        
    cmd_extract = [sys.executable, str(project_root / "src" / "rules" / "static_rule_extraction.py")]
    if not run_command(cmd_extract, dry_run):
        return False
        
    cmd_eval = [sys.executable, str(project_root / "src" / "evaluation" / "evaluate_milestone2.py")]
    if not run_command(cmd_eval, dry_run=dry_run):
        return False
        
    print("[Stage 4 Complete] Milestone 2 static rule extraction and static comparison finished.")
    return True

def run_milestone3(project_root, dry_run=False):
    print("\n" + "="*50)
    print("STAGE 5: Milestone 3 - Continual Learning & Evaluation")
    print("="*50)
    
    processed_dir = project_root / "data" / "processed"
    if not (processed_dir / "X_train.npy").exists() and not dry_run:
        print("ERROR: Missing required input 'data/processed/X_train.npy' for Milestone 3.", file=sys.stderr)
        print("Please run the 'preprocessing' stage first.", file=sys.stderr)
        return False
        
    scripts = [
        "src/continual/continual_pipeline.py",
        "src/continual/replay_finetune.py",
        "src/evaluation/run_milestone3_forgetting.py",
        "src/evaluation/run_final_comparison.py",
        "src/evaluation/run_ablation_study.py",
        "src/evaluation/run_external_validation.py"
    ]
    
    for script in scripts:
        cmd = [sys.executable, str(project_root / script)]
        expected_file = None
        if "replay_finetune.py" in script:
            expected_file = str(project_root / "results" / "continual" / "replay_seed42.csv")
        if "run_external_validation.py" in script:
            expected_file = str(project_root / "results" / "external_validation.md")
        if not run_command(cmd, expected_output_file=expected_file, dry_run=dry_run):
            return False
            
    print("[Stage 5 Complete] Continual learning pipelines and Milestone 3 evaluations finished.")
    return True

def run_milestone4(project_root, dry_run=False):
    print("\n" + "="*50)
    print("STAGE 6: Milestone 4 - Final Demo & Master Tests")
    print("="*50)
    
    processed_dir = project_root / "data" / "processed"
    if not (processed_dir / "X_test.npy").exists() and not dry_run:
        print("ERROR: Missing required input 'data/processed/X_test.npy' for Milestone 4.", file=sys.stderr)
        print("Please run the 'preprocessing' stage first.", file=sys.stderr)
        return False
        
    scripts = [
        ["src/evaluation/run_demo.py", "--batches", "5", "--speed", "0"],
        ["src/evaluation/final_tests.py"]
    ]
    
    for script_args in scripts:
        cmd = [sys.executable, str(project_root / script_args[0])] + script_args[1:]
        expected_file = None
        if "run_demo.py" in script_args[0]:
            expected_file = str(project_root / "results" / "demo_run.jsonl")
        elif "final_tests.py" in script_args[0]:
            expected_file = str(project_root / "results" / "final_test_report.md")
            
        if not run_command(cmd, expected_output_file=expected_file, dry_run=dry_run):
            return False
            
    print("[Stage 6 Complete] Final tests and demo execution finished.")
    return True

def main():
    parser = argparse.ArgumentParser(description="End-to-End Orchestrator for Adaptive Neuro-Symbolic IDS")
    parser.add_argument("--stage", type=str, choices=["all", "preprocessing", "baselines", "milestone2", "milestone3", "milestone4"], default="all", help="Specific stage to run")
    parser.add_argument("--dry-run", action="store_true", help="Print the order and check dependencies without executing")
    parser.add_argument("--resume", action="store_true", help="Skip stages that appear to be already complete based on output files")
    args = parser.parse_args()
    
    project_root = Path(os.getcwd())
    if project_root.name != "adaptive-neuro-symbolic-ids":
        for parent in project_root.parents:
            if parent.name == "adaptive-neuro-symbolic-ids":
                project_root = parent
                break
                
    # Force PYTHONPATH to project root so imports work
    os.environ["PYTHONPATH"] = str(project_root)
    
    stages_to_run = []
    if args.stage == "all":
        stages_to_run = ["preprocessing", "baselines", "milestone2", "milestone3", "milestone4"]
    else:
        stages_to_run = [args.stage]
        
    start_time = time.time()
    ran_stages = []
    skipped_stages = []
    
    results_dir = project_root / "results"
    
    for stage in stages_to_run:
        success = True
        
        if stage == "preprocessing":
            if args.resume and (project_root / "data" / "processed" / "X_train.npy").exists():
                print("Skipping preprocessing (--resume): data/processed/X_train.npy exists.")
                skipped_stages.append(stage)
                continue
            success = run_preprocessing(project_root, args.dry_run)
            
        elif stage == "baselines":
            success = run_baselines(project_root, args.dry_run)
            
        elif stage == "milestone2":
            if args.resume and (results_dir / "milestone2_summary.md").exists():
                print("Skipping milestone2 (--resume): results/milestone2_summary.md exists.")
                skipped_stages.append(stage)
                continue
            success = run_milestone2(project_root, args.dry_run)
            
        elif stage == "milestone3":
            if args.resume and (results_dir / "milestone3_forgetting.csv").exists():
                print("Skipping milestone3 (--resume): results/milestone3_forgetting.csv exists.")
                skipped_stages.append(stage)
                continue
            success = run_milestone3(project_root, args.dry_run)
            
        elif stage == "milestone4":
            if args.resume and (results_dir / "final_test_report.md").exists():
                print("Skipping milestone4 (--resume): results/final_test_report.md exists.")
                skipped_stages.append(stage)
                continue
            success = run_milestone4(project_root, args.dry_run)
            
        if success:
            ran_stages.append(stage)
        else:
            print(f"\n[!] Pipeline aborted at stage: {stage}")
            sys.exit(1)
            
    end_time = time.time()
    
    print("\n" + "="*50)
    print("ORCHESTRATOR SUMMARY")
    print("="*50)
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")
    print(f"Stages Executed: {', '.join(ran_stages) if ran_stages else 'None'}")
    print(f"Stages Skipped: {', '.join(skipped_stages) if skipped_stages else 'None'}")
    print(f"All outputs and reports are located in: {results_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
