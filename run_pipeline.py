"""
Complete Pipeline Runner for LLM Data Extraction and MIA Evaluation.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

from config import load_config, Config, save_config, get_default_extraction_config, get_default_mia_config


def run_command(command: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print('='*60)
    
    result = subprocess.run(command, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"Success: {description} completed")


def main():
    parser = argparse.ArgumentParser(description="Complete pipeline for LLM data extraction and MIA evaluation")
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Pipeline control
    parser.add_argument('--skip_extraction', action='store_true', help='Skip extraction step')
    parser.add_argument('--skip_mia', action='store_true', help='Skip MIA evaluation step')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--model', type=str, help='Model name or path')
    
    # Generation settings
    parser.add_argument('--num_trials', type=int, help='Number of generation trials')
    parser.add_argument('--val_set_num', type=int, help='Number of validation examples')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    
    # Checkpoint settings
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--force_restart', action='store_true', help='Force restart by removing checkpoint')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(args.config, args)
        print(f"Loaded configuration from {args.config}")
    else:
        print("No config file provided, using defaults with command-line overrides")
        config_dict = get_default_extraction_config()
        config = Config(config_dict)
        
        # Apply command-line overrides
        if args.experiment_name:
            config.experiment.name = args.experiment_name
        if args.dataset_dir:
            config.experiment.dataset_dir = args.dataset_dir
        if args.model:
            config.model.name = args.model
        if args.num_trials:
            config.generation.num_trials = args.num_trials
        if args.val_set_num:
            config.generation.val_set_num = args.val_set_num
        if args.batch_size:
            config.generation.batch_size = args.batch_size
        if args.resume:
            config.checkpoint.resume = True
        if args.force_restart:
            config.checkpoint.force_restart = True
    
    print("\nLLM Data Extraction and MIA Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.experiment.dataset_dir}")
    print(f"Experiment: {config.experiment.name}")
    print(f"Validation set size: {config.generation.val_set_num}")
    print(f"Number of trials: {config.generation.num_trials}")
    print(f"Checkpointing: {'Enabled' if config.checkpoint.save_checkpoints else 'Disabled'}")
    print(f"Resume: {'Enabled' if config.checkpoint.resume else 'Disabled'}")
    
    # Check dataset files
    dataset_path = Path(config.experiment.dataset_dir)
    required_files = ["train_prefix.npy", "train_dataset.npy"]
    
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            print(f"Error: Required file {file_name} not found in {config.experiment.dataset_dir}")
            sys.exit(1)
    
    # Handle checkpoint management
    experiment_path = Path("results") / config.experiment.name
    checkpoint_path = experiment_path / "checkpoint.pkl"
    
    if config.checkpoint.force_restart and checkpoint_path.exists():
        print("Force restart requested - removing existing checkpoint")
        checkpoint_path.unlink()
    
    if checkpoint_path.exists() and not config.checkpoint.resume:
        print(f"Warning: Checkpoint found at {checkpoint_path}")
        print("Use --resume to continue or --force_restart to start fresh")
    
    # Save config for this run
    os.makedirs(experiment_path, exist_ok=True)
    config_save_path = experiment_path / "pipeline_config.yaml"
    save_config(config, str(config_save_path))
    print(f"\nSaved pipeline configuration to {config_save_path}")
    
    # Step 1: Data Extraction
    if not args.skip_extraction:
        extraction_command = [
            sys.executable, "extract.py",
            "--config", str(config_save_path)
        ]
        
        run_command(extraction_command, "Data Extraction")
    
    # Step 2: MIA Evaluation
    if not args.skip_mia:
        guess_dir = os.path.join("results", config.experiment.name, "guess_files")
        
        if not os.path.exists(guess_dir):
            print(f"Error: Guess files directory not found at {guess_dir}")
            sys.exit(1)
        
        csv_files = list(Path(guess_dir).glob("*.csv"))
        if not csv_files:
            print(f"Error: No guess CSV files found in {guess_dir}")
            sys.exit(1)
        
        # Create MIA config
        mia_config_dict = get_default_mia_config()
        mia_config = Config(mia_config_dict)
        mia_config.model.name = config.model.name
        mia_config.experiment.dataset_dir = config.experiment.dataset_dir
        mia_config.evaluation.guess_dir = guess_dir
        mia_config.evaluation.batch_size = config.generation.batch_size
        
        mia_config_path = experiment_path / "mia_config.yaml"
        save_config(mia_config, str(mia_config_path))
        
        mia_command = [
            sys.executable, "evaluate_mia.py",
            "--config", str(mia_config_path)
        ]
        
        run_command(mia_command, "MIA Evaluation")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Summary
    print("\nGenerated outputs:")
    
    if not args.skip_extraction:
        if experiment_path.exists():
            print(f"- Experiment data: {experiment_path}")
            
            summary_csv = experiment_path / "extraction_metrics_summary.csv"
            if summary_csv.exists():
                print(f"- Extraction metrics: {summary_csv}")
            
            checkpoint_file = experiment_path / "checkpoint.pkl"
            if checkpoint_file.exists():
                print(f"- Checkpoint still exists: {checkpoint_file}")
                print("  (Extraction may not have completed)")
    
    if not args.skip_mia:
        mia_results_path = experiment_path / "mia_evaluation"
        if mia_results_path.exists():
            result_files = list(mia_results_path.glob("*.csv"))
            if result_files:
                print(f"- MIA evaluation results: {mia_results_path}")
                for result_file in result_files:
                    print(f"    - {result_file.name}")
    
    guess_path = experiment_path / "guess_files"
    if guess_path.exists():
        guess_files = list(guess_path.glob("*.csv"))
        if guess_files:
            print(f"- Guess files: {guess_path}")
            print(f"  - CSV files: {len(guess_files)}")


if __name__ == "__main__":
    main()
