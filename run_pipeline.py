#!/usr/bin/env python3
"""
Complete Pipeline Runner for LLM Data Extraction and MIA Evaluation with Checkpoint Support.

This script runs the complete pipeline:
1. Data extraction with multiple scoring methods (with checkpointing)
2. MIA evaluation of the extracted data
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path


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
    parser = argparse.ArgumentParser(
        description="Complete pipeline for LLM data extraction and MIA evaluation with checkpoint support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file (NEW)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (overrides other arguments)')
    
    # Pipeline control
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip extraction step and only run MIA evaluation')
    parser.add_argument('--skip_mia', action='store_true',
                       help='Skip MIA evaluation step and only run extraction')
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default="../datasets",
                       help='Path to dataset directory')
    parser.add_argument('--experiment_name', type=str, default='pipeline_experiment',
                       help='Name of the experiment')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B',
                       help='Model name or path')
    
    # Generation arguments
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of generation trials per prompt')
    parser.add_argument('--val_set_num', type=int, default=1000,
                       help='Number of validation examples to use')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k for generation')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation')
    parser.add_argument('--typical', type=float, default=1.0,
                       help='Typical p for generation for generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                       help='Repetition penalty for generation')
    
    # Saving arguments
    parser.add_argument('--save_all_generations_per_prompt', action='store_true',
                       help='Save guess CSVs for all generation count tiers (1, 5, 10, etc.)')
    parser.add_argument('--save_all_methods', action='store_true',
                       help='Save guess CSVs for all scoring methods, not just likelihood')
    parser.add_argument('--save_npy_files', action='store_true',
                       help='Save intermediate generation and loss .npy files')

    # Checkpoint arguments (NEW)
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='Save checkpoints after each trial for resumability')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoint if available')
    parser.add_argument('--force_restart', action='store_true',
                       help='Force restart by removing existing checkpoints')

    # Other arguments
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration from YAML if provided
    if args.config:
        try:
            from metric_loader import load_config
            config = load_config(args.config)
            
            # Override args with config values (command-line args take precedence)
            if 'global' in config:
                global_config = config['global']
                
                # Only override if not explicitly set via command line
                if args.dataset_dir == "../datasets" and 'dataset_dir' in global_config:
                    args.dataset_dir = global_config['dataset_dir']
                if args.model == 'EleutherAI/gpt-neo-1.3B' and 'model' in global_config:
                    args.model = global_config['model']
                if args.seed == 2022 and 'seed' in global_config:
                    args.seed = global_config['seed']
                if args.batch_size == 64 and 'batch_size' in global_config:
                    args.batch_size = global_config['batch_size']
                
                # Generation parameters
                if 'generation' in global_config:
                    gen_config = global_config['generation']
                    if args.num_trials == 5 and 'num_trials' in gen_config:
                        args.num_trials = gen_config['num_trials']
                    if args.val_set_num == 1000 and 'val_set_num' in gen_config:
                        args.val_set_num = gen_config['val_set_num']
                    if args.top_k == 50 and 'top_k' in gen_config:
                        args.top_k = gen_config['top_k']
                    if args.top_p == 1.0 and 'top_p' in gen_config:
                        args.top_p = gen_config['top_p']
                    if args.temperature == 1.0 and 'temperature' in gen_config:
                        args.temperature = gen_config['temperature']
                    if args.repetition_penalty == 1.0 and 'repetition_penalty' in gen_config:
                        args.repetition_penalty = gen_config['repetition_penalty']
                
                # Output parameters
                if 'output' in global_config:
                    out_config = global_config['output']
                    if args.experiment_name == 'pipeline_experiment' and 'experiment_name' in out_config:
                        args.experiment_name = out_config['experiment_name']
            
            print("Configuration loaded from:", args.config)
        except ImportError:
            print("Warning: metric_loader not found. Install pyyaml: pip install pyyaml")
            print("Falling back to command-line arguments only.")
        except Exception as e:
            print(f"Warning: Failed to load config from {args.config}: {e}")
            print("Falling back to command-line arguments only.")
    
    print("LLM Data Extraction and MIA Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Validation set size: {args.val_set_num}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Checkpointing: {'Enabled' if args.save_checkpoints else 'Disabled'}")
    print(f"Resume: {'Enabled' if args.resume else 'Disabled'}")
    
    # Check if required files exist
    dataset_path = Path(args.dataset_dir)
    required_files = ["train_prefix.npy", "train_dataset.npy"]
    
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            print(f"Error: Required file {file_name} not found in {args.dataset_dir}")
            sys.exit(1)
    
    # Handle checkpoint management
    experiment_path = Path("results") / args.experiment_name
    checkpoint_path = experiment_path / "checkpoint.pkl"
    
    if args.force_restart and checkpoint_path.exists():
        print("Force restart requested - removing existing checkpoint")
        checkpoint_path.unlink()
    
    if checkpoint_path.exists() and not args.resume:
        print(f"Warning: Checkpoint found at {checkpoint_path}")
        print("Use --resume to continue from checkpoint or --force_restart to start fresh")
    
    # Step 1: Data Extraction
    if not args.skip_extraction:
        extraction_command = [
            sys.executable, "extract.py",
            "--dataset_dir", args.dataset_dir,
            "--experiment_name", args.experiment_name,
            "--model", args.model,
            "--num_trials", str(args.num_trials),
            "--val_set_num", str(args.val_set_num),
            "--batch_size", str(args.batch_size),
            "--top_k", str(args.top_k),
            "--top_p", str(args.top_p),
            "--temperature", str(args.temperature),
            "--typical", str(args.typical),
            "--repetition_penalty", str(args.repetition_penalty),
            "--seed", str(args.seed)
        ]
        
        if args.save_all_generations_per_prompt:
            extraction_command.append("--save_all_generations_per_prompt")
        if args.save_all_methods:
            extraction_command.append("--save_all_methods")
        if args.save_npy_files:
            extraction_command.append("--save_npy_files")
        if args.save_checkpoints:
            extraction_command.append("--save_checkpoints")
        if args.resume:
            extraction_command.append("--resume")
        
        run_command(extraction_command, "Data Extraction")
    
    # Step 2: MIA Evaluation
    if not args.skip_mia:
        # Guess files are now saved in results/experiment-name/guess_files/
        guess_dir = os.path.join("results", args.experiment_name, "guess_files")
        
        if not os.path.exists(guess_dir):
            print(f"Error: Guess files directory not found at {guess_dir}")
            print("Either run extraction first or provide existing guess files")
            sys.exit(1)
        
        # Check if CSV files exist
        csv_files = list(Path(guess_dir).glob("*.csv"))
        if not csv_files:
            print(f"Error: No guess CSV files found in {guess_dir}")
            print("Either run extraction first or provide existing CSV files")
            sys.exit(1)
        
        mia_command = [
            sys.executable, "evaluate_mia.py",
            "--model", args.model,
            "--guess_dir", guess_dir,
            "--batch_size", str(args.batch_size),
            "--dataset_dir", args.dataset_dir
        ]
        
        run_command(mia_command, "MIA Evaluation")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Summary of outputs
    print("\nGenerated outputs:")
    
    experiment_path = Path("results") / args.experiment_name
    
    if not args.skip_extraction:
        if experiment_path.exists():
            print(f"- Experiment data: {experiment_path}")
            
            gen_files = list((experiment_path / "generations").glob("*.npy"))
            if gen_files:
                print(f"  - Generation files: {len(gen_files)}")

            loss_files = list((experiment_path / "losses").glob("*.npy"))
            if loss_files:
                print(f"  - Loss files: {len(loss_files)}")
            
            summary_csv = experiment_path / "extraction_metrics_summary.csv"
            if summary_csv.exists():
                print(f"- Extraction metrics summary: {summary_csv}")
            
            # Show checkpoint status
            checkpoint_file = experiment_path / "checkpoint.pkl"
            if checkpoint_file.exists():
                print(f"- Checkpoint file still exists: {checkpoint_file}")
                print("  (This indicates the extraction may not have completed)")
    
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
            print(f"- Guess files for MIA: {guess_path}")
            print(f"  - CSV files: {len(guess_files)}")


if __name__ == "__main__":
    main()