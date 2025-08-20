#!/usr/bin/env python3
"""
Complete Pipeline Runner for LLM Data Extraction and MIA Evaluation.

This script runs the complete pipeline:
1. Data extraction with multiple scoring methods
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
        description="Complete pipeline for LLM data extraction and MIA evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pipeline control
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip extraction step and only run MIA evaluation')
    parser.add_argument('--skip_mia', action='store_true',
                       help='Skip MIA evaluation step and only run extraction')
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default="../datasets",
                       help='Path to dataset directory')
    parser.add_argument('--root_dir', type=str, default="tmp/",
                       help='Root directory for results')
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
    
    # Generation parameters
    parser.add_argument('--top_k', type=int, default=24,
                       help='Top-k for generation')
    parser.add_argument('--top_p', type=float, default=0.8,
                       help='Top-p for generation')
    parser.add_argument('--temperature', type=float, default=0.58,
                       help='Temperature for generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.04,
                       help='Repetition penalty for generation')
    
    # Saving arguments
    parser.add_argument('--save_all_generations_per_prompt', action='store_true',
                       help='Save guess CSVs for all generation count tiers (1, 5, 10, etc.)')
    parser.add_argument('--save_all_methods', action='store_true',
                       help='Save guess CSVs for all scoring methods, not just likelihood')
    parser.add_argument('--save_npy_files', action='store_true',
                       help='Save intermediate generation and loss .npy files')

    # Other arguments
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("LLM Data Extraction and MIA Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Validation set size: {args.val_set_num}")
    print(f"Number of trials: {args.num_trials}")
    
    # Check if required files exist
    dataset_path = Path(args.dataset_dir)
    required_files = ["train_prefix.npy", "train_dataset.npy"]
    
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            print(f"Error: Required file {file_name} not found in {args.dataset_dir}")
            sys.exit(1)
    
    # Step 1: Data Extraction
    if not args.skip_extraction:
        extraction_command = [
            sys.executable, "extract.py",
            "--dataset_dir", args.dataset_dir,
            "--root_dir", args.root_dir,
            "--experiment_name", args.experiment_name,
            "--model", args.model,
            "--num_trials", str(args.num_trials),
            "--val_set_num", str(args.val_set_num),
            "--batch_size", str(args.batch_size),
            "--top_k", str(args.top_k),
            "--top_p", str(args.top_p),
            "--temperature", str(args.temperature),
            "--repetition_penalty", str(args.repetition_penalty),
            "--seed", str(args.seed)
        ]
        
        if args.save_all_generations_per_prompt:
            extraction_command.append("--save_all_generations_per_prompt")
        if args.save_all_methods:
            extraction_command.append("--save_all_methods")
        if args.save_npy_files:
            extraction_command.append("--save_npy_files")
        
        run_command(extraction_command, "Data Extraction")
        
        # Check if CSV files were generated
        csv_files = list(Path(".").glob("guess_*.csv"))
        if not csv_files:
            print("Warning: No guess CSV files found after extraction. This is expected if MIA evaluation is skipped.")
        else:
            print(f"Generated {len(csv_files)} guess files for MIA evaluation.")
    
    # Step 2: MIA Evaluation
    if not args.skip_mia:
        # Create a directory for guess files if it doesn't exist
        guess_dir = "guess_files"
        os.makedirs(guess_dir, exist_ok=True)
        
        # Move CSV files to guess directory for better organization
        csv_files = list(Path(".").glob("guess_*.csv"))
        if not csv_files:
            print("Error: No guess CSV files found for MIA evaluation")
            print("Either run extraction first or provide existing CSV files")
            sys.exit(1)
        
        for csv_file in csv_files:
            # Move file, overwriting if it exists from a previous run
            target_path = Path(guess_dir) / csv_file.name
            csv_file.replace(target_path)
        
        mia_command = [
            sys.executable, "evaluate_mia.py",
            "--model", args.model,
            "--guess_dir", guess_dir,
            "--batch_size", str(args.batch_size)
        ]
        
        run_command(mia_command, "MIA Evaluation")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Summary of outputs
    print("\nGenerated outputs:")
    
    if not args.skip_extraction:
        experiment_path = Path(args.root_dir) / args.experiment_name
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
    
    if not args.skip_mia:
        results_path = Path("results/mia_evaluation")
        if results_path.exists():
            result_files = list(results_path.glob("*.csv"))
            if result_files:
                print(f"- MIA evaluation results: {results_path}")
                for result_file in result_files:
                    print(f"    - {result_file.name}")
    
    guess_path = Path("guess_files")
    if guess_path.exists():
        guess_files = list(guess_path.glob("*.csv"))
        if guess_files:
            print(f"- Guess files for MIA: {guess_path}")
            print(f"  - CSV files: {len(guess_files)}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Complete Pipeline Runner for LLM Data Extraction and MIA Evaluation.

This script runs the complete pipeline:
1. Data extraction with multiple scoring methods
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
        description="Complete pipeline for LLM data extraction and MIA evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pipeline control
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip extraction step and only run MIA evaluation')
    parser.add_argument('--skip_mia', action='store_true',
                       help='Skip MIA evaluation step and only run extraction')
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default="../datasets",
                       help='Path to dataset directory')
    parser.add_argument('--root_dir', type=str, default="tmp/",
                       help='Root directory for results')
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
    
    # Generation parameters
    parser.add_argument('--top_k', type=int, default=24,
                       help='Top-k for generation')
    parser.add_argument('--top_p', type=float, default=0.8,
                       help='Top-p for generation')
    parser.add_argument('--temperature', type=float, default=0.58,
                       help='Temperature for generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.04,
                       help='Repetition penalty for generation')
    
    # Saving arguments
    parser.add_argument('--save_all_generations_per_prompt', action='store_true',
                       help='Save guess CSVs for all generation count tiers (1, 5, 10, etc.)')
    parser.add_argument('--save_all_methods', action='store_true',
                       help='Save guess CSVs for all scoring methods, not just likelihood')
    parser.add_argument('--save_npy_files', action='store_true',
                       help='Save intermediate generation and loss .npy files')

    # Other arguments
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("LLM Data Extraction and MIA Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Validation set size: {args.val_set_num}")
    print(f"Number of trials: {args.num_trials}")
    
    # Check if required files exist
    dataset_path = Path(args.dataset_dir)
    required_files = ["train_prefix.npy", "train_dataset.npy"]
    
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            print(f"Error: Required file {file_name} not found in {args.dataset_dir}")
            sys.exit(1)
    
    # Step 1: Data Extraction
    if not args.skip_extraction:
        extraction_command = [
            sys.executable, "extract.py",
            "--dataset_dir", args.dataset_dir,
            "--root_dir", args.root_dir,
            "--experiment_name", args.experiment_name,
            "--model", args.model,
            "--num_trials", str(args.num_trials),
            "--val_set_num", str(args.val_set_num),
            "--batch_size", str(args.batch_size),
            "--top_k", str(args.top_k),
            "--top_p", str(args.top_p),
            "--temperature", str(args.temperature),
            "--repetition_penalty", str(args.repetition_penalty),
            "--seed", str(args.seed)
        ]
        
        if args.save_all_generations_per_prompt:
            extraction_command.append("--save_all_generations_per_prompt")
        if args.save_all_methods:
            extraction_command.append("--save_all_methods")
        if args.save_npy_files:
            extraction_command.append("--save_npy_files")
        
        run_command(extraction_command, "Data Extraction")
        
        # Check if CSV files were generated
        csv_files = list(Path(".").glob("guess_*.csv"))
        if not csv_files:
            print("Warning: No guess CSV files found after extraction. This is expected if MIA evaluation is skipped.")
        else:
            print(f"Generated {len(csv_files)} guess files for MIA evaluation.")
    
    # Step 2: MIA Evaluation
    if not args.skip_mia:
        # Create a directory for guess files if it doesn't exist
        guess_dir = "guess_files"
        os.makedirs(guess_dir, exist_ok=True)
        
        # Move CSV files to guess directory for better organization
        csv_files = list(Path(".").glob("guess_*.csv"))
        if not csv_files:
            print("Error: No guess CSV files found for MIA evaluation")
            print("Either run extraction first or provide existing CSV files")
            sys.exit(1)
        
        for csv_file in csv_files:
            # Move file, overwriting if it exists from a previous run
            target_path = Path(guess_dir) / csv_file.name
            csv_file.replace(target_path)
        
        mia_command = [
            sys.executable, "evaluate_mia.py",
            "--model", args.model,
            "--guess_dir", guess_dir,
            "--batch_size", str(args.batch_size)
        ]
        
        run_command(mia_command, "MIA Evaluation")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Summary of outputs
    print("\nGenerated outputs:")
    
    if not args.skip_extraction:
        experiment_path = Path(args.root_dir) / args.experiment_name
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
    
    if not args.skip_mia:
        results_path = Path("results/mia_evaluation")
        if results_path.exists():
            result_files = list(results_path.glob("*.csv"))
            if result_files:
                print(f"- MIA evaluation results: {results_path}")
                for result_file in result_files:
                    print(f"    - {result_file.name}")
    
    guess_path = Path("guess_files")
    if guess_path.exists():
        guess_files = list(guess_path.glob("*.csv"))
        if guess_files:
            print(f"- Guess files for MIA: {guess_path}")
            print(f"  - CSV files: {len(guess_files)}")


if __name__ == "__main__":
    main()