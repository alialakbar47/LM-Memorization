#!/usr/bin/env python3
"""
Complete Pipeline Runner for LLM Data Extraction and MIA Evaluation with Multi-GPU Support.

This script runs the complete pipeline optimized for Kaggle's dual T4 setup:
1. Data extraction with multiple scoring methods using multi-GPU acceleration
2. MIA evaluation of the extracted data with multi-GPU support
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
import torch


def check_gpu_setup():
    """Check and report GPU configuration."""
    print("GPU Configuration:")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - will run on CPU (very slow)")
        return 0
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name} - {memory_gb:.1f} GB")
    
    if num_gpus > 1:
        print(f"🚀 Multi-GPU acceleration will be used across {num_gpus} GPUs")
    
    print("-" * 40)
    return num_gpus


def optimize_batch_size_for_gpus(base_batch_size: int, num_gpus: int, mode: str = "extraction") -> int:
    """Optimize batch size based on GPU count and operation type."""
    if num_gpus <= 1:
        return base_batch_size
    
    # For multi-GPU, ensure batch size is divisible by GPU count
    optimized_size = (base_batch_size // num_gpus) * num_gpus
    if optimized_size == 0:
        optimized_size = num_gpus
    
    # For evaluation, we can use smaller batches
    if mode == "evaluation" and optimized_size > 16:
        optimized_size = min(optimized_size, 32)
    
    return optimized_size


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
        description="Complete pipeline for LLM data extraction and MIA evaluation with multi-GPU support",
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
    parser.add_argument('--experiment_name', type=str, default='multi_gpu_pipeline',
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
                       help='Base batch size (will be auto-optimized for multi-GPU)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='Base evaluation batch size (will be auto-optimized for multi-GPU)')
    
    # Generation parameters
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k for generation')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation')
    parser.add_argument('--typical_p', type=float, default=1.0,
                       help='Typical p for generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
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
    
    print("LLM Data Extraction and MIA Evaluation Pipeline with Multi-GPU Support")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Validation set size: {args.val_set_num}")
    print(f"Number of trials: {args.num_trials}")
    
    # Check GPU setup and optimize batch sizes
    num_gpus = check_gpu_setup()
    
    extraction_batch_size = optimize_batch_size_for_gpus(args.batch_size, num_gpus, "extraction")
    eval_batch_size = optimize_batch_size_for_gpus(args.eval_batch_size, num_gpus, "evaluation")
    
    if num_gpus > 1:
        print(f"📊 Batch size optimization:")
        print(f"   Extraction: {args.batch_size} -> {extraction_batch_size}")
        print(f"   Evaluation: {args.eval_batch_size} -> {eval_batch_size}")
    
    # Check if required files exist
    dataset_path = Path(args.dataset_dir)
    required_files = ["train_prefix.npy", "train_dataset.npy"]
    
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            print(f"❌ Error: Required file {file_name} not found in {args.dataset_dir}")
            sys.exit(1)
    print("✅ All required dataset files found")
    
    # Step 1: Data Extraction
    if not args.skip_extraction:
        print(f"\n🎯 Starting data extraction with multi-GPU support...")
        
        extraction_command = [
            sys.executable, "extract.py",
            "--dataset_dir", args.dataset_dir,
            "--root_dir", args.root_dir,
            "--experiment_name", args.experiment_name,
            "--model", args.model,
            "--num_trials", str(args.num_trials),
            "--val_set_num", str(args.val_set_num),
            "--batch_size", str(extraction_batch_size),  # Use optimized batch size
            "--top_k", str(args.top_k),
            "--top_p", str(args.top_p),
            "--typical_p", str(args.typical_p),
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
        
        run_command(extraction_command, "Data Extraction with Multi-GPU")
        
        # Check if CSV files were generated
        csv_files = list(Path(".").glob("guess_*.csv"))
        if not csv_files:
            print("⚠️  Warning: No guess CSV files found after extraction. This is expected if MIA evaluation is skipped.")
        else:
            print(f"✅ Generated {len(csv_files)} guess files for MIA evaluation.")
    
    # Step 2: MIA Evaluation
    if not args.skip_mia:
        print(f"\n🔍 Starting MIA evaluation with multi-GPU support...")
        
        # Create a directory for guess files if it doesn't exist
        guess_dir = "guess_files"
        os.makedirs(guess_dir, exist_ok=True)
        
        # Move CSV files to guess directory for better organization
        csv_files = list(Path(".").glob("guess_*.csv"))
        if not csv_files:
            print("❌ Error: No guess CSV files found for MIA evaluation")
            print("Either run extraction first or provide existing CSV files")
            sys.exit(1)
        
        for csv_file in csv_files:
            # Move file, overwriting if it exists from a previous run
            target_path = Path(guess_dir) / csv_file.name
            csv_file.replace(target_path)
        
        print(f"📁 Moved {len(csv_files)} CSV files to {guess_dir}/")
        
        mia_command = [
            sys.executable, "evaluate_mia.py",
            "--model", args.model,
            "--guess_dir", guess_dir,
            "--batch_size", str(eval_batch_size)  # Use optimized evaluation batch size
        ]
        
        run_command(mia_command, "MIA Evaluation with Multi-GPU")
    
    print("\n" + "=" * 80)
    print("🎉 Pipeline completed successfully with multi-GPU acceleration!")
    print("=" * 80)
    
    # Performance summary
    if num_gpus > 1:
        print(f"\n⚡ Performance Summary:")
        print(f"   Used {num_gpus} GPUs in parallel")
        print(f"   Extraction batch size: {extraction_batch_size}")
        print(f"   Evaluation batch size: {eval_batch_size}")
        estimated_speedup = min(num_gpus * 0.85, num_gpus)  # Account for overhead
        print(f"   Estimated speedup: ~{estimated_speedup:.1f}x compared to single GPU")
    
    # Summary of outputs
    print("\n📊 Generated outputs:")
    
    if not args.skip_extraction:
        experiment_path = Path(args.root_dir) / args.experiment_name
        if experiment_path.exists():
            print(f"- 📁 Experiment data: {experiment_path}")
            
            gen_files = list((experiment_path / "generations").glob("*.npy"))
            if gen_files:
                print(f"  - 🔄 Generation files: {len(gen_files)}")

            loss_files = list((experiment_path / "losses").glob("*.npy"))
            if loss_files:
                print(f"  - 📉 Loss files: {len(loss_files)}")
            
            summary_csv = experiment_path / "extraction_metrics_summary.csv"
            if summary_csv.exists():
                print(f"- 📈 Extraction metrics summary: {summary_csv}")
    
    if not args.skip_mia:
        results_path = Path("results/mia_evaluation")
        if results_path.exists():
            result_files = list(results_path.glob("*.csv"))
            if result_files:
                print(f"- 🎯 MIA evaluation results: {results_path}")
                for result_file in result_files:
                    print(f"    - 📋 {result_file.name}")
    
    guess_path = Path("guess_files")
    if guess_path.exists():
        guess_files = list(guess_path.glob("*.csv"))
        if guess_files:
            print(f"- 🔍 Guess files for MIA: {guess_path}")
            print(f"  - 📄 CSV files: {len(guess_files)}")
    
    # Final GPU memory summary
    if torch.cuda.is_available():
        print(f"\n💾 Final GPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total")


if __name__ == "__main__":
    main()