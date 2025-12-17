"""
LLM Data Extraction with Multiple Scoring Methods - Enhanced with Checkpoint/Resume.

This script generates text continuations using various scoring methods
for membership inference attacks and data extraction evaluation.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils import (
    init_seeds, load_model_and_tokenizer, load_prompts, write_array,
    prepare_directories, write_guesses_to_csv
)
from metric_loader import load_config, load_all_enabled_metrics, get_enabled_metrics, update_metric_config_from_dataset

# Constants (can be overridden by config)
SUFFIX_LEN = 50
PREFIX_LEN = 50

# Enable TF32 for faster computation on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CheckpointManager:
    """Manages checkpointing and resuming for experiments."""
    
    def __init__(self, experiment_base: str, checkpoint_name: str = "checkpoint.pkl"):
        self.checkpoint_path = os.path.join(experiment_base, checkpoint_name)
        self.experiment_base = experiment_base
        
    def save_checkpoint(self, trial: int, all_generations: List, all_scores: Dict, 
                       args: argparse.Namespace, rng_states: Dict):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'trial': trial,
            'all_generations': all_generations,
            'all_scores': all_scores,
            'args': vars(args),
            'rng_states': rng_states
        }
        
        # Save to temporary file first, then rename (atomic operation)
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        os.rename(temp_path, self.checkpoint_path)
        print(f"Checkpoint saved after trial {trial + 1}")
        
    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("Checkpoint file cleaned up")

    def get_rng_states(self):
        """Get current random number generator states."""
        return {
            'python_random': np.random.get_state(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_random': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    
    def set_rng_states(self, rng_states: Dict):
        """Restore random number generator states."""
        if 'python_random' in rng_states:
            np.random.set_state(rng_states['python_random'])
        if 'numpy_random' in rng_states:
            np.random.set_state(rng_states['numpy_random'])
        if 'torch_random' in rng_states:
            torch.set_rng_state(rng_states['torch_random'])
        if 'torch_cuda_random' in rng_states and rng_states['torch_cuda_random'] is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['torch_cuda_random'])


def calculate_extraction_metrics(generations_dict: Dict, answers: np.ndarray) -> Dict:
    """Calculate extraction metrics for each method's generations."""
    metrics_dict = {}
    
    for method, generations in generations_dict.items():
        decoded_generations = [
            bytes(generation[generation != 0].tolist()).decode('utf-8', errors='ignore')
            for generation in generations
        ]
        decoded_answers = [
            bytes(answer[answer != 0].tolist()).decode('utf-8', errors='ignore')
            for answer in answers
        ]
        
        correct = sum(1 for gen, ans in zip(decoded_generations, decoded_answers) if gen == ans)
        total = len(decoded_answers)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics_dict[method] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
    
    return metrics_dict


@torch.no_grad()
def generate_and_score(prompts: np.ndarray, 
                      model, 
                      tokenizer,
                      metrics: Dict,
                      batch_size: int = 64,
                      skip_generation: bool = False,
                      generation_params: Dict = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate text continuations and calculate all scoring metrics using modular metric system.
    
    Args:
        prompts: Input prompts for generation
        model: Language model
        tokenizer: Tokenizer
        metrics: Dictionary of metric objects from metric_loader
        batch_size: Batch size for processing
        skip_generation: If True, use prompts as generations (for evaluation)
        generation_params: Parameters for text generation
    
    Returns:
        Tuple of (generations, scores_dict)
    """
    if generation_params is None:
        generation_params = {
            'max_length': SUFFIX_LEN + PREFIX_LEN,
            'do_sample': True,
            'top_k': 10,
            'top_p': 1.0,
            'typical_p': 1.0,
            'temperature': 1.0,
            'repetition_penalty': 1.0,
            'pad_token_id': 50256,
            'use_cache': True
        }
    
    device = next(model.parameters()).device
    generations = []
    scores = {metric_name: [] for metric_name in metrics.keys()}
    
    # Process prompts in batches
    for off in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        prompt_batch = prompts[off:off + batch_size]
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64, device=device)
        
        if not skip_generation:
            # Generate text continuations
            generated_tokens = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                **generation_params
            )
        else:
            generated_tokens = input_ids
        
        # Calculate scores using each metric
        for metric_name, metric_obj in metrics.items():
            try:
                batch_scores = metric_obj.compute_score(
                    generated_tokens=generated_tokens,
                    prefix_len=PREFIX_LEN,
                    suffix_len=SUFFIX_LEN
                )
                scores[metric_name].extend(batch_scores.cpu().numpy() if torch.is_tensor(batch_scores) else batch_scores)
            except Exception as e:
                print(f"Warning: Error computing {metric_name}: {e}")
                # Add zero scores for this batch
                scores[metric_name].extend([0.0] * generated_tokens.shape[0])

        generations.extend(generated_tokens.cpu().numpy())
    
    generations = np.array(generations)
    for method in metrics.keys():
        if scores[method]:
            scores[method] = np.array(scores[method])
    
    return generations, scores


def run_extraction(args):
    """Main extraction pipeline with checkpoint/resume capability."""
    print(f"Starting extraction with {args.model}")
    
    # Load configuration if provided
    if hasattr(args, 'config') and args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Update config with dataset paths
        dataset_paths = {
            'non_member_prefix': os.path.join(args.dataset_dir, 'non_member_prefix.npy'),
            'member_prefix': os.path.join(args.dataset_dir, 'member_prefix.npy')
        }
        config = update_metric_config_from_dataset(config, dataset_paths)
    else:
        # Use default config
        print("Loading default configuration from config.yaml")
        try:
            config = load_config('config.yaml')
            dataset_paths = {
                'non_member_prefix': os.path.join(args.dataset_dir, 'non_member_prefix.npy'),
                'member_prefix': os.path.join(args.dataset_dir, 'member_prefix.npy')
            }
            config = update_metric_config_from_dataset(config, dataset_paths)
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
            config = None
    
    # Initialize directories first - now under results/experiment-name/
    experiment_base, generations_base, losses_base = prepare_directories(
        "results", args.experiment_name
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(experiment_base)
    
    # Check for existing checkpoint
    checkpoint_data = checkpoint_manager.load_checkpoint()
    
    if checkpoint_data and args.resume:
        print("Found existing checkpoint, resuming...")
        
        # Restore RNG states for reproducibility
        checkpoint_manager.set_rng_states(checkpoint_data['rng_states'])
        
        # Restore progress
        start_trial = checkpoint_data['trial'] + 1
        all_generations = checkpoint_data['all_generations']
        all_scores = checkpoint_data['all_scores']
        
        print(f"Resuming from trial {start_trial + 1}/{args.num_trials}")
        
        # Verify args compatibility (important for reproducibility)
        saved_args = checkpoint_data['args']
        critical_args = ['model', 'seed', 'num_trials', 'val_set_num', 'batch_size', 
                        'top_k', 'top_p', 'temperature', 'typical', 'repetition_penalty']
        for arg in critical_args:
            if getattr(args, arg) != saved_args.get(arg):
                print(f"Warning: Argument {arg} has changed from {saved_args.get(arg)} to {getattr(args, arg)}")
                print("This may affect reproducibility. Consider using the same parameters as the checkpoint.")
    else:
        if checkpoint_data and not args.resume:
            print("Found existing checkpoint but --resume not specified. Starting fresh.")
        
        # Initialize seeds for fresh start
        init_seeds(args.seed)
        start_trial = 0
        all_generations = []
        all_scores = {}
    
    # Load model and data (this should be after RNG state restoration for consistency)
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load metrics using metric_loader
    if config:
        metrics = load_all_enabled_metrics(model, tokenizer, config)
        print(f"Loaded {len(metrics)} metrics: {list(metrics.keys())}")
    else:
        print("Error: No configuration available. Cannot load metrics.")
        print("Please ensure config.yaml exists or provide --config argument.")
        return
    
    # Initialize scores if starting fresh
    if not all_scores:
        all_scores = {name: [] for name in metrics.keys()}
    
    prompts = load_prompts(args.dataset_dir, "train_prefix.npy")[-args.val_set_num:]

    generation_params = {
        'max_length': SUFFIX_LEN + PREFIX_LEN,
        'do_sample': True,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'typical_p': args.typical,
        'repetition_penalty': args.repetition_penalty,
        'pad_token_id': tokenizer.pad_token_id,
        'use_cache': True
    }
    
    scoring_methods = list(metrics.keys())
    
    # Continue from where we left off
    for trial in range(start_trial, args.num_trials):
        print(f'Trial {trial + 1}/{args.num_trials}...')
        
        generations, scores = generate_and_score(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            metrics=metrics,
            batch_size=args.batch_size,
            generation_params=generation_params
        )
        
        if args.save_npy_files:
            write_array(os.path.join(generations_base, "{}.npy"), generations, trial)
            for method in scoring_methods:
                if len(scores.get(method, [])) > 0:
                    write_array(os.path.join(losses_base, f"{method}_{{}}.npy"), scores[method], trial)
        
        # Accumulate results
        all_generations.append(generations)
        for method in scoring_methods:
            if len(scores.get(method, [])) > 0:
                all_scores[method].append(scores[method])
        
        # Save checkpoint after each trial
        if args.save_checkpoints:
            rng_states = checkpoint_manager.get_rng_states()
            checkpoint_manager.save_checkpoint(trial, all_generations, all_scores, args, rng_states)
    
    # Clean up checkpoint after successful completion
    if args.save_checkpoints:
        checkpoint_manager.cleanup_checkpoint()
    
    # Convert to final format
    all_generations = np.stack(all_generations, axis=1)
    for method in scoring_methods:
        if all_scores.get(method):
            all_scores[method] = np.stack(all_scores[method], axis=1)
    
    print(f"Generated shape: {all_generations.shape}")
    
    answers = load_prompts(args.dataset_dir, "train_dataset.npy")[-args.val_set_num:, -100:]

    max_generations_per_prompt = all_generations.shape[1]
    gen_tiers = [1, 5, 10, 20, 50, max_generations_per_prompt]
    
    if args.save_all_generations_per_prompt:
        generations_to_process = [g for g in gen_tiers if g <= max_generations_per_prompt]
    else:
        generations_to_process = [max_generations_per_prompt]

    all_metrics_data = []
    full_generations_tiers = sorted(list(set([g for g in gen_tiers if g <= max_generations_per_prompt])))
    
    # Create guess_files directory within the experiment
    guess_files_dir = os.path.join(experiment_base, "guess_files")
    os.makedirs(guess_files_dir, exist_ok=True)
    
    for generations_per_prompt in full_generations_tiers:
        print(f"\nCalculating metrics for {generations_per_prompt} generations per prompt...")
        
        limited_generations = all_generations[:, :generations_per_prompt, :]
        generations_dict = {}
        
        valid_methods = [m for m in scoring_methods if all_scores.get(m) is not None and len(all_scores[m]) > 0]
        # Determine argmin/argmax based on metric properties
        argmin_methods = [name for name, metric in metrics.items() if metric.uses_argmin()]
        argmax_methods = [name for name, metric in metrics.items() if metric.uses_argmax()]
        
        for method in valid_methods:
            limited_scores = all_scores[method][:, :generations_per_prompt]
            
            best_indices = limited_scores.argmin(axis=1) if method in argmin_methods else limited_scores.argmax(axis=1)
            
            prompt_indices = np.arange(limited_generations.shape[0])
            generations_dict[method] = limited_generations[prompt_indices, best_indices, :]
        
        if generations_per_prompt in generations_to_process:
            methods_to_save_csv = valid_methods if args.save_all_methods else (["likelihood"] if "likelihood" in valid_methods else [])
            if methods_to_save_csv:
                # Updated to save in the experiment's guess_files directory
                write_guesses_to_csv(generations_per_prompt, generations_dict, answers, methods_to_save_csv, guess_files_dir)
        
        metrics_dict = calculate_extraction_metrics(generations_dict, answers)
        for method, method_metrics in metrics_dict.items():
            all_metrics_data.append({
                'generations_per_prompt': generations_per_prompt,
                'method': method,
                **method_metrics
            })

    df_metrics = pd.DataFrame(all_metrics_data)
    if not df_metrics.empty:
        results_csv_path = os.path.join(experiment_base, "extraction_metrics_summary.csv")
        df_metrics.to_csv(results_csv_path, index=False, float_format='%.4f')
        print(f"\nExtraction metrics summary saved to {results_csv_path}")

        print(f"\nResults for {max_generations_per_prompt} generations per prompt:")
        max_n_metrics = df_metrics[df_metrics['generations_per_prompt'] == max_generations_per_prompt]
        if not max_n_metrics.empty:
            print(max_n_metrics.to_string(index=False))
        else:
            print("No metrics calculated for the maximum number of generations.")
    else:
        print("No metrics were generated.")


def main():
    parser = argparse.ArgumentParser(description="LLM Data Extraction with Multiple Scoring Methods")
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default="datasets", 
                       help='Path to dataset directory')
    parser.add_argument('--experiment_name', type=str, default='extraction_experiment',
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

    # Checkpoint arguments
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='Save checkpoints after each trial for resumability')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoint if available')

    # Other arguments
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()