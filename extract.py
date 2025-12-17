"""
LLM Data Extraction with Multiple Scoring Methods.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config import load_config, Config, save_config
from metrics import get_all_metrics, get_metric_names
from utils.core import init_seeds, load_model_and_tokenizer, prepare_directories
from utils.data import load_prompts, write_array, write_guesses_to_csv
from utils.evaluation import calculate_metrics
from utils.checkpoint import CheckpointManager

# Enable TF32 for faster computation on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def generate_and_score(prompts: np.ndarray,
                      model,
                      tokenizer,
                      metrics_list: List,
                      config: Config,
                      batch_offset: int = 0,
                      skip_generation: bool = False,
                      non_member_prefix: np.ndarray = None,
                      member_prefix: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate text continuations and calculate all scoring metrics.
    """
    generation_params = {
        'max_length': config.metrics.suffix_len + config.metrics.prefix_len,
        'do_sample': True,
        'top_k': config.generation.top_k,
        'top_p': config.generation.top_p,
        'typical_p': config.generation.typical_p,
        'temperature': config.generation.temperature,
        'repetition_penalty': config.generation.repetition_penalty,
        'pad_token_id': tokenizer.pad_token_id,
        'use_cache': True
    }
    
    device = next(model.parameters()).device
    generations = []
    scores = {metric.name: [] for metric in metrics_list}
    
    # Process prompts in batches
    for off in tqdm(range(0, len(prompts), config.generation.batch_size), desc="Processing batches"):
        prompt_batch = prompts[off:off + config.generation.batch_size]
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64, device=device)
        
        if not skip_generation:
            generated_tokens = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                **generation_params
            )
        else:
            generated_tokens = input_ids
        
        # Forward pass for scoring
        outputs = model(generated_tokens, labels=generated_tokens)
        
        # Calculate normalized NLL for metrics that need it
        full_labels = generated_tokens[:, 1:].contiguous()
        mask = (full_labels != tokenizer.pad_token_id).float()
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token_flat = F.cross_entropy(
            full_logits,
            generated_tokens[:, 1:].flatten(),
            reduction='none'
        )
        original_nlls = (full_loss_per_token_flat.reshape(full_labels.shape) * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Compute metrics using ThreadPoolExecutor for recall-based metrics
        with ThreadPoolExecutor(max_workers=4) as executor:
            for metric in metrics_list:
                # Build kwargs for metric computation
                metric_kwargs = {
                    'input_ids': input_ids,
                    'suffix_len': config.metrics.suffix_len,
                    'non_member_prefix': non_member_prefix,
                    'member_prefix': member_prefix,
                    'original_nlls': original_nlls,
                    'batch_offset': off + batch_offset,
                }
                
                # Compute metric scores
                if metric.name in ['suffix_recall', 'recall', 'con_recall', 'suffix_conrecall']:
                    # These metrics require sequential processing
                    metric_scores = metric.compute(
                        model, tokenizer, generated_tokens, outputs, device, **metric_kwargs
                    )
                else:
                    # Other metrics can be computed in batch
                    metric_scores = metric.compute(
                        model, tokenizer, generated_tokens, outputs, device, **metric_kwargs
                    )
                
                scores[metric.name].extend(metric_scores)
        
        generations.extend(generated_tokens.cpu().numpy())
    
    generations = np.array(generations)
    for metric_name in scores:
        if scores[metric_name]:
            scores[metric_name] = np.array(scores[metric_name])
    
    return generations, scores


def run_extraction(config: Config):
    """Main extraction pipeline with checkpoint/resume capability."""
    print(f"Starting extraction with {config.model.name}")
    
    # Initialize directories
    experiment_base, generations_base, losses_base = prepare_directories(
        "results", config.experiment.name
    )
    
    # Save config
    save_config(config, os.path.join(experiment_base, "config.yaml"))
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(experiment_base)
    
    # Check for existing checkpoint
    checkpoint_data = checkpoint_manager.load_checkpoint()
    
    if checkpoint_data and config.checkpoint.resume:
        print("Found existing checkpoint, resuming...")
        checkpoint_manager.set_rng_states(checkpoint_data['rng_states'])
        start_trial = checkpoint_data['trial'] + 1
        all_generations = checkpoint_data['all_generations']
        all_scores = checkpoint_data['all_scores']
        print(f"Resuming from trial {start_trial + 1}/{config.generation.num_trials}")
    else:
        if checkpoint_data and not config.checkpoint.resume:
            print("Found existing checkpoint but resume not enabled. Starting fresh.")
        init_seeds(config.experiment.seed)
        start_trial = 0
        all_generations = []
        metric_names = get_metric_names(config.metrics.k_ratios)
        all_scores = {method: [] for method in metric_names}
    
    # Load model and data
    model, tokenizer = load_model_and_tokenizer(config.model.name)
    
    prompts = load_prompts(config.experiment.dataset_dir, "train_prefix.npy")[-config.generation.val_set_num:]
    
    # Load prefix data for recall calculations
    non_member_prefix, member_prefix = None, None
    try:
        non_member_prefix = load_prompts(config.experiment.dataset_dir, "non_member_prefix.npy", allow_pickle=True)
        print("Loaded non-member prefix for recall calculation.")
    except FileNotFoundError:
        print("Warning: non_member_prefix.npy not found.")
    try:
        member_prefix = load_prompts(config.experiment.dataset_dir, "member_prefix.npy", allow_pickle=True)
        print("Loaded member prefix for con_recall calculation.")
    except FileNotFoundError:
        print("Warning: member_prefix.npy not found.")
    
    # Get all metrics
    metrics_list = get_all_metrics(
        k_ratios=config.metrics.k_ratios,
        suffix_len=config.metrics.suffix_len,
        max_entropy=config.metrics.max_entropy
    )
    
    # Continue from where we left off
    for trial in range(start_trial, config.generation.num_trials):
        print(f'\nTrial {trial + 1}/{config.generation.num_trials}...')
        
        generations, scores = generate_and_score(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            metrics_list=metrics_list,
            config=config,
            batch_offset=trial * len(prompts),
            non_member_prefix=non_member_prefix,
            member_prefix=member_prefix
        )
        
        if config.saving.save_npy_files:
            write_array(os.path.join(generations_base, "{}.npy"), generations, trial)
            for metric_name, metric_scores in scores.items():
                if len(metric_scores) > 0:
                    write_array(os.path.join(losses_base, f"{metric_name}_{{}}.npy"), metric_scores, trial)
        
        # Accumulate results
        all_generations.append(generations)
        for metric_name in scores:
            if len(scores[metric_name]) > 0:
                all_scores[metric_name].append(scores[metric_name])
        
        # Save checkpoint after each trial
        if config.checkpoint.save_checkpoints:
            rng_states = checkpoint_manager.get_rng_states()
            checkpoint_manager.save_checkpoint(
                trial, all_generations, all_scores, config.to_dict(), rng_states
            )
    
    # Clean up checkpoint after successful completion
    if config.checkpoint.save_checkpoints:
        checkpoint_manager.cleanup_checkpoint()
    
    # Convert to final format
    all_generations = np.stack(all_generations, axis=1)
    for metric_name in all_scores:
        if all_scores[metric_name]:
            all_scores[metric_name] = np.stack(all_scores[metric_name], axis=1)
    
    print(f"Generated shape: {all_generations.shape}")
    
    # Load answers
    answers = load_prompts(config.experiment.dataset_dir, "train_dataset.npy")[-config.generation.val_set_num:, -100:]
    
    # Process results
    max_generations_per_prompt = all_generations.shape[1]
    gen_tiers = [1, 5, 10, 20, 50, max_generations_per_prompt]
    
    if config.saving.save_all_generations_per_prompt:
        generations_to_process = [g for g in gen_tiers if g <= max_generations_per_prompt]
    else:
        generations_to_process = [max_generations_per_prompt]
    
    all_metrics_data = []
    full_generations_tiers = sorted(list(set([g for g in gen_tiers if g <= max_generations_per_prompt])))
    
    # Create guess_files directory
    guess_files_dir = os.path.join(experiment_base, "guess_files")
    os.makedirs(guess_files_dir, exist_ok=True)
    
    # Create metric direction mapping
    metric_directions = {}
    for metric in metrics_list:
        metric_directions[metric.name] = metric.direction()
    
    for generations_per_prompt in full_generations_tiers:
        print(f"\nCalculating metrics for {generations_per_prompt} generations per prompt...")
        
        limited_generations = all_generations[:, :generations_per_prompt, :]
        generations_dict = {}
        
        valid_methods = [m for m in all_scores.keys() if all_scores.get(m) is not None and len(all_scores[m]) > 0]
        
        for method in valid_methods:
            limited_scores = all_scores[method][:, :generations_per_prompt]
            
            # Use direction from metric
            if metric_directions.get(method) == 'min':
                best_indices = limited_scores.argmin(axis=1)
            else:
                best_indices = limited_scores.argmax(axis=1)
            
            prompt_indices = np.arange(limited_generations.shape[0])
            generations_dict[method] = limited_generations[prompt_indices, best_indices, :]
        
        if generations_per_prompt in generations_to_process:
            methods_to_save_csv = valid_methods if config.saving.save_all_methods else (["likelihood"] if "likelihood" in valid_methods else [])
            if methods_to_save_csv:
                write_guesses_to_csv(generations_per_prompt, generations_dict, answers, methods_to_save_csv, guess_files_dir)
        
        metrics = calculate_metrics(generations_dict, answers)
        for method, method_metrics in metrics.items():
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


def main():
    parser = argparse.ArgumentParser(description="LLM Data Extraction")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--experiment_name', type=str, help='Name of experiment')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--num_trials', type=int, help='Number of trials')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config, args)
    else:
        # Use default config and override with args
        from config import get_default_extraction_config
        config_dict = get_default_extraction_config()
        config = Config(config_dict)
        if args:
            for key, value in vars(args).items():
                if value is not None and key != 'config':
                    # Handle nested config
                    if key in ['name', 'seed', 'dataset_dir']:
                        setattr(config.experiment, key, value)
                    elif key == 'model':
                        setattr(config.model, 'name', value)
                    elif key in ['num_trials', 'val_set_num', 'batch_size', 'top_k', 'top_p', 'temperature', 'typical_p', 'repetition_penalty']:
                        setattr(config.generation, key, value)
                    elif key in ['save_all_generations_per_prompt', 'save_all_methods', 'save_npy_files']:
                        setattr(config.saving, key, value)
                    elif key in ['save_checkpoints', 'resume', 'force_restart']:
                        setattr(config.checkpoint, key, value)
    
    run_extraction(config)


if __name__ == "__main__":
    main()
