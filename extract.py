"""
LLM Data Extraction with Multiple Scoring Methods.

This script generates text continuations using various scoring methods
for membership inference attacks and data extraction evaluation.
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils import (
    init_seeds, load_model_and_tokenizer, load_prompts, write_array,
    prepare_directories, calculate_likelihood_scores, calculate_recall_scores,
    calculate_original_recall, calculate_min_k_scores, calculate_zlib_scores,
    calculate_metric_scores, calculate_high_confidence_scores, write_guesses_to_csv,
    calculate_metrics, get_scoring_methods, get_argmin_methods, get_argmax_methods
)

# Constants
SUFFIX_LEN = 50
PREFIX_LEN = 50
K_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Enable TF32 for faster computation on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def generate_and_score(prompts: np.ndarray, 
                      model, 
                      tokenizer,
                      batch_size: int = 64,
                      skip_generation: bool = False,
                      non_member_prefix: np.ndarray = None,
                      generation_params: Dict = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate text continuations and calculate all scoring metrics.
    
    Args:
        prompts: Input prompts for generation
        model: Language model
        tokenizer: Tokenizer
        batch_size: Batch size for processing
        skip_generation: If True, use prompts as generations (for evaluation)
        non_member_prefix: Non-member prefixes for original recall calculation
        generation_params: Parameters for text generation
    
    Returns:
        Tuple of (generations, scores_dict)
    """
    if generation_params is None:
        generation_params = {
            'max_length': SUFFIX_LEN + PREFIX_LEN,
            'do_sample': True,
            'top_k': 24,
            'top_p': 0.8,
            'typical_p': 0.9,
            'temperature': 0.58,
            'repetition_penalty': 1.04,
            'pad_token_id': 50256,
            'use_cache': True
        }
    
    device = next(model.parameters()).device
    scoring_methods = get_scoring_methods(K_RATIOS)
    generations = []
    scores = {method: [] for method in scoring_methods}
    
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
        
        # Forward pass for scoring
        outputs = model(generated_tokens, labels=generated_tokens)
        
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token = torch.nn.functional.cross_entropy(
            full_logits, 
            generated_tokens[:, 1:].flatten(), 
            reduction='none'
        )

        high_conf_scores = calculate_high_confidence_scores(
            outputs.logits[:, :-1],
            full_loss_per_token,
            SUFFIX_LEN
        )
        scores["high_confidence"].extend(high_conf_scores)

        loss_per_token = full_loss_per_token.reshape(
            -1, generated_tokens.shape[1] - 1
        )[:, -SUFFIX_LEN:]
        likelihood = loss_per_token.mean(1)
        scores["likelihood"].extend(likelihood.cpu().numpy())
        
        zlib_scores = calculate_zlib_scores(generated_tokens, likelihood)
        scores["zlib"].extend(zlib_scores)
        
        metric_scores = calculate_metric_scores(loss_per_token)
        scores["metric"].extend(metric_scores)
        
        # Calculate recall scores using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            recall_futures = []
            original_recall_futures = []
            
            for batch_idx in range(generated_tokens.shape[0]):
                input_tokens = input_ids[batch_idx]
                suffix_tokens = generated_tokens[batch_idx, -SUFFIX_LEN:]
                
                # Regular recall scores
                recall_futures.append(
                    executor.submit(
                        calculate_recall_scores,
                        input_tokens,
                        suffix_tokens,
                        model,
                        device
                    )
                )
                
                # Original recall if non_member_prefix provided
                if non_member_prefix is not None:
                    non_member_prefix_tokens = torch.tensor(
                        non_member_prefix[batch_idx % len(non_member_prefix)], 
                        dtype=torch.int64, 
                        device=device
                    )
                    original_recall_futures.append(
                        executor.submit(
                            calculate_original_recall,
                            non_member_prefix_tokens,
                            input_tokens,
                            suffix_tokens,
                            model,
                            device
                        )
                    )
            
            # Process recall results
            for batch_idx, future in enumerate(recall_futures):
                ll_unconditional, ll_conditional = future.result()
                ll_full = -likelihood[batch_idx].item()
                
                recall_score = ll_full / ll_unconditional if ll_unconditional != 0 else 0
                recall2_score = ll_conditional / ll_unconditional if ll_unconditional != 0 else 0
                
                scores["-recall"].append(recall_score)
                scores["recall2"].append(recall2_score)
                scores["recall3"].append(recall2_score)
            
            # Process original recall results
            if non_member_prefix is not None:
                for future in original_recall_futures:
                    ll_unconditional, ll_conditional = future.result()
                    recall_score = ll_conditional / ll_unconditional if ll_unconditional != 0 else 0
                    scores["recall_original"].append(recall_score)
        
        # Calculate min_k scores
        logits_batch = outputs.logits[:, :-1]
        input_ids_batch = generated_tokens[:, 1:].unsqueeze(-1)
        token_log_probs, mink_plus = calculate_min_k_scores(logits_batch, input_ids_batch, device)
        
        # Process min_k scores for each sequence
        for batch_idx in range(token_log_probs.shape[0]):
            seq_token_log_probs = token_log_probs[batch_idx]
            seq_mink_plus = mink_plus[batch_idx]
            
            # Only consider suffix part
            suffix_token_log_probs = seq_token_log_probs[-SUFFIX_LEN:]
            suffix_mink_plus = seq_mink_plus[-SUFFIX_LEN:]
            
            for ratio in K_RATIOS:
                k_length = int(SUFFIX_LEN * ratio)
                bottomk_mink = np.sort(suffix_token_log_probs)[:k_length]
                bottomk_mink_plus = np.sort(suffix_mink_plus)[:k_length]
                
                scores[f'min_k_{ratio}'].append(np.mean(bottomk_mink))
                scores[f'min_k_plus_{ratio}'].append(np.mean(bottomk_mink_plus))
        
        generations.extend(generated_tokens.cpu().numpy())
    
    # Convert to numpy arrays
    generations = np.array(generations)
    for method in scoring_methods:
        if scores[method]:  # Only convert non-empty lists
            scores[method] = np.array(scores[method])
    
    return generations, scores


def run_extraction(args):
    """Main extraction pipeline."""
    print(f"Starting extraction with {args.model}")
    
    # Initialize
    init_seeds(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Prepare directories
    experiment_base, generations_base, losses_base = prepare_directories(
        args.root_dir, args.experiment_name
    )
    
    # Load data
    prompts = load_prompts(args.dataset_dir, "train_prefix.npy")[-args.val_set_num:]
    
    # Load non-member prefix if available
    non_member_prefix = None
    try:
        non_member_prefix = load_prompts(args.dataset_dir, "non_member_prefix.npy", allow_pickle=True)
        print("Loaded non-member prefix for original recall calculation")
    except FileNotFoundError:
        print("Warning: non_member_prefix.npy not found. Original recall will not be calculated.")
    
    # Generation parameters
    generation_params = {
        'max_length': SUFFIX_LEN + PREFIX_LEN,
        'do_sample': True,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'repetition_penalty': args.repetition_penalty,
        'pad_token_id': 50256,
        'use_cache': True
    }
    
    scoring_methods = get_scoring_methods(K_RATIOS)
    all_generations = []
    all_scores = {method: [] for method in scoring_methods}
    
    # Generate and score for multiple trials
    for trial in range(args.num_trials):
        print(f'Trial {trial + 1}/{args.num_trials}...')
        
        generations, scores = generate_and_score(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            non_member_prefix=non_member_prefix,
            generation_params=generation_params
        )
        
        if args.save_npy_files:
            generation_path = os.path.join(generations_base, "{}.npy")
            write_array(generation_path, generations, trial)
            for method in scoring_methods:
                if len(scores[method]) > 0:
                    losses_path = os.path.join(losses_base, f"{method}_{{}}.npy")
                    write_array(losses_path, scores[method], trial)
        
        for method in scoring_methods:
            if len(scores[method]) > 0:
                all_scores[method].append(scores[method])
        
        all_generations.append(generations)
    
    # Stack results
    all_generations = np.stack(all_generations, axis=1)
    for method in scoring_methods:
        if all_scores[method]:
            all_scores[method] = np.stack(all_scores[method], axis=1)
    
    print(f"Generated shape: {all_generations.shape}")
    
    # Load ground truth for evaluation
    answers = load_prompts(args.dataset_dir, "train_dataset.npy")[-args.val_set_num:, -100:]

    max_generations_per_prompt = all_generations.shape[1]
    if args.save_all_generations_per_prompt:
        generations_to_process = [1, 5, 10, 20, 50, max_generations_per_prompt]
        generations_to_process = [g for g in generations_to_process if g <= max_generations_per_prompt]
    else:
        generations_to_process = [max_generations_per_prompt]

    all_metrics_data = []
    
    # Always process all generation tiers for the final summary CSV
    full_generations_tiers = [1, 5, 10, 20, 50, max_generations_per_prompt]
    full_generations_tiers = sorted(list(set([g for g in full_generations_tiers if g <= max_generations_per_prompt])))
    
    for generations_per_prompt in full_generations_tiers:
        print(f"\nCalculating metrics for {generations_per_prompt} generations per prompt...")
        
        limited_generations = all_generations[:, :generations_per_prompt, :]
        generations_dict = {}
        
        valid_methods = [method for method in scoring_methods if len(all_scores[method]) > 0]
        argmin_methods = get_argmin_methods()
        argmax_methods = get_argmax_methods(K_RATIOS)
        
        for method in valid_methods:
            limited_scores = all_scores[method][:, :generations_per_prompt]
            
            if method in argmin_methods:
                best_indices = limited_scores.argmin(axis=1)
            else:
                best_indices = limited_scores.argmax(axis=1)
            
            prompt_indices = np.arange(limited_generations.shape[0])
            generations_dict[method] = limited_generations[prompt_indices, best_indices, :]
        
        # Decide which guess files to write based on flags
        if generations_per_prompt in generations_to_process:
            if args.save_all_methods:
                methods_to_save_csv = valid_methods
            else:
                methods_to_save_csv = ["likelihood"] if "likelihood" in valid_methods else []
            
            if methods_to_save_csv:
                write_guesses_to_csv(generations_per_prompt, generations_dict, answers, methods_to_save_csv)
        
        metrics = calculate_metrics(generations_dict, answers)
        for method, method_metrics in metrics.items():
            all_metrics_data.append({
                'generations_per_prompt': generations_per_prompt,
                'method': method,
                'precision': method_metrics['precision'],
                'hamming_distance': method_metrics['hamming_distance']
            })

    df_metrics = pd.DataFrame(all_metrics_data)
    if not df_metrics.empty:
        results_csv_path = os.path.join(experiment_base, "extraction_metrics_summary.csv")
        df_metrics.to_csv(results_csv_path, index=False, float_format='%.4f')
        print(f"\nExtraction metrics summary saved to {results_csv_path}")

    print(f"\nResults for {max_generations_per_prompt} generations per prompt:")
    if not df_metrics.empty:
        max_n_metrics = df_metrics[df_metrics['generations_per_prompt'] == max_generations_per_prompt]
        if not max_n_metrics.empty:
            print(max_n_metrics.to_string(index=False))
        else:
            print("No metrics calculated for the maximum number of generations.")
    else:
        print("No metrics were generated.")


def main():
    parser = argparse.ArgumentParser(description="LLM Data Extraction with Multiple Scoring Methods")
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default="../datasets", 
                       help='Path to dataset directory')
    parser.add_argument('--root_dir', type=str, default="tmp/", 
                       help='Root directory for results')
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
    run_extraction(args)


if __name__ == "__main__":
    main()