

"""
LLM Data Extraction with Multiple Scoring Methods.

This script generates text continuations using various scoring methods
for membership inference attacks and data extraction evaluation.
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

from utils import (
    init_seeds, load_model_and_tokenizer, load_prompts, write_array,
    prepare_directories, calculate_recall_scores, calculate_suffix_con_recall,
    calculate_recall, calculate_con_recall, calculate_min_k_scores, calculate_zlib_scores,
    calculate_metric_scores, calculate_high_confidence_scores, calculate_lowercase_score,
    write_guesses_to_csv, calculate_metrics, get_scoring_methods, 
    get_argmin_methods, get_argmax_methods
)

# Constants
SUFFIX_LEN = 50
PREFIX_LEN = 50
K_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_ENTROPY = 2.0

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
                      member_prefix: np.ndarray = None,
                      generation_params: Dict = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate text continuations and calculate all scoring metrics.
    
    Args:
        prompts: Input prompts for generation
        model: Language model
        tokenizer: Tokenizer
        batch_size: Batch size for processing
        skip_generation: If True, use prompts as generations (for evaluation)
        non_member_prefix: Non-member prefixes for recall calculation
        member_prefix: Member prefixes for con_recall calculation
        generation_params: Parameters for text generation
    
    Returns:
        Tuple of (generations, scores_dict)
    """
    if generation_params is None:
        generation_params = {
            'max_length': SUFFIX_LEN + PREFIX_LEN,
            'do_sample': True,
            'top_k': 10, # MODIFIED: Changed fallback default to match new baseline
            'top_p': 1.0, # MODIFIED: Changed fallback default to match new baseline
            'typical_p': 1.0, # MODIFIED: Changed fallback default to match new baseline
            'temperature': 1.0, # MODIFIED: Changed fallback default to match new baseline
            'repetition_penalty': 1.0, # MODIFIED: Changed fallback default to match new baseline
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
        full_loss_per_token_flat = F.cross_entropy(full_logits, generated_tokens[:, 1:].flatten(), reduction='none')

        high_conf_scores = calculate_high_confidence_scores(
            outputs.logits[:, :-1], full_loss_per_token_flat, SUFFIX_LEN
        )
        scores["high_confidence"].extend(high_conf_scores)

        loss_per_token = full_loss_per_token_flat.reshape(-1, generated_tokens.shape[1] - 1)[:, -SUFFIX_LEN:]
        likelihood = loss_per_token.mean(1)
        scores["likelihood"].extend(likelihood.cpu().numpy())
        
        scores["zlib"].extend(calculate_zlib_scores(generated_tokens, likelihood))
        scores["metric"].extend(calculate_metric_scores(loss_per_token))
        
        # Calculate NLL for lowercase and con_recall
        full_labels = generated_tokens[:, 1:].contiguous()
        mask = (full_labels != tokenizer.pad_token_id).float()
        original_nlls = (full_loss_per_token_flat.reshape(full_labels.shape) * mask).sum(dim=1)
        
        scores["lowercase"].extend(calculate_lowercase_score(
            generated_tokens, original_nlls, model, tokenizer, device
        ))

        # Calculate various recall scores using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'suffix_recall': [], 'recall': [], 'con_recall': [], 'suffix_conrecall': []
            }
            
            for batch_idx in range(generated_tokens.shape[0]):
                input_toks = input_ids[batch_idx]
                suffix_toks = generated_tokens[batch_idx, -SUFFIX_LEN:]
                
                futures['suffix_recall'].append(executor.submit(
                    calculate_recall_scores, input_toks, suffix_toks, model, device
                ))
                futures['suffix_conrecall'].append(executor.submit(
                    calculate_suffix_con_recall, input_toks, suffix_toks, model, tokenizer, device
                ))
                
                if non_member_prefix is not None:
                    nm_prefix = torch.tensor(non_member_prefix[batch_idx % len(non_member_prefix)], dtype=torch.int64)
                    futures['recall'].append(executor.submit(
                        calculate_recall, nm_prefix, input_toks, suffix_toks, model, device
                    ))
                    
                    if member_prefix is not None:
                        m_prefix = torch.tensor(member_prefix[batch_idx % len(member_prefix)], dtype=torch.int64)
                        futures['con_recall'].append(executor.submit(
                            calculate_con_recall, nm_prefix, m_prefix,
                            generated_tokens[batch_idx], original_nlls[batch_idx].item(),
                            model, device
                        ))

            for ll_u, ll_c in [f.result() for f in futures['suffix_recall']]:
                nll_unconditional = -ll_u
                nll_conditional = -ll_c
                scores["suffix_recall"].append(nll_unconditional / nll_conditional if nll_conditional != 0 else 0)
            for score in [f.result() for f in futures['suffix_conrecall']]:
                scores["suffix_conrecall"].append(score)
            for nll_u, nll_c in [f.result() for f in futures['recall']]:
                scores["recall"].append(nll_c / nll_u if nll_u != 0 else 0)
            for score in [f.result() for f in futures['con_recall']]:
                scores["con_recall"].append(score)

        # Calculate min_k, min_k_plus, and surprise scores
        logits_batch = outputs.logits[:, :-1]
        log_probs_batch = F.log_softmax(logits_batch, dim=-1)
        entropy_batch = (-torch.exp(log_probs_batch) * log_probs_batch).sum(dim=-1)
        
        input_ids_batch = generated_tokens[:, 1:].unsqueeze(-1)
        token_log_probs, mink_plus = calculate_min_k_scores(logits_batch, input_ids_batch, device)
        
        for batch_idx in range(token_log_probs.shape[0]):
            seq_token_log_probs = token_log_probs[batch_idx][-SUFFIX_LEN:]
            seq_mink_plus = mink_plus[batch_idx][-SUFFIX_LEN:]
            seq_entropy = entropy_batch[batch_idx][-SUFFIX_LEN:]
            
            for ratio in K_RATIOS:
                k_length = int(SUFFIX_LEN * ratio)
                if k_length == 0: continue
                
                scores[f'min_k_{ratio}'].append(np.mean(np.sort(seq_token_log_probs)[:k_length]))
                scores[f'min_k_plus_{ratio}'].append(np.mean(np.sort(seq_mink_plus)[:k_length]))

                mink_idx = np.argsort(seq_token_log_probs)[:k_length]
                entropy_idx = np.where(seq_entropy.cpu().numpy() < MAX_ENTROPY)[0]
                intersection = np.intersect1d(mink_idx, entropy_idx, assume_unique=True)

                surprise_score = np.mean(seq_token_log_probs[intersection]) if len(intersection) > 0 else -100.0
                scores[f'surprise_{ratio}'].append(surprise_score)

        generations.extend(generated_tokens.cpu().numpy())
    
    for ratio in K_RATIOS:
        if int(SUFFIX_LEN * ratio) == 0:
            num_missing = len(prompts) - len(scores[f'min_k_{ratio}'])
            scores[f'min_k_{ratio}'].extend([0.0] * num_missing)
            scores[f'min_k_plus_{ratio}'].extend([0.0] * num_missing)
            scores[f'surprise_{ratio}'].extend([0.0] * num_missing)

    generations = np.array(generations)
    for method in scoring_methods:
        if scores[method]:
            scores[method] = np.array(scores[method])
    
    return generations, scores


def run_extraction(args):
    """Main extraction pipeline."""
    print(f"Starting extraction with {args.model}")
    
    init_seeds(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    experiment_base, generations_base, losses_base = prepare_directories(
        args.root_dir, args.experiment_name
    )
    
    prompts = load_prompts(args.dataset_dir, "train_prefix.npy")[-args.val_set_num:]
    
    non_member_prefix, member_prefix = None, None
    try:
        non_member_prefix = load_prompts(args.dataset_dir, "non_member_prefix.npy", allow_pickle=True)
        print("Loaded non-member prefix for recall calculation.")
    except FileNotFoundError:
        print("Warning: non_member_prefix.npy not found. `recall` and `con_recall` will not be calculated.")
    try:
        member_prefix = load_prompts(args.dataset_dir, "member_prefix.npy", allow_pickle=True)
        print("Loaded member prefix for con_recall calculation.")
    except FileNotFoundError:
        print("Warning: member_prefix.npy not found. `con_recall` will not be calculated.")

    generation_params = {
        'max_length': SUFFIX_LEN + PREFIX_LEN,
        'do_sample': True,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'typical_p':args.typical,
        'repetition_penalty': args.repetition_penalty,
        'pad_token_id': tokenizer.pad_token_id,
        'use_cache': True
    }
    
    scoring_methods = get_scoring_methods(K_RATIOS)
    all_generations = []
    all_scores = {method: [] for method in scoring_methods}
    
    for trial in range(args.num_trials):
        print(f'Trial {trial + 1}/{args.num_trials}...')
        
        generations, scores = generate_and_score(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            non_member_prefix=non_member_prefix,
            member_prefix=member_prefix,
            generation_params=generation_params
        )
        
        if args.save_npy_files:
            write_array(os.path.join(generations_base, "{}.npy"), generations, trial)
            for method in scoring_methods:
                if len(scores.get(method, [])) > 0:
                    write_array(os.path.join(losses_base, f"{method}_{{}}.npy"), scores[method], trial)
        
        for method in scoring_methods:
            if len(scores.get(method, [])) > 0:
                all_scores[method].append(scores[method])
        
        all_generations.append(generations)
    
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
    
    for generations_per_prompt in full_generations_tiers:
        print(f"\nCalculating metrics for {generations_per_prompt} generations per prompt...")
        
        limited_generations = all_generations[:, :generations_per_prompt, :]
        generations_dict = {}
        
        valid_methods = [m for m in scoring_methods if all_scores.get(m) is not None and len(all_scores[m]) > 0]
        argmin_methods = get_argmin_methods()
        argmax_methods = get_argmax_methods(K_RATIOS)
        
        for method in valid_methods:
            limited_scores = all_scores[method][:, :generations_per_prompt]
            
            best_indices = limited_scores.argmin(axis=1) if method in argmin_methods else limited_scores.argmax(axis=1)
            
            prompt_indices = np.arange(limited_generations.shape[0])
            generations_dict[method] = limited_generations[prompt_indices, best_indices, :]
        
        if generations_per_prompt in generations_to_process:
            methods_to_save_csv = valid_methods if args.save_all_methods else (["likelihood"] if "likelihood" in valid_methods else [])
            if methods_to_save_csv:
                write_guesses_to_csv(generations_per_prompt, generations_dict, answers, methods_to_save_csv)
        
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
    # MODIFIED: Changed all defaults to new baseline
    parser.add_argument('--top_k', type=int, default=10,
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

    # Other arguments
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()