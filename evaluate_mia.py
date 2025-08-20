"""
Membership Inference Attack (MIA) Evaluation.

This script evaluates the effectiveness of different scoring methods
for membership inference attacks using extracted data.
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
import zlib

from utils import load_model_and_tokenizer, get_scoring_methods, calculate_suffix_con_recall

# Constants matching extract.py
K_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_guess_data(guess_file: str) -> pd.DataFrame:
    """Load guess data from CSV file."""
    df = pd.read_csv(guess_file)
    # Convert string representations back to numpy arrays
    df['Suffix Guess'] = df['Suffix Guess'].apply(eval)
    df['Ground Truth'] = df['Ground Truth'].apply(eval)
    return df


@torch.no_grad()
def calculate_scores_for_evaluation(model, tokenizer, df: pd.DataFrame, device: torch.device) -> dict:
    """Calculate scores for MIA evaluation."""
    scores = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Calculating scores'):
        guess = np.array(row['Suffix Guess'])
        ground_truth = np.array(row['Ground Truth'])
        
        # Split into prefix and suffix (assuming 50-50 split)
        prefix = guess[:50]
        suffix = guess[50:]
        
        # Convert to tensors
        prefix_ids = torch.tensor(prefix, dtype=torch.int64).unsqueeze(0).to(device)
        suffix_ids = torch.tensor(suffix, dtype=torch.int64).unsqueeze(0).to(device)
        full_ids = torch.tensor(guess, dtype=torch.int64).unsqueeze(0).to(device)
        
        # Calculate recall scores (using prefix-suffix split)
        # Unconditional: probability of suffix
        suffix_outputs = model(suffix_ids)
        suffix_logits = suffix_outputs.logits[:, :-1]
        suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)
        suffix_token_log_probs = suffix_log_probs.gather(
            dim=-1,
            index=suffix_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ll_unconditional = suffix_token_log_probs.mean().item()
        
        # Conditional: probability of suffix given prefix
        full_outputs = model(full_ids)
        full_logits = full_outputs.logits[:, :-1]
        full_log_probs = F.log_softmax(full_logits, dim=-1)
        # Only gather probabilities for suffix positions
        suffix_positions = torch.arange(50, len(guess)-1, device=device)
        full_token_log_probs = full_log_probs[0, suffix_positions].gather(
            dim=-1,
            index=suffix_ids[0, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ll_conditional = full_token_log_probs.mean().item()
        
        # Calculate suffix_recall score using the new formula
        nll_unconditional = -ll_unconditional
        nll_conditional = -ll_conditional
        suffix_recall_score = nll_unconditional / nll_conditional if nll_conditional != 0 else 0
        scores['suffix_recall'].append(suffix_recall_score)
        
        # Calculate suffix_con_recall score
        s_con_recall_score = calculate_suffix_con_recall(
            prefix_ids.squeeze(0), suffix_ids.squeeze(0), model, tokenizer, device
        )
        scores['suffix_conrecall'].append(s_con_recall_score)

        # Calculate other scores using the full sequence
        outputs = model(full_ids, labels=full_ids)
        loss, logits = outputs[:2]
        
        # 1. Likelihood - keep original scale (log likelihood)
        log_probs_full = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs_full = log_probs_full.gather(dim=-1, index=full_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        ll = token_log_probs_full.mean().item()
        scores['likelihood'].append(ll)
        
        # 2. Zlib
        text = tokenizer.decode(full_ids[0].cpu().numpy())
        compression_ratio = len(zlib.compress(text.encode('utf-8'))) / len(text.encode('utf-8'))
        scores['zlib'].append(ll * compression_ratio)
        
        # 3. Metric (mean with outlier removal)
        loss_per_token = F.cross_entropy(
            logits[0, :-1], 
            full_ids[0, 1:], 
            reduction='none'
        ).cpu().numpy()
        mean = np.mean(loss_per_token)
        std = np.std(loss_per_token)
        floor = mean - 3*std
        upper = mean + 3*std
        metric_loss = np.where(
            ((loss_per_token < floor) | (loss_per_token > upper)),
            mean,
            loss_per_token
        )
        scores['metric'].append(-np.mean(metric_loss))
        
        # 4. High confidence
        probs = F.softmax(logits[0, :-1], dim=-1)
        top_scores, _ = probs.topk(2, dim=-1)
        flag1 = (top_scores[:, 0] - top_scores[:, 1]) > 0.5
        flag2 = top_scores[:, 0] > 0
        conf_adjustment = (flag1.int() - flag2.int()) * mean * 0.15
        conf_loss = loss_per_token - conf_adjustment.cpu().numpy()
        scores['high_confidence'].append(-np.mean(conf_loss))
        
        # 5. Min-k, Min-k++, and Surprise scores
        logits_shifted = logits[0, :-1]
        log_probs = F.log_softmax(logits_shifted, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=full_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        
        mu = log_probs.mean(-1)
        sigma = log_probs.std(-1)
        mink_plus = (token_log_probs - mu) / sigma
        
        entropy = (-torch.exp(log_probs) * log_probs).sum(dim=-1)
        
        for ratio in K_RATIOS:
            k_length = int(len(token_log_probs) * ratio)
            if k_length > 0:
                # Min-k
                topk_mink = np.sort(token_log_probs.cpu())[:k_length]
                scores[f'min_k_{ratio}'].append(-np.mean(topk_mink).item())
                
                # Min-k++
                topk_mink_plus = np.sort(mink_plus.cpu())[:k_length]
                scores[f'min_k_plus_{ratio}'].append(-np.mean(topk_mink_plus).item())

                # Surprise
                mink_idx = np.argsort(token_log_probs.cpu().numpy())[:k_length]
                entropy_idx = np.where(entropy.cpu().numpy() < 2.0)[0]
                intersection = np.intersect1d(mink_idx, entropy_idx, assume_unique=True)
                
                if len(intersection) > 0:
                    score = np.mean(token_log_probs.cpu().numpy()[intersection])
                else:
                    score = -100.0
                scores[f'surprise_{ratio}'].append(score)

            else:
                scores[f'min_k_{ratio}'].append(0.0)
                scores[f'min_k_plus_{ratio}'].append(0.0)
                scores[f'surprise_{ratio}'].append(0.0)
        
        # 6. Lowercase score
        original_nll = F.cross_entropy(logits[0, :-1], full_ids[0, 1:], reduction='sum').item()
        
        decoded_text = tokenizer.decode(full_ids[0].cpu().numpy(), skip_special_tokens=True)
        lowercase_text = decoded_text.lower()
        lowercase_ids = tokenizer(lowercase_text, return_tensors='pt').input_ids.to(device)
        
        if lowercase_ids.shape[1] > 1:
            lowercase_outputs = model(lowercase_ids, labels=lowercase_ids)
            lowercase_logits = lowercase_outputs.logits
            lowercase_nll = F.cross_entropy(lowercase_logits[0, :-1], lowercase_ids[0, 1:], reduction='sum').item()
            lowercase_score = -original_nll / (lowercase_nll + 1e-9)
        else:
            lowercase_score = 0
        scores['lowercase'].append(lowercase_score)

    return scores


def get_metrics(scores: list, labels: list) -> dict:
    """Calculate MIA metrics including precision-recall metrics."""
    scores = np.array(scores)
    labels = np.array(labels)
    
    # ROC curve metrics
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    
    # Handle FPR95 (TPR >= 0.95)
    tpr_95_idx = np.where(tpr_list >= 0.95)[0]
    if len(tpr_95_idx) > 0:
        fpr95 = fpr_list[tpr_95_idx[0]]
    else:
        fpr95 = 1.0
    
    # Handle TPR05 (FPR <= 0.05)
    fpr_05_idx = np.where(fpr_list <= 0.05)[0]
    if len(fpr_05_idx) > 0:
        tpr05 = tpr_list[fpr_05_idx[-1]]
    else:
        tpr05 = 0.0
    
    # Precision-Recall metrics
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    # Calculate precision at different recall thresholds
    recall_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_at_recall = {}
    
    for r_threshold in recall_thresholds:
        mask = recall >= r_threshold
        if np.any(mask):
            precision_at_recall[f'precision_at_recall_{int(r_threshold*100)}'] = np.max(precision[mask])
        else:
            precision_at_recall[f'precision_at_recall_{int(r_threshold*100)}'] = 0.0
    
    # Calculate recall at high precision thresholds
    precision_thresholds = [0.9, 0.95, 0.99]
    recall_at_precision = {}
    
    for p_threshold in precision_thresholds:
        mask = precision >= p_threshold
        if np.any(mask):
            recall_at_precision[f'recall_at_precision_{int(p_threshold*100)}'] = np.max(recall[mask])
        else:
            recall_at_precision[f'recall_at_precision_{int(p_threshold*100)}'] = 0.0
    
    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'tpr05': tpr05,
        'avg_precision': avg_precision,
        **precision_at_recall,
        **recall_at_precision
    }


def run_mia_evaluation(args):
    """Main MIA evaluation pipeline."""
    print(f"Starting MIA evaluation with {args.model}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    device = next(model.parameters()).device
    
    # Get all scoring methods
    scoring_methods = get_scoring_methods(K_RATIOS)
    
    # Process each guess file
    results = defaultdict(list)
    
    guess_files = [f for f in os.listdir(args.guess_dir) if f.endswith('.csv')]
    
    if not guess_files:
        print(f"No CSV files found in {args.guess_dir}")
        return
    
    for guess_file in guess_files:
        print(f"\nProcessing {guess_file}")
        df = load_guess_data(os.path.join(args.guess_dir, guess_file))
        
        # Calculate scores using all methods
        scores = calculate_scores_for_evaluation(model, tokenizer, df, device)
        
        # Calculate metrics using ground truth labels
        labels = df['Is Correct'].values
        
        for method in scoring_methods:
            if method not in scores or len(scores[method]) == 0:
                continue
                
            # For MIA, higher score should always indicate membership.
            # Some calculated scores are losses (lower is better), so we invert them.
            lower_is_better_methods = (
                [f"min_k_{r}" for r in K_RATIOS] + 
                [f"min_k_plus_{r}" for r in K_RATIOS]
            )
            
            if method in lower_is_better_methods:
                method_scores = [-s for s in scores[method]]
            else:
                method_scores = scores[method]
                
            metrics = get_metrics(method_scores, labels)
            
            # Store results
            results['file'].append(os.path.splitext(guess_file)[0])
            results['method'].append(method)
            results['auroc'].append(f"{metrics['auroc']:.3f}")
            results['fpr95'].append(f"{metrics['fpr95']:.3f}")
            results['tpr05'].append(f"{metrics['tpr05']:.3f}")
            results['avg_precision'].append(f"{metrics['avg_precision']:.3f}")
            
            # Add precision at different recall thresholds
            for r_threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                key = f'precision_at_recall_{r_threshold}'
                results[key].append(f"{metrics[key]:.3f}")
            
            # Add recall at different precision thresholds
            for p_threshold in [90, 95, 99]:
                key = f'recall_at_precision_{p_threshold}'
                results[key].append(f"{metrics[key]:.3f}")
    
    if not results:
        print("\nNo results were generated from the provided guess files.")
        return

    df_results = pd.DataFrame(results)

    # Save full results to CSV
    save_root = "results/mia_evaluation"
    os.makedirs(save_root, exist_ok=True)
    model_id = args.model.split('/')[-1]
    output_file = os.path.join(save_root, f"{model_id}_mia_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nFull MIA evaluation results saved to {output_file}")

    # Print a cleaner summary to the console
    print("\nResults Summary:")
    summary_cols = ['file', 'method', 'auroc', 'avg_precision', 'fpr95', 'tpr05']
    display_cols = [col for col in summary_cols if col in df_results.columns]
    if display_cols:
        print(df_results[display_cols].to_string(index=False))
    else:
        print("No summary columns found in results.")


def main():
    parser = argparse.ArgumentParser(description="MIA Evaluation for Extracted Data")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B',
                       help='Model name or path')
    
    # Data arguments
    parser.add_argument('--guess_dir', type=str, required=True,
                       help='Directory containing guess CSV files')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    run_mia_evaluation(args)


if __name__ == "__main__":
    main()