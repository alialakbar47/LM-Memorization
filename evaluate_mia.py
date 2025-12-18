"""
Membership Inference Attack (MIA) Evaluation.
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch

from config import load_config, Config
from metrics import get_all_metrics
from utils.core import load_model_and_tokenizer
from utils.data import load_prompts
from utils.evaluation import get_mia_metrics


def load_guess_data(guess_file: str) -> pd.DataFrame:
    """Load guess data from CSV file."""
    df = pd.read_csv(guess_file)
    df['Suffix Guess'] = df['Suffix Guess'].apply(eval)
    df['Ground Truth'] = df['Ground Truth'].apply(eval)
    return df


@torch.no_grad()
def calculate_scores_for_evaluation(model, tokenizer, df: pd.DataFrame, 
                                   device: torch.device,
                                   metrics_list: list,
                                   suffix_len: int,
                                   non_member_prefix: np.ndarray = None,
                                   member_prefix: np.ndarray = None) -> dict:
    """Calculate scores for MIA evaluation."""
    import torch.nn.functional as F
    scores = defaultdict(list)
    
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc='Calculating scores')):
        guess = np.array(row['Suffix Guess'])
        
        # Convert to tensor
        generated_tokens = torch.tensor(guess, dtype=torch.int64).unsqueeze(0).to(device)
        
        # Assume prefix and suffix split
        input_ids = generated_tokens[:, :-suffix_len] if generated_tokens.shape[1] > suffix_len else generated_tokens[:, :1]
        
        # Forward pass
        outputs = model(generated_tokens, labels=generated_tokens)
        
        # ===== Compute all shared values (same as extract.py) =====
        # Logits and labels
        logits = outputs.logits[:, :-1].contiguous()
        labels = generated_tokens[:, 1:].contiguous()
        
        # Reshape for loss calculation
        full_logits_flat = logits.reshape((-1, logits.shape[-1])).float()
        labels_flat = labels.flatten()
        
        # Loss per token
        full_loss_per_token_flat = F.cross_entropy(
            full_logits_flat,
            labels_flat,
            reduction='none'
        )
        loss_per_token = full_loss_per_token_flat.reshape(labels.shape)
        
        # Log probabilities
        log_probs_batch = F.log_softmax(logits, dim=-1)
        input_ids_batch = labels.unsqueeze(-1)
        token_log_probs = log_probs_batch.gather(dim=-1, index=input_ids_batch).squeeze(-1)
        
        # Mu and sigma for min-k++
        mu = log_probs_batch.mean(dim=-1)
        sigma = log_probs_batch.std(dim=-1)
        
        # Mask and normalized NLL
        mask = (labels != tokenizer.pad_token_id).float()
        original_nlls = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Build shared context (same as extract.py)
        shared_context = {
            'generated_tokens': generated_tokens,
            'input_ids': input_ids,
            'outputs': outputs,
            'logits': logits,
            'labels': labels,
            'loss_per_token': loss_per_token,
            'full_loss_per_token_flat': full_loss_per_token_flat,
            'log_probs_batch': log_probs_batch,
            'token_log_probs': token_log_probs,
            'mu': mu,
            'sigma': sigma,
            'mask': mask,
            'original_nlls': original_nlls,
            'suffix_len': suffix_len,
            'non_member_prefix': non_member_prefix,
            'member_prefix': member_prefix,
            'batch_offset': i,
        }
        
        # Compute each metric using shared context
        for metric in metrics_list:
            try:
                metric_score = metric.compute(model, tokenizer, device, shared_context)
                if isinstance(metric_score, np.ndarray):
                    scores[metric.name].append(metric_score[0])
                elif isinstance(metric_score, list):
                    scores[metric.name].append(metric_score[0])
                else:
                    scores[metric.name].append(metric_score)
            except Exception as e:
                print(f"Warning: Error computing {metric.name}: {e}")
                scores[metric.name].append(0.0)
    
    return scores


def run_mia_evaluation(config: Config):
    """Main MIA evaluation pipeline."""
    print(f"Starting MIA evaluation with {config.model.name}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model.name)
    device = next(model.parameters()).device
    
    # Load prefix data
    non_member_prefix, member_prefix = None, None
    if config.experiment.dataset_dir:
        try:
            non_member_prefix = load_prompts(config.experiment.dataset_dir, "non_member_prefix.npy", allow_pickle=True)
            print(f"Loaded non-member prefix.")
        except FileNotFoundError:
            print("Warning: non_member_prefix.npy not found.")
        try:
            member_prefix = load_prompts(config.experiment.dataset_dir, "member_prefix.npy", allow_pickle=True)
            print(f"Loaded member prefix.")
        except FileNotFoundError:
            print("Warning: member_prefix.npy not found.")
    
    # Get all metrics
    metrics_list = get_all_metrics(
        k_ratios=config.metrics.k_ratios,
        suffix_len=config.metrics.suffix_len,
        max_entropy=config.metrics.max_entropy
    )
    
    # Process each guess file
    results = defaultdict(list)
    
    guess_files = [f for f in os.listdir(config.evaluation.guess_dir) if f.endswith('.csv')]
    
    if not guess_files:
        print(f"No CSV files found in {config.evaluation.guess_dir}")
        return
    
    for guess_file in guess_files:
        print(f"\nProcessing {guess_file}")
        df = load_guess_data(os.path.join(config.evaluation.guess_dir, guess_file))
        
        # Calculate scores
        scores = calculate_scores_for_evaluation(
            model, tokenizer, df, device,
            metrics_list=metrics_list,
            suffix_len=config.metrics.suffix_len,
            non_member_prefix=non_member_prefix,
            member_prefix=member_prefix
        )
        
        # Calculate metrics using ground truth labels
        labels = df['Is Correct'].values
        
        for metric in metrics_list:
            if metric.name not in scores or len(scores[metric.name]) == 0:
                continue
            
            # For MIA, higher score should indicate membership
            # Min-based metrics need to be inverted
            if metric.direction() == 'min' and 'min_k' in metric.name:
                method_scores = [-s for s in scores[metric.name]]
            else:
                method_scores = scores[metric.name]
            
            metrics_result = get_mia_metrics(method_scores, labels)
            
            # Store results
            results['file'].append(os.path.splitext(guess_file)[0])
            results['method'].append(metric.name)
            results['auroc'].append(f"{metrics_result['auroc']:.3f}")
            results['fpr95'].append(f"{metrics_result['fpr95']:.3f}")
            results['tpr05'].append(f"{metrics_result['tpr05']:.3f}")
            results['avg_precision'].append(f"{metrics_result['avg_precision']:.3f}")
            
            for r_threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                key = f'precision_at_recall_{r_threshold}'
                results[key].append(f"{metrics_result[key]:.3f}")
            
            for p_threshold in [90, 95, 99]:
                key = f'recall_at_precision_{p_threshold}'
                results[key].append(f"{metrics_result[key]:.3f}")
    
    if not results:
        print("\nNo results generated.")
        return
    
    df_results = pd.DataFrame(results)
    
    # Save results
    experiment_name = os.path.basename(config.evaluation.guess_dir)
    if experiment_name == "guess_files":
        experiment_name = os.path.basename(os.path.dirname(config.evaluation.guess_dir))
    
    save_root = os.path.join("results", experiment_name, "mia_evaluation")
    os.makedirs(save_root, exist_ok=True)
    model_id = config.model.name.split('/')[-1]
    output_file = os.path.join(save_root, f"{model_id}_mia_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nFull MIA evaluation results saved to {output_file}")
    
    # Print summary
    print("\nResults Summary:")
    summary_cols = ['file', 'method', 'auroc', 'avg_precision', 'fpr95', 'tpr05']
    display_cols = [col for col in summary_cols if col in df_results.columns]
    if display_cols:
        print(df_results[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="MIA Evaluation")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--guess_dir', type=str, help='Directory containing guess CSV files')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config, args)
    else:
        from config import get_default_mia_config
        config_dict = get_default_mia_config()
        config = Config(config_dict)
        if args:
            for key, value in vars(args).items():
                if value is not None and key != 'config':
                    if key == 'dataset_dir':
                        setattr(config.experiment, key, value)
                    elif key == 'model':
                        setattr(config.model, 'name', value)
                    elif key in ['batch_size', 'guess_dir']:
                        setattr(config.evaluation, key, value)
    
    run_mia_evaluation(config)


if __name__ == "__main__":
    main()
