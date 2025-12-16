#!/usr/bin/env python3
"""
Example script demonstrating the new metric system.
This shows how to use the configuration-based approach for running metrics.
"""

import torch
import numpy as np
from metric_loader import load_config, load_all_enabled_metrics, update_metric_config_from_dataset
from utils import load_model_and_tokenizer
import os


def main():
    print("=" * 60)
    print("LLM Memorization Metrics - Configuration-Based Example")
    print("=" * 60)
    
    # 1. Load configuration
    config_path = "config.yaml"
    print(f"\n1. Loading configuration from {config_path}")
    config = load_config(config_path)
    print(f"   - Model: {config['global']['model']}")
    print(f"   - Seed: {config['global']['seed']}")
    print(f"   - Batch size: {config['global']['batch_size']}")
    
    # 2. Update configuration with dataset paths
    print("\n2. Loading dataset paths")
    dataset_dir = config['global']['dataset_dir']
    dataset_paths = {
        'non_member_prefix': os.path.join(dataset_dir, 'non_member_prefix.npy'),
        'member_prefix': os.path.join(dataset_dir, 'member_prefix.npy')
    }
    config = update_metric_config_from_dataset(config, dataset_paths)
    
    # 3. Load model and tokenizer
    print(f"\n3. Loading model: {config['global']['model']}")
    model, tokenizer = load_model_and_tokenizer(config['global']['model'])
    print(f"   - Model device: {model.device}")
    print(f"   - Vocab size: {tokenizer.vocab_size}")
    
    # 4. Load all enabled metrics
    print("\n4. Loading enabled metrics")
    metrics = load_all_enabled_metrics(model, tokenizer, config)
    print(f"   - Loaded {len(metrics)} metrics:")
    for metric_name in metrics.keys():
        print(f"     * {metric_name}")
    
    # 5. Generate some example tokens for testing
    print("\n5. Generating example tokens for testing")
    batch_size = 4
    seq_len = 100
    example_tokens = torch.randint(
        0, 
        tokenizer.vocab_size, 
        (batch_size, seq_len), 
        device=model.device
    )
    print(f"   - Shape: {example_tokens.shape}")
    
    # 6. Run each metric on the example
    print("\n6. Computing scores for each metric")
    print("-" * 60)
    
    results = {}
    for metric_name, metric in metrics.items():
        print(f"\n   Metric: {metric_name}")
        print(f"   - Uses argmin: {metric.uses_argmin()}")
        print(f"   - Uses argmax: {metric.uses_argmax()}")
        
        try:
            scores = metric.compute_score(example_tokens)
            results[metric_name] = scores
            print(f"   - Score shape: {scores.shape}")
            print(f"   - Sample scores: {scores[:2].cpu().numpy()}")
        except Exception as e:
            print(f"   - Error: {str(e)}")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully computed {len(results)}/{len(metrics)} metrics")
    print(f"\nTo customize metrics:")
    print(f"  1. Edit config.yaml to enable/disable metrics")
    print(f"  2. Adjust hyperparameters in each metric's 'config' section")
    print(f"  3. Re-run this script to see changes")
    print("\nTo add a new metric:")
    print(f"  1. Create metrics/my_metric.py with MyMetric class")
    print(f"  2. Add entry to config.yaml metrics section")
    print(f"  3. Metric will be automatically loaded")
    
    return results


if __name__ == "__main__":
    results = main()
