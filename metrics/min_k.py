"""
Min-k based metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric


class MinKMetric(BaseMetric):
    """Min-k metric - averages k lowest token log probabilities."""
    
    def __init__(self, ratio: float, suffix_len: int = 50, **kwargs):
        super().__init__(name=f"min_k_{ratio}", **kwargs)
        self.ratio = ratio
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute min-k scores."""
        # Use pre-computed token_log_probs from shared context
        token_log_probs = shared_context['token_log_probs']
        suffix_len = shared_context['suffix_len']
        
        scores = []
        for batch_idx in range(token_log_probs.shape[0]):
            seq_token_log_probs = token_log_probs[batch_idx][-suffix_len:].cpu().numpy()
            
            k_length = int(suffix_len * self.ratio)
            if k_length == 0:
                scores.append(0.0)
            else:
                # Return positive mean - matches extract_old.py line 245
                # Higher log prob (less negative) = more memorized
                score = np.mean(np.sort(seq_token_log_probs)[:k_length])
                scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better (argmax in extract)


class MinKPlusMetric(BaseMetric):
    """Min-k++ metric - normalized min-k using mean and std."""
    
    def __init__(self, ratio: float, suffix_len: int = 50, **kwargs):
        super().__init__(name=f"min_k_plus_{ratio}", **kwargs)
        self.ratio = ratio
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute min-k++ scores."""
        # Use pre-computed values from shared context
        token_log_probs = shared_context['token_log_probs']
        mu = shared_context['mu']
        sigma = shared_context['sigma']
        suffix_len = shared_context['suffix_len']
        
        # Calculate normalized scores
        mink_plus = (token_log_probs - mu) / sigma
        
        scores = []
        for batch_idx in range(mink_plus.shape[0]):
            seq_mink_plus = mink_plus[batch_idx][-suffix_len:].cpu().numpy()
            
            k_length = int(suffix_len * self.ratio)
            if k_length == 0:
                scores.append(0.0)
            else:
                # Return positive mean - matches extract_old.py line 246
                # Higher normalized score = more memorized
                score = np.mean(np.sort(seq_mink_plus)[:k_length])
                scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better (argmax in extract)


class SurpriseMetric(BaseMetric):
    """Surprise metric - min-k with entropy filtering."""
    
    def __init__(self, ratio: float, suffix_len: int = 50, max_entropy: float = 2.0, **kwargs):
        super().__init__(name=f"surprise_{ratio}", **kwargs)
        self.ratio = ratio
        self.suffix_len = suffix_len
        self.max_entropy = max_entropy
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute surprise scores."""
        # Use pre-computed values from shared context
        token_log_probs = shared_context['token_log_probs']
        log_probs_batch = shared_context['log_probs_batch']
        suffix_len = shared_context['suffix_len']
        
        # Calculate entropy (only thing we can't pre-compute for all metrics)
        entropy_batch = (-torch.exp(log_probs_batch) * log_probs_batch).sum(dim=-1)
        
        scores = []
        for batch_idx in range(token_log_probs.shape[0]):
            seq_token_log_probs = token_log_probs[batch_idx][-suffix_len:].cpu().numpy()
            seq_entropy = entropy_batch[batch_idx][-suffix_len:].cpu().numpy()
            
            k_length = int(suffix_len * self.ratio)
            if k_length == 0:
                scores.append(0.0)
            else:
                mink_idx = np.argsort(seq_token_log_probs)[:k_length]
                entropy_idx = np.where(seq_entropy < self.max_entropy)[0]
                intersection = np.intersect1d(mink_idx, entropy_idx, assume_unique=True)
                
                if len(intersection) > 0:
                    score = np.mean(seq_token_log_probs[intersection])
                else:
                    score = -100.0
                scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better
