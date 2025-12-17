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
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute min-k scores."""
        logits_batch = outputs.logits[:, :-1]
        log_probs_batch = F.log_softmax(logits_batch, dim=-1)
        
        input_ids_batch = generated_tokens[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs_batch.gather(dim=-1, index=input_ids_batch).squeeze(-1)
        
        scores = []
        for batch_idx in range(token_log_probs.shape[0]):
            seq_token_log_probs = token_log_probs[batch_idx][-self.suffix_len:].cpu().numpy()
            
            k_length = int(self.suffix_len * self.ratio)
            if k_length == 0:
                scores.append(0.0)
            else:
                score = np.mean(np.sort(seq_token_log_probs)[:k_length])
                scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "min"  # Lower is better (note: will be inverted for MIA)


class MinKPlusMetric(BaseMetric):
    """Min-k++ metric - normalized min-k using mean and std."""
    
    def __init__(self, ratio: float, suffix_len: int = 50, **kwargs):
        super().__init__(name=f"min_k_plus_{ratio}", **kwargs)
        self.ratio = ratio
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute min-k++ scores."""
        logits_batch = outputs.logits[:, :-1]
        log_probs_batch = F.log_softmax(logits_batch, dim=-1)
        
        input_ids_batch = generated_tokens[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs_batch.gather(dim=-1, index=input_ids_batch).squeeze(-1)
        
        # Calculate mu and sigma
        mu = log_probs_batch.mean(dim=-1)
        sigma = log_probs_batch.std(dim=-1)
        
        mink_plus = (token_log_probs - mu) / sigma
        
        scores = []
        for batch_idx in range(mink_plus.shape[0]):
            seq_mink_plus = mink_plus[batch_idx][-self.suffix_len:].cpu().numpy()
            
            k_length = int(self.suffix_len * self.ratio)
            if k_length == 0:
                scores.append(0.0)
            else:
                score = np.mean(np.sort(seq_mink_plus)[:k_length])
                scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "min"  # Lower is better (note: will be inverted for MIA)


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
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute surprise scores."""
        logits_batch = outputs.logits[:, :-1]
        log_probs_batch = F.log_softmax(logits_batch, dim=-1)
        
        input_ids_batch = generated_tokens[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs_batch.gather(dim=-1, index=input_ids_batch).squeeze(-1)
        
        # Calculate entropy
        entropy_batch = (-torch.exp(log_probs_batch) * log_probs_batch).sum(dim=-1)
        
        scores = []
        for batch_idx in range(token_log_probs.shape[0]):
            seq_token_log_probs = token_log_probs[batch_idx][-self.suffix_len:].cpu().numpy()
            seq_entropy = entropy_batch[batch_idx][-self.suffix_len:].cpu().numpy()
            
            k_length = int(self.suffix_len * self.ratio)
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
