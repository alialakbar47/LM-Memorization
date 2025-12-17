"""
High confidence metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric


class HighConfidenceMetric(BaseMetric):
    """High confidence metric - adjusts loss based on prediction confidence."""
    
    def __init__(self, suffix_len: int = 50, **kwargs):
        super().__init__(name="high_confidence", **kwargs)
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute high confidence scores."""
        # Use pre-computed values from shared context
        logits = shared_context['logits']  # Shape: [batch, seq_len-1, vocab]
        full_loss_per_token_flat = shared_context['full_loss_per_token_flat']  # Shape: [batch * seq_len]
        suffix_len = shared_context['suffix_len']
        
        # Compute flags on the full logits (before suffix extraction)
        # This matches the old implementation exactly
        top_scores, _ = logits.topk(2, dim=-1)  # Shape: [batch, seq_len-1, 2]
        flag1 = (top_scores[:, :, 0] - top_scores[:, :, 1]) > 0.5
        flag2 = top_scores[:, :, 0] > 0
        flat_flag1 = flag1.reshape(-1)  # Flatten to match full_loss_per_token_flat
        flat_flag2 = flag2.reshape(-1)
        
        # Calculate mean batch loss for adjustment
        mean_batch_loss = full_loss_per_token_flat.mean()
        
        # Apply adjustment to the full flat loss (EXACT OLD LOGIC)
        loss_adjusted_flat = full_loss_per_token_flat - (flat_flag1.int() - flat_flag2.int()) * mean_batch_loss * 0.15
        
        # Reshape back to [batch, seq_len-1] and extract suffix
        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        loss_adjusted_reshaped = loss_adjusted_flat.reshape(batch_size, seq_len)
        loss_adjusted_suffix = loss_adjusted_reshaped[:, -suffix_len:]
        
        return loss_adjusted_suffix.mean(1).cpu().numpy()
    
    def direction(self) -> str:
        return "min"  # Lower is better
