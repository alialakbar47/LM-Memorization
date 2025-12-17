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
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute high confidence scores."""
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token_flat = F.cross_entropy(
            full_logits, 
            generated_tokens[:, 1:].flatten(), 
            reduction='none'
        )
        
        # Get suffix logits and compute flags
        suffix_logits = outputs.logits[:, -self.suffix_len-1:-1]
        top_scores, _ = suffix_logits.topk(2, dim=-1)
        flag1 = (top_scores[:, :, 0] - top_scores[:, :, 1]) > 0.5
        flag2 = top_scores[:, :, 0] > 0
        flat_flag1 = flag1.reshape(-1)
        flat_flag2 = flag2.reshape(-1)
        
        # Calculate mean batch loss for adjustment
        mean_batch_loss = full_loss_per_token_flat.mean()
        
        # Apply adjustment
        loss_adjusted_flat = full_loss_per_token_flat - (flat_flag1.int() - flat_flag2.int()) * mean_batch_loss * 0.15
        loss_adjusted_reshaped = loss_adjusted_flat.reshape(outputs.logits.shape[0], -1)
        loss_adjusted_suffix = loss_adjusted_reshaped[:, -self.suffix_len:]
        
        return loss_adjusted_suffix.mean(1).cpu().numpy()
    
    def direction(self) -> str:
        return "min"  # Lower is better
