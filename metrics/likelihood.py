"""
Likelihood-based metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric


class LikelihoodMetric(BaseMetric):
    """Likelihood metric - measures average token loss."""
    
    def __init__(self, suffix_len: int = 50, **kwargs):
        super().__init__(name="likelihood", **kwargs)
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute likelihood scores (mean loss per token on suffix)."""
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token_flat = F.cross_entropy(
            full_logits, 
            generated_tokens[:, 1:].flatten(), 
            reduction='none'
        )
        
        loss_per_token = full_loss_per_token_flat.reshape(-1, generated_tokens.shape[1] - 1)[:, -self.suffix_len:]
        likelihood = loss_per_token.mean(1)
        
        return likelihood.cpu().numpy()
    
    def direction(self) -> str:
        return "min"  # Lower loss is better
