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
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute likelihood scores (mean loss per token on suffix).
        OLD: loss_per_token = full_loss_per_token_flat.reshape(-1, seq_len-1)[:, -SUFFIX_LEN:]
              likelihood = loss_per_token.mean(1)
        """
        # Use pre-computed loss_per_token from shared context
        loss_per_token = shared_context['loss_per_token']
        suffix_len = shared_context['suffix_len']
        
        # Extract suffix portion - matches old implementation
        loss_per_token_suffix = loss_per_token[:, -suffix_len:]
        likelihood = loss_per_token_suffix.mean(1)
        
        return likelihood.cpu().numpy()
    
    def direction(self) -> str:
        return "min"  # Lower loss is better
