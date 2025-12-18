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
        Returns positive loss values for argmin selection in extraction.
        """
        # Use pre-computed loss_per_token from shared context
        loss_per_token = shared_context['loss_per_token']
        
        # loss_per_token is already suffix-only from shared_context in evaluate_mia
        # In extract.py it's computed on full sequence
        likelihood = loss_per_token.mean(1)
        
        return likelihood.cpu().numpy()  # Returns positive loss values
    
    def direction(self) -> str:
        return "min"  # Lower loss is better
