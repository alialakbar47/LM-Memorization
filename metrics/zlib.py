"""
Zlib compression-based metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
import zlib
from .base import BaseMetric


class ZlibMetric(BaseMetric):
    """Zlib metric - combines likelihood with compression ratio."""
    
    def __init__(self, suffix_len: int = 50, **kwargs):
        super().__init__(name="zlib", **kwargs)
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute zlib compression scores."""
        # Use pre-computed loss_per_token from shared context
        loss_per_token = shared_context['loss_per_token']
        generated_tokens = shared_context['generated_tokens']
        suffix_len = shared_context['suffix_len']
        
        # Extract suffix portion and compute likelihood
        loss_per_token_suffix = loss_per_token[:, -suffix_len:]
        likelihood = loss_per_token_suffix.mean(1)
        
        # Calculate zlib compression scores
        zlib_likelihood = np.zeros_like(likelihood.cpu().numpy())
        for batch_i in range(likelihood.shape[0]):
            prompt = generated_tokens[batch_i].cpu().numpy()
            compressed_len = len(zlib.compress(prompt.tobytes()))
            zlib_likelihood[batch_i] = likelihood[batch_i].item() * compressed_len
        
        return zlib_likelihood
    
    def direction(self) -> str:
        return "min"  # Lower is better
