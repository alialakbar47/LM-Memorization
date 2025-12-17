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
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute zlib compression scores."""
        # First compute likelihood
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token_flat = F.cross_entropy(
            full_logits, 
            generated_tokens[:, 1:].flatten(), 
            reduction='none'
        )
        
        loss_per_token = full_loss_per_token_flat.reshape(-1, generated_tokens.shape[1] - 1)[:, -self.suffix_len:]
        likelihood = loss_per_token.mean(1)
        
        # Calculate zlib compression scores
        zlib_likelihood = np.zeros_like(likelihood.cpu().numpy())
        for batch_i in range(likelihood.shape[0]):
            prompt = generated_tokens[batch_i].cpu().numpy()
            compressed_len = len(zlib.compress(prompt.tobytes()))
            zlib_likelihood[batch_i] = likelihood[batch_i].item() * compressed_len
        
        return zlib_likelihood
    
    def direction(self) -> str:
        return "min"  # Lower is better
