"""
Metric score with outlier removal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric


class MetricMetric(BaseMetric):
    """Metric score - likelihood with outlier removal."""
    
    def __init__(self, suffix_len: int = 50, **kwargs):
        super().__init__(name="metric", **kwargs)
        self.suffix_len = suffix_len
    
    def compute(self, 
                model,
                tokenizer,
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Compute metric scores with outlier removal."""
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token_flat = F.cross_entropy(
            full_logits, 
            generated_tokens[:, 1:].flatten(), 
            reduction='none'
        )
        
        loss_per_token = full_loss_per_token_flat.reshape(-1, generated_tokens.shape[1] - 1)[:, -self.suffix_len:]
        loss_per_token_np = loss_per_token.cpu().numpy()
        
        # Outlier removal
        mean = np.mean(loss_per_token_np, axis=-1, keepdims=True)
        std = np.std(loss_per_token_np, axis=-1, keepdims=True)
        floor = mean - 3*std
        upper = mean + 3*std
        
        metric_loss = np.where(
            ((loss_per_token_np < floor) | (loss_per_token_np > upper)),
            mean,
            loss_per_token_np
        )
        
        return metric_loss.mean(1)
    
    def direction(self) -> str:
        return "min"  # Lower is better
