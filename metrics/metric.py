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
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute metric scores with outlier removal."""
        # Use pre-computed loss_per_token from shared context
        loss_per_token = shared_context['loss_per_token']
        suffix_len = shared_context['suffix_len']
        
        # Extract suffix portion
        loss_per_token_suffix = loss_per_token[:, -suffix_len:]
        loss_per_token_np = loss_per_token_suffix.cpu().numpy()
        
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
