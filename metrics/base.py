"""
Base metric class for all scoring methods.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
    
    @abstractmethod
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: Dict[str, Any]) -> np.ndarray:
        """
        Compute the metric score using pre-computed shared values.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            device: Device for computation
            shared_context: Dictionary containing pre-computed values:
                - generated_tokens: Generated token IDs
                - input_ids: Input token IDs (prompts)
                - outputs: Model outputs from forward pass
                - logits: Logits from model [:, :-1]
                - labels: Target labels (shifted tokens) [:, 1:]
                - loss_per_token: Cross-entropy loss per token (2D)
                - full_loss_per_token_flat: Flattened loss per token (1D)
                - log_probs_batch: Log probabilities for all vocab
                - token_log_probs: Log probs for actual tokens
                - mu: Mean log probability per position
                - sigma: Std log probability per position
                - mask: Padding mask
                - original_nlls: Normalized negative log-likelihood
                - suffix_len: Length of suffix
                - non_member_prefix: Non-member prefix data
                - member_prefix: Member prefix data
                - batch_offset: Batch offset for indexing
            
        Returns:
            Array of scores for each sequence
        """
        pass
    
    @abstractmethod
    def direction(self) -> str:
        """
        Return the direction of optimization ('min' or 'max').
        
        For MIA, higher scores should indicate membership.
        Returns:
            'min' if lower scores are better (argmin)
            'max' if higher scores are better (argmax)
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
