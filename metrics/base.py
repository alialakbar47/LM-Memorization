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
                generated_tokens: torch.Tensor,
                outputs: Any,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """
        Compute the metric score.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            generated_tokens: Generated token sequences
            outputs: Model outputs from forward pass
            device: Device for computation
            **kwargs: Additional arguments
            
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
