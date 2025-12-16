"""
Base class for all scoring metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import hashlib
import json


class AbstractMetric(ABC):
    """Base class for all scoring metrics."""
    
    @abstractmethod
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.name = name

    @abstractmethod
    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the metric score for generated sequences.
        
        Args:
            generated_tokens: Tensor of generated token IDs
            **kwargs: Additional arguments specific to each metric
            
        Returns:
            Tensor of scores for each sequence
        """
        pass

    def signature(self, dataset_info: str = "") -> str:
        """Generate a unique signature for this metric configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        encoded = (dataset_info + self.name + config_str).encode()
        hash_obj = hashlib.sha256(encoded)
        return hash_obj.hexdigest()[:32]
    
    def uses_argmin(self) -> bool:
        """Return True if this metric uses argmin for selection (lower is better)."""
        return True
    
    def uses_argmax(self) -> bool:
        """Return True if this metric uses argmax for selection (higher is better)."""
        return False
