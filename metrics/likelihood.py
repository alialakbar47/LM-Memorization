"""
Likelihood-based scoring metric.
"""

import torch
import torch.nn.functional as F
from metrics import AbstractMetric
from typing import Dict, Any


class LikelihoodMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute negative log-likelihood scores.
        Lower scores indicate higher likelihood (more memorized).
        """
        outputs = self.model(generated_tokens, labels=generated_tokens)
        logits = outputs.logits[:, :-1, :]
        shift_labels = generated_tokens[:, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
        loss = loss.view(shift_labels.size())
        
        # Mean loss per sequence (negative log-likelihood)
        scores = loss.mean(dim=1)
        return scores
    
    def uses_argmin(self) -> bool:
        return True
