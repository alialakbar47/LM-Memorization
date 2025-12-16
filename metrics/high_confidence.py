"""
High confidence scoring metric.
"""

import torch
import torch.nn.functional as F
from metrics import AbstractMetric
from typing import Dict, Any


class HighConfidenceMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.adjustment_factor = config.get('adjustment_factor', 0.15)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute high confidence scores.
        Adjusts loss based on prediction confidence.
        """
        suffix_len = kwargs.get('suffix_len', 50)
        
        outputs = self.model(generated_tokens, labels=generated_tokens)
        full_logits = outputs.logits[:, :-1, :]
        shift_labels = generated_tokens[:, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(full_logits.reshape(-1, full_logits.size(-1)), shift_labels.reshape(-1))
        full_loss_per_token = loss.view(shift_labels.size())
        
        # Calculate confidence flags
        top_scores, _ = full_logits.topk(2, dim=-1)
        flag1 = (top_scores[:, :, 0] - top_scores[:, :, 1]) > self.confidence_threshold
        flag2 = top_scores[:, :, 0] > 0
        flat_flag1 = flag1.reshape(-1)
        flat_flag2 = flag2.reshape(-1)
        
        # Adjust loss based on confidence
        mean_batch_loss = full_loss_per_token.mean()
        loss_adjusted_flat = full_loss_per_token - (flat_flag1.int() - flat_flag2.int()) * mean_batch_loss * self.adjustment_factor
        loss_adjusted_reshaped = loss_adjusted_flat.reshape(full_logits.shape[0], -1)
        loss_adjusted_suffix = loss_adjusted_reshaped[:, -suffix_len:]
        
        scores = loss_adjusted_suffix.mean(1)
        return scores
    
    def uses_argmin(self) -> bool:
        return True
