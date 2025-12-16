"""
Suffix recall scoring metric.
"""

import torch
import torch.nn.functional as F
from metrics import AbstractMetric
from typing import Dict, Any
from transformers.cache_utils import DynamicCache


class SuffixRecallMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.prefix_len = config.get('prefix_len', 50)
        self.suffix_len = config.get('suffix_len', 50)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute suffix recall scores.
        Compares conditional vs unconditional likelihood of suffix.
        """
        scores = []
        
        for i in range(generated_tokens.shape[0]):
            full_seq = generated_tokens[i]
            prefix_tokens = full_seq[:self.prefix_len]
            suffix_tokens = full_seq[self.prefix_len:self.prefix_len + self.suffix_len]
            
            # Unconditional likelihood of suffix
            suffix_outputs = self.model(suffix_tokens.unsqueeze(0), labels=suffix_tokens.unsqueeze(0))
            ll_unconditional = -suffix_outputs.loss.item()
            
            # Conditional likelihood of suffix given prefix
            prefix_outputs = self.model(prefix_tokens.unsqueeze(0))
            cache = DynamicCache.from_legacy_cache(prefix_outputs.past_key_values)
            suffix_outputs_cond = self.model(suffix_tokens.unsqueeze(0), 
                                            past_key_values=cache, 
                                            labels=suffix_tokens.unsqueeze(0))
            ll_conditional = -suffix_outputs_cond.loss.item()
            
            # Recall score: conditional - unconditional (higher is better)
            score = ll_conditional - ll_unconditional
            scores.append(score)
        
        return torch.tensor(scores, device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
