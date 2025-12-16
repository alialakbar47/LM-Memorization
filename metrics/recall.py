"""
Recall scoring metric.
"""

import torch
from metrics import AbstractMetric
from typing import Dict, Any


class RecallMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.prefix_len = config.get('prefix_len', 50)
        self.non_member_prefix_pool = config.get('non_member_prefix_pool', None)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute recall scores using non-member prefix conditioning.
        Compares NLL with and without non-member prefix.
        """
        prefix_tokens_batch = kwargs.get('prefix_tokens', None)
        suffix_tokens_batch = kwargs.get('suffix_tokens', None)
        
        if prefix_tokens_batch is None or suffix_tokens_batch is None:
            # Fall back to splitting generated tokens
            prefix_tokens_batch = generated_tokens[:, :self.prefix_len]
            suffix_tokens_batch = generated_tokens[:, self.prefix_len:]
        
        scores = []
        non_member_prefix = self._get_non_member_prefix()
        
        for i in range(generated_tokens.shape[0]):
            input_tokens = prefix_tokens_batch[i]
            suffix_tokens = suffix_tokens_batch[i]
            
            # NLL without non-member prefix
            full_sequence = torch.cat((input_tokens, suffix_tokens))
            outputs = self.model(full_sequence.unsqueeze(0), labels=full_sequence.unsqueeze(0))
            nll_unconditional = outputs.loss.item()
            
            # NLL with non-member prefix
            full_sequence_with_prefix = torch.cat((non_member_prefix, input_tokens, suffix_tokens))
            outputs_with_prefix = self.model(full_sequence_with_prefix.unsqueeze(0), 
                                            labels=full_sequence_with_prefix.unsqueeze(0))
            nll_conditional = outputs_with_prefix.loss.item()
            
            # Recall score (higher is better for members)
            score = nll_conditional - nll_unconditional
            scores.append(score)
        
        return torch.tensor(scores, device=self.device)
    
    def _get_non_member_prefix(self):
        """Get or generate non-member prefix tokens."""
        if self.non_member_prefix_pool is not None:
            # Use first prefix from pool
            return torch.tensor(self.non_member_prefix_pool[0], 
                              dtype=torch.int64, 
                              device=self.device)
        else:
            # Generate a simple non-member prefix
            return torch.randint(0, self.tokenizer.vocab_size, 
                               (self.prefix_len,), 
                               device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
