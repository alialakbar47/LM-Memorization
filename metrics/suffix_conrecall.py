"""
Suffix contrastive recall (suffix_conrecall) scoring metric.
"""

import torch
import torch.nn.functional as F
from metrics import AbstractMetric
from typing import Dict, Any


class SuffixConRecallMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.non_member_prefix_pool = config.get('non_member_prefix_pool', None)
        self.prefix_len = config.get('prefix_len', 50)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute suffix-based contrastive recall scores.
        Compares suffix likelihood with original vs non-member prefix.
        """
        scores = []
        
        for i in range(generated_tokens.shape[0]):
            full_seq = generated_tokens[i]
            prefix_tokens = full_seq[:self.prefix_len]
            suffix_tokens = full_seq[self.prefix_len:]
            
            # Original NLL of suffix alone (for normalization)
            suffix_outputs = self.model(suffix_tokens.unsqueeze(0), labels=suffix_tokens.unsqueeze(0))
            original_nll = suffix_outputs.loss.item()
            
            # Get non-member prefix
            non_member_prefix = self._get_non_member_prefix(i, len(prefix_tokens))
            
            # NLL of suffix with member prefix (original)
            member_sequence = torch.cat([prefix_tokens, suffix_tokens])
            member_outputs = self.model(member_sequence.unsqueeze(0), labels=member_sequence.unsqueeze(0))
            
            # Extract loss for suffix portion only
            member_logits = member_outputs.logits[0, len(prefix_tokens)-1:-1]
            member_loss = F.cross_entropy(member_logits, suffix_tokens, reduction='mean')
            nll_member = member_loss.item()
            
            # NLL of suffix with non-member prefix
            non_member_sequence = torch.cat([non_member_prefix, suffix_tokens])
            non_member_outputs = self.model(non_member_sequence.unsqueeze(0), 
                                          labels=non_member_sequence.unsqueeze(0))
            
            # Extract loss for suffix portion only
            non_member_logits = non_member_outputs.logits[0, len(non_member_prefix)-1:-1]
            non_member_loss = F.cross_entropy(non_member_logits, suffix_tokens, reduction='mean')
            nll_non_member = non_member_loss.item()
            
            # Contrastive score
            score = (nll_non_member - nll_member) / (original_nll + 1e-9)
            scores.append(score)
        
        return torch.tensor(scores, device=self.device)
    
    def _get_non_member_prefix(self, idx: int, target_length: int):
        """Get non-member prefix tokens matching target length."""
        if self.non_member_prefix_pool is not None:
            pool_idx = idx % len(self.non_member_prefix_pool)
            selected = torch.tensor(self.non_member_prefix_pool[pool_idx], 
                                   dtype=torch.int64, 
                                   device=self.device)
            
            # Adjust length to match
            if len(selected) == target_length:
                return selected
            elif len(selected) > target_length:
                return selected[:target_length]
            else:
                # Repeat to reach target length
                repeats = (target_length + len(selected) - 1) // len(selected)
                repeated = selected.repeat(repeats)
                return repeated[:target_length]
        else:
            return torch.randint(0, self.tokenizer.vocab_size, 
                               (target_length,), 
                               device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
