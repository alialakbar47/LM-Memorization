"""
Contrastive recall (con_recall) scoring metric.
"""

import torch
import torch.nn.functional as F
from metrics import AbstractMetric
from typing import Dict, Any


class ConRecallMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.non_member_prefix_pool = config.get('non_member_prefix_pool', None)
        self.member_prefix_pool = config.get('member_prefix_pool', None)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute contrastive recall scores.
        Compares NLL when prefixed with member vs non-member contexts.
        """
        scores = []
        
        # Get original NLL for normalization
        outputs = self.model(generated_tokens, labels=generated_tokens)
        original_nlls = outputs.loss.unsqueeze(0).expand(generated_tokens.shape[0])
        
        for i in range(generated_tokens.shape[0]):
            full_sequence = generated_tokens[i]
            
            # Get prefixes
            non_member_prefix = self._get_non_member_prefix(i)
            member_prefix = self._get_member_prefix(i)
            
            # NLL with non-member prefix
            nm_sequence = torch.cat((non_member_prefix, full_sequence))
            nm_outputs = self.model(nm_sequence.unsqueeze(0), labels=nm_sequence.unsqueeze(0))
            nll_non_member = nm_outputs.loss.item()
            
            # NLL with member prefix
            m_sequence = torch.cat((member_prefix, full_sequence))
            m_outputs = self.model(m_sequence.unsqueeze(0), labels=m_sequence.unsqueeze(0))
            nll_member = m_outputs.loss.item()
            
            # Contrastive score normalized by original NLL
            score = (nll_non_member - nll_member) / (original_nlls[i].item() + 1e-9)
            scores.append(score)
        
        return torch.tensor(scores, device=self.device)
    
    def _get_non_member_prefix(self, idx: int):
        """Get non-member prefix tokens."""
        if self.non_member_prefix_pool is not None:
            pool_idx = idx % len(self.non_member_prefix_pool)
            return torch.tensor(self.non_member_prefix_pool[pool_idx], 
                              dtype=torch.int64, 
                              device=self.device)
        else:
            return torch.randint(0, self.tokenizer.vocab_size, 
                               (50,), 
                               device=self.device)
    
    def _get_member_prefix(self, idx: int):
        """Get member prefix tokens."""
        if self.member_prefix_pool is not None:
            pool_idx = idx % len(self.member_prefix_pool)
            return torch.tensor(self.member_prefix_pool[pool_idx], 
                              dtype=torch.int64, 
                              device=self.device)
        else:
            # Generate random member prefix
            return torch.randint(0, self.tokenizer.vocab_size, 
                               (50,), 
                               device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
