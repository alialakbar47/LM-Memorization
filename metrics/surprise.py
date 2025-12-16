"""
Surprise (min-k with entropy filtering) scoring metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
from metrics import AbstractMetric
from typing import Dict, Any


class SurpriseMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.k_ratio = config.get('k_ratio', 0.2)
        self.max_entropy = config.get('max_entropy', 2.0)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute surprise scores (min-k with entropy threshold).
        Only considers tokens with entropy below max_entropy threshold.
        """
        outputs = self.model(generated_tokens, labels=generated_tokens)
        logits = outputs.logits[:, :-1, :]
        shift_labels = generated_tokens[:, 1:]
        
        log_probs_all_vocab = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all_vocab.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Calculate entropy for filtering
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all_vocab).sum(dim=-1)
        
        # Filter by entropy and calculate surprise scores
        token_log_probs_np = token_log_probs.cpu().numpy()
        entropy_np = entropy.cpu().numpy()
        
        scores = []
        for i in range(len(token_log_probs_np)):
            # Filter tokens by entropy threshold
            valid_mask = entropy_np[i] < self.max_entropy
            valid_log_probs = token_log_probs_np[i][valid_mask]
            
            if len(valid_log_probs) > 0:
                sorted_log_probs = np.sort(valid_log_probs)
                k = int(self.k_ratio * len(sorted_log_probs))
                k = max(1, k)
                surprise_score = sorted_log_probs[:k].mean()
            else:
                # If no tokens pass entropy threshold, use mean of all
                surprise_score = token_log_probs_np[i].mean()
            
            scores.append(surprise_score)
        
        return torch.tensor(scores, device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
