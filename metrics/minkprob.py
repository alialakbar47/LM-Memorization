"""
Min-k% probability scoring metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
from metrics import AbstractMetric
from typing import Dict, Any


class MinkprobMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.k_ratio = config.get('k_ratio', 0.2)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute min-k% probability scores.
        Averages the lowest k% of token log probabilities.
        """
        outputs = self.model(generated_tokens, labels=generated_tokens)
        logits = outputs.logits[:, :-1, :]
        shift_labels = generated_tokens[:, 1:]
        
        log_probs_all_vocab = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all_vocab.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Calculate min-k scores
        token_log_probs_np = token_log_probs.cpu().numpy()
        sorted_log_probs = np.sort(token_log_probs_np, axis=1)
        
        scores = []
        for i in range(len(sorted_log_probs)):
            seq_len = shift_labels[i].shape[0]
            k = int(self.k_ratio * seq_len)
            k = max(1, k)  # At least one token
            min_k_score = sorted_log_probs[i][:k].mean()
            scores.append(min_k_score)
        
        return torch.tensor(scores, device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
