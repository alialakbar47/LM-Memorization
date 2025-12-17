"""
Min-k++ (normalized) scoring metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
from metrics import AbstractMetric
from typing import Dict, Any


class MinkplusplusMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.k_ratio = config.get('k_ratio', 0.2)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute min-k++ scores (normalized version of min-k).
        Normalizes token log probabilities before selecting minimum k%.
        """
        outputs = self.model(generated_tokens, labels=generated_tokens)
        logits = outputs.logits[:, :-1, :]
        shift_labels = generated_tokens[:, 1:]
        
        log_probs_all_vocab = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all_vocab.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Normalize: (log_prob - mean) / std
        mu = log_probs_all_vocab.mean(dim=-1)
        sigma = log_probs_all_vocab.std(dim=-1)
        normalized_scores = (token_log_probs - mu) / (sigma + 1e-10)
        
        # Calculate min-k++ scores
        normalized_scores_np = normalized_scores.cpu().numpy()
        sorted_scores = np.sort(normalized_scores_np, axis=1)
        
        scores = []
        for i in range(len(sorted_scores)):
            seq_len = shift_labels[i].shape[0]
            k = int(self.k_ratio * seq_len)
            k = max(1, k)
            min_k_plus_score = sorted_scores[i][:k].mean()
            scores.append(min_k_plus_score)
        
        return torch.tensor(scores, device=self.device)
    
    def uses_argmax(self) -> bool:
        return True
