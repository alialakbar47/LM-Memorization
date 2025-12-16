"""
Zlib compression-based scoring metric.
"""

import torch
import zlib
from metrics import AbstractMetric
from typing import Dict, Any


class ZlibMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute zlib compression scores.
        Combines likelihood with compression ratio.
        """
        # First get likelihood scores
        outputs = self.model(generated_tokens, labels=generated_tokens)
        likelihood = outputs.loss.unsqueeze(0).expand(generated_tokens.shape[0])
        
        # Calculate compression for each sequence
        zlib_scores = []
        for batch_i in range(generated_tokens.shape[0]):
            prompt = generated_tokens[batch_i].cpu().numpy()
            compressed_len = len(zlib.compress(prompt.tobytes()))
            zlib_score = likelihood[batch_i].item() * compressed_len
            zlib_scores.append(zlib_score)
        
        return torch.tensor(zlib_scores, device=self.device)
    
    def uses_argmin(self) -> bool:
        return True
