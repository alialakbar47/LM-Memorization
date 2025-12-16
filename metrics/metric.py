"""
Metric scoring with outlier removal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from metrics import AbstractMetric
from typing import Dict, Any


class MetricMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)
        self.num_std = config.get('num_std', 3)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute metric scores with outlier removal.
        Removes tokens that are more than num_std standard deviations from mean.
        """
        outputs = self.model(generated_tokens, labels=generated_tokens)
        logits = outputs.logits[:, :-1, :]
        shift_labels = generated_tokens[:, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
        loss_per_token = loss.view(shift_labels.size())
        
        # Convert to numpy for easier manipulation
        loss_per_token_np = loss_per_token.cpu().numpy()
        mean = np.mean(loss_per_token_np, axis=-1, keepdims=True)
        std = np.std(loss_per_token_np, axis=-1, keepdims=True)
        
        floor = mean - self.num_std * std
        upper = mean + self.num_std * std
        
        # Replace outliers with mean
        metric_loss = np.where(
            ((loss_per_token_np < floor) | (loss_per_token_np > upper)),
            mean,
            loss_per_token_np
        )
        
        scores = torch.tensor(metric_loss.mean(1), device=self.device)
        return scores
    
    def uses_argmin(self) -> bool:
        return True
