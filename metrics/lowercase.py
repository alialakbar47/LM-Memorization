"""
Lowercase metric.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric


class LowercaseMetric(BaseMetric):
    """Lowercase metric - compares original vs lowercase NLL."""
    
    def __init__(self, **kwargs):
        super().__init__(name="lowercase", **kwargs)
    
    @torch.no_grad()
    def compute(self, 
                model,
                tokenizer,
                generated_tokens: torch.Tensor,
                outputs,
                device: torch.device,
                **kwargs) -> np.ndarray:
        """Calculate lowercase scores for a batch of generated sequences."""
        # Calculate original NLLs
        full_labels = generated_tokens[:, 1:].contiguous()
        mask = (full_labels != tokenizer.pad_token_id).float()
        
        full_logits = outputs.logits[:, :-1].reshape((-1, outputs.logits.shape[-1])).float()
        full_loss_per_token_flat = F.cross_entropy(
            full_logits, 
            generated_tokens[:, 1:].flatten(), 
            reduction='none'
        )
        
        original_nlls = (full_loss_per_token_flat.reshape(full_labels.shape) * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Calculate lowercase NLLs
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        lowercase_texts = [text.lower() for text in decoded_texts]
        
        lowercase_inputs = tokenizer(
            lowercase_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=generated_tokens.shape[1]
        ).to(device)
        
        lowercase_outputs = model(lowercase_inputs.input_ids, labels=lowercase_inputs.input_ids)
        
        lowercase_logits = lowercase_outputs.logits
        shift_logits = lowercase_logits[..., :-1, :].contiguous()
        shift_labels = lowercase_inputs.input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        
        mask = (shift_labels != tokenizer.pad_token_id).float()
        lowercase_nlls = (loss * mask).sum(dim=1)
        
        # Score = -original_nll / (lowercase_nll + 1e-9)
        scores = -original_nlls / (lowercase_nlls + 1e-9)
        return scores.cpu().numpy()
    
    def direction(self) -> str:
        return "max"  # Higher is better
