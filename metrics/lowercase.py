"""
Lowercase perturbation scoring metric.
"""

import torch
from metrics import AbstractMetric
from typing import Dict, Any


class LowercaseMetric(AbstractMetric):
    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any]):
        super().__init__(name, model, tokenizer, config)

    def compute_score(self, generated_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute lowercase perturbation scores.
        Compares original vs lowercase version likelihood.
        """
        # Get original NLLs
        outputs = self.model(generated_tokens, labels=generated_tokens)
        original_nlls = []
        
        for i in range(generated_tokens.shape[0]):
            seq_outputs = self.model(generated_tokens[i].unsqueeze(0), 
                                    labels=generated_tokens[i].unsqueeze(0))
            original_nlls.append(seq_outputs.loss.item())
        
        original_nlls = torch.tensor(original_nlls, device=self.device)
        
        # Decode to text and lowercase
        decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        lowercase_texts = [text.lower() for text in decoded_texts]
        
        # Re-encode lowercase versions
        lowercase_inputs = self.tokenizer(
            lowercase_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=generated_tokens.shape[1]
        ).to(self.device)
        
        # Get lowercase NLLs
        lowercase_outputs = self.model(lowercase_inputs.input_ids, 
                                      labels=lowercase_inputs.input_ids)
        
        lowercase_logits = lowercase_outputs.logits[..., :-1, :].contiguous()
        shift_labels = lowercase_inputs.input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(lowercase_logits.view(-1, lowercase_logits.size(-1)), 
                       shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        lowercase_nlls = (loss * mask).sum(dim=1)
        
        # Score = -original_nll / (lowercase_nll + eps)
        scores = -original_nlls / (lowercase_nlls + 1e-9)
        return scores
    
    def uses_argmax(self) -> bool:
        return True
