"""
Recall-based metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric
from transformers.cache_utils import DynamicCache


class SuffixRecallMetric(BaseMetric):
    """Suffix recall metric - ratio of unconditional to conditional likelihood."""
    
    def __init__(self, **kwargs):
        super().__init__(name="suffix_recall", **kwargs)
    
    @torch.no_grad()
    def _get_ll(self, model, tokens: torch.Tensor, device: torch.device) -> float:
        """Helper to get the mean log-likelihood of a sequence."""
        outputs = model(tokens.unsqueeze(0).to(device))
        logits = outputs.logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=tokens[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
        return token_log_probs.mean().item()
    
    @torch.no_grad()
    def _get_cond_ll(self, model, prefix: torch.Tensor, suffix: torch.Tensor, device: torch.device) -> float:
        """Helper to get the mean conditional log-likelihood of a suffix given a prefix."""
        prefix_outputs = model(prefix.unsqueeze(0).to(device))
        cache = DynamicCache.from_legacy_cache(prefix_outputs.past_key_values)
        suffix_outputs = model(suffix.unsqueeze(0).to(device), past_key_values=cache)
        logits = suffix_outputs.logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=suffix[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
        return token_log_probs.mean().item()
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute suffix recall scores."""
        generated_tokens = shared_context['generated_tokens']
        input_ids = shared_context['input_ids']
        suffix_len = shared_context['suffix_len']
        
        scores = []
        
        for batch_idx in range(generated_tokens.shape[0]):
            prefix = input_ids[batch_idx]
            suffix = generated_tokens[batch_idx, -suffix_len:]
            
            ll_unconditional = self._get_ll(model, suffix, device)
            ll_conditional = self._get_cond_ll(model, prefix, suffix, device)
            
            nll_unconditional = -ll_unconditional
            nll_conditional = -ll_conditional
            score = nll_unconditional / nll_conditional if nll_conditional != 0 else 0
            scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better


class RecallMetric(BaseMetric):
    """Recall metric using non-member prefix."""
    
    def __init__(self, **kwargs):
        super().__init__(name="recall", **kwargs)
    
    @torch.no_grad()
    def _calculate_recall(self, non_member_prefix_tokens: torch.Tensor, 
                         input_tokens: torch.Tensor, 
                         suffix_tokens: torch.Tensor, 
                         model, device: torch.device):
        """Calculate recall score based on NLL ratio."""
        non_member_prefix_tokens = non_member_prefix_tokens.to(device)
        input_tokens = input_tokens.to(device)
        suffix_tokens = suffix_tokens.to(device)
        
        full_sequence = torch.cat((input_tokens, suffix_tokens))
        outputs = model(full_sequence.unsqueeze(0), labels=full_sequence.unsqueeze(0))
        nll_unconditional = outputs.loss.item()
        
        full_sequence_with_prefix = torch.cat((non_member_prefix_tokens, input_tokens, suffix_tokens))
        outputs_with_prefix = model(full_sequence_with_prefix.unsqueeze(0), labels=full_sequence_with_prefix.unsqueeze(0))
        nll_conditional = outputs_with_prefix.loss.item()
        
        return nll_unconditional, nll_conditional
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute recall scores."""
        generated_tokens = shared_context['generated_tokens']
        input_ids = shared_context['input_ids']
        non_member_prefix = shared_context['non_member_prefix']
        suffix_len = shared_context['suffix_len']
        batch_offset = shared_context['batch_offset']
        
        if non_member_prefix is None:
            return np.zeros(generated_tokens.shape[0])
        
        scores = []
        
        for batch_idx in range(generated_tokens.shape[0]):
            prefix = input_ids[batch_idx]
            suffix = generated_tokens[batch_idx, -suffix_len:]
            
            nm_prefix_idx = (batch_offset + batch_idx) % len(non_member_prefix)
            nm_prefix = torch.tensor(non_member_prefix[nm_prefix_idx], dtype=torch.int64).to(device)
            
            nll_u, nll_c = self._calculate_recall(nm_prefix, prefix, suffix, model, device)
            score = nll_c / nll_u if nll_u != 0 else 0
            scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better


class ConRecallMetric(BaseMetric):
    """Contrastive recall metric."""
    
    def __init__(self, **kwargs):
        super().__init__(name="con_recall", **kwargs)
    
    @torch.no_grad()
    def _calculate_con_recall(self, non_member_prefix_tokens: torch.Tensor,
                             member_prefix_tokens: torch.Tensor,
                             full_sequence_tokens: torch.Tensor,
                             original_nll: float,
                             model, device: torch.device):
        """Calculate contrastive recall score."""
        non_member_prefix_tokens = non_member_prefix_tokens.to(device)
        member_prefix_tokens = member_prefix_tokens.to(device)
        full_sequence_tokens = full_sequence_tokens.to(device)
        
        nm_prefixed_sequence = torch.cat((non_member_prefix_tokens, full_sequence_tokens))
        nm_outputs = model(nm_prefixed_sequence.unsqueeze(0), labels=nm_prefixed_sequence.unsqueeze(0))
        nll_non_member = nm_outputs.loss.item()
        
        m_prefixed_sequence = torch.cat((member_prefix_tokens, full_sequence_tokens))
        m_outputs = model(m_prefixed_sequence.unsqueeze(0), labels=m_prefixed_sequence.unsqueeze(0))
        nll_member = m_outputs.loss.item()
        
        score = (nll_non_member - nll_member) / (original_nll + 1e-9)
        return score
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute contrastive recall scores."""
        generated_tokens = shared_context['generated_tokens']
        non_member_prefix = shared_context['non_member_prefix']
        member_prefix = shared_context['member_prefix']
        original_nlls = shared_context['original_nlls']
        batch_offset = shared_context['batch_offset']
        
        if non_member_prefix is None or member_prefix is None:
            return np.zeros(generated_tokens.shape[0])
        
        scores = []
        
        for batch_idx in range(generated_tokens.shape[0]):
            nm_prefix_idx = (batch_offset + batch_idx) % len(non_member_prefix)
            nm_prefix = torch.tensor(non_member_prefix[nm_prefix_idx], dtype=torch.int64).to(device)
            
            m_prefix_idx = (batch_offset + batch_idx) % len(member_prefix)
            m_prefix = torch.tensor(member_prefix[m_prefix_idx], dtype=torch.int64).to(device)
            
            score = self._calculate_con_recall(
                nm_prefix, m_prefix,
                generated_tokens[batch_idx], original_nlls[batch_idx].item(),
                model, device
            )
            scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better


class SuffixConRecallMetric(BaseMetric):
    """Suffix-based contrastive recall metric."""
    
    def __init__(self, **kwargs):
        super().__init__(name="suffix_conrecall", **kwargs)
    
    @torch.no_grad()
    def _calculate_suffix_con_recall(self, prefix_tokens: torch.Tensor, suffix_tokens: torch.Tensor,
                                    model, tokenizer, device: torch.device,
                                    non_member_prefix_pool: np.ndarray = None,
                                    example_id: int = 0):
        """Calculate suffix-based contrastive recall."""
        prefix_tokens = prefix_tokens.to(device)
        suffix_tokens = suffix_tokens.to(device)
        
        # Calculate unconditional NLL of suffix
        suffix_outputs = model(suffix_tokens.unsqueeze(0), labels=suffix_tokens.unsqueeze(0))
        original_nll = suffix_outputs.loss.item()
        
        # Get non-member prefix
        if non_member_prefix_pool is not None:
            pool_idx = example_id % len(non_member_prefix_pool)
            selected_non_member = torch.tensor(non_member_prefix_pool[pool_idx], dtype=torch.int64, device=device)
            
            target_length = len(prefix_tokens)
            if len(selected_non_member) == target_length:
                non_member_prefix = selected_non_member
            elif len(selected_non_member) > target_length:
                non_member_prefix = selected_non_member[:target_length]
            else:
                repeats_needed = (target_length + len(selected_non_member) - 1) // len(selected_non_member)
                repeated = selected_non_member.repeat(repeats_needed)
                non_member_prefix = repeated[:target_length]
        else:
            non_member_prefix = (prefix_tokens + 1000) % tokenizer.vocab_size
        
        # Calculate conditional NLLs
        prefix_len = len(prefix_tokens)
        
        # Member context
        member_sequence = torch.cat([prefix_tokens, suffix_tokens])
        member_outputs = model(member_sequence.unsqueeze(0), labels=member_sequence.unsqueeze(0))
        member_logits = member_outputs.logits[0, prefix_len-1:-1]
        member_loss = F.cross_entropy(member_logits, suffix_tokens, reduction='mean')
        nll_member = member_loss.item()
        
        # Non-member context
        non_member_sequence = torch.cat([non_member_prefix, suffix_tokens])
        non_member_outputs = model(non_member_sequence.unsqueeze(0), labels=non_member_sequence.unsqueeze(0))
        non_member_logits = non_member_outputs.logits[0, prefix_len-1:-1]
        non_member_loss = F.cross_entropy(non_member_logits, suffix_tokens, reduction='mean')
        nll_non_member = non_member_loss.item()
        
        score = (nll_non_member - nll_member) / (original_nll + 1e-9)
        return score
    
    def compute(self, 
                model,
                tokenizer,
                device: torch.device,
                shared_context: dict) -> np.ndarray:
        """Compute suffix contrastive recall scores."""
        generated_tokens = shared_context['generated_tokens']
        input_ids = shared_context['input_ids']
        non_member_prefix = shared_context['non_member_prefix']
        suffix_len = shared_context['suffix_len']
        batch_offset = shared_context['batch_offset']
        
        scores = []
        
        for batch_idx in range(generated_tokens.shape[0]):
            prefix = input_ids[batch_idx]
            suffix = generated_tokens[batch_idx, -suffix_len:]
            
            score = self._calculate_suffix_con_recall(
                prefix, suffix, model, tokenizer, device,
                non_member_prefix, batch_offset + batch_idx
            )
            scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better
