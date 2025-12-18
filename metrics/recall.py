"""
Recall-based metrics - exact match with old utils_old.py implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseMetric


class SuffixRecallMetric(BaseMetric):
    """Suffix recall metric - ratio of unconditional to conditional likelihood.
    
    OLD FORMULA (utils_old.py lines 133-168):
    ll_unconditional = get_ll(model, suffix_tokens, device)  # P(suffix)
    ll_conditional = get_cond_ll(model, prefix_tokens, suffix_tokens, device)  # P(suffix | prefix)
    score = nll_unconditional / nll_conditional
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="suffix_recall", **kwargs)
    
    @torch.no_grad()
    def _get_ll(self, model, tokens: torch.Tensor, device: torch.device) -> float:
        """Get log-likelihood P(tokens) - matches get_ll from utils_old.py."""
        tokens = tokens.to(device)
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=tokens[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
        return token_log_probs.mean().item()
    
    @torch.no_grad()
    def _get_cond_ll(self, model, prefix: torch.Tensor, suffix: torch.Tensor, device: torch.device) -> float:
        """Get conditional log-likelihood P(suffix | prefix) - matches get_cond_ll from utils_old.py."""
        prefix = prefix.to(device)
        suffix = suffix.to(device)
        
        # Forward pass on full sequence
        full_sequence = torch.cat([prefix, suffix])
        outputs = model(full_sequence.unsqueeze(0))
        
        # Get logits for suffix prediction (after prefix)
        prefix_len = len(prefix)
        logits = outputs.logits[0, prefix_len-1:-1]  # Logits that predict suffix tokens
        
        # Calculate log probs for suffix tokens
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=suffix.unsqueeze(-1)).squeeze()
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
    """Recall metric using non-member prefix.
    
    OLD FORMULA (utils_old.py lines 205-222):
    full_sequence = input_tokens + suffix_tokens
    nll_unconditional = model(full_sequence).loss
    
    full_sequence_with_prefix = non_member_prefix + input_tokens + suffix_tokens
    nll_conditional = model(full_sequence_with_prefix).loss
    
    score = nll_c / nll_u
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="recall", **kwargs)
    
    @torch.no_grad()
    def _calculate_recall(self, non_member_prefix_tokens: torch.Tensor, 
                         input_tokens: torch.Tensor, 
                         suffix_tokens: torch.Tensor, 
                         model, device: torch.device):
        """Calculate recall score - exact match with utils_old.py calculate_recall."""
        non_member_prefix_tokens = non_member_prefix_tokens.to(device)
        input_tokens = input_tokens.to(device)
        suffix_tokens = suffix_tokens.to(device)
        
        # Unconditional: full_sequence (input + suffix)
        full_sequence = torch.cat((input_tokens, suffix_tokens))
        outputs = model(full_sequence.unsqueeze(0), labels=full_sequence.unsqueeze(0))
        nll_unconditional = outputs.loss.item()
        
        # Conditional: prepend non_member_prefix
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
            nm_prefix = torch.tensor(non_member_prefix[nm_prefix_idx], dtype=torch.int64, device=device)
            
            nll_u, nll_c = self._calculate_recall(nm_prefix, prefix, suffix, model, device)
            score = nll_c / nll_u if nll_u != 0 else 0
            scores.append(score)
        
        return np.array(scores)
    
    def direction(self) -> str:
        return "max"  # Higher is better


class ConRecallMetric(BaseMetric):
    """Contrastive recall metric.
    
    OLD FORMULA (utils_old.py lines 235-261):
    nm_prefixed = non_member_prefix + full_sequence
    nll_non_member = model(nm_prefixed).loss
    
    m_prefixed = member_prefix + full_sequence
    nll_member = model(m_prefixed).loss
    
    score = (nll_non_member - nll_member) / (original_nll + 1e-9)
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="con_recall", **kwargs)
    
    @torch.no_grad()
    def _calculate_con_recall(self, non_member_prefix_tokens: torch.Tensor,
                             member_prefix_tokens: torch.Tensor,
                             full_sequence_tokens: torch.Tensor,
                             original_nll: float,
                             model, device: torch.device):
        """Calculate contrastive recall - exact match with utils_old.py calculate_con_recall."""
        non_member_prefix_tokens = non_member_prefix_tokens.to(device)
        member_prefix_tokens = member_prefix_tokens.to(device)
        full_sequence_tokens = full_sequence_tokens.to(device)
        
        # Non-member prefixed sequence
        nm_prefixed_sequence = torch.cat((non_member_prefix_tokens, full_sequence_tokens))
        nm_outputs = model(nm_prefixed_sequence.unsqueeze(0), labels=nm_prefixed_sequence.unsqueeze(0))
        nll_non_member = nm_outputs.loss.item()
        
        # Member prefixed sequence
        m_prefixed_sequence = torch.cat((member_prefix_tokens, full_sequence_tokens))
        m_outputs = model(m_prefixed_sequence.unsqueeze(0), labels=m_prefixed_sequence.unsqueeze(0))
        nll_member = m_outputs.loss.item()
        
        # Score formula
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
            nm_prefix = torch.tensor(non_member_prefix[nm_prefix_idx], dtype=torch.int64, device=device)
            
            m_prefix_idx = (batch_offset + batch_idx) % len(member_prefix)
            m_prefix = torch.tensor(member_prefix[m_prefix_idx], dtype=torch.int64, device=device)
            
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
    """Suffix-based contrastive recall metric.
    
    OLD FORMULA (utils_old.py lines 170-202):
    original_nll = model(suffix_tokens).loss  # just suffix
    
    member_sequence = prefix_tokens + suffix_tokens
    nll_member = model(member_sequence).loss
    
    non_member_sequence = non_member_prefix + suffix_tokens
    nll_non_member = model(non_member_sequence).loss
    
    score = (nll_non_member - nll_member) / (original_nll + 1e-9)
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="suffix_conrecall", **kwargs)
    
    @torch.no_grad()
    def _calculate_suffix_con_recall(self, prefix_tokens: torch.Tensor, 
                                    suffix_tokens: torch.Tensor,
                                    model, tokenizer, device: torch.device,
                                    non_member_prefix_pool: np.ndarray = None,
                                    example_id: int = 0):
        """Calculate suffix contrastive recall - exact match with utils_old.py calculate_suffix_con_recall."""
        prefix_tokens = prefix_tokens.to(device)
        suffix_tokens = suffix_tokens.to(device)
        
        # Original NLL (just suffix, no context)
        suffix_outputs = model(suffix_tokens.unsqueeze(0), labels=suffix_tokens.unsqueeze(0))
        original_nll = suffix_outputs.loss.item()
        
        # Member NLL (prefix + suffix)
        member_sequence = torch.cat([prefix_tokens, suffix_tokens])
        member_outputs = model(member_sequence.unsqueeze(0), labels=member_sequence.unsqueeze(0))
        nll_member = member_outputs.loss.item()
        
        # Non-member NLL (non_member_prefix + suffix)
        if non_member_prefix_pool is not None:
            nm_prefix_idx = example_id % len(non_member_prefix_pool)
            non_member_prefix = torch.tensor(non_member_prefix_pool[nm_prefix_idx], dtype=torch.int64, device=device)
        else:
            # Fallback if no pool provided
            non_member_prefix = prefix_tokens
        
        non_member_sequence = torch.cat([non_member_prefix, suffix_tokens])
        non_member_outputs = model(non_member_sequence.unsqueeze(0), labels=non_member_sequence.unsqueeze(0))
        nll_non_member = non_member_outputs.loss.item()
        
        # Score formula (same as con_recall)
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
