"""
Utility functions for LLM data extraction and MIA evaluation.
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import zlib
import random
from typing import Tuple, Dict, List, Union
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def init_seeds(seed: int):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


@functools.lru_cache(maxsize=1)
def load_model_and_tokenizer(model_name: str = "EleutherAI/gpt-neo-1.3B"):
    """Load model and tokenizer with caching."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Set a default pad token id if none exists
            tokenizer.pad_token_id = 50256
    return model, tokenizer


def load_prompts(dir_path: str, file_name: str, allow_pickle: bool = False) -> np.ndarray:
    """Load prompts from numpy file."""
    try:
        return np.load(os.path.join(dir_path, file_name)).astype(np.int64)
    except ValueError as e:
        if "allow_pickle=False" in str(e):
            data = np.load(os.path.join(dir_path, file_name), allow_pickle=True)
            if data.dtype == np.dtype('O'):
                return np.array([np.array(x, dtype=np.int64) for x in data], dtype=np.int64)
            return data.astype(np.int64)
        raise


def write_array(file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Write numpy array to file with unique ID."""
    file_name = file_path.format(unique_id)
    np.save(file_name, array)


def prepare_directories(root_dir: str, experiment_name: str):
    """Create necessary directories for experiment."""
    experiment_base = os.path.join(root_dir, experiment_name)
    generations_base = os.path.join(experiment_base, "generations")
    losses_base = os.path.join(experiment_base, "losses")
    
    os.makedirs(generations_base, exist_ok=True)
    os.makedirs(losses_base, exist_ok=True)
    
    return experiment_base, generations_base, losses_base


def perturb_prefix(prefix_tokens: torch.Tensor, vocab_size: int, p: float = 0.2) -> torch.Tensor:
    """Randomly replaces a fraction `p` of tokens in a prefix."""
    perturbed_tokens = prefix_tokens.clone()
    num_to_perturb = int(len(prefix_tokens) * p)
    if num_to_perturb == 0 and len(prefix_tokens) > 0:
        num_to_perturb = 1
    
    indices_to_perturb = torch.randperm(len(prefix_tokens))[:num_to_perturb]
    replacements = torch.randint(0, vocab_size, (num_to_perturb,), device=prefix_tokens.device)
    perturbed_tokens[indices_to_perturb] = replacements
    return perturbed_tokens


@torch.no_grad()
def get_ll(model, tokens: torch.Tensor, device: torch.device) -> float:
    """Helper to get the mean log-likelihood of a sequence."""
    outputs = model(tokens.unsqueeze(0).to(device))
    logits = outputs.logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=tokens[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
    return token_log_probs.mean().item()


@torch.no_grad()
def get_cond_ll(model, prefix: torch.Tensor, suffix: torch.Tensor, device: torch.device) -> float:
    """Helper to get the mean conditional log-likelihood of a suffix given a prefix."""
    prefix_outputs = model(prefix.unsqueeze(0).to(device))
    cache = DynamicCache.from_legacy_cache(prefix_outputs.past_key_values)
    suffix_outputs = model(suffix.unsqueeze(0).to(device), past_key_values=cache)
    logits = suffix_outputs.logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=suffix[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
    return token_log_probs.mean().item()


@torch.no_grad()
def calculate_recall_scores(prefix_tokens: torch.Tensor, suffix_tokens: torch.Tensor, 
                          model, device: torch.device) -> Tuple[float, float]:
    """Calculate recall scores using prefix-suffix split."""
    ll_unconditional = get_ll(model, suffix_tokens, device)
    ll_conditional = get_cond_ll(model, prefix_tokens, suffix_tokens, device)
    return ll_unconditional, ll_conditional


@torch.no_grad()
def calculate_suffix_con_recall(prefix_tokens: torch.Tensor, suffix_tokens: torch.Tensor,
                               model, tokenizer, device: torch.device) -> float:
    """Calculate suffix-based contrastive recall."""
    ll_unconditional = get_ll(model, suffix_tokens, device)
    ll_cond_member = get_cond_ll(model, prefix_tokens, suffix_tokens, device)
    
    perturbed_prefix = perturb_prefix(prefix_tokens, tokenizer.vocab_size)
    ll_cond_non_member = get_cond_ll(model, perturbed_prefix, suffix_tokens, device)
    
    return (ll_cond_member - ll_cond_non_member) / (abs(ll_unconditional) + 1e-9)


@torch.no_grad()
def calculate_recall(non_member_prefix_tokens: torch.Tensor, 
                     input_tokens: torch.Tensor, 
                     suffix_tokens: torch.Tensor, 
                     model, device: torch.device) -> Tuple[float, float]:
    """
    Calculate recall score based on NLL ratio, matching the provided recall.py logic.
    Returns (unconditional_nll, conditional_nll).
    """
    # [FIXED] Ensure all tensors are on the same device before concatenation.
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


@torch.no_grad()
def calculate_con_recall(
    non_member_prefix_tokens: torch.Tensor,
    member_prefix_tokens: torch.Tensor,
    full_sequence_tokens: torch.Tensor,
    original_nll: float,
    model,
    device: torch.device
) -> float:
    """Calculate contrastive recall score."""
    # [FIXED] Ensure all tensors are on the same device before concatenation.
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


@torch.no_grad()
def calculate_min_k_scores(logits_batch: torch.Tensor, 
                         input_ids_batch: torch.Tensor, 
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate min_k and min_k_plus scores efficiently."""
    log_probs_all_vocab = F.log_softmax(logits_batch, dim=-1)
    token_log_probs = log_probs_all_vocab.gather(dim=-1, index=input_ids_batch).squeeze(-1)
    
    mu = log_probs_all_vocab.mean(dim=-1)
    sigma = log_probs_all_vocab.std(dim=-1)
    
    mink_plus = (token_log_probs - mu) / sigma
    
    return token_log_probs.cpu().numpy(), mink_plus.cpu().numpy()


def calculate_zlib_scores(generated_tokens: torch.Tensor, likelihood: torch.Tensor) -> np.ndarray:
    """Calculate zlib compression scores."""
    zlib_likelihood = np.zeros_like(likelihood.cpu().numpy())
    for batch_i in range(likelihood.shape[0]):
        prompt = generated_tokens[batch_i].cpu().numpy()
        compressed_len = len(zlib.compress(prompt.tobytes()))
        zlib_likelihood[batch_i] = likelihood[batch_i].item() * compressed_len
    return zlib_likelihood


def calculate_metric_scores(loss_per_token: torch.Tensor) -> np.ndarray:
    """Calculate metric scores with outlier removal."""
    loss_per_token_np = loss_per_token.cpu().numpy()
    mean = np.mean(loss_per_token_np, axis=-1, keepdims=True)
    std = np.std(loss_per_token_np, axis=-1, keepdims=True)
    floor = mean - 3*std
    upper = mean + 3*std
    
    metric_loss = np.where(
        ((loss_per_token_np < floor) | (loss_per_token_np > upper)),
        mean,
        loss_per_token_np
    )
    return metric_loss.mean(1)


def calculate_high_confidence_scores(
    full_logits: torch.Tensor, 
    full_loss_per_token: torch.Tensor, 
    suffix_len: int
) -> np.ndarray:
    """Calculate high confidence scores, matching the logic from baseline_highconf."""
    top_scores, _ = full_logits.topk(2, dim=-1)
    flag1 = (top_scores[:, :, 0] - top_scores[:, :, 1]) > 0.5
    flag2 = top_scores[:, :, 0] > 0
    flat_flag1 = flag1.reshape(-1)
    flat_flag2 = flag2.reshape(-1)

    mean_batch_loss = full_loss_per_token.mean()
    loss_adjusted_flat = full_loss_per_token - (flat_flag1.int() - flat_flag2.int()) * mean_batch_loss * 0.15
    loss_adjusted_reshaped = loss_adjusted_flat.reshape(full_logits.shape[0], -1)
    loss_adjusted_suffix = loss_adjusted_reshaped[:, -suffix_len:]
    
    return loss_adjusted_suffix.mean(1).cpu().numpy()


@torch.no_grad()
def calculate_lowercase_score(
    generated_tokens: torch.Tensor,
    original_nlls: torch.Tensor,
    model,
    tokenizer,
    device: torch.device
) -> np.ndarray:
    """Calculate lowercase scores for a batch of generated sequences."""
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

    scores = -original_nlls / (lowercase_nlls + 1e-9)
    return scores.cpu().numpy()


def write_guesses_to_csv(generations_per_prompt: int, 
                        generations_dict: Dict[str, np.ndarray], 
                        answers: np.ndarray, 
                        methods: List[str]):
    """Write guesses with ground truth labels to CSV files."""
    for method in methods:
        filename = f"guess_{method}_{generations_per_prompt}.csv"
        with open(filename, "w", newline='') as file_handle:
            print(f"Writing {filename}")
            writer = csv.writer(file_handle)
            writer.writerow(["Example ID", "Suffix Guess", "Ground Truth", "Is Correct"])

            for example_id in range(len(generations_dict[method])):
                guess = generations_dict[method][example_id]
                ground_truth = answers[example_id]
                is_correct = np.all(guess == ground_truth)
                
                row_output = [
                    example_id, 
                    str(list(guess)).replace(" ", ""),
                    str(list(ground_truth)).replace(" ", ""),
                    int(is_correct)
                ]
                writer.writerow(row_output)


def calculate_metrics(generations_dict: Dict[str, np.ndarray], 
                     answers: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Calculate various evaluation metrics."""
    results = {}
    
    for method in generations_dict:
        generations = generations_dict[method]
        
        # Precision (exact match)
        precision = np.sum(np.all(generations == answers, axis=-1)) / generations.shape[0]
        
        # Hamming distance
        hamming_dist = (answers != generations).sum(1).mean()
        
        results[method] = {
            'precision': precision,
            'hamming_distance': hamming_dist
        }
    
    return results


def get_scoring_methods(k_ratios: List[float]) -> List[str]:
    """Get all available scoring methods."""
    base_methods = [
        "likelihood", "zlib", "metric", "high_confidence", 
        "suffix_recall", "recall", "lowercase", "con_recall", "suffix_conrecall"
    ]
    min_k_methods = [f"min_k_{r}" for r in k_ratios]
    min_k_plus_methods = [f"min_k_plus_{r}" for r in k_ratios]
    surprise_methods = [f"surprise_{r}" for r in k_ratios]
    
    return base_methods + min_k_methods + min_k_plus_methods + surprise_methods


def get_argmin_methods() -> List[str]:
    """Get methods that use argmin for selection."""
    return ["likelihood", "zlib", "metric", "high_confidence"]


def get_argmax_methods(k_ratios: List[float]) -> List[str]:
    """Get methods that use argmax for selection."""
    min_k_methods = [f"min_k_{r}" for r in k_ratios]
    min_k_plus_methods = [f"min_k_plus_{r}" for r in k_ratios]
    surprise_methods = [f"surprise_{r}" for r in k_ratios]
    return ["suffix_recall", "recall", "lowercase", "con_recall", "suffix_conrecall"] + min_k_methods + min_k_plus_methods + surprise_methods