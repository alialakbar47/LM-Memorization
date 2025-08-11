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


@torch.no_grad()
def calculate_likelihood_scores(model_outputs, generated_tokens: torch.Tensor, suffix_len: int) -> torch.Tensor:
    """Calculate log-likelihood scores efficiently."""
    logits = model_outputs.logits[:, :-1].reshape((-1, model_outputs.logits.shape[-1])).float()
    
    loss_per_token = F.cross_entropy(
        logits, 
        generated_tokens[:, 1:].flatten(), 
        reduction='none'
    ).reshape((-1, generated_tokens.shape[1] - 1))[:, -suffix_len:]
    
    return loss_per_token


@torch.no_grad()
def calculate_recall_scores(prefix_tokens: torch.Tensor, suffix_tokens: torch.Tensor, 
                          model, device: torch.device) -> Tuple[float, float]:
    """Calculate recall scores using prefix-suffix split."""
    # Unconditional LL (suffix only)
    suffix_outputs = model(suffix_tokens.unsqueeze(0).to(device))
    suffix_logits = suffix_outputs.logits[:, :-1]
    suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)
    suffix_token_log_probs = suffix_log_probs.gather(
        dim=-1,
        index=suffix_tokens[1:].unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1)
    ll_unconditional = suffix_token_log_probs.mean().item()
    
    # Conditional LL using KV cache
    prefix_outputs = model(prefix_tokens.unsqueeze(0).to(device))
    cache = DynamicCache.from_legacy_cache(prefix_outputs.past_key_values)
    suffix_outputs = model(suffix_tokens.unsqueeze(0).to(device), past_key_values=cache)
    suffix_logits = suffix_outputs.logits[:, :-1]
    suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)
    suffix_token_log_probs = suffix_log_probs.gather(
        dim=-1,
        index=suffix_tokens[1:].unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1)
    ll_conditional = suffix_token_log_probs.mean().item()
    
    return ll_unconditional, ll_conditional


@torch.no_grad()
def calculate_original_recall(non_member_prefix_tokens: torch.Tensor, 
                            input_tokens: torch.Tensor, 
                            suffix_tokens: torch.Tensor, 
                            model, device: torch.device) -> Tuple[float, float]:
    """Calculate original recall using non-member prefix."""
    # Unconditional LL (input + suffix)
    full_sequence = torch.cat((input_tokens.unsqueeze(0), suffix_tokens.unsqueeze(0)), dim=1)
    outputs = model(full_sequence.to(device), labels=full_sequence.to(device))
    ll_unconditional = -outputs.loss.item()
    
    # Conditional LL (non_member_prefix + input + suffix)
    full_sequence_with_prefix = torch.cat((
        non_member_prefix_tokens.unsqueeze(0),
        input_tokens.unsqueeze(0),
        suffix_tokens.unsqueeze(0)
    ), dim=1)
    
    labels = full_sequence_with_prefix.clone()
    labels[:, :non_member_prefix_tokens.size(0)] = -100
    
    outputs = model(full_sequence_with_prefix.to(device), labels=labels.to(device))
    ll_conditional = -outputs.loss.item()
    
    return ll_unconditional, ll_conditional


@torch.no_grad()
def calculate_min_k_scores(logits_batch: torch.Tensor, 
                         input_ids_batch: torch.Tensor, 
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate min_k and min_k_plus scores efficiently."""
    probs = F.softmax(logits_batch, dim=-1)
    log_probs = F.log_softmax(logits_batch, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids_batch).squeeze(-1)
    
    # Calculate mu and sigma efficiently
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    
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
        "-recall", "recall2", "recall3", "recall_original"
    ]
    min_k_methods = [f"min_k_{ratio}" for ratio in k_ratios]
    min_k_plus_methods = [f"min_k_plus_{ratio}" for ratio in k_ratios]
    
    return base_methods + min_k_methods + min_k_plus_methods


def get_argmin_methods() -> List[str]:
    """Get methods that use argmin for selection."""
    return ["likelihood", "zlib", "metric", "high_confidence", "-recall", "recall2"]


def get_argmax_methods(k_ratios: List[float]) -> List[str]:
    """Get methods that use argmax for selection."""
    min_k_methods = [f"min_k_{ratio}" for ratio in k_ratios]
    min_k_plus_methods = [f"min_k_plus_{ratio}" for ratio in k_ratios]
    return ["recall3", "recall_original"] + min_k_methods + min_k_plus_methods
