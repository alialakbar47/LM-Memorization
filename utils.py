"""
Utility functions for LLM data extraction and MIA evaluation.
Enhanced for multi-GPU support.
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
    """Load model and tokenizer with enhanced multi-GPU support."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available with {num_gpus} GPU(s):")
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / 1024**3
            print(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f} GB)")
        device = torch.device("cuda:0")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    print(f"Loading model '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)
    print(f"Model loaded to {device}")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        print("DataParallel enabled successfully")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

    model.eval()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    return model, tokenizer


def load_prompts(dir_path: str, file_name: str, allow_pickle: bool = False) -> np.ndarray:
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
    file_name = file_path.format(unique_id)
    np.save(file_name, array)


def prepare_directories(root_dir: str, experiment_name: str):
    experiment_base = os.path.join(root_dir, experiment_name)
    generations_base = os.path.join(experiment_base, "generations")
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(generations_base, exist_ok=True)
    os.makedirs(losses_base, exist_ok=True)
    return experiment_base, generations_base, losses_base


def get_model_device(model):
    return next(model.module.parameters()).device if isinstance(model, torch.nn.DataParallel) else next(model.parameters()).device


def get_model_vocab_size(model):
    return model.module.config.vocab_size if isinstance(model, torch.nn.DataParallel) else model.config.vocab_size


def perturb_prefix(prefix_tokens: torch.Tensor, vocab_size: int, p: float = 0.2) -> torch.Tensor:
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
    outputs = model(input_ids=tokens.unsqueeze(0).to(device))
    logits = outputs.logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=tokens[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
    return token_log_probs.mean().item()


@torch.no_grad()
def get_cond_ll(model, prefix: torch.Tensor, suffix: torch.Tensor, device: torch.device) -> float:
    prefix_outputs = model(input_ids=prefix.unsqueeze(0).to(device))
    past_key_values = getattr(prefix_outputs, 'past_key_values', None)
    if past_key_values is not None:
        cache = DynamicCache.from_legacy_cache(past_key_values)
        suffix_outputs = model(input_ids=suffix.unsqueeze(0).to(device), past_key_values=cache)
    else:
        full_sequence = torch.cat([prefix, suffix])
        full_outputs = model(input_ids=full_sequence.unsqueeze(0).to(device))
        prefix_len = len(prefix)
        suffix_logits = full_outputs.logits[0, prefix_len-1:-1]
        log_probs = F.log_softmax(suffix_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=suffix[1:].unsqueeze(-1)).squeeze()
        return token_log_probs.mean().item()
    logits = suffix_outputs.logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=suffix[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
    return token_log_probs.mean().item()


@torch.no_grad()
def calculate_recall_scores(prefix_tokens: torch.Tensor, suffix_tokens: torch.Tensor, 
                          model, device: torch.device) -> Tuple[float, float]:
    ll_unconditional = get_ll(model, suffix_tokens, device)
    ll_conditional = get_cond_ll(model, prefix_tokens, suffix_tokens, device)
    return ll_unconditional, ll_conditional


@torch.no_grad()
def calculate_suffix_con_recall(prefix_tokens: torch.Tensor, suffix_tokens: torch.Tensor, 
                               model, tokenizer, device: torch.device,
                               non_member_prefix_pool: torch.Tensor = None,
                               example_id: int = 0) -> float:
    prefix_tokens = prefix_tokens.to(device)
    suffix_tokens = suffix_tokens.to(device)
    suffix_outputs = model(input_ids=suffix_tokens.unsqueeze(0), labels=suffix_tokens.unsqueeze(0))
    original_nll = suffix_outputs.loss.item()
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
        vocab_size = get_model_vocab_size(model)
        non_member_prefix = (prefix_tokens + 1000) % vocab_size
    member_sequence = torch.cat([prefix_tokens, suffix_tokens])
    member_outputs = model(input_ids=member_sequence.unsqueeze(0), labels=member_sequence.unsqueeze(0))
    prefix_len = len(prefix_tokens)
    member_logits = member_outputs.logits[0, prefix_len-1:-1]
    member_loss = F.cross_entropy(member_logits, suffix_tokens, reduction='mean')
    nll_member = member_loss.item()
    non_member_sequence = torch.cat([non_member_prefix, suffix_tokens])
    non_member_outputs = model(input_ids=non_member_sequence.unsqueeze(0), labels=non_member_sequence.unsqueeze(0))
    non_member_logits = non_member_outputs.logits[0, prefix_len-1:-1]
    non_member_loss = F.cross_entropy(non_member_logits, suffix_tokens, reduction='mean')
    nll_non_member = non_member_loss.item()
    return (nll_non_member - nll_member) / (original_nll + 1e-9)


@torch.no_grad()
def calculate_recall(non_member_prefix_tokens: torch.Tensor, 
                     input_tokens: torch.Tensor, 
                     suffix_tokens: torch.Tensor, 
                     model, device: torch.device) -> Tuple[float, float]:
    non_member_prefix_tokens = non_member_prefix_tokens.to(device)
    input_tokens = input_tokens.to(device)
    suffix_tokens = suffix_tokens.to(device)
    full_sequence = torch.cat((input_tokens, suffix_tokens))
    outputs = model(input_ids=full_sequence.unsqueeze(0), labels=full_sequence.unsqueeze(0))
    nll_unconditional = outputs.loss.item()
    full_sequence_with_prefix = torch.cat((non_member_prefix_tokens, input_tokens, suffix_tokens))
    outputs_with_prefix = model(input_ids=full_sequence_with_prefix.unsqueeze(0), labels=full_sequence_with_prefix.unsqueeze(0))
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
    non_member_prefix_tokens = non_member_prefix_tokens.to(device)
    member_prefix_tokens = member_prefix_tokens.to(device)
    full_sequence_tokens = full_sequence_tokens.to(device)
    nm_prefixed_sequence = torch.cat((non_member_prefix_tokens, full_sequence_tokens))
    nm_outputs = model(input_ids=nm_prefixed_sequence.unsqueeze(0), labels=nm_prefixed_sequence.unsqueeze(0))
    nll_non_member = nm_outputs.loss.item()
    m_prefixed_sequence = torch.cat((member_prefix_tokens, full_sequence_tokens))
    m_outputs = model(input_ids=m_prefixed_sequence.unsqueeze(0), labels=m_prefixed_sequence.unsqueeze(0))
    nll_member = m_outputs.loss.item()
    return (nll_non_member - nll_member) / (original_nll + 1e-9)


@torch.no_grad()
def calculate_min_k_scores(logits_batch: torch.Tensor, 
                         input_ids_batch: torch.Tensor, 
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    log_probs_all_vocab = F.log_softmax(logits_batch, dim=-1)
    token_log_probs = log_probs_all_vocab.gather(dim=-1, index=input_ids_batch).squeeze(-1)
    mu = log_probs_all_vocab.mean(dim=-1)
    sigma = log_probs_all_vocab.std(dim=-1)
    mink_plus = (token_log_probs - mu) / sigma
    return token_log_probs.cpu().numpy(), mink_plus.cpu().numpy()


def calculate_zlib_scores(generated_tokens: torch.Tensor, likelihood: torch.Tensor) -> np.ndarray:
    zlib_likelihood = np.zeros_like(likelihood.cpu().numpy())
    for batch_i in range(likelihood.shape[0]):
        prompt = generated_tokens[batch_i].cpu().numpy()
        compressed_len = len(zlib.compress(prompt.tobytes()))
        zlib_likelihood[batch_i] = likelihood[batch_i].item() * compressed_len
    return zlib_likelihood


def calculate_metric_scores(loss_per_token: torch.Tensor) -> np.ndarray:
    loss_per_token_np = loss_per_token.cpu().numpy()
    mean = np.mean(loss_per_token_np, axis=-1, keepdims=True)
    std = np.std(loss_per_token_np, axis=-1, keepdims=True)
    floor = mean - 3*std
    upper = mean + 3*std
    metric_loss = np.where(((loss_per_token_np < floor) | (loss_per_token_np > upper)), mean, loss_per_token_np)
    return metric_loss.mean(1)


def calculate_high_confidence_scores(
    full_logits: torch.Tensor, 
    full_loss_per_token: torch.Tensor, 
    suffix_len: int
) -> np.ndarray:
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
    decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    lowercase_texts = [text.lower() for text in decoded_texts]
    lowercase_inputs = tokenizer(lowercase_texts, return_tensors='pt', padding=True, truncation=True, max_length=generated_tokens.shape[1]).to(device)
    lowercase_outputs = model(input_ids=lowercase_inputs.input_ids, labels=lowercase_inputs.input_ids)
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


def write_guesses_to_csv(generations_per_prompt: int, generations_dict: Dict[str, np.ndarray], 
                        answers: np.ndarray, methods: List[str]):
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
                row_output = [example_id, str(list(guess)).replace(" ", ""), str(list(ground_truth)).replace(" ", ""), int(is_correct)]
                writer.writerow(row_output)


def calculate_metrics(generations_dict: Dict[str, np.ndarray], answers: np.ndarray) -> Dict[str, Dict[str, float]]:
    results = {}
    for method in generations_dict:
        generations = generations_dict[method]
        precision = np.sum(np.all(generations == answers, axis=-1)) / generations.shape[0]
        hamming_dist = (answers != generations).sum(1).mean()
        results[method] = {'precision': precision, 'hamming_distance': hamming_dist}
    return results


def get_scoring_methods(k_ratios: List[float]) -> List[str]:
    base_methods = ["likelihood", "zlib", "metric", "high_confidence", 
                    "suffix_recall", "recall", "lowercase", "con_recall", "suffix_conrecall"]
    min_k_methods = [f"min_k_{r}" for r in k_ratios]
    min_k_plus_methods = [f"min_k_plus_{r}" for r in k_ratios]
    surprise_methods = [f"surprise_{r}" for r in k_ratios]
    return base_methods + min_k_methods + min_k_plus_methods + surprise_methods


def get_argmin_methods() -> List[str]:
    return ["likelihood", "zlib", "metric", "high_confidence"]


def get_argmax_methods(k_ratios: List[float]) -> List[str]:
    min_k_methods = [f"min_k_{r}" for r in k_ratios]
    min_k_plus_methods = [f"min_k_plus_{r}" for r in k_ratios]
    surprise_methods = [f"surprise_{r}" for r in k_ratios]
    return ["suffix_recall", "recall", "lowercase", "con_recall", "suffix_conrecall"] + min_k_methods + min_k_plus_methods + surprise_methods
