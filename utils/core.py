"""
Core utility functions.
"""

import os
import random
import functools
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            tokenizer.pad_token_id = 50256
    return model, tokenizer


def prepare_directories(root_dir: str, experiment_name: str):
    """Create necessary directories for experiment under results/experiment-name/."""
    experiment_base = os.path.join(root_dir, experiment_name)
    generations_base = os.path.join(experiment_base, "generations")
    losses_base = os.path.join(experiment_base, "losses")
    
    os.makedirs(generations_base, exist_ok=True)
    os.makedirs(losses_base, exist_ok=True)
    
    return experiment_base, generations_base, losses_base
