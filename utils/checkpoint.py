"""
Checkpoint management for resumable experiments.
"""

import os
import pickle
import numpy as np
import torch
from typing import Dict, List, Any


class CheckpointManager:
    """Manages checkpointing and resuming for experiments."""
    
    def __init__(self, experiment_base: str, checkpoint_name: str = "checkpoint.pkl"):
        self.checkpoint_path = os.path.join(experiment_base, checkpoint_name)
        self.experiment_base = experiment_base
        
    def save_checkpoint(self, trial: int, all_generations: List, all_scores: Dict, 
                       config_dict: Dict, rng_states: Dict):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'trial': trial,
            'all_generations': all_generations,
            'all_scores': all_scores,
            'config': config_dict,
            'rng_states': rng_states
        }
        
        # Save to temporary file first, then rename (atomic operation)
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        os.rename(temp_path, self.checkpoint_path)
        print(f"Checkpoint saved after trial {trial + 1}")
        
    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("Checkpoint file cleaned up")

    def get_rng_states(self):
        """Get current random number generator states."""
        return {
            'python_random': np.random.get_state(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_random': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    
    def set_rng_states(self, rng_states: Dict):
        """Restore random number generator states."""
        if 'python_random' in rng_states:
            np.random.set_state(rng_states['python_random'])
        if 'numpy_random' in rng_states:
            np.random.set_state(rng_states['numpy_random'])
        if 'torch_random' in rng_states:
            torch.set_rng_state(rng_states['torch_random'])
        if 'torch_cuda_random' in rng_states and rng_states['torch_cuda_random'] is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['torch_cuda_random'])
