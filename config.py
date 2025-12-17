"""
Configuration management for LLM data extraction and MIA evaluation.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for managing experiment settings."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
        # Set attributes from config dictionary
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __setattr__(self, key, value):
        """Override setattr to keep _config in sync."""
        if key == '_config':
            object.__setattr__(self, key, value)
        else:
            # Update the underlying dictionary
            if hasattr(self, '_config'):
                if isinstance(value, Config):
                    self._config[key] = value.to_dict()
                else:
                    self._config[key] = value
            object.__setattr__(self, key, value)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __contains__(self, key):
        return key in self._config
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        """Convert config back to dictionary recursively."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, dict):
                # If we have a nested Config object, convert it
                if hasattr(self, key) and isinstance(getattr(self, key), Config):
                    result[key] = getattr(self, key).to_dict()
                else:
                    result[key] = value
            else:
                # Use the current attribute value (may have been updated)
                if hasattr(self, key):
                    result[key] = getattr(self, key)
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create config from argparse arguments."""
        return cls(vars(args))


def load_config(config_path: str = None, args: argparse.Namespace = None) -> Config:
    """
    Load configuration from YAML file and optionally override with command-line args.
    
    Args:
        config_path: Path to YAML configuration file
        args: Command-line arguments to override config
        
    Returns:
        Config object
    """
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config({})
    
    # Override with command-line arguments if provided
    if args:
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        config._config.update(args_dict)
        for key, value in args_dict.items():
            setattr(config, key, value)
    
    return config


def get_default_extraction_config() -> Dict[str, Any]:
    """Get default extraction configuration."""
    return {
        'experiment': {
            'name': 'extraction_experiment',
            'seed': 2022,
            'dataset_dir': '../datasets',
        },
        'model': {
            'name': 'EleutherAI/gpt-neo-1.3B',
        },
        'generation': {
            'num_trials': 5,
            'val_set_num': 1000,
            'batch_size': 64,
            'top_k': 50,
            'top_p': 1.0,
            'temperature': 1.0,
            'typical_p': 1.0,
            'repetition_penalty': 1.0,
        },
        'saving': {
            'save_all_generations_per_prompt': False,
            'save_all_methods': False,
            'save_npy_files': False,
        },
        'checkpoint': {
            'save_checkpoints': True,
            'resume': False,
            'force_restart': False,
        },
        'metrics': {
            'suffix_len': 50,
            'prefix_len': 50,
            'k_ratios': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'max_entropy': 2.0,
        }
    }


def get_default_mia_config() -> Dict[str, Any]:
    """Get default MIA evaluation configuration."""
    return {
        'experiment': {
            'dataset_dir': '../datasets',
        },
        'model': {
            'name': 'EleutherAI/gpt-neo-1.3B',
        },
        'evaluation': {
            'batch_size': 32,
            'guess_dir': None,  # Must be provided
        },
        'metrics': {
            'suffix_len': 50,
            'k_ratios': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    }


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
