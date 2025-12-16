"""
Metric loader and configuration utilities.
"""

import importlib
import yaml
from typing import Dict, Any, List
from metrics import AbstractMetric


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def get_enabled_metrics(config: Dict[str, Any]) -> List[str]:
    """Get list of enabled metric names from configuration."""
    if 'metrics' not in config:
        return []
    
    enabled = []
    for metric_name, metric_config in config['metrics'].items():
        if metric_config.get('enabled', False):
            enabled.append(metric_name)
    
    return enabled


def load_metric(
    metric_name: str,
    model,
    tokenizer,
    config: Dict[str, Any]
) -> AbstractMetric:
    """
    Dynamically load a metric based on its configuration.
    
    Args:
        metric_name: Name of the metric from config
        model: The language model
        tokenizer: The tokenizer
        config: Full configuration dictionary
        
    Returns:
        Instantiated metric object
    """
    if 'metrics' not in config:
        raise ValueError("No 'metrics' section found in configuration")
    
    if metric_name not in config['metrics']:
        raise ValueError(f"Metric '{metric_name}' not found in configuration")
    
    metric_config = config['metrics'][metric_name]
    module_name = metric_config.get('module')
    
    if not module_name:
        raise ValueError(f"No 'module' specified for metric '{metric_name}'")
    
    try:
        # Import the module dynamically
        module = importlib.import_module(f"metrics.{module_name}")
        
        # Find the metric class in the module
        # Convention: Module "likelihood" contains "LikelihoodMetric"
        class_name = ''.join(word.capitalize() for word in module_name.split('_')) + 'Metric'
        
        metric_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, AbstractMetric) and 
                attr is not AbstractMetric and
                attr_name == class_name):
                metric_class = attr
                break
        
        if metric_class is None:
            raise ValueError(f"No metric class '{class_name}' found in module '{module_name}'")
        
        # Instantiate the metric
        metric_instance = metric_class(
            name=metric_name,
            model=model,
            tokenizer=tokenizer,
            config=metric_config.get('config', {})
        )
        
        return metric_instance
        
    except ImportError as e:
        raise ValueError(f"Failed to import module 'metrics.{module_name}': {str(e)}")


def load_all_enabled_metrics(
    model,
    tokenizer,
    config: Dict[str, Any]
) -> Dict[str, AbstractMetric]:
    """
    Load all enabled metrics from configuration.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        config: Full configuration dictionary
        
    Returns:
        Dictionary mapping metric names to metric instances
    """
    enabled_metrics = get_enabled_metrics(config)
    
    metrics = {}
    for metric_name in enabled_metrics:
        try:
            metric = load_metric(metric_name, model, tokenizer, config)
            metrics[metric_name] = metric
            print(f"Loaded metric: {metric_name}")
        except Exception as e:
            print(f"Warning: Failed to load metric '{metric_name}': {str(e)}")
    
    return metrics


def get_metric_config(config: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    """Get configuration for a specific metric."""
    if 'metrics' not in config or metric_name not in config['metrics']:
        return {}
    return config['metrics'][metric_name].get('config', {})


def update_metric_config_from_dataset(
    config: Dict[str, Any],
    dataset_paths: Dict[str, str]
) -> Dict[str, Any]:
    """
    Update metric configurations with dataset-specific information.
    
    Args:
        config: Configuration dictionary
        dataset_paths: Dictionary with paths to dataset components
        
    Returns:
        Updated configuration
    """
    import numpy as np
    
    # Load dataset components if they exist
    non_member_prefix = None
    member_prefix = None
    
    if 'non_member_prefix' in dataset_paths:
        try:
            non_member_prefix = np.load(dataset_paths['non_member_prefix'])
            print(f"Loaded non-member prefix pool: {non_member_prefix.shape}")
        except Exception as e:
            print(f"Warning: Could not load non-member prefix: {e}")
    
    if 'member_prefix' in dataset_paths:
        try:
            member_prefix = np.load(dataset_paths['member_prefix'])
            print(f"Loaded member prefix pool: {member_prefix.shape}")
        except Exception as e:
            print(f"Warning: Could not load member prefix: {e}")
    
    # Update configs for metrics that need these
    if non_member_prefix is not None:
        for metric_name in ['recall', 'con_recall', 'suffix_conrecall']:
            if metric_name in config.get('metrics', {}):
                config['metrics'][metric_name]['config']['non_member_prefix_pool'] = non_member_prefix
    
    if member_prefix is not None:
        if 'con_recall' in config.get('metrics', {}):
            config['metrics']['con_recall']['config']['member_prefix_pool'] = member_prefix
    
    return config
