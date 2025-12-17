"""
Metric registry for easy access to all metrics.
"""

from .base import BaseMetric
from .likelihood import LikelihoodMetric
from .zlib import ZlibMetric
from .metric import MetricMetric
from .high_confidence import HighConfidenceMetric
from .recall import (SuffixRecallMetric, RecallMetric, ConRecallMetric, SuffixConRecallMetric)
from .lowercase import LowercaseMetric
from .min_k import MinKMetric, MinKPlusMetric, SurpriseMetric

# Metric registry
METRIC_REGISTRY = {
    'likelihood': LikelihoodMetric,
    'zlib': ZlibMetric,
    'metric': MetricMetric,
    'high_confidence': HighConfidenceMetric,
    'suffix_recall': SuffixRecallMetric,
    'recall': RecallMetric,
    'con_recall': ConRecallMetric,
    'suffix_conrecall': SuffixConRecallMetric,
    'lowercase': LowercaseMetric,
    'min_k': MinKMetric,
    'min_k_plus': MinKPlusMetric,
    'surprise': SurpriseMetric,
}


def get_metric(metric_name: str, **kwargs) -> BaseMetric:
    """Get metric instance by name."""
    # Handle ratio-based metrics
    if metric_name.startswith('min_k_plus_'):
        ratio = float(metric_name.split('_')[-1])
        return MinKPlusMetric(ratio=ratio, **kwargs)
    elif metric_name.startswith('min_k_'):
        ratio = float(metric_name.split('_')[-1])
        return MinKMetric(ratio=ratio, **kwargs)
    elif metric_name.startswith('surprise_'):
        ratio = float(metric_name.split('_')[-1])
        return SurpriseMetric(ratio=ratio, **kwargs)
    
    # Handle regular metrics
    if metric_name in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric_name](**kwargs)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def get_all_metrics(k_ratios=None, suffix_len=50, max_entropy=2.0):
    """Get all available metrics."""
    if k_ratios is None:
        k_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    metrics = []
    
    # Base metrics
    metrics.append(LikelihoodMetric(suffix_len=suffix_len))
    metrics.append(ZlibMetric(suffix_len=suffix_len))
    metrics.append(MetricMetric(suffix_len=suffix_len))
    metrics.append(HighConfidenceMetric(suffix_len=suffix_len))
    metrics.append(SuffixRecallMetric())
    metrics.append(RecallMetric())
    metrics.append(ConRecallMetric())
    metrics.append(SuffixConRecallMetric())
    metrics.append(LowercaseMetric())
    
    # Ratio-based metrics
    for ratio in k_ratios:
        metrics.append(MinKMetric(ratio=ratio, suffix_len=suffix_len))
        metrics.append(MinKPlusMetric(ratio=ratio, suffix_len=suffix_len))
        metrics.append(SurpriseMetric(ratio=ratio, suffix_len=suffix_len, max_entropy=max_entropy))
    
    return metrics


def get_metric_names(k_ratios=None):
    """Get names of all available metrics."""
    if k_ratios is None:
        k_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    base_names = [
        "likelihood", "zlib", "metric", "high_confidence",
        "suffix_recall", "recall", "lowercase", "con_recall", "suffix_conrecall"
    ]
    
    ratio_names = []
    for ratio in k_ratios:
        ratio_names.extend([
            f"min_k_{ratio}",
            f"min_k_plus_{ratio}",
            f"surprise_{ratio}"
        ])
    
    return base_names + ratio_names


__all__ = [
    'BaseMetric',
    'LikelihoodMetric',
    'ZlibMetric',
    'MetricMetric',
    'HighConfidenceMetric',
    'SuffixRecallMetric',
    'RecallMetric',
    'ConRecallMetric',
    'SuffixConRecallMetric',
    'LowercaseMetric',
    'MinKMetric',
    'MinKPlusMetric',
    'SurpriseMetric',
    'METRIC_REGISTRY',
    'get_metric',
    'get_all_metrics',
    'get_metric_names',
]
