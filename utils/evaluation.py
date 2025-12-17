"""
Evaluation utilities.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


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


def get_mia_metrics(scores: list, labels: list) -> dict:
    """Calculate MIA metrics including precision-recall metrics."""
    scores = np.array(scores)
    labels = np.array(labels)
    
    # ROC curve metrics
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    
    # Handle FPR95 (TPR >= 0.95)
    tpr_95_idx = np.where(tpr_list >= 0.95)[0]
    if len(tpr_95_idx) > 0:
        fpr95 = fpr_list[tpr_95_idx[0]]
    else:
        fpr95 = 1.0
    
    # Handle TPR05 (FPR <= 0.05)
    fpr_05_idx = np.where(fpr_list <= 0.05)[0]
    if len(fpr_05_idx) > 0:
        tpr05 = tpr_list[fpr_05_idx[-1]]
    else:
        tpr05 = 0.0
    
    # Precision-Recall metrics
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    # Calculate precision at different recall thresholds
    recall_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_at_recall = {}
    
    for r_threshold in recall_thresholds:
        mask = recall >= r_threshold
        if np.any(mask):
            precision_at_recall[f'precision_at_recall_{int(r_threshold*100)}'] = np.max(precision[mask])
        else:
            precision_at_recall[f'precision_at_recall_{int(r_threshold*100)}'] = 0.0
    
    # Calculate recall at high precision thresholds
    precision_thresholds = [0.9, 0.95, 0.99]
    recall_at_precision = {}
    
    for p_threshold in precision_thresholds:
        mask = precision >= p_threshold
        if np.any(mask):
            recall_at_precision[f'recall_at_precision_{int(p_threshold*100)}'] = np.max(recall[mask])
        else:
            recall_at_precision[f'recall_at_precision_{int(p_threshold*100)}'] = 0.0
    
    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'tpr05': tpr05,
        'avg_precision': avg_precision,
        **precision_at_recall,
        **recall_at_precision
    }
