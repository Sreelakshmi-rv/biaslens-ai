import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

class FairnessCalculator:
    """Calculate various fairness metrics for model evaluation"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, y_true, y_pred, sensitive_attr):
        """Calculate all fairness metrics"""
        privileged_group = 1  # Assuming binary sensitive attribute where 1 is privileged
        unprivileged_group = 0
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'disparate_impact': self.disparate_impact(y_pred, sensitive_attr, unprivileged_group, privileged_group),
            'statistical_parity_difference': self.statistical_parity_difference(y_pred, sensitive_attr, unprivileged_group, privileged_group),
            'equal_opportunity_difference': self.equal_opportunity_difference(y_true, y_pred, sensitive_attr, unprivileged_group, privileged_group),
            'average_odds_difference': self.average_odds_difference(y_true, y_pred, sensitive_attr, unprivileged_group, privileged_group),
            'theil_index': self.theil_index(y_pred, sensitive_attr)
        }
        
        return metrics
    
    def disparate_impact(self, y_pred, sensitive_attr, unprivileged_group, privileged_group):
        """Calculate disparate impact ratio"""
        unprivileged_rate = np.mean(y_pred[sensitive_attr == unprivileged_group])
        privileged_rate = np.mean(y_pred[sensitive_attr == privileged_group])
        
        if privileged_rate == 0:
            return float('inf')
        
        return unprivileged_rate / privileged_rate
    
    def statistical_parity_difference(self, y_pred, sensitive_attr, unprivileged_group, privileged_group):
        """Calculate statistical parity difference"""
        unprivileged_rate = np.mean(y_pred[sensitive_attr == unprivileged_group])
        privileged_rate = np.mean(y_pred[sensitive_attr == privileged_group])
        
        return unprivileged_rate - privileged_rate
    
    def equal_opportunity_difference(self, y_true, y_pred, sensitive_attr, unprivileged_group, privileged_group):
        """Calculate equal opportunity difference (TPR difference)"""
        # True positive rates
        tpr_unprivileged = self.true_positive_rate(y_true, y_pred, sensitive_attr == unprivileged_group)
        tpr_privileged = self.true_positive_rate(y_true, y_pred, sensitive_attr == privileged_group)
        
        return tpr_unprivileged - tpr_privileged
    
    def average_odds_difference(self, y_true, y_pred, sensitive_attr, unprivileged_group, privileged_group):
        """Calculate average odds difference"""
        # True positive rate difference
        tpr_diff = self.equal_opportunity_difference(y_true, y_pred, sensitive_attr, unprivileged_group, privileged_group)
        
        # False positive rate difference
        fpr_unprivileged = self.false_positive_rate(y_true, y_pred, sensitive_attr == unprivileged_group)
        fpr_privileged = self.false_positive_rate(y_true, y_pred, sensitive_attr == privileged_group)
        fpr_diff = fpr_unprivileged - fpr_privileged
        
        return (tpr_diff + fpr_diff) / 2
    
    def theil_index(self, y_pred, sensitive_attr):
        """Calculate Theil index for inequality measurement"""
        group_means = []
        for group in np.unique(sensitive_attr):
            group_pred = y_pred[sensitive_attr == group]
            if len(group_pred) > 0:
                group_means.append(np.mean(group_pred))
        
        if len(group_means) == 0:
            return 0
        
        overall_mean = np.mean(group_means)
        if overall_mean == 0:
            return 0
        
        # Calculate Theil index
        theil = 0
        for mean in group_means:
            if mean > 0:
                theil += (mean / overall_mean) * np.log(mean / overall_mean)
        
        return theil / len(group_means)
    
    def true_positive_rate(self, y_true, y_pred, mask):
        """Calculate true positive rate for a subgroup"""
        if np.sum(mask) == 0:
            return 0
        
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        
        if len(y_true_sub) == 0:
            return 0
        
        tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def false_positive_rate(self, y_true, y_pred, mask):
        """Calculate false positive rate for a subgroup"""
        if np.sum(mask) == 0:
            return 0
        
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        
        if len(y_true_sub) == 0:
            return 0
        
        tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0