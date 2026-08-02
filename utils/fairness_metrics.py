"""
utils/fairness_metrics.py — rebuilt

Fixes W7: privileged/unprivileged group is no longer implicit.
Rule: privileged = majority class (highest count) in the sensitive column,
computed on the TEST split actually used, not global data. Rule is stored
on the object so it can be reported to the user, not silently applied.
"""
import numpy as np
import pandas as pd


class FairnessCalculator:
    def __init__(self, privileged_group=None):
        """
        privileged_group: value to treat as privileged. If None, defaults to
        majority class in sensitive_attr at calculate time (documented, not hidden).
        """
        self.privileged_group = privileged_group
        self.last_group_assignment = None  # audit trail

    def _resolve_groups(self, sensitive_attr: pd.Series):
        if self.privileged_group is not None:
            priv_val = self.privileged_group
            rule = "user_specified"
        else:
            priv_val = sensitive_attr.value_counts().idxmax()
            rule = "majority_class"

        priv_mask = sensitive_attr == priv_val
        unpriv_mask = ~priv_mask

        self.last_group_assignment = {
            'rule': rule,
            'privileged_value': priv_val,
            'privileged_n': int(priv_mask.sum()),
            'unprivileged_n': int(unpriv_mask.sum()),
        }
        return priv_mask, unpriv_mask

    def calculate_all_metrics(self, y_true, y_pred, sensitive_attr):
        y_true = pd.Series(np.asarray(y_true)).reset_index(drop=True)
        y_pred = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
        sensitive_attr = pd.Series(np.asarray(sensitive_attr)).reset_index(drop=True)

        priv_mask, unpriv_mask = self._resolve_groups(sensitive_attr)

        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            raise ValueError(
                "Sensitive attribute has only one group in this split — "
                "cannot compute group fairness metrics."
            )

        accuracy = (y_true == y_pred).mean()

        # Positive prediction rates per group
        p_priv = y_pred[priv_mask].mean()
        p_unpriv = y_pred[unpriv_mask].mean()

        disparate_impact = (p_unpriv / p_priv) if p_priv > 0 else np.nan
        statistical_parity_difference = p_unpriv - p_priv

        # True positive rate (recall) per group — needs actual positives present
        def tpr(mask):
            actual_pos = (y_true[mask] == 1)
            if actual_pos.sum() == 0:
                return np.nan
            return (y_pred[mask][actual_pos] == 1).mean()

        def fpr(mask):
            actual_neg = (y_true[mask] == 0)
            if actual_neg.sum() == 0:
                return np.nan
            return (y_pred[mask][actual_neg] == 1).mean()

        tpr_priv, tpr_unpriv = tpr(priv_mask), tpr(unpriv_mask)
        fpr_priv, fpr_unpriv = fpr(priv_mask), fpr(unpriv_mask)

        equal_opportunity_difference = tpr_unpriv - tpr_priv
        average_odds_difference = 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))

        theil_index = self._theil_index(y_pred)

        return {
            'accuracy': float(accuracy),
            'disparate_impact': float(disparate_impact) if not np.isnan(disparate_impact) else None,
            'statistical_parity_difference': float(statistical_parity_difference),
            'equal_opportunity_difference': float(equal_opportunity_difference) if not np.isnan(equal_opportunity_difference) else None,
            'average_odds_difference': float(average_odds_difference) if not (np.isnan(fpr_priv) or np.isnan(fpr_unpriv) or np.isnan(tpr_priv) or np.isnan(tpr_unpriv)) else None,
            'theil_index': float(theil_index),
            'group_assignment': self.last_group_assignment,  # audit trail, W7 fix
        }

    @staticmethod
    def _theil_index(y_pred):
        b = np.asarray(y_pred, dtype=float) + 1.0  # shift to avoid log(0)
        mean_b = b.mean()
        if mean_b == 0:
            return 0.0
        ratio = b / mean_b
        ratio = np.where(ratio == 0, 1e-10, ratio)
        return float(np.mean(ratio * np.log(ratio)))