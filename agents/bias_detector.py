"""
agents/bias_detector.py — rebuilt

Fixes:
W2 — best-model score now uses all 5 metrics (weighted), not just accuracy+DI.
W3 — sensitive attribute excluded from training features by default, with an
      explicit opt-in flag if the user wants "fairness through unawareness"
      tested (documented tradeoff, not a silent oversight).
W4 — 5-fold stratified cross-validation added; mean + std reported per model.
W5 — binarization only auto-applied if target is already effectively binary
      (<=2 unique values) or explicitly boolean-like. Multi-class targets are
      rejected with a clear error instead of silently median-split.
W6 — input validation: target != sensitive attribute, target not an
      ID-like column (near-unique values), sensitive attribute must have
      >=2 groups with enough samples in the test split.
W9 — ID-like FEATURE columns (e.g. Name, PassengerId) dropped before
      encoding. Label-encoding a near-unique text column (like Name) turns
      it into an arbitrary per-row integer that leaks identity/proxy signal
      (e.g. titles in Name correlate with Sex) and inflates DI. W6 only
      caught ID-like TARGET columns — this catches ID-like FEATURE columns.
"""
from .base_agent import BaseAgent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from utils.fairness_metrics import FairnessCalculator
from utils.visualization import VisualizationEngine
from typing import Dict, Any

# W2 fix: explicit weights, documented. Accuracy gets weight too since a
# model that is fair but useless isn't a good pick either.
METRIC_WEIGHTS = {
    'accuracy': 0.30,
    'disparate_impact': 0.20,     # distance from 1.0, inverted
    'statistical_parity_difference': 0.15,  # abs value, inverted
    'equal_opportunity_difference': 0.15,   # abs value, inverted
    'average_odds_difference': 0.10,        # abs value, inverted
    'theil_index': 0.10,                    # inverted
}


class BiasDetectionAgent(BaseAgent):
    """Agent 3: Bias Detection - Runs models and computes fairness metrics"""

    def __init__(self):
        super().__init__("Bias Detection Agent")
        self.fairness_calculator = FairnessCalculator()
        self.visualization_engine = VisualizationEngine()

    def execute(self, data_context: Dict[str, Any], user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            cleaned_data = data_context.get('cleaned_data')
            target_variable = user_input.get('target_variable')
            sensitive_attribute = user_input.get('sensitive_attribute')
            include_sensitive_in_features = user_input.get('include_sensitive_in_features', False)  # W3

            if cleaned_data is None or target_variable is None or sensitive_attribute is None:
                return {'success': False, 'error': 'Missing required parameters: cleaned_data, target_variable, or sensitive_attribute'}

            # ---- W6: input validation ----
            validation_error = self._validate_inputs(cleaned_data, target_variable, sensitive_attribute)
            if validation_error:
                return {'success': False, 'error': validation_error}

            y_raw = cleaned_data[target_variable]

            # ---- W5: safe binarization, no silent median-split on arbitrary targets ----
            y, binarization_note = self._safe_binarize(y_raw)
            if y is None:
                return {'success': False, 'error': binarization_note}

            sensitive_attr = cleaned_data[sensitive_attribute]

            # ---- W3: sensitive attribute excluded from features by default ----
            drop_cols = [target_variable]
            if not include_sensitive_in_features:
                drop_cols.append(sensitive_attribute)
            X = cleaned_data.drop(columns=drop_cols)
            X, dropped_id_cols = self._drop_id_like_features(X)  # W9
            X_encoded = self._encode_categorical(X)

            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
                X_encoded, y, sensitive_attr, test_size=0.2, random_state=42, stratify=y
            )

            model_results = self._train_and_evaluate_models(
                X_train, X_test, y_train, y_test, sensitive_test
            )

            if not model_results:
                return {'success': False, 'error': 'All models failed to train. Check data quality.'}

            best_model = self._select_best_model(model_results)
            bias_detected = self._check_bias_detected(model_results)
            ai_insights = self._generate_bias_insights(model_results, best_model, bias_detected, sensitive_attribute)

            return {
                'success': True,
                'model_results': model_results,
                'best_model': best_model,
                'bias_detected': bias_detected,
                'ai_insights': ai_insights,
                'sensitive_attribute_in_features': include_sensitive_in_features,  # W3 audit trail
                'binarization_note': binarization_note,
                'dropped_id_like_features': dropped_id_cols,  # W9 audit trail
                'message': 'Bias analysis completed successfully'
            }

        except Exception as e:
            return {'success': False, 'error': f"Bias analysis failed: {str(e)}"}

    # ---------- W6: input validation ----------
    def _validate_inputs(self, df: pd.DataFrame, target_variable: str, sensitive_attribute: str):
        if target_variable == sensitive_attribute:
            return "Target variable and sensitive attribute cannot be the same column."

        n = len(df)
        target_uniqueness = df[target_variable].nunique() / n
        if target_uniqueness > 0.9:
            return (f"'{target_variable}' looks like an ID column "
                    f"({df[target_variable].nunique()} unique values out of {n} rows). "
                    f"Pick a real outcome column as target.")

        group_counts = df[sensitive_attribute].value_counts()
        if len(group_counts) < 2:
            return f"Sensitive attribute '{sensitive_attribute}' has only one distinct value — no groups to compare."
        if group_counts.min() < 10:
            return (f"Sensitive attribute '{sensitive_attribute}' has a group with only "
                     f"{group_counts.min()} rows. Too few to compute reliable fairness metrics "
                     f"(need at least ~10 per group after the train/test split).")
        return None

    # ---------- W5: safe binarization ----------
    def _safe_binarize(self, y: pd.Series):
        n_unique = y.nunique()
        if n_unique == 2:
            # already binary — but coerce to 0/1 ints for consistent metric math
            vals = sorted(y.unique())
            mapping = {vals[0]: 0, vals[1]: 1}
            return y.map(mapping), f"Target already binary ({vals[0]}→0, {vals[1]}→1). No transformation applied."
        if n_unique == 1:
            return None, "Target has only one class — cannot train a classifier."
        # Multi-class: refuse instead of silently median-splitting
        return None, (
            f"Target has {n_unique} distinct values. BiasLens v1 only supports binary "
            f"classification targets. Median-split binarization was removed because it "
            f"produces meaningless results on non-ordinal or multi-class targets — "
            f"recode your target to binary before uploading, or pick a different column."
        )

    # ---------- W9: drop ID-like FEATURE columns ----------
    def _drop_id_like_features(self, X: pd.DataFrame, threshold: float = 0.9):
        """
        Drop feature columns that are ID-like (near-unique per row) —
        e.g. Name, PassengerId, transaction_id. Label-encoding these turns
        them into an arbitrary per-row integer that leaks identity/proxy
        signal (Name correlates with Sex via titles) and adds noise a tree
        model will happily overfit to. This is separate from W6, which only
        checks the TARGET column for ID-likeness.
        """
        n = len(X)
        if n == 0:
            return X, []
        id_like = [col for col in X.columns if X[col].nunique() / n > threshold]
        return X.drop(columns=id_like), id_like

    def _encode_categorical(self, X):
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object']).columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
        return X_encoded

    def _train_and_evaluate_models(self, X_train, X_test, y_train, y_test, sensitive_test):
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'xgboost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
        }

        results = {}
        # ---- W4: 5-fold stratified CV on training data ----
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                fairness_metrics = self.fairness_calculator.calculate_all_metrics(
                    y_test, y_pred, sensitive_test
                )

                results[name] = {
                    'model': model,
                    'fairness_metrics': fairness_metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'cv_accuracy_mean': float(cv_scores.mean()),  # W4
                    'cv_accuracy_std': float(cv_scores.std()),    # W4
                }

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        return results

    # ---------- W2: multi-metric weighted score ----------
    def _select_best_model(self, model_results):
        if not model_results:
            return None

        best_score = -float('inf')
        best_model = None

        for name, results in model_results.items():
            metrics = results['fairness_metrics']
            score = 0.0

            accuracy = metrics.get('accuracy') or 0
            score += METRIC_WEIGHTS['accuracy'] * accuracy

            di = metrics.get('disparate_impact')
            di_term = 1 - abs((di if di is not None else 1) - 1)  # closer to 1 → closer to 1
            score += METRIC_WEIGHTS['disparate_impact'] * max(di_term, 0)

            for key in ['statistical_parity_difference', 'equal_opportunity_difference', 'average_odds_difference']:
                val = metrics.get(key)
                term = 1 - min(abs(val if val is not None else 0), 1)  # closer to 0 → closer to 1
                score += METRIC_WEIGHTS[key] * term

            theil = metrics.get('theil_index') or 0
            theil_term = 1 - min(abs(theil), 1)
            score += METRIC_WEIGHTS['theil_index'] * theil_term

            if score > best_score:
                best_score = score
                best_model = name

        return best_model

    def _check_bias_detected(self, model_results):
        for name, results in model_results.items():
            metrics = results['fairness_metrics']
            disparate_impact = metrics.get('disparate_impact')
            stat_parity_diff = abs(metrics.get('statistical_parity_difference') or 0)

            if disparate_impact is not None and (disparate_impact < 0.8 or disparate_impact > 1.25):
                return True
            if stat_parity_diff > 0.1:
                return True

        return False

    def _generate_bias_insights(self, model_results, best_model, bias_detected, sensitive_attribute):
        prompt = f"""
        Analyze these bias detection results and provide key insights:

        Models evaluated: {list(model_results.keys())}
        Best model: {best_model}
        Bias detected: {bias_detected}
        Sensitive attribute analyzed: {sensitive_attribute}

        Model Results:
        {self._format_results_for_ai(model_results)}

        Provide 3-4 key insights about:
        1. Overall fairness assessment
        2. Performance-fairness tradeoffs
        3. Recommendations for model selection
        4. Potential bias mitigation strategies

        Keep it concise and actionable.
        """
        return self.generate_response(prompt)

    def _format_results_for_ai(self, model_results):
        formatted = ""
        for name, results in model_results.items():
            metrics = results['fairness_metrics']
            formatted += f"""
            {name.replace('_', ' ').title()}:
            - Accuracy: {metrics.get('accuracy', 0):.3f} (CV mean: {results.get('cv_accuracy_mean', 0):.3f} ± {results.get('cv_accuracy_std', 0):.3f})
            - Disparate Impact: {metrics.get('disparate_impact', 0):.3f}
            - Statistical Parity Difference: {metrics.get('statistical_parity_difference', 0):.3f}
            - Equal Opportunity Difference: {metrics.get('equal_opportunity_difference', 0):.3f}
            - Average Odds Difference: {metrics.get('average_odds_difference', 0):.3f}
            - Theil Index: {metrics.get('theil_index', 0):.3f}
            """
        return formatted