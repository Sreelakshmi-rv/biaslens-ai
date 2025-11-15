from .base_agent import BaseAgent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from utils.fairness_metrics import FairnessCalculator
from utils.visualization import VisualizationEngine
from typing import Dict, Any

class BiasDetectionAgent(BaseAgent):
    """Agent 3: Bias Detection - Runs models and computes fairness metrics"""
    
    def __init__(self):
        super().__init__("Bias Detection Agent")
        self.fairness_calculator = FairnessCalculator()
        self.visualization_engine = VisualizationEngine()
    
    def execute(self, data_context: Dict[str, Any], user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main execution method for bias detection"""
        try:
            cleaned_data = data_context.get('cleaned_data')
            target_variable = user_input.get('target_variable')
            sensitive_attribute = user_input.get('sensitive_attribute')
            
            if cleaned_data is None or target_variable is None or sensitive_attribute is None:
                return {
                    'success': False,
                    'error': 'Missing required parameters: cleaned_data, target_variable, or sensitive_attribute'
                }
            
            # Prepare data for modeling
            X = cleaned_data.drop(columns=[target_variable])
            y = cleaned_data[target_variable]
            sensitive_attr = cleaned_data[sensitive_attribute]
            
            # Ensure target is binary for classification
            if y.nunique() > 2:
                # Convert to binary (you might want more sophisticated handling)
                y = (y > y.median()).astype(int)
            
            # Encode categorical variables
            X_encoded = self._encode_categorical(X)
            
            # Split data
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
                X_encoded, y, sensitive_attr, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train models and compute fairness metrics
            model_results = self._train_and_evaluate_models(
                X_train, X_test, y_train, y_test, sensitive_test
            )
            
            # Determine best model
            best_model = self._select_best_model(model_results)
            
            # Check for bias
            bias_detected = self._check_bias_detected(model_results)
            
            # Generate AI insights
            ai_insights = self._generate_bias_insights(model_results, best_model, bias_detected, sensitive_attribute)
            
            return {
                'success': True,
                'model_results': model_results,
                'best_model': best_model,
                'bias_detected': bias_detected,
                'ai_insights': ai_insights,
                'message': 'Bias analysis completed successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Bias analysis failed: {str(e)}"
            }
    
    def _encode_categorical(self, X):
        """Encode categorical variables"""
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object']).columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
        return X_encoded
    
    def _train_and_evaluate_models(self, X_train, X_test, y_train, y_test, sensitive_test):
        """Train multiple models and compute fairness metrics"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'xgboost': XGBClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Compute fairness metrics
                fairness_metrics = self.fairness_calculator.calculate_all_metrics(
                    y_test, y_pred, sensitive_test
                )
                
                results[name] = {
                    'model': model,
                    'fairness_metrics': fairness_metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        return results
    
    def _select_best_model(self, model_results):
        """Select the best model based on fairness-accuracy tradeoff"""
        if not model_results:
            return None
        
        best_score = -float('inf')
        best_model = None
        
        for name, results in model_results.items():
            metrics = results['fairness_metrics']
            accuracy = metrics.get('accuracy', 0)
            disparate_impact = abs(metrics.get('disparate_impact', 1) - 1)  # Distance from 1
            
            # Combined score favoring both accuracy and fairness
            score = accuracy - disparate_impact
            
            if score > best_score:
                best_score = score
                best_model = name
        
        return best_model
    
    def _check_bias_detected(self, model_results):
        """Check if significant bias is detected in any model"""
        for name, results in model_results.items():
            metrics = results['fairness_metrics']
            disparate_impact = metrics.get('disparate_impact', 1)
            stat_parity_diff = abs(metrics.get('statistical_parity_difference', 0))
            
            # Bias thresholds
            if disparate_impact < 0.8 or disparate_impact > 1.25 or stat_parity_diff > 0.1:
                return True
        
        return False
    
    def _generate_bias_insights(self, model_results, best_model, bias_detected, sensitive_attribute):
        """Generate AI insights about the bias analysis"""
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
        """Format model results for AI analysis"""
        formatted = ""
        for name, results in model_results.items():
            metrics = results['fairness_metrics']
            formatted += f"""
            {name.replace('_', ' ').title()}:
            - Accuracy: {metrics.get('accuracy', 0):.3f}
            - Disparate Impact: {metrics.get('disparate_impact', 0):.3f}
            - Statistical Parity Difference: {metrics.get('statistical_parity_difference', 0):.3f}
            - Equal Opportunity Difference: {metrics.get('equal_opportunity_difference', 0):.3f}
            """
        return formatted