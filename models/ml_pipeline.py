from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class MLPipeline:
    """Machine learning pipeline for model training and evaluation"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'svm': SVC(random_state=random_state, probability=True),
            'xgboost': XGBClassifier(random_state=random_state, n_estimators=100)
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
    
    def train_models(self, X, y, test_size=0.2):
        """Train all models and return results"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Train model
                if model_name == 'svm':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Store trained model
                self.trained_models[model_name] = model
                
                results[model_name] = {
                    'model': model,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'feature_names': X.columns.tolist() if hasattr(X, 'columns') else []
                }
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def get_model(self, model_name):
        """Get trained model by name"""
        return self.trained_models.get(model_name)