import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# App Configuration
APP_CONFIG = {
    'max_file_size': 200,  # MB
    'supported_formats': ['.csv', '.xlsx'],
    'default_test_size': 0.2,
    'random_state': 42
}

# Model Configuration
MODEL_CONFIG = {
    'models_to_train': ['logistic_regression', 'random_forest', 'svm', 'xgboost'],
    'hyperparameters': {
        'logistic_regression': {'C': 1.0, 'max_iter': 1000},
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'svm': {'C': 1.0, 'kernel': 'rbf'},
        'xgboost': {'n_estimators': 100, 'max_depth': 6}
    }
}

# Fairness Metrics
FAIRNESS_METRICS = [
    'disparate_impact',
    'statistical_parity_difference', 
    'equal_opportunity_difference',
    'average_odds_difference',
    'theil_index'
]