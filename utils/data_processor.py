import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Utility class for data processing operations"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def detect_data_types(self, df: pd.DataFrame) -> dict:
        """Detect column data types"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'total_columns': len(df.columns),
            'total_rows': len(df)
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['object']:
                    # Categorical - fill with mode
                    df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
                else:
                    # Numerical - fill with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get feature column names"""
        return [col for col in df.columns if col not in ['target', 'sensitive_attribute']]