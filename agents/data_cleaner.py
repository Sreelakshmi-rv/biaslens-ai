import pandas as pd
from .base_agent import BaseAgent
from utils.data_processor import DataProcessor

class DataCleaningAgent(BaseAgent):
    """Agent 2: Data Cleaning - Prepares data for analysis"""
    
    def __init__(self):
        super().__init__("Data Cleaning Agent")
        self.data_processor = DataProcessor()
    
    def execute(self, data_context: dict, user_input: dict = None) -> dict:
        """Main execution method for data cleaning"""
        raw_data = data_context.get('raw_data')
        
        if raw_data is None:
            return {
                'success': False,
                'error': 'No data provided for cleaning',
                'cleaned_data': None,
                'cleaning_report': {}
            }
        
        try:
            # Create a copy for cleaning
            df_clean = raw_data.copy()
            
            # Track cleaning operations
            operations_performed = []
            
            # Handle missing values
            missing_before = df_clean.isnull().sum().sum()
            df_clean = self.data_processor.handle_missing_values(df_clean)
            missing_after = df_clean.isnull().sum().sum()
            
            if missing_before > 0:
                operations_performed.append(
                    f"Handled {missing_before - missing_after} missing values"
                )
            
            # Encode categorical variables
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                df_clean = self.data_processor.encode_categorical(df_clean)
                operations_performed.append(
                    f"Encoded {len(categorical_cols)} categorical columns"
                )
            
            # Generate cleaning report
            cleaning_report = self._generate_cleaning_report(
                raw_data, df_clean, operations_performed
            )
            
            result = {
                'success': True,
                'cleaned_data': df_clean,
                'cleaning_report': cleaning_report
            }
            
            # Update memory
            self.update_memory('last_cleaning', result)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Cleaning failed: {str(e)}",
                'cleaned_data': None,
                'cleaning_report': {}
            }
    
    def _generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, operations: list) -> dict:
        """Generate comprehensive cleaning report"""
        return {
            'operations_performed': operations,
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'missing_values_removed': original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
            'data_types_changed': f"{len(original_df.select_dtypes(include=['object']).columns)} categorical → encoded",
            'cleaning_summary': self._generate_llm_cleaning_summary(original_df, cleaned_df, operations)
        }
    
    def _generate_llm_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, operations: list) -> str:
        """Generate LLM-powered cleaning summary"""
        prompt = f"""
        Summarize this data cleaning process in 2-3 sentences:
        
        Operations performed: {operations}
        Original data shape: {original_df.shape}
        Cleaned data shape: {cleaned_df.shape}
        Missing values before: {original_df.isnull().sum().sum()}
        Missing values after: {cleaned_df.isnull().sum().sum()}
        
        Provide a natural language summary of what was cleaned and how it improves data quality for fairness analysis.
        """
        
        return self.generate_response(prompt)