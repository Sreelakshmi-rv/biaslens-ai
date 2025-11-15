import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from utils.data_processor import DataProcessor

class DataProfilerAgent(BaseAgent):
    """Agent 1: Data Profiler - Analyzes and understands the dataset"""
    
    def __init__(self):
        super().__init__("Data Profiler Agent")
        self.data_processor = DataProcessor()
    
    def execute(self, data_context: dict, user_input: dict = None) -> dict:
        """Main execution method for data profiling"""
        raw_data = data_context.get('raw_data')
        
        if raw_data is None:
            return {
                'success': False,
                'error': 'No data provided for profiling',
                'data_profile': {}
            }
        
        try:
            # Basic data information
            profile = self._generate_basic_profile(raw_data)
            
            # Data type analysis
            data_types = self.data_processor.detect_data_types(raw_data)
            
            # Sensitive attribute detection
            sensitive_attrs = self._detect_sensitive_attributes(raw_data)
            
            # Data quality assessment
            quality_report = self._assess_data_quality(raw_data)
            
            # Generate LLM insights
            llm_insights = self._generate_llm_insights(raw_data, profile)
            
            result = {
                'success': True,
                'data_profile': {
                    'basic_info': profile,
                    'data_types': data_types,
                    'sensitive_attributes_suggested': sensitive_attrs,
                    'quality_assessment': quality_report,
                    'llm_insights': llm_insights
                }
            }
            
            # Update memory
            self.update_memory('last_profile', result)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Profiling failed: {str(e)}",
                'data_profile': {}
            }
    
    def _generate_basic_profile(self, df: pd.DataFrame) -> dict:
        """Generate basic dataset profile"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'duplicate_rows': df.duplicated().sum(),
            'total_missing_values': df.isnull().sum().sum()
        }
    
    def _detect_sensitive_attributes(self, df: pd.DataFrame) -> list:
        """Auto-detect potential sensitive attributes"""
        sensitive_keywords = [
            'gender', 'sex', 'race', 'ethnic', 'age', 'religion', 
            'disability', 'marital', 'country', 'nationality'
        ]
        
        potential_attrs = []
        for col in df.columns:
            col_lower = col.lower()
            for keyword in sensitive_keywords:
                if keyword in col_lower:
                    potential_attrs.append({
                        'column': col,
                        'reason': f"Contains sensitive keyword: '{keyword}'",
                        'unique_values': df[col].nunique() if df[col].dtype == 'object' else 'numeric'
                    })
                    break
        
        return potential_attrs
    
    def _assess_data_quality(self, df: pd.DataFrame) -> dict:
        """Assess data quality"""
        missing_per_col = df.isnull().sum()
        missing_percentage = (missing_per_col / len(df)) * 100
        
        quality_score = 100
        issues = []
        
        # Check for missing values
        high_missing = missing_percentage[missing_percentage > 20]
        if not high_missing.empty:
            quality_score -= 20
            issues.append(f"High missing values in: {list(high_missing.index)}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_score -= 10
            issues.append(f"Constant columns: {constant_cols}")
        
        return {
            'quality_score': quality_score,
            'issues_found': issues,
            'missing_values_summary': missing_percentage.to_dict(),
            'recommendations': self._generate_quality_recommendations(issues)
        }
    
    def _generate_quality_recommendations(self, issues: list) -> list:
        """Generate data quality recommendations"""
        recommendations = []
        
        if any("missing" in issue.lower() for issue in issues):
            recommendations.append("Consider imputing missing values or removing rows with high missingness")
        
        if any("constant" in issue.lower() for issue in issues):
            recommendations.append("Remove constant columns as they don't provide predictive value")
        
        if not recommendations:
            recommendations.append("Data quality looks good! Ready for analysis.")
        
        return recommendations
    
    def _generate_llm_insights(self, df: pd.DataFrame, profile: dict) -> str:
        """Generate LLM-powered insights about the dataset"""
        sample_data = df.head(3).to_string()
        
        prompt = f"""
        Analyze this dataset and provide 2-3 key insights:
        
        Dataset Shape: {profile['shape']}
        Columns: {profile['columns']}
        Sample Data:
        {sample_data}
        
        Provide brief, actionable insights about:
        1. What this dataset might be about
        2. Potential fairness considerations
        3. Data quality observations
        
        Keep it concise and focused on fairness analysis.
        """
        
        return self.generate_response(prompt)