class PromptTemplates:
    """Templates for LLM prompts used across agents"""
    
    @staticmethod
    def data_profiling_insights(shape, columns, sample_data):
        return f"""
        Analyze this dataset for fairness analysis preparation:
        
        Dataset Shape: {shape}
        Columns: {columns}
        Sample Data (first 3 rows):
        {sample_data}
        
        Provide 2-3 concise insights about:
        1. What this dataset might represent
        2. Potential fairness considerations based on column names
        3. Data quality observations
        
        Keep responses brief and focused on bias detection preparation.
        """
    
    @staticmethod
    def data_cleaning_summary(operations, original_shape, cleaned_shape):
        return f"""
        Summarize this data cleaning process in 2-3 sentences:
        
        Operations performed: {operations}
        Original data: {original_shape}
        Cleaned data: {cleaned_shape}
        
        Provide a natural language summary of what was cleaned and how it improves data quality for fairness analysis.
        Focus on why these cleaning steps are important for detecting bias.
        """
    
    @staticmethod
    def fairness_report_technical(model_results, dataset_info):
        return f"""
        Create a technical fairness analysis report based on these results:
        
        Dataset: {dataset_info}
        Model Results: {model_results}
        
        Include:
        1. Executive summary of key findings
        2. Model performance comparison (accuracy and fairness)
        3. Detailed fairness metric analysis
        4. Bias patterns detected
        5. Technical recommendations for mitigation
        
        Format as a professional technical report.
        """
    
    @staticmethod
    def fairness_report_simple(model_results, dataset_info):
        return f"""
        Explain these fairness analysis results in simple, non-technical language:
        
        Context: {dataset_info}
        Results: {model_results}
        
        Requirements:
        - Use everyday analogies and simple terms
        - Avoid technical jargon
        - Focus on real-world impact
        - Keep it under 200 words
        - Explain what the bias means for different groups of people
        
        Example style: "The analysis shows that the computer model tends to favor [group] over [group] when predicting [outcome]. This means that..."
        """
    
    @staticmethod
    def answer_question(question, context, complexity="simple"):
        complexity_instructions = {
            "simple": "Explain in very simple terms that a non-technical person can understand. Use analogies and avoid technical terms.",
            "business": "Explain in business terms focusing on impact and decisions.",
            "technical": "Provide detailed technical explanation with metrics and methodology."
        }
        
        return f"""
        Answer this question about the fairness analysis: "{question}"
        
        Analysis Context: {context}
        
        Instructions: {complexity_instructions.get(complexity, complexity_instructions['simple'])}
        
        Provide a clear, helpful answer based on the analysis results.
        """