from .base_agent import BaseAgent
import pandas as pd
from typing import Dict, Any

class ReportGenerationAgent(BaseAgent):
    """Agent 4: Report Generation - Creates comprehensive fairness reports"""
    
    def __init__(self):
        super().__init__("Report Generation Agent")
    
    def execute(self, data_context: Dict[str, Any], user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive fairness analysis reports"""
        try:
            analysis_result = data_context.get('analysis_result', {})
            cleaned_data = data_context.get('cleaned_data')
            profile_result = data_context.get('profile_result', {})
            
            report_type = user_input.get('report_type', 'Comprehensive Report')
            complexity_level = user_input.get('complexity_level', 'Business')
            
            # Generate reports using AI
            reports = self._generate_ai_reports(analysis_result, profile_result, report_type, complexity_level)
            
            return {
                'success': True,
                'reports': reports,
                'message': f'Successfully generated {report_type}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Report generation failed: {str(e)}",
                'reports': {}
            }
    
    def _generate_ai_reports(self, analysis_result: Dict, profile_result: Dict, report_type: str, complexity_level: str) -> Dict[str, str]:
        """Generate AI-powered reports using Groq"""
        
        # Prepare context for the AI
        context = self._prepare_report_context(analysis_result, profile_result)
        
        reports = {}
        
        # Generate comprehensive report
        if report_type == "Comprehensive Report":
            comprehensive_prompt = self._create_comprehensive_report_prompt(context, complexity_level)
            reports['comprehensive_report'] = self.generate_response(comprehensive_prompt)
        
        # Generate technical report
        technical_prompt = self._create_technical_report_prompt(context, complexity_level)
        reports['technical_report'] = self.generate_response(technical_prompt)
        
        # Generate executive summary
        executive_prompt = self._create_executive_summary_prompt(context, complexity_level)
        reports['executive_summary'] = self.generate_response(executive_prompt)
        
        # Generate mitigation recommendations
        mitigation_prompt = self._create_mitigation_prompt(context, complexity_level)
        reports['mitigation_recommendations'] = self.generate_response(mitigation_prompt)
        
        return reports
    
    def _prepare_report_context(self, analysis_result: Dict, profile_result: Dict) -> str:
        """Prepare context data for report generation"""
        
        # Extract key information from analysis results
        model_results = analysis_result.get('model_results', {})
        best_model = analysis_result.get('best_model', 'Unknown')
        bias_detected = analysis_result.get('bias_detected', False)
        
        # Extract profile information
        profile_data = profile_result.get('data_profile', {})
        basic_info = profile_data.get('basic_info', {})
        sensitive_attrs = profile_data.get('sensitive_attributes_suggested', [])
        
        # Prepare model comparison data
        model_comparison = []
        for model_name, model_data in model_results.items():
            metrics = model_data.get('fairness_metrics', {})
            model_comparison.append({
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'disparate_impact': metrics.get('disparate_impact', 0),
                'statistical_parity_difference': metrics.get('statistical_parity_difference', 0),
                'equal_opportunity_difference': metrics.get('equal_opportunity_difference', 0)
            })
        
        context = f"""
        FAIRNESS ANALYSIS CONTEXT:
        
        Dataset Information:
        - Shape: {basic_info.get('shape', 'Unknown')}
        - Total Rows: {basic_info.get('shape', [0, 0])[0] if basic_info.get('shape') else 0}
        - Total Columns: {basic_info.get('shape', [0, 0])[1] if basic_info.get('shape') else 0}
        - Sensitive Attributes: {[attr['column'] for attr in sensitive_attrs]}
        
        Analysis Results:
        - Best Performing Model: {best_model}
        - Bias Detected: {bias_detected}
        
        Model Performance Comparison:
        {self._format_model_comparison(model_comparison)}
        
        Key Fairness Metrics Interpretation:
        - Disparate Impact: Should be close to 1.0 (0.8-1.25 is generally acceptable)
        - Statistical Parity Difference: Should be close to 0.0
        - Equal Opportunity Difference: Should be close to 0.0
        - Accuracy: Higher is better, but consider fairness trade-offs
        """
        
        return context
    
    def _format_model_comparison(self, model_comparison: list) -> str:
        """Format model comparison data for the AI"""
        comparison_text = ""
        for model in model_comparison:
            comparison_text += f"""
            {model['model'].replace('_', ' ').title()}:
              - Accuracy: {model['accuracy']:.3f}
              - Disparate Impact: {model['disparate_impact']:.3f}
              - Statistical Parity Difference: {model['statistical_parity_difference']:.3f}
              - Equal Opportunity Difference: {model['equal_opportunity_difference']:.3f}
            """
        return comparison_text
    
    def _create_comprehensive_report_prompt(self, context: str, complexity_level: str) -> str:
        """Create prompt for comprehensive report"""
        
        complexity_instructions = {
            "Simple": "Use simple language that anyone can understand. Avoid technical jargon.",
            "Business": "Focus on business impact and decision-making implications.",
            "Technical": "Include technical details, metrics, and methodology explanations."
        }
        
        return f"""
        Generate a comprehensive fairness analysis report based on the following analysis results.
        
        {context}
        
        Instructions for {complexity_level} level:
        {complexity_instructions.get(complexity_level, complexity_instructions['Business'])}
        
        Please structure the report as follows:
        
        1. EXECUTIVE SUMMARY
           - Brief overview of findings
           - Key fairness issues identified
           - Overall assessment
        
        2. METHODOLOGY OVERVIEW
           - Analysis approach
           - Models evaluated
           - Fairness metrics used
        
        3. DETAILED FINDINGS
           - Model performance comparison
           - Fairness metric analysis
           - Bias patterns detected
        
        4. IMPACT ASSESSMENT
           - Potential real-world implications
           - Affected groups
           - Severity assessment
        
        5. RECOMMENDATIONS
           - Immediate actions
           - Long-term strategies
           - Monitoring suggestions
        
        Make the report professional, actionable, and focused on practical insights.
        """
    
    def _create_technical_report_prompt(self, context: str, complexity_level: str) -> str:
        """Create prompt for technical report"""
        return f"""
        Generate a technical fairness analysis report for data scientists and ML engineers.
        
        {context}
        
        Focus on:
        - Technical implementation details
        - Model architecture comparisons
        - Fairness metric calculations
        - Statistical significance
        - Algorithmic considerations
        - Code-level recommendations
        
        Include specific technical recommendations for bias mitigation in machine learning pipelines.
        """
    
    def _create_executive_summary_prompt(self, context: str, complexity_level: str) -> str:
        """Create prompt for executive summary"""
        return f"""
        Generate an executive summary of the fairness analysis for business stakeholders.
        
        {context}
        
        Requirements:
        - Maximum 300 words
        - Focus on business impact
        - Avoid technical jargon
        - Highlight key risks and opportunities
        - Provide clear recommendations
        
        Structure:
        1. Key Findings (2-3 bullet points)
        2. Business Impact
        3. Recommended Actions
        4. Risk Level Assessment
        """
    
    def _create_mitigation_prompt(self, context: str, complexity_level: str) -> str:
        """Create prompt for mitigation recommendations"""
        return f"""
        Based on the fairness analysis results, provide specific bias mitigation recommendations.
        
        {context}
        
        Provide recommendations in these categories:
        
        1. DATA-LEVEL MITIGATION
           - Data collection improvements
           - Feature engineering suggestions
           - Sampling strategies
        
        2. MODEL-LEVEL MITIGATION  
           - Algorithm selection guidance
           - Hyperparameter tuning
           - Fairness constraints
        
        3. POST-PROCESSING MITIGATION
           - Output calibration
           - Decision threshold optimization
           - Ensemble methods
        
        4. ORGANIZATIONAL MITIGATION
           - Monitoring frameworks
           - Governance policies
           - Team training
        
        Make each recommendation specific, actionable, and prioritized by impact.
        """