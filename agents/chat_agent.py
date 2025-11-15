from .base_agent import BaseAgent
from typing import Dict, Any

class ConversationalAgent(BaseAgent):
    """Agent 5: Conversational Interface - Answers questions naturally"""
    
    def __init__(self):
        super().__init__("Conversational Agent")
        self.conversation_history = []
    
    def execute(self, data_context: Dict[str, Any], user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main execution method for conversational interface"""
        try:
            user_question = user_input.get('question', '')
            analysis_result = data_context.get('analysis_result', {})
            profile_result = data_context.get('profile_result', {})
            cleaned_data = data_context.get('cleaned_data')
            
            if not user_question:
                return {
                    'success': True,
                    'response': "Hello! I'm your BiasLens assistant. Ask me anything about your fairness analysis, the models, bias metrics, or recommendations.",
                    'suggested_questions': self._get_suggested_questions()
                }
            
            # Generate response using AI with context
            response = self._generate_chat_response(user_question, analysis_result, profile_result, cleaned_data)
            
            # Update conversation history
            self.conversation_history.append({
                'question': user_question,
                'response': response
            })
            
            return {
                'success': True,
                'response': response,
                'suggested_questions': self._get_suggested_questions()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Chat failed: {str(e)}",
                'response': "I'm having trouble processing your question right now. Please try again."
            }
    
    def _generate_chat_response(self, question: str, analysis_result: Dict, profile_result: Dict, cleaned_data) -> str:
        """Generate AI response with analysis context"""
        
        # Prepare context from analysis
        context = self._prepare_chat_context(analysis_result, profile_result, cleaned_data)
        
        prompt = f"""
        You are a helpful AI assistant specialized in fairness analysis and bias detection. 
        A user is asking about their machine learning fairness analysis.
        
        {context}
        
        User Question: "{question}"
        
        Please provide a helpful, accurate response that:
        1. Directly answers the user's question
        2. Uses the analysis results to support your answer
        3. Explains concepts in simple, understandable terms
        4. Provides practical insights and recommendations when relevant
        5. If the question is outside the analysis scope, politely explain what you can help with
        
        Keep your response conversational but informative.
        """
        
        return self.generate_response(prompt)
    
    def _prepare_chat_context(self, analysis_result: Dict, profile_result: Dict, cleaned_data) -> str:
        """Prepare context for the chat agent"""
        
        if not analysis_result:
            return "No analysis has been completed yet. The user may ask general questions about fairness analysis."
        
        # Extract key information
        model_results = analysis_result.get('model_results', {})
        best_model = analysis_result.get('best_model', 'Unknown')
        bias_detected = analysis_result.get('bias_detected', False)
        ai_insights = analysis_result.get('ai_insights', '')
        
        # Extract profile information
        profile_data = profile_result.get('data_profile', {})
        basic_info = profile_data.get('basic_info', {})
        sensitive_attrs = profile_data.get('sensitive_attributes_suggested', [])
        
        # Prepare model summary
        model_summary = ""
        for model_name, results in model_results.items():
            metrics = results.get('fairness_metrics', {})
            model_summary += f"""
            {model_name.replace('_', ' ').title()}:
            - Accuracy: {metrics.get('accuracy', 0):.3f}
            - Disparate Impact: {metrics.get('disparate_impact', 0):.3f}
            - Statistical Parity Difference: {metrics.get('statistical_parity_difference', 0):.3f}
            - Equal Opportunity Difference: {metrics.get('equal_opportunity_difference', 0):.3f}
            """
        
        context = f"""
        FAIRNESS ANALYSIS CONTEXT:
        
        Dataset Overview:
        - Shape: {basic_info.get('shape', 'Unknown')}
        - Sensitive Attributes Analyzed: {[attr['column'] for attr in sensitive_attrs]}
        
        Key Findings:
        - Best Performing Model: {best_model}
        - Bias Detected: {bias_detected}
        - AI Insights: {ai_insights}
        
        Model Performance Summary:
        {model_summary}
        
        Fairness Metrics Guide:
        - Disparate Impact: Ratio of positive outcomes between groups (ideal: 1.0)
        - Statistical Parity Difference: Difference in positive outcome rates (ideal: 0.0)
        - Equal Opportunity Difference: Difference in true positive rates (ideal: 0.0)
        - Accuracy: Overall prediction correctness
        
        Bias Thresholds:
        - Disparate Impact < 0.8 or > 1.25 indicates potential bias
        - Statistical Parity Difference > 0.1 indicates significant difference
        - Equal Opportunity Difference > 0.1 indicates unequal opportunities
        """
        
        return context
    
    def _get_suggested_questions(self) -> list:
        """Get list of suggested questions for the user"""
        return [
            "What bias was detected in my analysis?",
            "Which model performed the best and why?",
            "Explain disparate impact in simple terms",
            "How can I reduce bias in my models?",
            "What do the fairness metrics mean?",
            "Which sensitive attributes showed the most bias?",
            "Should I be concerned about the bias findings?",
            "What are the limitations of this analysis?",
            "How reliable are the fairness metrics?",
            "What are the next steps after this analysis?"
        ]
    
    def get_conversation_history(self) -> list:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []