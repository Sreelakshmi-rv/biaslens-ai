import pandas as pd
from .base_agent import BaseAgent

class ReportGenerationAgent(BaseAgent):
    """Agent 4: Report Generation – Produces complete textual fairness reports"""

    def __init__(self):
        super().__init__("Report Generation Agent")

    def execute(self, data_context: dict, user_input: dict = None) -> dict:
        try:
            analysis = data_context.get("analysis_result")
            cleaned_data = data_context.get("cleaned_data")
            profile = data_context.get("profile_result")

            if analysis is None or cleaned_data is None or profile is None:
                return {
                    "success": False,
                    "error": "Missing analysis_result, cleaned_data, or profile_result"
                }

            report_type = user_input.get("report_type", "Comprehensive Report")
            complexity = user_input.get("complexity_level", "Simple")

            comprehensive = self._make_comprehensive_report(analysis, profile, complexity)
            technical = self._make_technical_report(analysis, profile)
            executive = self._make_executive_summary(analysis, profile)

            recommendations = self._make_mitigation_recommendations(analysis)

            return {
                "success": True,
                "reports": {
                    "comprehensive_report": comprehensive,
                    "technical_report": technical,
                    "executive_summary": executive,
                    "mitigation_recommendations": recommendations
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Report generation failed: {str(e)}"
            }

    # ------------------------
    # REPORT BUILDING METHODS
    # ------------------------

    def _make_comprehensive_report(self, analysis, profile, complexity):
        """Full-length report with model summaries and fairness interpretation."""

        models = analysis.get("model_results", {})
        best = analysis.get("best_model", "Unknown")
        bias = analysis.get("bias_detected", False)
        ai = analysis.get("ai_insights", "")

        intro = f"# Comprehensive Fairness Analysis Report\n\n"
        intro += f"**Best Model:** {best.replace('_',' ').title()}\n"
        intro += f"**Bias Detected:** {'Yes' if bias else 'No'}\n\n"

        # Model table
        model_details = "## Model Performance & Fairness Metrics\n"
        for name, res in models.items():
            m = res["fairness_metrics"]
            model_details += f"""
### {name.replace('_',' ').title()}
- Accuracy: **{m.get('accuracy',0):.3f}**
- Disparate Impact: **{m.get('disparate_impact',0):.3f}**
- Statistical Parity Difference: **{m.get('statistical_parity_difference',0):.3f}**
- Equal Opportunity Difference: **{m.get('equal_opportunity_difference',0):.3f}**
"""

        # LLM-generated interpretation
        interpretation_prompt = f"""
Generate a complete fairness analysis narrative based on:

Best model: {best}
Bias detected: {bias}
AI insights: {ai}

Explain the fairness results in a clear,
{complexity.lower()} style.
"""
        interpretation = self.generate_response(interpretation_prompt)

        return intro + model_details + "\n\n## Interpretation\n" + interpretation

    # ------------------------

    def _make_technical_report(self, analysis, profile):
        """Technical deep-dive report — metrics only, no storytelling."""
        models = analysis.get("model_results", {})
        text = "# Technical Fairness Report\n\n"

        for name, res in models.items():
            m = res["fairness_metrics"]
            text += f"""
## {name.replace('_',' ').title()}
Accuracy: {m.get('accuracy',0):.4f}
Disparate Impact: {m.get('disparate_impact',0):.4f}
Statistical Parity Difference: {m.get('statistical_parity_difference',0):.4f}
Equal Opportunity Difference: {m.get('equal_opportunity_difference',0):.4f}

"""
        return text

    # ------------------------

    def _make_executive_summary(self, analysis, profile):
        """Short, business-friendly summary."""
        best = analysis.get("best_model", "Unknown")
        bias = analysis.get("bias_detected", False)

        prompt = f"""
Create a short executive summary (5-7 sentences) covering:

- Whether bias was detected: {bias}
- Which model performed best: {best}
- High-level fairness implications
- High-level recommendations

Keep it business-friendly and non-technical.
"""

        return self.generate_response(prompt)

    # ------------------------

    def _make_mitigation_recommendations(self, analysis):
        """LLM-generated bias reduction strategies."""

        models = analysis.get("model_results", {})
        best = analysis.get("best_model", "Unknown")

        prompt = f"""
Suggest practical bias mitigation strategies based on:

Models evaluated: {list(models.keys())}
Best model: {best}
Fairness metrics: {analysis}

Provide actionable steps such as reweighing, balancing, feature auditing,
model choice recommendations, or preprocessing techniques.
"""

        return self.generate_response(prompt)
