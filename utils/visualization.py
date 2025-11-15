import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

class VisualizationEngine:
    """Generate visualizations for fairness analysis"""
    
    def __init__(self):
        self.figures = {}
    
    def create_fairness_comparison_chart(self, model_results):
        """Create bar chart comparing fairness metrics across models"""
        metrics_df = self._prepare_metrics_dataframe(model_results)
        
        fig = go.Figure()
        
        for metric in ['disparate_impact', 'statistical_parity_difference', 'equal_opportunity_difference']:
            if metric in metrics_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=metrics_df.index,
                    y=metrics_df[metric]
                ))
        
        fig.update_layout(
            title="Fairness Metrics Comparison Across Models",
            xaxis_title="Models",
            yaxis_title="Metric Values",
            barmode='group'
        )
        
        return fig
    
    def create_accuracy_fairness_tradeoff(self, model_results):
        """Create scatter plot showing accuracy vs fairness tradeoff"""
        metrics_df = self._prepare_metrics_dataframe(model_results)
        
        fig = px.scatter(
            metrics_df, 
            x='accuracy', 
            y='statistical_parity_difference',
            size='disparate_impact',
            color=metrics_df.index,
            title="Accuracy vs Fairness Trade-off",
            labels={
                'accuracy': 'Accuracy',
                'statistical_parity_difference': 'Statistical Parity Difference',
                'disparate_impact': 'Disparate Impact (size)'
            }
        )
        
        # Add ideal point reference
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Perfect Fairness")
        
        return fig
    
    def create_metric_radar_chart(self, model_results):
        """Create radar chart for model comparison"""
        metrics_df = self._prepare_metrics_dataframe(model_results)
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_df = self._normalize_metrics(metrics_df)
        
        fig = go.Figure()
        
        for model in normalized_df.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized_df.loc[model].values,
                theta=normalized_df.columns,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title="Model Comparison - Radar Chart",
            showlegend=True
        )
        
        return fig
    
    def _prepare_metrics_dataframe(self, model_results):
        """Prepare metrics data for visualization"""
        data = {}
        for model_name, results in model_results.items():
            if 'fairness_metrics' in results:
                data[model_name] = results['fairness_metrics']
        
        return pd.DataFrame(data).T
    
    def _normalize_metrics(self, df):
        """Normalize metrics to 0-1 scale for radar chart"""
        # For metrics where 1 is ideal (accuracy, disparate_impact)
        normalized_df = df.copy()
        
        for col in normalized_df.columns:
            if col in ['accuracy', 'disparate_impact']:
                # Normalize with 1 as best
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            else:
                # For difference metrics, we want values close to 0
                abs_max = normalized_df[col].abs().max()
                if abs_max > 0:
                    normalized_df[col] = 1 - (normalized_df[col].abs() / abs_max)
        
        return normalized_df