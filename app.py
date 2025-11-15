import streamlit as st
import pandas as pd
import sys
import os

api_key = st.secrets["GROQ_API_KEY"]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import DataProfilerAgent, DataCleaningAgent, BiasDetectionAgent, ReportGenerationAgent, ConversationalAgent
import config

class BiasLensApp:
    def __init__(self):
        self.setup_page()
        self.initialize_agents()
        self.initialize_session_state()
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="BiasLens 2.0 - AI Fairness Analysis",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔍 BiasLens 2.0")
        st.markdown("### AI-Powered Fairness Analysis Platform")
    
    def initialize_agents(self):
        """Initialize all agents"""
        if 'agents' not in st.session_state:
            st.session_state.agents = {
                'profiler': DataProfilerAgent(),
                'cleaner': DataCleaningAgent(), 
                'detector': BiasDetectionAgent(),
                'reporter': ReportGenerationAgent(),
                'chat': ConversationalAgent()
            }
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data_uploaded': False,
            'raw_data': None,
            'cleaned_data': None,
            'analysis_complete': False,
            'current_step': 1,
            'analysis_context': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Render sidebar with progress and info"""
        with st.sidebar:
            st.header("🔍 Analysis Progress")
            
            steps = [
                "1. Data Upload & Profiling",
                "2. Data Cleaning", 
                "3. Bias Analysis",
                "4. Report Generation",
                "5. Interactive Q&A"
            ]
            
            current_step = st.session_state.get('current_step', 1)
            
            for i, step in enumerate(steps, 1):
                if i == current_step:
                    st.markdown(f"**▶️ {step}**")
                elif i < current_step:
                    st.markdown(f"✅ {step}")
                else:
                    st.markdown(f"⏸️ {step}")
            
            st.divider()
            st.markdown("### About BiasLens 2.0")
            st.markdown("""
            An agentic AI system that:
            - 🤖 Uses 5 specialized agents
            - 📊 Analyzes model fairness automatically  
            - 💬 Explains results in simple language
            - 🛠️ Suggests bias mitigation strategies
            """)
    
    def run(self):
        """Main application loop"""
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Data Upload", 
            "🔧 Data Cleaning", 
            "📊 Bias Analysis",
            "📋 Reports", 
            "💬 Ask Questions"
        ])
        
        with tab1:
            self.render_data_upload()
        
        with tab2:
            self.render_data_cleaning()
        
        with tab3:
            self.render_bias_analysis()
        
        with tab4:
            self.render_reports()
        
        with tab5:
            self.render_chat_interface()
    
    def render_data_upload(self):
        st.header("📁 Data Upload & Profiling")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file",
            type=['csv'],
            help="Upload any CSV dataset for bias analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Simple CSV reading with headers
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_data = df
                st.session_state.data_uploaded = True
                
                # Display basic file info
                st.subheader("📊 Dataset Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Show column names
                st.write("**Columns:**", list(df.columns))
                
                # Data preview
                with st.expander("👀 Data Preview (First 10 rows)"):
                    st.dataframe(df.head(10), width='stretch')
                
                # Run Data Profiler Agent
                if st.button("🚀 Run Data Profiling", type="primary"):
                    with st.spinner("🤖 Data Profiler Agent is analyzing your dataset..."):
                        profiler_agent = st.session_state.agents['profiler']
                        profile_result = profiler_agent.execute({
                            'raw_data': df
                        })
                    
                    if profile_result['success']:
                        st.session_state.profile_result = profile_result
                        st.session_state.current_step = 2
                        st.success("✅ Data profiling completed!")
                        st.rerun()
                    else:
                        st.error(f"Profiling failed: {profile_result.get('error', 'Unknown error')}")
                
                # Display existing profiling results if available
                if 'profile_result' in st.session_state and st.session_state.profile_result['success']:
                    self._display_profiling_results(st.session_state.profile_result['data_profile'])
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("💡 Make sure your CSV file has headers in the first row")
        
        else:
            st.info("👆 Drag and drop a CSV file to begin analysis")
    
    def _display_profiling_results(self, profile):
        """Display data profiling results"""
        st.subheader("📊 Data Profile Summary")
        
        # Basic Information
        with st.expander("📋 Basic Dataset Information", expanded=True):
            basic_info = profile['basic_info']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Shape", f"{basic_info['shape'][0]} × {basic_info['shape'][1']}")
            with col2:
                st.metric("Memory", basic_info['memory_usage'])
            with col3:
                st.metric("Duplicates", basic_info['duplicate_rows'])
            with col4:
                st.metric("Missing Values", basic_info['total_missing_values'])
        
        # Data Types
        with st.expander("🔧 Data Types Analysis"):
            data_types = profile['data_types']
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numeric Columns:**", len(data_types['numeric']))
                if data_types['numeric']:
                    st.write(data_types['numeric'])
            with col2:
                st.write("**Categorical Columns:**", len(data_types['categorical']))
                if data_types['categorical']:
                    st.write(data_types['categorical'])
        
        # Sensitive Attributes Detection
        with st.expander("⚖️ Sensitive Attributes Detection"):
            sensitive_attrs = profile['sensitive_attributes_suggested']
            if sensitive_attrs:
                st.success("🔍 Potential sensitive attributes detected:")
                for attr in sensitive_attrs:
                    st.write(f"- **{attr['column']}**: {attr['reason']} ({attr['unique_values']} unique values)")
            else:
                st.info("No obvious sensitive attributes detected. You'll be able to select manually in the analysis tab.")
        
        # NOTE: Data Quality Assessment removed as requested by user
        
        # LLM Insights - CORRECTED INDENTATION
        with st.expander("🤖 AI Insights"):
            insights = profile.get('llm_insights', 'No insights generated')
            if "Error generating response" in insights:
                st.warning("⚠️ AI insights unavailable - API key issue")
                st.info("Make sure your Groq API key is correctly set in the .env file")
            else:
                st.write(insights)
        
        # Next steps
        st.success("🎉 Data profiling completed! Move to the 'Data Cleaning' tab to continue.")
    
    def render_data_cleaning(self):
        st.header("🔧 Data Cleaning")
        
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Please upload and profile your data in the 'Data Upload' tab first.")
            return
        
        st.info("Data Cleaning Agent will prepare your data for bias analysis")
        
        # Only keep the Run Data Cleaning button (removed Current Data Status metrics)
        if st.button("🔄 Run Data Cleaning", type="primary"):
            with st.spinner("🤖 Data Cleaning Agent is preparing your data..."):
                cleaner_agent = st.session_state.agents['cleaner']
                cleaning_result = cleaner_agent.execute({
                    'raw_data': st.session_state.raw_data
                })
            
            if cleaning_result['success']:
                st.session_state.cleaned_data = cleaning_result['cleaned_data']
                st.session_state.cleaning_report = cleaning_result['cleaning_report']
                st.success("✅ Data cleaning completed!")
                
                # Display cleaning results
                st.subheader("Cleaning Report")
                report = cleaning_result['cleaning_report']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Shape", f"{report['original_shape'][0]} × {report['original_shape'][1]}")
                with col2:
                    st.metric("Cleaned Shape", f"{report['cleaned_shape'][0]} × {report['cleaned_shape'][1]}")
                with col3:
                    st.metric("Missing Values Fixed", report['missing_values_removed'])
                
                st.write("**Operations Performed:**")
                for operation in report['operations_performed']:
                    st.write(f"✅ {operation}")
                
                st.write("**Data Types Changed:**")
                st.write(f"📊 {report['data_types_changed']}")
                
                if 'cleaning_summary' in report:
                    st.write("**Summary:**")
                    st.info(report['cleaning_summary'])
                
                # Show cleaned data preview
                with st.expander("👀 Preview Cleaned Data"):
                    st.dataframe(st.session_state.cleaned_data.head(10), width='stretch')
                
                st.session_state.current_step = 3
                st.success("🎉 Ready for bias analysis! Move to the 'Bias Analysis' tab.")
                
            else:
                st.error(f"Cleaning failed: {cleaning_result.get('error', 'Unknown error')}")
    
    def render_bias_analysis(self):
        st.header("📊 Bias Analysis")
        
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Please upload and profile your data in the 'Data Upload' tab first.")
            return
        
        if st.session_state.cleaned_data is None:
            st.warning("⚠️ Please clean your data in the 'Data Cleaning' tab first.")
            return
        
        st.info("Bias Detection Agent will analyze fairness across multiple ML models")
        
        # Configuration section
        st.subheader("⚙️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target variable selection
            target_options = list(st.session_state.cleaned_data.columns)
            target_variable = st.selectbox(
                "Select target variable:",
                options=target_options,
                help="The column you want to predict"
            )
        
        with col2:
            # Sensitive attribute selection
            sensitive_options = list(st.session_state.cleaned_data.columns)
            sensitive_attribute = st.selectbox(
                "Select sensitive attribute:",
                options=sensitive_options,
                help="The column to check for bias (e.g., gender, race, age)"
            )
        
        if st.button("🔍 Run Bias Analysis", type="primary"):
            with st.spinner("🤖 Bias Detection Agent is analyzing fairness across models..."):
                detector_agent = st.session_state.agents['detector']
                analysis_result = detector_agent.execute(
                    data_context={
                        'cleaned_data': st.session_state.cleaned_data
                    },
                    user_input={
                        'target_variable': target_variable,
                        'sensitive_attribute': sensitive_attribute
                    }
                )
            
            if analysis_result['success']:
                st.session_state.analysis_result = analysis_result
                st.session_state.current_step = 4
                st.session_state.analysis_complete = True
                st.success("✅ Bias analysis completed!")
                st.rerun()
            else:
                st.error(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        
        # Display existing results if available
        if 'analysis_result' in st.session_state and st.session_state.analysis_result['success']:
            st.subheader("📈 Analysis Results")
            
            result = st.session_state.analysis_result
            
            # Show success message
            st.success("✅ " + result.get('message', 'Bias analysis completed successfully!'))
            
            # Show best model
            best_model = result.get('best_model')
            if best_model:
                st.info(f"🏆 **Best Model**: {best_model.replace('_', ' ').title()}")
            
            # Show bias detection result
            bias_detected = result.get('bias_detected', False)
            if bias_detected:
                st.error("🚨 **Significant bias detected** in the models")
            else:
                st.success("✅ **No significant bias detected**")
            
            # Show model performance comparison
            st.subheader("📊 Model Performance Comparison")
            
            if 'model_results' in result:
                model_results = result['model_results']
                
                # Create a comparison table
                comparison_data = []
                for model_name, model_data in model_results.items():
                    metrics = model_data.get('fairness_metrics', {})
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                        'Disparate Impact': f"{metrics.get('disparate_impact', 0):.3f}",
                        'Statistical Parity Diff': f"{metrics.get('statistical_parity_difference', 0):.3f}",
                        'Equal Opportunity Diff': f"{metrics.get('equal_opportunity_difference', 0):.3f}"
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch')
            
            # Show AI Insights
            if 'ai_insights' in result:
                with st.expander("🤖 AI Analysis Insights", expanded=True):
                    st.write(result['ai_insights'])
            
            # Show detailed results for each model
            with st.expander("🔍 Detailed Model Results"):
                if 'model_results' in result:
                    for model_name, model_data in result['model_results'].items():
                        st.write(f"### {model_name.replace('_', ' ').title()}")
                        metrics = model_data.get('fairness_metrics', {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                        with col2:
                            st.metric("Disparate Impact", f"{metrics.get('disparate_impact', 0):.3f}")
                        with col3:
                            st.metric("Stat Parity Diff", f"{metrics.get('statistical_parity_difference', 0):.3f}")
                        with col4:
                            st.metric("Equal Opp Diff", f"{metrics.get('equal_opportunity_difference', 0):.3f}")
                        
                        st.divider()
            st.success("🎉 Bias analysis completed! Move to the 'Reports' tab to generate comprehensive fairness reports.")
    
    def render_reports(self):
        st.header("📋 Generated Reports")
        
        if not st.session_state.get('analysis_complete', False):
            st.warning("⚠️ Please complete the bias analysis first.")
            return
        
        st.info("Report Generation Agent creates comprehensive fairness analysis reports")
        
        # Show current analysis status
        st.subheader("📊 Analysis Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Processed", "✅ Ready")
        with col2:
            st.metric("Bias Analysis", "✅ Complete")
        with col3:
            st.metric("Reports", "🔄 Ready to Generate")
        
        # Report type selection
        st.subheader("📝 Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Select report type:",
                ["Comprehensive Report", "Technical Report", "Executive Summary"],
                help="Choose the type of report to generate"
            )
        
        with col2:
            complexity_level = st.selectbox(
                "Explanation level:",
                ["Simple", "Business", "Technical"],
                help="Choose the complexity level for explanations"
            )
        
        # Generate reports
        if st.button("📄 Generate Fairness Reports", type="primary"):
            with st.spinner("🤖 Report Generation Agent is creating comprehensive reports..."):
                reporter_agent = st.session_state.agents['reporter']
                report_result = reporter_agent.execute(
                    data_context={
                        'analysis_result': st.session_state.analysis_result,
                        'cleaned_data': st.session_state.cleaned_data,
                        'profile_result': st.session_state.profile_result
                    },
                    user_input={
                        'report_type': report_type,
                        'complexity_level': complexity_level
                    }
                )
            
            if report_result['success']:
                st.session_state.report_result = report_result
                st.session_state.current_report_type = report_type
                st.session_state.current_complexity = complexity_level
                st.success("✅ Reports generated successfully!")
                st.rerun()
            else:
                st.error(f"Report generation failed: {report_result.get('error', 'Unknown error')}")
        
        # Display generated reports if available
        if 'report_result' in st.session_state and st.session_state.report_result['success']:
            st.subheader("📋 Generated Reports")
            
            report_data = st.session_state.report_result
            report_type = st.session_state.current_report_type
            complexity_level = st.session_state.current_complexity
            
            # Show available reports
            available_reports = report_data.get('reports', {})
            
            if available_reports:
                # Show the selected report type first
                if report_type == "Comprehensive Report":
                    report_content = available_reports.get('comprehensive_report', 'Report not generated yet.')
                    st.markdown("### Comprehensive Fairness Analysis Report")
                    # use a large text_area to prevent truncation/cut-off mid-sentence
                    st.text_area("Report Content", value=report_content, height=400)
                    
                    # Download button for comprehensive report
                    st.download_button(
                        label="📥 Download Comprehensive Report",
                        data=report_content,
                        file_name="biaslens_comprehensive_report.md",
                        mime="text/markdown"
                    )
                
                elif report_type == "Technical Report":
                    report_content = available_reports.get('technical_report', 'Report not generated yet.')
                    st.markdown("### Technical Report")
                    st.text_area("Report Content", value=report_content, height=400)
                    
                    # Download button for technical report
                    st.download_button(
                        label="📥 Download Technical Report",
                        data=report_content,
                        file_name="biaslens_technical_report.md",
                        mime="text/markdown"
                    )
                
                elif report_type == "Executive Summary":
                    report_content = available_reports.get('executive_summary', 'Report not generated yet.')
                    st.markdown("### Executive Summary")
                    st.text_area("Report Content", value=report_content, height=400)
                    
                    # Download button for executive summary
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=report_content,
                        file_name="biaslens_executive_summary.md",
                        mime="text/markdown"
                    )
                
                # Show bias mitigation recommendations below the main report
                if 'mitigation_recommendations' in available_reports:
                    st.markdown("---")
                    with st.expander("🛠️ Bias Mitigation Recommendations", expanded=True):
                        st.markdown("### Recommended Bias Mitigation Strategies")
                        st.markdown(available_reports['mitigation_recommendations'])
            
            st.success("🎉 All reports generated! Move to the 'Ask Questions' tab for interactive analysis.")
    
    def render_chat_interface(self):
        # Top header and intro removed as requested
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'suggested_questions' not in st.session_state:
            st.session_state.suggested_questions = [
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
        
        # Display conversation history
        st.subheader("💭 Conversation")
        
        if not st.session_state.chat_history:
            st.info("💡 Start by asking a question or clicking a suggested question below!")
        
        # Display chat messages
        for i, chat in enumerate(st.session_state.chat_history):
            if chat['type'] == 'user':
                with st.chat_message("user"):
                    st.write(chat['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(chat['content'])
        
        # Suggested questions
        st.subheader("🎯 Suggested Questions")
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggested_{i}"):
                    # Use a different approach for suggested questions
                    self._handle_suggested_question(question)
        
        # Chat input - use a different key approach
        st.subheader("💬 Ask Your Question")
        
        # Create a unique key for this session
        chat_key = f"chat_input_{len(st.session_state.chat_history)}"
        
        user_question = st.text_input(
            "Type your question here:",
            key=chat_key,
            placeholder="e.g., What bias was found in the Random Forest model?"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            ask_clicked = st.button("🚀 Ask", type="primary", key="ask_button")
        
        with col2:
            clear_clicked = st.button("🗑️ Clear Chat", key="clear_button")
        
        # Handle ask button click
        if ask_clicked and user_question:
            with st.spinner("🤖 Thinking..."):
                chat_agent = st.session_state.agents['chat']
                chat_result = chat_agent.execute(
                    data_context={
                        'analysis_result': st.session_state.analysis_result,
                        'profile_result': st.session_state.profile_result,
                        'cleaned_data': st.session_state.cleaned_data
                    },
                    user_input={
                        'question': user_question
                    }
                )
            
            if chat_result['success']:
                # Add user question to history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_question
                })
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': chat_result['response']
                })
                
                # Rerun to clear the input and show new messages
                st.rerun()
            else:
                st.error(f"Chat failed: {chat_result.get('error', 'Unknown error')}")
        
        # Handle clear button click
        if clear_clicked:
            st.session_state.chat_history = []
            st.rerun()
        
        # Current Analysis Context removed as requested

    def _handle_suggested_question(self, question):
        """Handle suggested question selection"""
        with st.spinner("🤖 Thinking..."):
            chat_agent = st.session_state.agents['chat']
            chat_result = chat_agent.execute(
                data_context={
                    'analysis_result': st.session_state.analysis_result,
                    'profile_result': st.session_state.profile_result,
                    'cleaned_data': st.session_state.cleaned_data
                },
                user_input={
                    'question': question
                }
            )
        
        if chat_result['success']:
            # Add user question to history
            st.session_state.chat_history.append({
                'type': 'user',
                'content': question
            })
            
            # Add AI response to history
            st.session_state.chat_history.append({
                'type': 'assistant',
                'content': chat_result['response']
            })
            
            # Rerun to show new messages
            st.rerun()
        else:
            st.error(f"Chat failed: {chat_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    app = BiasLensApp()
    app.run()
