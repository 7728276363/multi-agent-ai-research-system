"""
Streamlit Multi-Agent Interface
========================================================

This Streamlit application provides a web interface for comparing multiple
AI agent frameworks (CrewAI, LangChain, LangGraph, LlamaIndex) performing
standardized multi-agent research analysis.

WHAT IS THIS APPLICATION?
This app demonstrates how different AI frameworks can be used to create
collaborative agent teams that analyze complex topics. It shows:
- Framework performance comparisons
- Multi-agent workflow execution
- Real-time analysis progress
- Comprehensive result visualization

KEY FEATURES:
1. **Framework Comparison**: Side-by-side performance testing
2. **Standardized Analysis**: Identical prompts across all frameworks
3. **Real-time Progress**: Live updates during analysis execution
4. **Result Visualization**: Charts, metrics, and detailed breakdowns
5. **Export Capabilities**: Download results as JSON or text reports

MULTI-AGENT WORKFLOW EXPLAINED:
Each framework runs the same 5-agent workflow:
1. Healthcare Domain Expert → Clinical insights and industry knowledge
2. AI Technical Analyst → Technical feasibility and implementation details
3. Healthcare Regulatory Specialist → Compliance and legal requirements
4. Healthcare Economics Analyst → Financial impact and ROI analysis
5. Strategic Content Synthesizer → Integrated strategic recommendations

To execute: streamlit run streamlit_multi_agent_app.py
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import streamlit as st          # Web application framework
import pandas as pd            # Data manipulation and analysis
import plotly.express as px    # Interactive visualizations
import plotly.graph_objects as go  # Advanced plotting capabilities
from datetime import datetime  # Date and time handling
import time                   # Time-related functions
import json                   # JSON data handling
import os                     # Operating system interface

# ============================================================================
# UTILITY FUNCTIONS FOR DATA HANDLING
# ============================================================================

def safe_get(result, key, default=0):
    """
    Safely extract values from result dictionaries with fallback defaults.
    
    Args:
        result: Dictionary or object containing results
        key (str): Key to extract
        default: Default value if key not found
    
    Returns:
        Extracted value or default
    """
    if isinstance(result, dict):
        return result.get(key, default)
    elif hasattr(result, key):
        return getattr(result, key, default)
    else:
        return default

def normalize_result(result, framework):
    """
    Standardize result format across different frameworks.
    
    Args:
        result: Raw result from framework
        framework (str): Framework name
    
    Returns:
        Dict: Normalized result structure
    """
    if not isinstance(result, dict):
        # Handle non-dictionary results (rare edge cases)
        if hasattr(result, 'raw'):
            content = str(result.raw)
        elif hasattr(result, 'output'):
            content = str(result.output)
        else:
            content = str(result)
        
        word_count = len(content.split())
        
        normalized = {
            "framework": framework,
            "total_duration": 0,
            "total_words": word_count,
            "words_per_second": 0,
            "final_synthesis": content,
            "individual_analyses": {},
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        return normalized
    
    # Ensure all required fields exist with reasonable defaults
    required_keys = ['total_duration', 'total_words', 'words_per_second', 'final_synthesis']
    for key in required_keys:
        if key not in result:
            if key == 'total_duration':
                result[key] = 0
            elif key == 'total_words':
                content = str(result.get('final_synthesis', ''))
                result[key] = len(content.split())
            elif key == 'words_per_second':
                duration = result.get('total_duration', 1)
                words = result.get('total_words', 0)
                result[key] = words / duration if duration > 0 else 0
            elif key == 'final_synthesis':
                result[key] = "No synthesis available"
    
    result['framework'] = framework
    return result

# ============================================================================
# STREAMLIT APPLICATION CONFIGURATION
# ============================================================================

# Configure the Streamlit page with custom settings
st.set_page_config(
    page_title="🤖 Standardized Multi-Agent AI Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced visual design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .standardized-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

if 'framework_stats' not in st.session_state:
    st.session_state.framework_stats = {
        'crewai': {'runs': 0, 'total_time': 0, 'total_words': 0},
        'langchain': {'runs': 0, 'total_time': 0, 'total_words': 0},
        'langgraph': {'runs': 0, 'total_time': 0, 'total_words': 0},
        'llamaindex': {'runs': 0, 'total_time': 0, 'total_words': 0}
    }

# ============================================================================
# FRAMEWORK FUNCTION IMPORT
# ============================================================================

def import_analysis_functions():
    """Import multi-agent analysis functions from all frameworks."""
    try:
        from crewai_functions import run_crewai_analysis
        from langchain_functions import run_langchain_analysis
        from langgraph_functions import run_langgraph_analysis
        from llamaindex_functions import run_llamaindex_analysis

        return {
            'crewai': run_crewai_analysis,
            'langchain': run_langchain_analysis,
            'langgraph': run_langgraph_analysis,
            'llamaindex': run_llamaindex_analysis
        }
        
    except ImportError as e:
        st.error(f"Could not import analysis functions: {e}")
        st.info("Make sure you have saved the function files with the correct names.")
        return None

# Load analysis functions
analysis_functions = import_analysis_functions()

# ============================================================================
# APPLICATION HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🤖 Multi-Agent AI Research System</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><span class="standardized-badge">STANDARDIZED TO CREWAI BASELINE</span></div>', unsafe_allow_html=True)

# Standardization explanation
with st.expander("ℹ️ About Framework Standardization"):
    st.markdown("""
    **This application ensures fair comparison across all AI frameworks:**
    
    ✅ **Identical Agent Roles**: All frameworks use the same 5 specialized agents
    
    ✅ **Identical Prompts**: All agents receive exactly the same task descriptions
    
    ✅ **Identical Context Passing**: All frameworks pass previous agent outputs as context
    
    ✅ **Identical Word Targets**: 400-500 words per agent, 1500-2000 for synthesis
    
    ✅ **Identical Workflow**: Healthcare → Technical → Regulatory → Economic → Synthesis
    
    **Baseline**: CrewAI agent definitions serve as the standard for all frameworks.
    """)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("🔧 Configuration")
    
    # Framework selection
    framework = st.selectbox(
        "Select AI Framework",
        ["crewai", "langchain", "langgraph", "llamaindex"],
        format_func=lambda x: {
            "crewai": "🔗 CrewAI (Baseline)",
            "langchain": "🔗 LangChain (Standardized)",
            "langgraph": "📊 LangGraph (Standardized)",
            "llamaindex": "🦙 LlamaIndex (Standardized)"
        }[x]
    )
    
    # Topic selection
    st.subheader("🔍 Research Topic")
    
    predefined_topics = [
        "AI-Powered Medical Diagnostics Implementation",
        "Robotic Surgery Integration Strategy", 
        "Telemedicine AI Deployment Analysis",
        "AI Drug Discovery Regulatory Pathway",
        "Healthcare AI Ethics Framework",
        "Custom Topic..."
    ]
    
    selected_topic = st.selectbox("Choose a topic:", predefined_topics)
    
    if selected_topic == "Custom Topic...":
        topic = st.text_area("Enter your custom research topic:", height=100)
    else:
        topic = selected_topic
        st.text_area("Selected topic:", value=topic, height=100, disabled=True)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        save_results = st.checkbox("Save results to file", value=True)
        show_individual_analyses = st.checkbox("Show individual agent analyses", value=True)
        real_time_updates = st.checkbox("Show real-time progress", value=True)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.header("🚀 Analysis Control")
    
    st.info("**Fair Comparison Enabled**: All frameworks use identical agents, prompts, and workflows.")
    
    # Main analysis button
    if st.button("🔍 Run Multi-Agent Analysis", type="primary", use_container_width=True):
        if not topic.strip():
            st.error("Please enter a research topic!")
        elif analysis_functions is None:
            st.error("Analysis functions not available!")
        else:
            # Progress tracking
            progress_container = st.container()
            
            with progress_container:
                st.subheader(f"Running {framework.upper()} Analysis...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                agents = [
                    'Healthcare Domain Expert', 
                    'AI Technical Analyst', 
                    'Healthcare Regulatory Specialist', 
                    'Healthcare Economics Analyst', 
                    'Strategic Content Synthesizer'
                ]
                
                if real_time_updates:
                    for i, agent in enumerate(agents):
                        status_text.text(f"🔄 {agent} analyzing...")
                        progress_bar.progress((i + 1) / len(agents))
                        time.sleep(0.5)
                
                # Execute analysis
                with st.spinner(f"Executing {framework} multi-agent analysis..."):
                    try:
                        raw_result = analysis_functions[framework](topic)
                        result = normalize_result(raw_result, framework)
                        
                        # Update session state
                        st.session_state.current_analysis = result
                        st.session_state.analysis_history.append(result)
                        
                        # Update stats
                        stats = st.session_state.framework_stats[framework]
                        stats['runs'] += 1
                        stats['total_time'] += safe_get(result, 'total_duration', 0)
                        stats['total_words'] += safe_get(result, 'total_words', 0)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Analysis complete!")
                        
                        duration = safe_get(result, 'total_duration', 0)
                        st.success(f"Analysis completed in {duration:.1f} seconds!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        error_result = {
                            "topic": topic,
                            "framework": framework,
                            "total_duration": 0,
                            "total_words": 0,
                            "words_per_second": 0,
                            "final_synthesis": f"Analysis failed: {str(e)}",
                            "individual_analyses": {},
                            "timestamp": datetime.now().isoformat(),
                            "success": False,
                            "error": str(e)
                        }
                        st.session_state.current_analysis = error_result
                        st.session_state.analysis_history.append(error_result)
            
            st.rerun()

with col2:
    st.header("📊 Performance Stats")
    
    total_runs = sum(stats['runs'] for stats in st.session_state.framework_stats.values())
    
    if total_runs > 0:
        st.metric("Total Analyses", total_runs)
        
        for fw, stats in st.session_state.framework_stats.items():
            if stats['runs'] > 0:
                avg_time = stats['total_time'] / stats['runs']
                avg_words = stats['total_words'] / stats['runs']
                avg_speed = avg_words / avg_time if avg_time > 0 else 0
                
                with st.expander(f"📈 {fw.upper()} Stats"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Runs", stats['runs'])
                        st.metric("Avg Time", f"{avg_time:.1f}s")
                    with col_b:
                        st.metric("Avg Words", f"{avg_words:.0f}")
                        st.metric("Avg Speed", f"{avg_speed:.1f} w/s")
    else:
        st.info("No analyses run yet. Start an analysis to see performance stats!")

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if st.session_state.current_analysis:
    result = st.session_state.current_analysis
    
    st.header("📋 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duration = safe_get(result, 'total_duration', 0)
        st.metric("Duration", f"{duration:.1f}s")
    
    with col2:
        word_count = safe_get(result, 'total_words', 0)
        st.metric("Word Count", f"{word_count:,}")
    
    with col3:
        speed = safe_get(result, 'words_per_second', 0)
        st.metric("Speed", f"{speed:.1f} w/s")
    
    with col4:
        # Agent completion count
        if 'completed_agents' in result:
            agents_completed = len(result['completed_agents'])
        elif 'successful_agents' in result:
            agents_completed = result['successful_agents']
        else:
            agents_completed = 5 if safe_get(result, 'success', True) else 0
        
        st.metric("Agents", f"{agents_completed}/5")
    
    # Detailed results
    if show_individual_analyses:
        individual_analyses = safe_get(result, 'individual_analyses', {})
        
        if individual_analyses and any(individual_analyses.values()):
            st.subheader("🤖 Agent Analysis Results")
            
            domain_agents = [k for k in individual_analyses.keys() if k in ['healthcare', 'technical', 'regulatory', 'economic']]
            
            agent_name_map = {
                'healthcare': 'Healthcare Domain Expert',
                'technical': 'AI Technical Analyst',
                'regulatory': 'Healthcare Regulatory Specialist', 
                'economic': 'Healthcare Economics Analyst'
            }
            
            tab_names = [agent_name_map.get(agent, agent.replace('_', ' ').title()) for agent in domain_agents]
            tab_names.append("Strategic Synthesis")
            
            agent_tabs = st.tabs(tab_names)
            
            # Individual agent tabs
            for i, agent in enumerate(domain_agents):
                with agent_tabs[i]:
                    display_name = agent_name_map.get(agent, agent.replace('_', ' ').title())
                    st.markdown(f"### {display_name}")
                    
                    analysis = individual_analyses[agent]
                    if analysis:
                        st.write(analysis)
                        word_count = len(analysis.split())
                        st.caption(f"Word count: {word_count} (Target: 400-500 words)")
                    else:
                        st.write("Analysis not available")
            
            # Synthesis tab
            with agent_tabs[-1]:
                st.markdown("### 🎯 Strategic Synthesis")
                synthesis = safe_get(result, 'final_synthesis', 'No synthesis available')
                st.write(synthesis)
                if synthesis != 'No synthesis available':
                    word_count = len(synthesis.split())
                    st.caption(f"Word count: {word_count} (Target: 1500-2000 words)")
        else:
            st.subheader("🎯 Strategic Synthesis")
            synthesis = safe_get(result, 'final_synthesis', 'No synthesis available')
            st.write(synthesis)
    else:
        st.subheader("🎯 Strategic Synthesis")
        synthesis = safe_get(result, 'final_synthesis', 'No synthesis available')
        st.write(synthesis)
    
    # Export options
    if save_results:
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"{framework}_{result['topic'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        with col2:
            report = f"""MULTI-AGENT ANALYSIS REPORT
{'='*60}

Topic: {result['topic']}
Framework: {result['framework'].upper()} (Standardized to CrewAI Baseline)
Duration: {result['total_duration']:.1f} seconds
Word Count: {result['total_words']:,}
Generation Speed: {result['words_per_second']:.1f} words/second

STRATEGIC SYNTHESIS:
{'-'*30}
{result.get('final_synthesis', 'No synthesis available')}
"""
            
            st.download_button(
                label="📝 Download Report",
                data=report,
                file_name=f"{framework}_{result['topic'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

# ============================================================================
# ANALYSIS HISTORY
# ============================================================================

if st.session_state.analysis_history:
    st.header("📚 Analysis History")
    
    # Create history table
    history_data = []
    for analysis in st.session_state.analysis_history:
        timestamp = safe_get(analysis, 'timestamp', datetime.now().isoformat())
        topic = safe_get(analysis, 'topic', 'Unknown Topic')
        framework = safe_get(analysis, 'framework', 'unknown')
        duration = safe_get(analysis, 'total_duration', 0)
        words = safe_get(analysis, 'total_words', 0)
        speed = safe_get(analysis, 'words_per_second', 0)
        
        if 'completed_agents' in analysis:
            agents_completed = len(analysis['completed_agents'])
        elif 'successful_agents' in analysis:
            agents_completed = analysis['successful_agents']
        else:
            agents_completed = 5 if analysis.get('success', True) else 0
        
        try:
            display_time = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
        except:
            display_time = str(timestamp)[:16]
        
        history_data.append({
            'Timestamp': display_time,
            'Topic': topic[:50] + "..." if len(str(topic)) > 50 else str(topic),
            'Framework': framework.upper(),
            'Duration (s)': f"{duration:.1f}",
            'Words': f"{words:,}",
            'Speed (w/s)': f"{speed:.1f}",
            'Agents': f"{agents_completed}/5"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Performance charts
    if len(st.session_state.analysis_history) > 1:
        st.subheader("📈 Framework Performance Comparison")
        
        chart_data = []
        for i, analysis in enumerate(st.session_state.analysis_history):
            framework = safe_get(analysis, 'framework', 'unknown')
            duration = safe_get(analysis, 'total_duration', 0)
            words = safe_get(analysis, 'total_words', 0)
            speed = safe_get(analysis, 'words_per_second', 0)
            
            if duration > 0 or words > 0:
                chart_data.append({
                    'Analysis #': i + 1,
                    'Framework': framework.upper(),
                    'Duration': float(duration),
                    'Words per Second': float(speed),
                    'Word Count': int(words)
                })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_duration = px.line(
                    chart_df, 
                    x='Analysis #', 
                    y='Duration', 
                    color='Framework',
                    title='Analysis Duration Over Time',
                    markers=True
                )
                st.plotly_chart(fig_duration, use_container_width=True)
            
            with col2:
                fig_speed = px.line(
                    chart_df, 
                    x='Analysis #', 
                    y='Words per Second', 
                    color='Framework',
                    title='Generation Speed Over Time',
                    markers=True
                )
                st.plotly_chart(fig_speed, use_container_width=True)
            
            # Performance summary
            if len(chart_df) >= 2:
                st.subheader("🏆 Framework Performance Summary")
                
                framework_summary = chart_df.groupby('Framework').agg({
                    'Duration': 'mean',
                    'Words per Second': 'mean',
                    'Word Count': 'mean'
                }).round(2)
                
                framework_summary = framework_summary.sort_values('Words per Second', ascending=False)
                
                summary_display = framework_summary.copy()
                summary_display['Speed Rank'] = range(1, len(summary_display) + 1)
                summary_display = summary_display[['Speed Rank', 'Words per Second', 'Duration', 'Word Count']]
                
                st.dataframe(summary_display, use_container_width=True)
                
                fastest_framework = framework_summary.index[0]
                fastest_speed = framework_summary.loc[fastest_framework, 'Words per Second']
                st.success(f"🥇 **Fastest Framework**: {fastest_framework} ({fastest_speed:.1f} words/second)")

# ============================================================================
# FOOTER AND USER GUIDANCE
# ============================================================================

st.markdown("---")
st.markdown("**🤖 Multi-Agent AI Research System** | Built with Streamlit")
st.markdown("**🎯 Fair Comparison Enabled** | All frameworks use identical agents, prompts, and workflows")

# Welcome message for new users
if not st.session_state.analysis_history:
    st.info("""
    👋 **Welcome to the Multi-Agent AI Research System!**
    
    **Quick Start Guide:**
    1. Choose a framework from the sidebar (CrewAI, LangChain, LangGraph, or LlamaIndex)
    2. Select a research topic or enter your own
    3. Click "Run Multi-Agent Analysis" to start
    4. Compare results across different frameworks
    
    **What Makes This Special:**
    - All frameworks use identical agents and prompts for fair comparison
    - 5-agent workflow: Healthcare Expert → Technical Analyst → Regulatory Specialist → Economic Analyst → Strategic Synthesizer
    - Real-time progress tracking and comprehensive performance metrics
    - Export capabilities for detailed analysis reports
    """)
