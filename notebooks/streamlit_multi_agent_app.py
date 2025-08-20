# Streamlit Multi-Agent Interface
# Save as: streamlit_multi_agent_app.py

#
# to execute: streamlit run streamlit_multi_agent_app.py
#

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import os

# HELPER FUNCTIONS
def safe_get(result, key, default=0):
    """Safely get a value from result dict, handling different formats"""
    if isinstance(result, dict):
        return result.get(key, default)
    elif hasattr(result, key):
        return getattr(result, key, default)
    else:
        return default

def normalize_result(result, framework):
    """Normalize result format across different frameworks"""
    if not isinstance(result, dict):
        # Convert non-dict results to dict format
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
    
    # Ensure required keys exist
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

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Standardized Multi-Agent AI Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .framework-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .standardized-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .agent-status {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-success { background-color: #28a745; }
    .status-error { background-color: #dc3545; }
    .status-running { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Import analysis functions
def import_analysis_functions():
    """Import standardized analysis functions"""
    try:
        # Import functions
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

# Header
st.markdown('<h1 class="main-header">🤖 Multi-Agent AI Research System</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><span class="standardized-badge">TO CREWAI BASELINE</span></div>', unsafe_allow_html=True)

# Add standardization info box
with st.expander("ℹ️ About Standardization"):
    st.markdown("""
    **This version ensures fair comparison across all frameworks:**
    
    ✅ **Identical Agent Roles**: All frameworks use the same 5 agents with identical roles and goals
    
    ✅ **Identical Prompts**: All agents receive exactly the same prompts and instructions
    
    ✅ **Identical Context Passing**: All frameworks pass previous agent outputs as context
    
    ✅ **Identical Word Targets**: All agents target 400-500 words, synthesis targets 1500-2000 words
    
    ✅ **Identical Workflow**: Sequential execution: Healthcare → Technical → Regulatory → Economic → Synthesis
    
    **Baseline**: CrewAI agent definitions and prompts used as the standard for all frameworks.
    """)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Framework Selection
    framework = st.selectbox(
        "Select Framework",
        ["crewai", "langchain", "langgraph", "llamaindex"],
        format_func=lambda x: {
            "crewai": "🔗 CrewAI (Baseline)",
            "langchain": "🔗 LangChain (Standardized)",
            "langgraph": "📊 LangGraph (Standardized)",
            "llamaindex": "🦙 LlamaIndex (Standardized)"
        }[x]
    )
    
    # Topic Input
    st.subheader("📝 Research Topic")
    
    # Predefined topics
    predefined_topics = [
        "AI-Powered Medical Diagnostics Implementation",
        "Robotic Surgery Integration Strategy", 
        "Telemedicine AI Deployment Analysis",
        "AI Drug Discovery Regulatory Pathway",
        "Healthcare AI Ethics Framework",
        "Custom Topic..."
    ]
    
    selected_topic = st.selectbox("Choose a topic or select custom:", predefined_topics)
    
    if selected_topic == "Custom Topic...":
        topic = st.text_area("Enter your custom research topic:", height=100)
    else:
        topic = selected_topic
        st.text_area("Selected topic:", value=topic, height=100, disabled=True)
    
    # Advanced Options
    with st.expander("⚙️ Advanced Options"):
        save_results = st.checkbox("Save results to file", value=True)
        show_individual_analyses = st.checkbox("Show individual agent analyses", value=True)
        real_time_updates = st.checkbox("Show real-time progress", value=True)
    
    # Framework Information
    st.subheader("ℹ️ Standardized Framework Info")
    
    standardized_info = {
        "crewai": {
            "description": "Original CrewAI implementation (BASELINE)",
            "standardization": "This is the baseline - all other frameworks match this exactly"
        },
        "langchain": {
            "description": "LangChain standardized to match CrewAI baseline",
            "standardization": "Agents, prompts, and workflow now identical to CrewAI"
        },
        "langgraph": {
            "description": "LangGraph standardized to match CrewAI baseline", 
            "standardization": "State management preserved, but agents/prompts now match CrewAI"
        },
        "llamaindex": {
            "description": "LlamaIndex standardized to match CrewAI baseline",
            "standardization": "Tool-based approach preserved, but prompts now match CrewAI"
        }
    }
    
    info = standardized_info[framework]
    st.markdown(f"**{info['description']}**")
    st.write(f"**Standardization:** {info['standardization']}")
    
    # Show agent list
    with st.expander("View Agents"):
        agents = [
            "1. Healthcare Domain Expert",
            "2. AI Technical Analyst", 
            "3. Healthcare Regulatory Specialist",
            "4. Healthcare Economics Analyst",
            "5. Strategic Content Synthesizer"
        ]
        for agent in agents:
            st.write(f"• {agent}")

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🚀 Analysis Control")
    
    # Add comparison info
    st.info("""
    **Fair Comparison Enabled**: All frameworks now use identical agents, prompts, and workflows.
    Performance differences reflect framework architecture, not prompt variations.
    """)
    
    # Run Analysis Button
    if st.button("🔍 Run Multi-Agent Analysis", type="primary", use_container_width=True):
        if not topic.strip():
            st.error("Please enter a research topic!")
        elif analysis_functions is None:
            st.error("Analysis functions not available!")
        else:
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                st.subheader(f"Running {framework.upper()} Analysis...")
                st.caption("All frameworks use identical agents and prompts for fair comparison")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Agent progress
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
                        time.sleep(0.5)  # Simulate processing time
                
                # Run actual analysis with error handling
                with st.spinner(f"Executing {framework} multi-agent analysis..."):
                    try:
                        raw_result = analysis_functions[framework](topic)
                        result = normalize_result(raw_result, framework)
                        
                        # Update session state
                        st.session_state.current_analysis = result
                        st.session_state.analysis_history.append(result)
                        
                        # Update framework stats with safe extraction
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
                        # Create error result
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
    
    if st.session_state.framework_stats:
        # Create performance metrics
        total_runs = sum(stats['runs'] for stats in st.session_state.framework_stats.values())
        
        if total_runs > 0:
            # Overall metrics
            st.metric("Total Analyses", total_runs)
            
            # Framework-specific metrics
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

# Results Display
if st.session_state.current_analysis:
    result = st.session_state.current_analysis
    
    st.header("📋 Analysis Results")
    
    # Show standardization badge
    st.markdown('<div style="margin-bottom: 1rem;"><span class="standardized-badge">RESULTS</span></div>', unsafe_allow_html=True)
    
    # Key Metrics with safe extraction
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
        # Handle different ways agents might be tracked
        if 'completed_agents' in result:
            agents_completed = len(result['completed_agents'])
        elif 'individual_analyses' in result:
            agents_completed = len([a for a in result['individual_analyses'].values() if a])
        elif 'successful_agents' in result:
            agents_completed = result['successful_agents']
        else:
            agents_completed = 5 if safe_get(result, 'success', True) else 0
        
        st.metric("Agents", f"{agents_completed}/5")
    
    # Agent Status - handle different result formats
    if show_individual_analyses:
        individual_analyses = safe_get(result, 'individual_analyses', {})
        
        if individual_analyses and any(individual_analyses.values()):
            st.subheader("🤖 Agent Analysis Results")
            st.caption("All agents use identical prompts across frameworks")
            
            # Create tabs for the 4 domain agents only (exclude any synthesis key) + Final Synthesis
            domain_agents = [k for k in individual_analyses.keys() if k in ['healthcare', 'technical', 'regulatory', 'economic']]




            
            print("****************************************************")
            print("****************************************************")
            print("****************************************************")
            print("Streamlit")
            print(individual_analyses)
            print("****************************************************")
            print("****************************************************")
            print("****************************************************")



            
            # Agent name mapping
            agent_name_map = {
                'healthcare': 'Healthcare Domain Expert',
                'technical': 'AI Technical Analyst',
                'regulatory': 'Healthcare Regulatory Specialist', 
                'economic': 'Healthcare Economics Analyst'
            }
            
            # Create tab names
            tab_names = [agent_name_map.get(agent, agent.replace('_', ' ').title()) for agent in domain_agents]
            tab_names.append("Final Synthesis")
            
            agent_tabs = st.tabs(tab_names)
            
            # Display domain agent analyses
            for i, agent in enumerate(domain_agents):
                with agent_tabs[i]:
                    display_name = agent_name_map.get(agent, agent.replace('_', ' ').title())
                    st.markdown(f"### {display_name}")
                    
                    analysis = individual_analyses[agent]
                    if analysis:
                        st.write(analysis)
                        # Show word count
                        word_count = len(analysis.split())
                        st.caption(f"Word count: {word_count} (Target: 400-500 words)")
                    else:
                        st.write("Analysis not available")
            
            # Final synthesis tab (always last tab)
            with agent_tabs[-1]:
                st.markdown("### 🎯 Strategic Synthesis")
                synthesis = safe_get(result, 'final_synthesis', 'No synthesis available')
                st.write(synthesis)
                if synthesis != 'No synthesis available':
                    word_count = len(synthesis.split())
                    st.caption(f"Word count: {word_count} (Target: 1500-2000 words)")
        else:
            # Show just the final synthesis
            st.subheader("🎯 Final Strategic Synthesis")
            synthesis = safe_get(result, 'final_synthesis', 'No synthesis available')
            st.write(synthesis)
    else:
        # Show just the final synthesis
        st.subheader("🎯 Final Strategic Synthesis")
        synthesis = safe_get(result, 'final_synthesis', 'No synthesis available')
        st.write(synthesis)
    
    # Download Results
    if save_results:
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"{framework}_{result['topic'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        with col2:
            # Text report download
            report = f"""
MULTI-AGENT ANALYSIS REPORT
{'='*60}

Topic: {result['topic']}
Framework: {result['framework'].upper()} (Standardized to CrewAI Baseline)
Duration: {result['total_duration']:.1f} seconds
Word Count: {result['total_words']:,}
Generation Speed: {result['words_per_second']:.1f} words/second

STANDARDIZATION NOTES:
- All frameworks use identical agent roles and prompts
- Sequential workflow: Healthcare → Technical → Regulatory → Economic → Synthesis
- Context passing enabled between all agents
- Target word counts: 400-500 per agent, 1500-2000 for synthesis

STRATEGIC SYNTHESIS:
{'-'*30}
{result.get('final_synthesis', 'No synthesis available')}

INDIVIDUAL ANALYSES:
{'-'*30}
"""
            
            if 'individual_analyses' in result:
                agent_name_map = {
                    'healthcare': 'Healthcare Domain Expert',
                    'technical': 'AI Technical Analyst',
                    'regulatory': 'Healthcare Regulatory Specialist',
                    'economic': 'Healthcare Economics Analyst'
                }
                
                for agent, analysis in result['individual_analyses'].items():
                    # Skip synthesis since it's already shown in the main synthesis section
                    if agent != 'synthesis':
                        display_name = agent_name_map.get(agent, agent.upper())
                        report += f"\n{display_name.upper()}:\n{analysis}\n\n"
            
            st.download_button(
                label="📝 Download Report",
                data=report,
                file_name=f"{framework}_{result['topic'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

# Debug information for LlamaIndex
if st.session_state.current_analysis and st.session_state.current_analysis.get('framework') == 'llamaindex':
    result = st.session_state.current_analysis
    
    if 'debug_log' in result or 'diagnostic_info' in result:
        with st.expander("🔍 LlamaIndex Debug Information"):
            
            # Show diagnostic info
            if 'diagnostic_info' in result:
                st.subheader("📊 Diagnostic Status")
                diag = result['diagnostic_info']
                for key, value in diag.items():
                    status = "✅" if value else "❌"
                    st.write(f"{status} {key}: {value}")
            
            # Show debug log
            if 'debug_log' in result:
                st.subheader("📝 Debug Log")
                for log_entry in result['debug_log']:
                    st.text(log_entry)
            
            # Show traceback if available
            if 'traceback' in result:
                st.subheader("🐛 Error Traceback")
                st.code(result['traceback'])

            if 'initialization_log' in result:
                st.subheader("🔧 Agent Initialization Log")
                for log_entry in result['initialization_log']:
                    st.text(log_entry)
            
            if 'available_agents' in result:
                st.subheader("🤖 Available Agents")
                st.write(f"Created agents: {result['available_agents']}")
                st.write(f"Agent count: {len(result['available_agents'])}")

# Analysis History
if st.session_state.analysis_history:
    st.header("📚 Analysis History")
    st.caption("All results use standardized prompts for fair comparison")

    # Create DataFrame for history with safe extraction
    history_data = []
    for analysis in st.session_state.analysis_history:
        # Use safe_get to extract values regardless of result format
        timestamp = safe_get(analysis, 'timestamp', datetime.now().isoformat())
        topic = safe_get(analysis, 'topic', 'Unknown Topic')
        framework = safe_get(analysis, 'framework', 'unknown')
        duration = safe_get(analysis, 'total_duration', 0)
        words = safe_get(analysis, 'total_words', 0)
        speed = safe_get(analysis, 'words_per_second', 0)
        
        # Format timestamp for display
        try:
            if isinstance(timestamp, str):
                display_time = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
            else:
                display_time = str(timestamp)[:16]  # Truncate if needed
        except:
            display_time = str(timestamp)[:16]
        
        history_data.append({
            'Timestamp': display_time,
            'Topic': topic[:50] + "..." if len(str(topic)) > 50 else str(topic),
            'Framework': framework.upper(),
            'Duration (s)': f"{duration:.1f}",
            'Words': f"{words:,}",
            'Speed (w/s)': f"{speed:.1f}"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Performance comparison chart
    if len(st.session_state.analysis_history) > 1:
        st.subheader("📈 Framework Performance Comparison")
        st.caption("Fair comparison enabled - all frameworks use identical prompts")
        
        # Create performance chart data with safe extraction
        chart_data = []
        for i, analysis in enumerate(st.session_state.analysis_history):
            # Extract data safely
            framework = safe_get(analysis, 'framework', 'unknown')
            duration = safe_get(analysis, 'total_duration', 0)
            words = safe_get(analysis, 'total_words', 0)
            speed = safe_get(analysis, 'words_per_second', 0)
            
            # Only include valid data points
            if duration > 0 or words > 0:
                chart_data.append({
                    'Analysis #': i + 1,
                    'Framework': framework.upper(),
                    'Duration': float(duration),
                    'Words per Second': float(speed),
                    'Word Count': int(words)
                })
        
        if chart_data:  # Only create charts if we have valid data
            chart_df = pd.DataFrame(chart_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not chart_df.empty and 'Duration' in chart_df.columns:
                    fig_duration = px.line(
                        chart_df, 
                        x='Analysis #', 
                        y='Duration', 
                        color='Framework',
                        title='Analysis Duration Over Time (Standardized)',
                        markers=True
                    )
                    fig_duration.update_layout(
                        xaxis_title="Analysis Number",
                        yaxis_title="Duration (seconds)"
                    )
                    st.plotly_chart(fig_duration, use_container_width=True)
                else:
                    st.info("No duration data available for charting")
            
            with col2:
                if not chart_df.empty and 'Words per Second' in chart_df.columns:
                    fig_speed = px.line(
                        chart_df, 
                        x='Analysis #', 
                        y='Words per Second', 
                        color='Framework',
                        title='Generation Speed Over Time (Standardized)',
                        markers=True
                    )
                    fig_speed.update_layout(
                        xaxis_title="Analysis Number",
                        yaxis_title="Words per Second"
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)
                else:
                    st.info("No speed data available for charting")
            
            # Framework comparison summary
            if len(chart_df) >= 2:
                st.subheader("🏆 Framework Performance Summary")
                
                # Calculate averages by framework
                framework_summary = chart_df.groupby('Framework').agg({
                    'Duration': 'mean',
                    'Words per Second': 'mean',
                    'Word Count': 'mean'
                }).round(2)
                
                # Sort by speed (descending)
                framework_summary = framework_summary.sort_values('Words per Second', ascending=False)
                
                # Display as table with rankings
                summary_display = framework_summary.copy()
                summary_display['Speed Rank'] = range(1, len(summary_display) + 1)
                summary_display = summary_display[['Speed Rank', 'Words per Second', 'Duration', 'Word Count']]
                
                st.dataframe(summary_display, use_container_width=True)
                
                # Winner announcement
                fastest_framework = framework_summary.index[0]
                fastest_speed = framework_summary.loc[fastest_framework, 'Words per Second']
                st.success(f"🥇 **Fastest Framework**: {fastest_framework} ({fastest_speed:.1f} words/second)")
        else:
            st.info("No valid performance data available for trending charts")
    else:
        st.info("Run multiple analyses to see performance trends and comparisons")

# Footer
st.markdown("---")
st.markdown("**🤖 Multi-Agent AI Research System** | Built with Streamlit | Powered by Ollama + RTX 4070")
st.markdown("**🎯 Fair Comparison Enabled** | All frameworks use identical agents, prompts, and workflows")

# Instructions for first-time users
if not st.session_state.analysis_history:
    st.info("""
    👋 **Welcome to the Multi-Agent AI Research System!**
    
    **What's New:**
    🎯 **Fair Comparison**: All frameworks now use identical agents, prompts, and workflows
    📊 **Results**: Performance differences reflect framework architecture, not prompt variations
    🔧 **CrewAI Baseline**: All frameworks standardized to match CrewAI's agent definitions
    
    **Quick Start:**
    1. Choose a framework from the sidebar
    2. Select a research topic or enter a custom one
    3. Click "Run Multi-Agent Analysis" to start
    4. Compare results fairly across frameworks
    
    **Pro Tips:**
    - Run the same topic across different frameworks for direct comparison
    - All agents now target 400-500 words, synthesis targets 1500-2000 words
    - Context is passed between agents in all frameworks for consistent quality
    """)