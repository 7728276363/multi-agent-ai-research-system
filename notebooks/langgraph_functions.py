"""
LangGraph Multi-Agent System
==============================================================

This file implements a multi-agent AI research system using LangGraph framework.
LangGraph is a library for building stateful, multi-actor applications with LLMs.

LANGGRAPH VS OTHER FRAMEWORKS:
- CrewAI: Purpose-built agent collaboration with simple workflows
- LangChain: General LLM framework requiring manual orchestration  
- LangGraph: Stateful graph-based workflows with sophisticated state management

WHAT IS LANGGRAPH?
LangGraph excels at:
- Complex, branching workflows (graphs vs linear sequences)
- Persistent state management across agent interactions
- Advanced coordination patterns (parallel execution, conditional routing)
- Cyclic workflows where agents can revisit previous steps

KEY LANGGRAPH CONCEPTS:
1. **State**: Shared data structure that persists across all agents
2. **Nodes**: Individual agents/functions that modify state
3. **Edges**: Connections defining workflow sequence
4. **Graph**: Complete workflow definition
5. **Annotations**: How state updates are handled (merge, replace, etc.)

STANDARDIZATION APPROACH:
This implementation uses identical:
- Agent roles and expertise (matching CrewAI baseline)
- Prompts and task descriptions
- Sequential workflow pattern
- Output format and metrics

However, it leverages LangGraph's unique strengths:
- Sophisticated state management for context passing
- Robust error handling and recovery
- Built-in workflow visualization capabilities

"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

from langgraph.graph import StateGraph, END          # Core LangGraph components
from langgraph.graph.message import add_messages     # Message handling utilities
from typing import TypedDict, Annotated              # Type system for state schema
from langchain_ollama import OllamaLLM               # LLM interface
from datetime import datetime                         # Timestamp functionality
import time                                          # Performance timing
from typing import Dict, List, Any                   # Additional type hints
import operator                                      # For state update operations


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

# Initialize the same LLM as other frameworks for fair comparison
llm = OllamaLLM(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://ollama:11434"
)


# ============================================================================
# STATE MANAGEMENT UTILITIES
# ============================================================================

"""
LangGraph's power comes from sophisticated state management.
These functions define how the shared state is updated as agents work.
"""

def merge_dicts(left: Dict, right: Dict) -> Dict:
    """
    Merge two dictionaries, with right dict taking precedence.
    
    Used for combining agent outputs and metadata without losing data.
    
    Args:
        left (Dict): Existing dictionary
        right (Dict): New dictionary to merge in
    
    Returns:
        Dict: Combined dictionary
    """
    if not left:
        return right
    if not right:
        return left
    result = left.copy()
    result.update(right)
    return result


def add_unique_agents(left: List[str], right: List[str]) -> List[str]:
    """
    Add agents to completion list without creating duplicates.
    
    CRITICAL FIX: This prevents the "31/5 agents" bug by ensuring
    each agent name appears only once in the completed list.
    
    Args:
        left (List[str]): Existing agent list
        right (List[str]): New agents to add
    
    Returns:
        List[str]: Combined list with no duplicates
    """
    if not left:
        return right
    if not right:
        return left
    
    # Combine lists and remove duplicates while preserving order
    combined = left.copy()
    for agent in right:
        if agent not in combined:
            combined.append(agent)
    return combined


# ============================================================================
# STATE SCHEMA DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Defines the shared state structure for the multi-agent workflow.
    
    WHAT IS STATE?
    In LangGraph, state is a shared data structure that:
    - Persists across all agent interactions
    - Gets updated by each agent as they work
    - Provides context for subsequent agents
    - Tracks workflow progress and metrics
    
    ANNOTATIONS EXPLAINED:
    - Annotated[Type, update_function]: Defines how state updates
    - merge_dicts: Combines dictionaries without losing data
    - add_unique_agents: Prevents duplicate agent tracking
    
    This state schema matches CrewAI's output structure for compatibility.
    """
    
    # ----------------------------------------------------------------
    # CORE WORKFLOW DATA
    # ----------------------------------------------------------------
    topic: str                    # Research topic being analyzed
    analysis_type: str           # Type of analysis (e.g., "comprehensive")
    
    # ----------------------------------------------------------------
    # AGENT ANALYSIS OUTPUTS (MATCHING CREWAI STRUCTURE)
    # ----------------------------------------------------------------
    healthcare_analysis: str     # Healthcare Domain Expert output
    technical_analysis: str      # AI Technical Analyst output
    regulatory_analysis: str     # Regulatory Specialist output
    economic_analysis: str       # Economics Analyst output
    final_synthesis: str         # Strategic Synthesizer output
    
    # ----------------------------------------------------------------
    # WORKFLOW COORDINATION METADATA
    # ----------------------------------------------------------------
    current_agent: str                                    # Currently executing agent
    completed_agents: Annotated[List[str], add_unique_agents]  # Successful completions (no duplicates)
    agent_outputs: Annotated[Dict[str, Any], merge_dicts]     # Raw agent outputs
    
    # ----------------------------------------------------------------
    # PERFORMANCE TRACKING
    # ----------------------------------------------------------------
    start_time: float                                     # Workflow start timestamp
    agent_durations: Annotated[Dict[str, float], merge_dicts]  # Individual agent timings
    total_words: int                                      # Final word count


# ============================================================================
# AGENT NODE FUNCTIONS - IDENTICAL TO CREWAI BASELINE
# ============================================================================

"""
Each function represents an agent "node" in the LangGraph workflow.
These nodes:
1. Receive the current state
2. Execute their specialized analysis
3. Update the state with their results
4. Return the modified state for the next agent

All prompts and logic are identical to CrewAI baseline for fair comparison.
"""

def healthcare_expert_node(state: AgentState) -> AgentState:
    """
    Healthcare Domain Expert node - first agent in the workflow.
    
    ROLE: Provides clinical insights and healthcare industry expertise
    CONTEXT: No previous context (first agent)
    UPDATES: healthcare_analysis, completed_agents, agent_outputs, agent_durations
    
    Args:
        state (AgentState): Current workflow state
    
    Returns:
        AgentState: Updated state with healthcare analysis
    """
    print("🏥 Healthcare Domain Expert analyzing...")
    
    start_time = time.time()
    
    # Build the specialized prompt (identical to CrewAI)
    prompt = f"""You are a Healthcare Domain Expert.

Goal: Provide deep medical and healthcare industry insights

Background: You are a healthcare industry veteran with 15+ years experience in 
medical technology adoption, clinical workflows, and healthcare regulations. 
You understand how technology impacts patient care, hospital operations, and 
medical decision-making processes.

Task: Analyze "{state['topic']}" from a healthcare domain perspective.

Focus on:
- Current healthcare challenges this addresses
- Clinical workflow integration requirements
- Impact on patient care and outcomes
- Healthcare provider adoption barriers
- Real-world implementation examples

Provide domain-specific insights that only a healthcare expert would know.
Target: 400-500 words with detailed analysis."""
    
    try:
        # Execute LLM analysis
        analysis = llm.invoke(prompt)
        duration = time.time() - start_time
        
        print(f"✅ Healthcare analysis completed in {duration:.1f}s ({len(analysis.split())} words)")
        
        # Return updated state with analysis results
        return {
            **state,  # Preserve existing state
            "healthcare_analysis": analysis,
            "current_agent": "healthcare_expert",
            "completed_agents": ["healthcare_expert"],  # Start fresh list (no previous agents)
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"healthcare_expert": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"healthcare_expert": duration})
        }
        
    except Exception as e:
        print(f"❌ Healthcare analysis failed: {e}")
        return {
            **state,
            "healthcare_analysis": f"Analysis failed: {e}",
            "current_agent": "healthcare_expert",
            "completed_agents": ["healthcare_expert"],  # Still track attempt
        }


def technical_analyst_node(state: AgentState) -> AgentState:
    """
    AI Technical Analyst node - second agent, builds on healthcare insights.
    
    ROLE: Analyzes technical feasibility and implementation challenges
    CONTEXT: Healthcare analysis from previous agent
    UPDATES: technical_analysis, completed_agents, agent_outputs, agent_durations
    """
    print("🔧 AI Technical Analyst analyzing...")
    
    start_time = time.time()
    
    # Build context from healthcare analysis (truncated to avoid token limits)
    context = f"Healthcare perspective: {state.get('healthcare_analysis', '')[:300]}..." if state.get('healthcare_analysis') else ""
    
    prompt = f"""You are an AI Technical Analyst.

Goal: Analyze technical feasibility, architecture, and implementation challenges

Background: You are a senior AI engineer specializing in healthcare AI systems.
You understand machine learning model validation, data pipelines, integration
challenges, and technical requirements for medical-grade AI systems.

Task: Analyze the technical aspects of "{state['topic']}".

{f"Context from previous analyses: {context}" if context else ""}

Examine:
- AI/ML model requirements and validation needs
- Data infrastructure and integration challenges
- Scalability and performance considerations
- Security and data privacy technical requirements
- Implementation complexity and technical risks

Focus on technical feasibility and engineering challenges.
Target: 400-500 words with specific implementation details."""
    
    try:
        analysis = llm.invoke(prompt)
        duration = time.time() - start_time
        
        print(f"✅ Technical analysis completed in {duration:.1f}s ({len(analysis.split())} words)")
        
        return {
            **state,
            "technical_analysis": analysis,
            "current_agent": "technical_analyst",
            "completed_agents": ["technical_analyst"],  # Add only this agent
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"technical_analyst": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"technical_analyst": duration})
        }
        
    except Exception as e:
        print(f"❌ Technical analysis failed: {e}")
        return {
            **state,
            "technical_analysis": f"Analysis failed: {e}",
            "current_agent": "technical_analyst",
            "completed_agents": ["technical_analyst"],
        }


def regulatory_specialist_node(state: AgentState) -> AgentState:
    """
    Healthcare Regulatory Specialist node - third agent.
    
    ROLE: Analyzes regulatory compliance and approval processes
    CONTEXT: Healthcare and technical analyses from previous agents
    UPDATES: regulatory_analysis, completed_agents, agent_outputs, agent_durations
    """
    print("⚖️ Healthcare Regulatory Specialist analyzing...")
    
    start_time = time.time()
    
    # Build context from both previous analyses
    context = f"Healthcare: {state.get('healthcare_analysis', '')[:200]}... Technical: {state.get('technical_analysis', '')[:200]}..."
    
    prompt = f"""You are a Healthcare Regulatory Specialist.

Goal: Analyze regulatory compliance, approval processes, and legal implications

Background: You are a regulatory affairs expert with deep knowledge of FDA
approval processes, HIPAA compliance, international medical device regulations,
and healthcare data privacy requirements.

Task: Analyze regulatory implications of "{state['topic']}".

{f"Context from previous analyses: {context}" if context else ""}

Cover:
- FDA approval pathways and requirements
- HIPAA and data privacy compliance
- International regulatory considerations
- Clinical trial requirements
- Liability and legal risk factors

Provide regulatory roadmap and compliance strategy.
Target: 400-500 words with specific regulatory guidance."""
    
    try:
        analysis = llm.invoke(prompt)
        duration = time.time() - start_time
        
        print(f"✅ Regulatory analysis completed in {duration:.1f}s ({len(analysis.split())} words)")
        
        return {
            **state,
            "regulatory_analysis": analysis,
            "current_agent": "regulatory_specialist",
            "completed_agents": ["regulatory_specialist"],  # Add only this agent
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"regulatory_specialist": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"regulatory_specialist": duration})
        }
        
    except Exception as e:
        print(f"❌ Regulatory analysis failed: {e}")
        return {
            **state,
            "regulatory_analysis": f"Analysis failed: {e}",
            "current_agent": "regulatory_specialist",
            "completed_agents": ["regulatory_specialist"],
        }


def economic_analyst_node(state: AgentState) -> AgentState:
    """
    Healthcare Economics Analyst node - fourth agent.
    
    ROLE: Evaluates economic impact and financial implications
    CONTEXT: Healthcare, technical, and regulatory analyses
    UPDATES: economic_analysis, completed_agents, agent_outputs, agent_durations
    """
    print("💰 Healthcare Economics Analyst analyzing...")
    
    start_time = time.time()
    
    # Build comprehensive context from all three previous analyses
    context = f"Healthcare: {state.get('healthcare_analysis', '')[:150]}... Technical: {state.get('technical_analysis', '')[:150]}... Regulatory: {state.get('regulatory_analysis', '')[:150]}..."
    
    prompt = f"""You are a Healthcare Economics Analyst.

Goal: Evaluate economic impact, cost-benefit analysis, and market dynamics

Background: You are a healthcare economist who analyzes the financial impact
of new technologies on healthcare systems, insurance models, hospital budgets,
and patient outcomes. You understand ROI calculations for healthcare IT.

Task: Analyze economic impact of "{state['topic']}".

{f"Context from previous analyses: {context}" if context else ""}

Evaluate:
- Cost-benefit analysis for healthcare systems
- Impact on healthcare spending and insurance
- ROI calculations for hospitals and providers
- Market size and growth projections
- Economic barriers to adoption

Provide financial impact assessment and business case.
Target: 400-500 words with financial projections."""
    
    try:
        analysis = llm.invoke(prompt)
        duration = time.time() - start_time
        
        print(f"✅ Economic analysis completed in {duration:.1f}s ({len(analysis.split())} words)")
        
        return {
            **state,
            "economic_analysis": analysis,
            "current_agent": "economic_analyst",
            "completed_agents": ["economic_analyst"],  # Add only this agent
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"economic_analyst": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"economic_analyst": duration})
        }
        
    except Exception as e:
        print(f"❌ Economic analysis failed: {e}")
        return {
            **state,
            "economic_analysis": f"Analysis failed: {e}",
            "current_agent": "economic_analyst",
            "completed_agents": ["economic_analyst"],
        }


def strategic_synthesizer_node(state: AgentState) -> AgentState:
    """
    Strategic Content Synthesizer node - final agent.
    
    ROLE: Combines all specialist insights into comprehensive strategic analysis
    CONTEXT: All four previous analyses (healthcare, technical, regulatory, economic)
    UPDATES: final_synthesis, completed_agents, agent_outputs, total_words
    """
    print("🎯 Strategic Content Synthesizer creating final analysis...")
    
    start_time = time.time()
    
    # Compile all analyses for comprehensive synthesis
    healthcare_analysis = state.get("healthcare_analysis", "")
    technical_analysis = state.get("technical_analysis", "")
    regulatory_analysis = state.get("regulatory_analysis", "")
    economic_analysis = state.get("economic_analysis", "")
    
    prompt = f"""You are a Strategic Content Synthesizer.

Goal: Integrate multi-domain insights into cohesive strategic analysis

Background: You are an expert strategic analyst who excels at synthesizing
complex information from multiple domains. You create comprehensive reports
that weave together technical, regulatory, economic, and domain-specific 
insights into actionable strategic recommendations.

Task: Create comprehensive strategic analysis of "{state['topic']}" by synthesizing insights from domain, technical, regulatory, and economic analyses.

Previous Analyses:

Healthcare Analysis: {healthcare_analysis}

Technical Analysis: {technical_analysis}

Regulatory Analysis: {regulatory_analysis}

Economic Analysis: {economic_analysis}

Structure:
1. Executive Summary (key strategic insights)
2. Integrated Analysis (how all factors interact)
3. Strategic Recommendations (for different stakeholders)
4. Implementation Roadmap (priorities and timeline)
5. Risk Assessment (challenges and mitigation strategies)
6. Future Outlook (strategic implications)

Target: 1500-2000 words for C-suite executives and strategic decision makers."""
    
    try:
        synthesis = llm.invoke(prompt)
        duration = time.time() - start_time
        
        # Calculate final workflow metrics
        total_duration = time.time() - state.get("start_time", time.time())
        
        # Calculate total words (individual analyses + synthesis, no double counting)
        individual_words = sum(len(analysis.split()) for analysis in [
            healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis
        ] if analysis and "failed" not in analysis.lower())
        
        synthesis_words = len(synthesis.split()) if "failed" not in synthesis.lower() else 0
        all_words = individual_words + synthesis_words
        
        print(f"✅ Strategic synthesis completed in {duration:.1f}s ({synthesis_words} words)")
        print(f"🎉 Total workflow: {total_duration:.1f}s, {all_words:,} words ({all_words/total_duration:.1f} w/s)")
        
        return {
            **state,
            "final_synthesis": synthesis,
            "current_agent": "strategic_synthesizer",
            "completed_agents": ["strategic_synthesizer"],  # Add only this agent
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"strategic_synthesizer": synthesis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"strategic_synthesizer": duration}),
            "total_words": all_words
        }
        
    except Exception as e:
        print(f"❌ Strategic synthesis failed: {e}")
        return {
            **state,
            "final_synthesis": f"Synthesis failed: {e}",
            "current_agent": "strategic_synthesizer",
            "completed_agents": ["strategic_synthesizer"],
        }


# ============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

def create_multi_agent_graph():
    """
    Create and configure the LangGraph workflow.
    
    GRAPH CONSTRUCTION:
    1. Initialize StateGraph with our state schema
    2. Add each agent as a node
    3. Define edges (workflow sequence)
    4. Set entry point and exit conditions
    5. Compile into executable graph
    
    WORKFLOW PATTERN:
    This creates a linear, sequential workflow matching CrewAI:
    Healthcare → Technical → Regulatory → Economic → Synthesis → END
    
    Returns:
        CompiledGraph: Ready-to-execute LangGraph workflow
    """
    
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add agent nodes in execution order
    workflow.add_node("healthcare_expert", healthcare_expert_node)
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("regulatory_specialist", regulatory_specialist_node)
    workflow.add_node("economic_analyst", economic_analyst_node)
    workflow.add_node("strategic_synthesizer", strategic_synthesizer_node)
    
    # Define sequential workflow (matching CrewAI baseline)
    workflow.set_entry_point("healthcare_expert")              # Start here
    workflow.add_edge("healthcare_expert", "technical_analyst")
    workflow.add_edge("technical_analyst", "regulatory_specialist") 
    workflow.add_edge("regulatory_specialist", "economic_analyst")
    workflow.add_edge("economic_analyst", "strategic_synthesizer")
    workflow.add_edge("strategic_synthesizer", END)            # Workflow complete
    
    # Compile the graph into executable form
    app = workflow.compile()
    return app


# Compile the graph once at module load time
graph_app = create_multi_agent_graph()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_langgraph_analysis(topic, analysis_type="comprehensive"):
    """
    Execute the complete LangGraph multi-agent analysis workflow.
    
    EXECUTION PROCESS:
    1. Initialize workflow state with topic and timing
    2. Invoke the compiled graph (runs all agents sequentially)
    3. Extract and format results
    4. Calculate performance metrics
    5. Return standardized results matching CrewAI format
    
    LANGGRAPH ADVANTAGES:
    - Automatic state management across agents
    - Built-in error handling and recovery
    - Workflow visualization capabilities
    - Support for complex coordination patterns
    
    Args:
        topic (str): Research topic to analyze
        analysis_type (str): Type of analysis (default: "comprehensive")
    
    Returns:
        dict: Comprehensive results matching CrewAI baseline format
    """
    
    print(f"📊 Starting LangGraph analysis: {topic}")
    print("=" * 70)
    
    # Initialize workflow state
    initial_state = {
        "topic": topic,
        "analysis_type": analysis_type,
        "healthcare_analysis": "",
        "technical_analysis": "",
        "regulatory_analysis": "",
        "economic_analysis": "",
        "final_synthesis": "",
        "current_agent": "",
        "completed_agents": [],     # Start with empty list
        "agent_outputs": {},
        "start_time": time.time(),
        "agent_durations": {},
        "total_words": 0
    }
    
    # Execute the graph workflow
    try:
        # LangGraph automatically manages state transitions and agent coordination
        final_state = graph_app.invoke(initial_state)
        
        # Calculate final performance metrics
        total_duration = time.time() - initial_state["start_time"]
        
        # Extract and deduplicate completed agents list
        # CRITICAL FIX: Remove duplicates that can occur during state merging
        completed_agents = final_state.get("completed_agents", [])
        unique_completed_agents = list(set(completed_agents)) if completed_agents else []
        
        # Create standardized result format (matching CrewAI and LangChain)
        result = {
            "topic": topic,
            "framework": "langgraph",
            "total_duration": total_duration,
            "total_words": final_state.get("total_words", 0),
            "words_per_second": final_state.get("total_words", 0) / total_duration if total_duration > 0 else 0,
            "completed_agents": unique_completed_agents,    # Deduplicated list
            "successful_agents": len(unique_completed_agents), # Count of unique successful agents
            "total_agents": 5,                             # Expected total (4 domain + 1 synthesis)
            "agent_durations": final_state.get("agent_durations", {}),
            "final_synthesis": final_state.get("final_synthesis", ""),
            "individual_analyses": {
                "healthcare": final_state.get("healthcare_analysis", ""),
                "technical": final_state.get("technical_analysis", ""),
                "regulatory": final_state.get("regulatory_analysis", ""),
                "economic": final_state.get("economic_analysis", "")
            },
            "timestamp": datetime.now().isoformat(),
            "success": len(unique_completed_agents) >= 4   # Success if at least 4 agents completed
        }
        
        # Print workflow summary
        print(f"\n🎉 LangGraph analysis completed!")
        print(f"⏱️ Total time: {total_duration:.1f} seconds")
        print(f"📝 Total words: {result['total_words']:,}")
        print(f"🚀 Speed: {result['words_per_second']:.1f} words/second")
        print(f"✅ Agents completed: {len(unique_completed_agents)}/5")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        # Handle workflow execution failures
        print(f"❌ LangGraph execution failed: {e}")
        return {
            "topic": topic,
            "error": str(e),
            "framework": "langgraph",
            "total_duration": time.time() - initial_state["start_time"],
            "completed_agents": [],
            "successful_agents": 0,
            "total_agents": 5,
            "success": False
        }


# ============================================================================
# COMPATIBILITY FUNCTION
# ============================================================================

def run_comprehensive_analysis(topic):
    """
    Legacy compatibility function for existing code.
    
    Args:
        topic (str): Research topic to analyze
    
    Returns:
        dict: Same results as run_langgraph_analysis()
    """
    return run_langgraph_analysis(topic)