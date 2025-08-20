# LangGraph Multi-Agent System - MATCHED TO CREWAI BASELINE
# Save as: langgraph_functions.py

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_ollama import OllamaLLM
from datetime import datetime
import time
from typing import Dict, List, Any
import operator

# Initialize LLM
llm = OllamaLLM(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://ollama:11434"
)

# STATE SCHEMA
# =========================

def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Helper function to merge dictionaries"""
    if not left:
        return right
    if not right:
        return left
    result = left.copy()
    result.update(right)
    return result

class AgentState(TypedDict):
    """Standardized state schema matching CrewAI baseline"""
    
    # Core information
    topic: str
    analysis_type: str
    
    # Agent outputs (matching CrewAI structure)
    healthcare_analysis: str
    technical_analysis: str
    regulatory_analysis: str
    economic_analysis: str
    final_synthesis: str
    
    # Workflow metadata
    current_agent: str
    completed_agents: Annotated[List[str], operator.add]
    agent_outputs: Annotated[Dict[str, Any], merge_dicts]
    
    # Performance tracking
    start_time: float
    agent_durations: Annotated[Dict[str, float], merge_dicts]
    total_words: int

# AGENT NODE FUNCTIONS (MATCHING CREWAI PROMPTS)
# ============================================================

def healthcare_expert_node(state: AgentState) -> AgentState:
    """Healthcare Domain Expert - matching CrewAI baseline exactly"""
    print("🏥 Healthcare Domain Expert analyzing...")
    
    start_time = time.time()
    
    # Build context from previous analyses
    context = ""
    # No previous context for first agent
    
    prompt = f"""You are a Healthcare Domain Expert.

Goal: Provide deep medical and healthcare industry insights

Background: You are a healthcare industry veteran with 15+ years experience in 
medical technology adoption, clinical workflows, and healthcare regulations. 
You understand how technology impacts patient care, hospital operations, and 
medical decision-making processes.

Task: Analyze "{state['topic']}" from a healthcare domain perspective.

{f"Context from previous analyses: {context}" if context else ""}

Focus on:
- Current healthcare challenges this addresses
- Clinical workflow integration requirements
- Impact on patient care and outcomes
- Healthcare provider adoption barriers
- Real-world implementation examples

Provide domain-specific insights that only a healthcare expert would know.
Target: 400-500 words with detailed analysis."""
    
    try:
        analysis = llm.invoke(prompt)
        duration = time.time() - start_time
        
        print(f"✅ Healthcare analysis completed in {duration:.1f}s ({len(analysis.split())} words)")
        
        return {
            **state,
            "healthcare_analysis": analysis,
            "current_agent": "healthcare_expert",
            "completed_agents": (state.get("completed_agents", []) + ["healthcare_expert"]),
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"healthcare_expert": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"healthcare_expert": duration})
        }
        
    except Exception as e:
        print(f"❌ Healthcare analysis failed: {e}")
        return {
            **state,
            "healthcare_analysis": f"Analysis failed: {e}",
            "current_agent": "healthcare_expert",
            "completed_agents": (state.get("completed_agents", []) + ["healthcare_expert"]),
        }

def technical_analyst_node(state: AgentState) -> AgentState:
    """AI Technical Analyst - matching CrewAI baseline exactly"""
    print("🔧 AI Technical Analyst analyzing...")
    
    start_time = time.time()
    
    # Include context from healthcare analysis
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
            "completed_agents": (state.get("completed_agents", []) + ["technical_analyst"]),
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"technical_analyst": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"technical_analyst": duration})
        }
        
    except Exception as e:
        print(f"❌ Technical analysis failed: {e}")
        return {
            **state,
            "technical_analysis": f"Analysis failed: {e}",
            "current_agent": "technical_analyst",
            "completed_agents": (state.get("completed_agents", []) + ["technical_analyst"]),
        }

def regulatory_specialist_node(state: AgentState) -> AgentState:
    """Healthcare Regulatory Specialist - matching CrewAI baseline exactly"""
    print("⚖️ Healthcare Regulatory Specialist analyzing...")
    
    start_time = time.time()
    
    # Build context from previous analyses
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
            "completed_agents": (state.get("completed_agents", []) + ["regulatory_specialist"]),
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"regulatory_specialist": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"regulatory_specialist": duration})
        }
        
    except Exception as e:
        print(f"❌ Regulatory analysis failed: {e}")
        return {
            **state,
            "regulatory_analysis": f"Analysis failed: {e}",
            "current_agent": "regulatory_specialist",
            "completed_agents": (state.get("completed_agents", []) + ["regulatory_specialist"]),
        }

def economic_analyst_node(state: AgentState) -> AgentState:
    """Healthcare Economics Analyst - matching CrewAI baseline exactly"""
    print("💰 Healthcare Economics Analyst analyzing...")
    
    start_time = time.time()
    
    # Build comprehensive context
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
            "completed_agents": (state.get("completed_agents", []) + ["economic_analyst"]),
            "agent_outputs": merge_dicts(state.get("agent_outputs", {}), {"economic_analyst": analysis}),
            "agent_durations": merge_dicts(state.get("agent_durations", {}), {"economic_analyst": duration})
        }
        
    except Exception as e:
        print(f"❌ Economic analysis failed: {e}")
        return {
            **state,
            "economic_analysis": f"Analysis failed: {e}",
            "current_agent": "economic_analyst",
            "completed_agents": (state.get("completed_agents", []) + ["economic_analyst"]),
        }

def strategic_synthesizer_node(state: AgentState) -> AgentState:
    """Strategic Content Synthesizer - matching CrewAI baseline exactly"""
    print("🎯 Strategic Content Synthesizer creating final analysis...")
    
    start_time = time.time()
    
    # Compile all analyses exactly like CrewAI
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
        
        # Calculate final metrics
        total_duration = time.time() - state.get("start_time", time.time())
        # Calculate words from individual analyses + synthesis (don't double count)
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
            "completed_agents": (state.get("completed_agents", []) + ["strategic_synthesizer"]),
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
            "completed_agents": (state.get("completed_agents", []) + ["strategic_synthesizer"]),
        }

# WORKFLOW CREATION
# ===============================

def create_multi_agent_graph():
    """Create LangGraph workflow matching CrewAI baseline"""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add agent nodes in standardized order
    workflow.add_node("healthcare_expert", healthcare_expert_node)
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("regulatory_specialist", regulatory_specialist_node)
    workflow.add_node("economic_analyst", economic_analyst_node)
    workflow.add_node("strategic_synthesizer", strategic_synthesizer_node)
    
    # Define sequential workflow matching CrewAI
    workflow.set_entry_point("healthcare_expert")
    workflow.add_edge("healthcare_expert", "technical_analyst")
    workflow.add_edge("technical_analyst", "regulatory_specialist")
    workflow.add_edge("regulatory_specialist", "economic_analyst")
    workflow.add_edge("economic_analyst", "strategic_synthesizer")
    workflow.add_edge("strategic_synthesizer", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

# Compile the graph
graph_app = create_multi_agent_graph()

# EXECUTION FUNCTION
# ================================

def run_langgraph_analysis(topic, analysis_type="comprehensive"):
    """Run LangGraph analysis matching CrewAI baseline exactly"""
    
    print(f"📊 Starting LangGraph analysis: {topic}")
    print("=" * 70)
    
    # Initialize state
    initial_state = {
        "topic": topic,
        "analysis_type": analysis_type,
        "healthcare_analysis": "",
        "technical_analysis": "",
        "regulatory_analysis": "",
        "economic_analysis": "",
        "final_synthesis": "",
        "current_agent": "",
        "completed_agents": [],
        "agent_outputs": {},
        "start_time": time.time(),
        "agent_durations": {},
        "total_words": 0
    }
    
    # Execute the graph
    try:
        final_state = graph_app.invoke(initial_state)
        
        # Calculate final metrics
        total_duration = time.time() - initial_state["start_time"]
        
        result = {
            "topic": topic,
            "framework": "langgraph",
            "total_duration": total_duration,
            "total_words": final_state.get("total_words", 0),
            "words_per_second": final_state.get("total_words", 0) / total_duration if total_duration > 0 else 0,
            "completed_agents": final_state.get("completed_agents", []),
            "agent_durations": final_state.get("agent_durations", {}),
            "final_synthesis": final_state.get("final_synthesis", ""),
            "individual_analyses": {
                "healthcare": final_state.get("healthcare_analysis", ""),
                "technical": final_state.get("technical_analysis", ""),
                "regulatory": final_state.get("regulatory_analysis", ""),
                "economic": final_state.get("economic_analysis", "")
            },
            "timestamp": datetime.now().isoformat(),
            "success": len(final_state.get("completed_agents", [])) >= 4
        }
        
        print(f"\n🎉 LangGraph analysis completed!")
        print(f"⏱️ Total time: {total_duration:.1f} seconds")
        print(f"📝 Total words: {result['total_words']:,}")
        print(f"🚀 Speed: {result['words_per_second']:.1f} words/second")
        print(f"✅ Agents completed: {len(result['completed_agents'])}/5")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print(f"❌ LangGraph execution failed: {e}")
        return {
            "topic": topic,
            "error": str(e),
            "framework": "langgraph",
            "total_duration": time.time() - initial_state["start_time"],
            "success": False
        }

# REMOVED THE DUPLICATE FUNCTION DEFINITION THAT WAS CAUSING RECURSION
# The duplicate function at the end has been removed