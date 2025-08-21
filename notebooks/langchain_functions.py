"""
LangChain Multi-Agent System
==============================================================

This file implements a multi-agent AI research system using LangChain framework.
LangChain is a framework for developing applications with large language models (LLMs).

LANGCHAIN VS CREWAI DIFFERENCES:
- CrewAI: Purpose-built for agent collaboration with built-in workflows
- LangChain: General-purpose LLM framework, requires custom agent implementation
- This implementation: Custom agents that match CrewAI's behavior exactly

FRAMEWORK STANDARDIZATION:
To ensure fair comparison across frameworks, this LangChain implementation uses:
- Identical agent roles and expertise as CrewAI baseline
- Identical prompts and task descriptions
- Identical sequential workflow with context passing
- Identical output structure and metrics

WHY STANDARDIZE?
Without standardization, performance differences could be due to:
- Different prompts or agent roles
- Different workflows or context passing
- Different output formats
By standardizing, we ensure performance differences reflect framework architecture only.

LANGCHAIN MULTI-AGENT APPROACH:
1. Custom Agent class that mimics CrewAI Agent behavior
2. Manual workflow orchestration (vs CrewAI's built-in Process)
3. Manual context passing between agents
4. Compatible result format for fair comparison

"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

from langchain_ollama import OllamaLLM  # LangChain's Ollama integration
from datetime import datetime           # For timestamps
import time                            # For timing analysis duration
from typing import Dict, List, Any     # Type hints for better code documentation


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

print("🔧 Initializing LangChain LLM for workflows...")

# LangChain uses a different LLM interface than CrewAI
# This connects to the same Ollama instance but through LangChain's interface
llm = OllamaLLM(
    model="llama3.1:8b-instruct-q4_K_M",  # Same model as CrewAI for fair comparison
    base_url="http://ollama:11434"         # Same Ollama endpoint
)


# ============================================================================
# CUSTOM AGENT CLASS - MIMICKING CREWAI BEHAVIOR
# ============================================================================

class Agent:
    """
    Custom Agent class that replicates CrewAI Agent functionality.
    
    WHAT THIS CLASS DOES:
    Since LangChain doesn't have built-in collaborative agents like CrewAI,
    we create our own Agent class that:
    - Stores agent role, goal, and expertise (like CrewAI)
    - Executes tasks using the same prompt structure
    - Tracks performance and memory
    - Returns compatible results
    
    WHY CUSTOM CLASS?
    LangChain is a general framework - it provides LLM interfaces but doesn't
    have CrewAI's specialized agent collaboration features. This custom class
    bridges that gap for fair comparison.
    """
    
    def __init__(self, role: str, goal: str, backstory: str, llm):
        """
        Initialize an agent with role, expertise, and LLM connection.
        
        Args:
            role (str): Agent's job title/specialty
            goal (str): What the agent is trying to achieve
            backstory (str): Agent's experience and knowledge base
            llm: Language model instance to power the agent
        """
        self.role = role           # e.g., "Healthcare Domain Expert"
        self.goal = goal           # e.g., "Provide deep medical insights"
        self.backstory = backstory # Detailed expertise description
        self.llm = llm            # LLM connection
        self.memory = []          # Store analysis history (for future enhancements)
        
    def analyze(self, task_description: str) -> Dict[str, Any]:
        """
        Execute an analysis task using this agent's expertise.
        
        HOW IT WORKS:
        1. Builds a structured prompt combining agent context + task
        2. Sends prompt to LLM for processing
        3. Times the execution and tracks word count
        4. Returns structured results compatible with CrewAI format
        
        Args:
            task_description (str): Detailed task instructions
        
        Returns:
            Dict containing:
                - agent: Agent role name
                - analysis: Generated text analysis
                - duration: Time taken in seconds
                - word_count: Number of words generated
                - success: Whether analysis completed successfully
        """
        
        # Build the prompt using the same structure as CrewAI
        # This ensures identical behavior across frameworks
        full_prompt = f"""You are a {self.role}.

Goal: {self.goal}

Background: {self.backstory}

Task: {task_description}

Provide your expert analysis focusing on your area of expertise. Be detailed, specific, and practical."""

        start_time = time.time()
        
        try:
            # Execute the LLM call
            response = self.llm.invoke(full_prompt)
            duration = time.time() - start_time
            
            # Create structured result
            analysis_result = {
                "agent": self.role,
                "analysis": response,
                "duration": duration,
                "word_count": len(response.split()),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # Store in memory for potential future use
            self.memory.append(analysis_result)
            
            print(f"✅ {self.role} completed analysis in {duration:.1f}s ({len(response.split())} words)")
            return analysis_result
            
        except Exception as e:
            # Handle LLM failures gracefully
            print(f"❌ {self.role} analysis failed: {e}")
            return {
                "agent": self.role,
                "analysis": f"Analysis failed: {e}",
                "duration": time.time() - start_time,
                "word_count": 0,
                "error": str(e),
                "success": False
            }


# ============================================================================
# AGENT INSTANCES - IDENTICAL TO CREWAI BASELINE
# ============================================================================

"""
These agent instances use exactly the same roles, goals, and backstories
as the CrewAI implementation to ensure fair comparison.
"""

# Agent 1: Healthcare Domain Expert
healthcare_expert = Agent(
    role="Healthcare Domain Expert",
    goal="Provide deep medical and healthcare industry insights",
    backstory="""You are a healthcare industry veteran with 15+ years experience in 
    medical technology adoption, clinical workflows, and healthcare regulations. 
    You understand how technology impacts patient care, hospital operations, and 
    medical decision-making processes.""",
    llm=llm
)

# Agent 2: AI Technical Analyst
technical_analyst = Agent(
    role="AI Technical Analyst",
    goal="Analyze technical feasibility, architecture, and implementation challenges",
    backstory="""You are a senior AI engineer specializing in healthcare AI systems.
    You understand machine learning model validation, data pipelines, integration
    challenges, and technical requirements for medical-grade AI systems.""",
    llm=llm
)

# Agent 3: Healthcare Regulatory Specialist
regulatory_specialist = Agent(
    role="Healthcare Regulatory Specialist",
    goal="Analyze regulatory compliance, approval processes, and legal implications",
    backstory="""You are a regulatory affairs expert with deep knowledge of FDA
    approval processes, HIPAA compliance, international medical device regulations,
    and healthcare data privacy requirements.""",
    llm=llm
)

# Agent 4: Healthcare Economics Analyst
economic_analyst = Agent(
    role="Healthcare Economics Analyst",
    goal="Evaluate economic impact, cost-benefit analysis, and market dynamics",
    backstory="""You are a healthcare economist who analyzes the financial impact
    of new technologies on healthcare systems, insurance models, hospital budgets,
    and patient outcomes. You understand ROI calculations for healthcare IT.""",
    llm=llm
)

# Agent 5: Strategic Content Synthesizer
strategic_synthesizer = Agent(
    role="Strategic Content Synthesizer",
    goal="Integrate multi-domain insights into cohesive strategic analysis",
    backstory="""You are an expert strategic analyst who excels at synthesizing
    complex information from multiple domains. You create comprehensive reports
    that weave together technical, regulatory, economic, and domain-specific 
    insights into actionable strategic recommendations.""",
    llm=llm
)


# ============================================================================
# TASK TEMPLATE FUNCTIONS - IDENTICAL TO CREWAI BASELINE
# ============================================================================

"""
These functions create task descriptions that are identical to CrewAI's
to ensure agents receive the same instructions across frameworks.
"""

def create_healthcare_task(topic, context=""):
    """Create task description for Healthcare Domain Expert."""
    return f"""Analyze "{topic}" from a healthcare domain perspective.
    
    {f"Context from previous analyses: {context}" if context else ""}
    
    Focus on:
    - Current healthcare challenges this addresses
    - Clinical workflow integration requirements
    - Impact on patient care and outcomes
    - Healthcare provider adoption barriers
    - Real-world implementation examples
    
    Provide domain-specific insights that only a healthcare expert would know.
    Target: 400-500 words with detailed analysis."""


def create_technical_task(topic, context=""):
    """Create task description for AI Technical Analyst."""
    return f"""Analyze the technical aspects of "{topic}".
    
    {f"Context from previous analyses: {context}" if context else ""}
    
    Examine:
    - AI/ML model requirements and validation needs
    - Data infrastructure and integration challenges
    - Scalability and performance considerations
    - Security and data privacy technical requirements
    - Implementation complexity and technical risks
    
    Focus on technical feasibility and engineering challenges.
    Target: 400-500 words with specific implementation details."""


def create_regulatory_task(topic, context=""):
    """Create task description for Healthcare Regulatory Specialist."""
    return f"""Analyze regulatory implications of "{topic}".
    
    {f"Context from previous analyses: {context}" if context else ""}
    
    Cover:
    - FDA approval pathways and requirements
    - HIPAA and data privacy compliance
    - International regulatory considerations
    - Clinical trial requirements
    - Liability and legal risk factors
    
    Provide regulatory roadmap and compliance strategy.
    Target: 400-500 words with specific regulatory guidance."""


def create_economic_task(topic, context=""):
    """Create task description for Healthcare Economics Analyst."""
    return f"""Analyze economic impact of "{topic}".
    
    {f"Context from previous analyses: {context}" if context else ""}
    
    Evaluate:
    - Cost-benefit analysis for healthcare systems
    - Impact on healthcare spending and insurance
    - ROI calculations for hospitals and providers
    - Market size and growth projections
    - Economic barriers to adoption
    
    Provide financial impact assessment and business case.
    Target: 400-500 words with financial projections."""


def create_synthesis_task(topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis):
    """Create task description for Strategic Content Synthesizer."""
    return f"""Create comprehensive strategic analysis of "{topic}" by synthesizing insights from domain, technical, regulatory, and economic analyses.
    
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


# ============================================================================
# MAIN WORKFLOW EXECUTION FUNCTION
# ============================================================================

def run_langchain_analysis(topic):
    """
    Execute the complete LangChain multi-agent analysis workflow.
    
    WORKFLOW DESIGN:
    This function manually orchestrates the same 5-phase workflow as CrewAI:
    1. Healthcare Domain Expert → provides clinical insights
    2. Technical Analyst → builds on healthcare context
    3. Regulatory Specialist → considers both previous analyses  
    4. Economic Analyst → incorporates all three perspectives
    5. Strategic Synthesizer → creates final integrated report
    
    MANUAL ORCHESTRATION:
    Unlike CrewAI which has built-in workflow management, LangChain requires
    manual coordination of:
    - Agent execution order
    - Context passing between agents
    - Error handling and recovery
    - Result aggregation and formatting
    
    CONTEXT PASSING STRATEGY:
    Each agent receives truncated context from previous agents to:
    - Provide relevant background information
    - Avoid token limit issues with very long contexts
    - Maintain workflow continuity
    
    Args:
        topic (str): Research topic to analyze
    
    Returns:
        dict: Comprehensive results matching CrewAI format
    """
    
    print(f"🔗 Starting LangChain analysis: {topic}")
    print("=" * 70)
    
    # Initialize tracking variables
    start_time = time.time()
    individual_outputs = {}   # Store each agent's analysis
    results = []             # Store detailed results from each agent
    completed_agents = []    # Track successful completions
    
    try:
        # ----------------------------------------------------------------
        # PHASE 1: HEALTHCARE DOMAIN ANALYSIS
        # ----------------------------------------------------------------
        print("🏥 Phase 1/5: Healthcare Domain Expert")
        
        # Create and execute healthcare analysis task
        healthcare_task = create_healthcare_task(topic)
        healthcare_result = healthcare_expert.analyze(healthcare_task)
        results.append(healthcare_result)
        
        # Process results and track completion
        if healthcare_result["success"]:
            healthcare_analysis = healthcare_result["analysis"]
            individual_outputs['healthcare'] = healthcare_analysis
            completed_agents.append('healthcare_expert')
        else:
            # Handle failure but continue workflow
            healthcare_analysis = ""
            individual_outputs['healthcare'] = f"Failed: {healthcare_result['analysis']}"
        
        # ----------------------------------------------------------------
        # PHASE 2: TECHNICAL ANALYSIS (WITH HEALTHCARE CONTEXT)
        # ----------------------------------------------------------------
        print("🔧 Phase 2/5: AI Technical Analyst")
        
        # Build context from healthcare analysis (truncated to avoid token limits)
        context = f"Healthcare perspective: {healthcare_analysis[:300]}..." if healthcare_analysis else ""
        
        technical_task = create_technical_task(topic, context)
        technical_result = technical_analyst.analyze(technical_task)
        results.append(technical_result)
        
        if technical_result["success"]:
            technical_analysis = technical_result["analysis"]
            individual_outputs['technical'] = technical_analysis
            completed_agents.append('technical_analyst')
        else:
            technical_analysis = ""
            individual_outputs['technical'] = f"Failed: {technical_result['analysis']}"
        
        # ----------------------------------------------------------------
        # PHASE 3: REGULATORY ANALYSIS (WITH ACCUMULATED CONTEXT)
        # ----------------------------------------------------------------
        print("⚖️ Phase 3/5: Healthcare Regulatory Specialist")
        
        # Build context from both previous analyses
        context = f"Healthcare: {healthcare_analysis[:200]}... Technical: {technical_analysis[:200]}..."
        
        regulatory_task = create_regulatory_task(topic, context)
        regulatory_result = regulatory_specialist.analyze(regulatory_task)
        results.append(regulatory_result)
        
        if regulatory_result["success"]:
            regulatory_analysis = regulatory_result["analysis"]
            individual_outputs['regulatory'] = regulatory_analysis
            completed_agents.append('regulatory_specialist')
        else:
            regulatory_analysis = ""
            individual_outputs['regulatory'] = f"Failed: {regulatory_result['analysis']}"
        
        # ----------------------------------------------------------------
        # PHASE 4: ECONOMIC ANALYSIS (WITH ALL CONTEXT)
        # ----------------------------------------------------------------
        print("💰 Phase 4/5: Healthcare Economics Analyst")
        
        # Build comprehensive context from all three previous analyses
        context = f"Healthcare: {healthcare_analysis[:150]}... Technical: {technical_analysis[:150]}... Regulatory: {regulatory_analysis[:150]}..."
        
        economic_task = create_economic_task(topic, context)
        economic_result = economic_analyst.analyze(economic_task)
        results.append(economic_result)
        
        if economic_result["success"]:
            economic_analysis = economic_result["analysis"]
            individual_outputs['economic'] = economic_analysis
            completed_agents.append('economic_analyst')
        else:
            economic_analysis = ""
            individual_outputs['economic'] = f"Failed: {economic_result['analysis']}"
        
        # ----------------------------------------------------------------
        # PHASE 5: STRATEGIC SYNTHESIS (COMBINING ALL INSIGHTS)
        # ----------------------------------------------------------------
        print("🎯 Phase 5/5: Strategic Content Synthesizer")
        
        # Create synthesis task with ALL previous analyses
        synthesis_task = create_synthesis_task(
            topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis
        )
        
        synthesis_result = strategic_synthesizer.analyze(synthesis_task)
        results.append(synthesis_result)
        
        if synthesis_result["success"]:
            final_synthesis = synthesis_result["analysis"]
            completed_agents.append('strategic_synthesizer')  # Track synthesis completion
        else:
            final_synthesis = f"Synthesis failed: {synthesis_result['analysis']}"
        
        # ----------------------------------------------------------------
        # CALCULATE METRICS AND FORMAT RESULTS
        # ----------------------------------------------------------------
        
        total_duration = time.time() - start_time
        
        # Calculate word counts (avoid double counting synthesis)
        # Individual analyses: sum of first 4 successful results
        individual_words = sum(r.get('word_count', 0) for r in results[:-1] if r.get('success', False))
        
        # Synthesis words: count separately
        synthesis_words = len(final_synthesis.split()) if 'failed' not in final_synthesis.lower() else 0
        total_words = individual_words + synthesis_words
        
        # Create standardized result format (matching CrewAI)
        workflow_result = {
            "topic": topic,
            "framework": "langchain",
            "total_duration": total_duration,
            "total_words": total_words,
            "words_per_second": total_words / total_duration if total_duration > 0 else 0,
            "completed_agents": completed_agents,           # List of successful agents
            "successful_agents": len(completed_agents),    # Count for compatibility
            "total_agents": 5,                            # Expected total
            "individual_analyses": individual_outputs,
            "final_synthesis": final_synthesis,
            "timestamp": datetime.now().isoformat(),
            "success": len(completed_agents) == 5         # True if all 5 agents completed
        }
        
        # Print summary
        print(f"\n⏱️ LangChain analysis completed in {total_duration:.1f} seconds")
        print(f"📝 Generated {total_words:,} words ({total_words/total_duration:.1f} words/second)")
        print(f"✅ {len(completed_agents)}/5 agents completed successfully")
        print("=" * 70)
        
        return workflow_result
        
    except Exception as e:
        # Handle any unexpected errors during workflow execution
        duration = time.time() - start_time
        print(f"❌ LangChain analysis failed: {e}")
        
        return {
            "topic": topic,
            "framework": "langchain",
            "total_duration": duration,
            "total_words": 0,
            "words_per_second": 0,
            "final_synthesis": f"Analysis failed: {e}",
            "individual_analyses": {},
            "completed_agents": completed_agents,     # Include partial completion
            "successful_agents": len(completed_agents), # Count partial completions
            "total_agents": 5,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
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
        dict: Same results as run_langchain_analysis()
    """
    return run_langchain_analysis(topic)