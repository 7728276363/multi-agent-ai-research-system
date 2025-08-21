"""
CrewAI Multi-Agent System
===================================================

This file implements a multi-agent AI research system using CrewAI framework.
CrewAI is designed for creating collaborative AI agent teams that work together
on complex tasks through structured workflows.

WHAT IS A MULTI-AGENT SYSTEM?
A multi-agent system uses multiple specialized AI agents that:
- Each have specific roles and expertise (like a team of specialists)
- Work sequentially or collaboratively on complex problems
- Pass context and insights between each other
- Combine their outputs into comprehensive analysis

WHY USE MULTIPLE AGENTS INSTEAD OF ONE?
- Specialization: Each agent focuses on their domain expertise
- Quality: Domain-specific prompts produce better results
- Structure: Clear workflow and responsibilities
- Scalability: Easy to add/modify specialist agents

CREWAI FRAMEWORK OVERVIEW:
CrewAI organizes work through:
- Agents: AI specialists with defined roles and expertise
- Tasks: Specific assignments given to agents
- Crews: Teams of agents working on tasks
- Process: How agents collaborate (sequential, hierarchical, etc.)

This implementation creates 5 specialist agents for healthcare AI analysis:
1. Healthcare Domain Expert - Clinical and industry insights
2. AI Technical Analyst - Technical feasibility and implementation
3. Healthcare Regulatory Specialist - Compliance and legal requirements
4. Healthcare Economics Analyst - Financial impact and ROI analysis
5. Strategic Content Synthesizer - Combines all insights into strategy

"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

from crewai import Agent, Task, Crew, Process  # CrewAI framework components
from crewai.llm import LLM                     # Large Language Model interface
import time                                    # For timing analysis duration
from datetime import datetime                  # For timestamps


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

print("🔧 Initializing LLM for CrewAI workflows...")

# CrewAI uses an LLM (Large Language Model) to power all agents
# This connects to a local Ollama instance running Llama 3.1
llm = LLM(
    model="ollama/llama3.1:8b-instruct-q4_K_M",  # Specific model version
    base_url="http://ollama:11434"                # Ollama API endpoint
)


# ============================================================================
# AGENT DEFINITIONS - THE SPECIALIST TEAM
# ============================================================================

"""
WHAT ARE AGENTS?
Agents are AI specialists with:
- role: Their job title/specialty
- goal: What they're trying to achieve
- backstory: Their experience and knowledge base
- llm: The language model that powers them

Each agent acts like hiring a specialist consultant with years of experience
in their field. The backstory gives them context and expertise.
"""

# Agent 1: Healthcare Domain Expert
# Specializes in medical technology, clinical workflows, and healthcare systems
healthcare_expert = Agent(
    role="Healthcare Domain Expert",
    goal="Provide deep medical and healthcare industry insights",
    backstory="""You are a healthcare industry veteran with 15+ years experience in 
    medical technology adoption, clinical workflows, and healthcare regulations. 
    You understand how technology impacts patient care, hospital operations, and 
    medical decision-making processes.""",
    verbose=True,  # Shows detailed output during execution
    llm=llm        # Uses the LLM we configured above
)

# Agent 2: AI Technical Analyst  
# Specializes in AI/ML implementation, data pipelines, and technical architecture
technical_analyst = Agent(
    role="AI Technical Analyst", 
    goal="Analyze technical feasibility, architecture, and implementation challenges",
    backstory="""You are a senior AI engineer specializing in healthcare AI systems.
    You understand machine learning model validation, data pipelines, integration
    challenges, and technical requirements for medical-grade AI systems.""",
    verbose=True,
    llm=llm
)

# Agent 3: Healthcare Regulatory Specialist
# Specializes in FDA compliance, HIPAA, and medical device regulations
regulatory_specialist = Agent(
    role="Healthcare Regulatory Specialist",
    goal="Analyze regulatory compliance, approval processes, and legal implications", 
    backstory="""You are a regulatory affairs expert with deep knowledge of FDA
    approval processes, HIPAA compliance, international medical device regulations,
    and healthcare data privacy requirements.""",
    verbose=True,
    llm=llm
)

# Agent 4: Healthcare Economics Analyst
# Specializes in financial impact, ROI analysis, and healthcare economics
economic_analyst = Agent(
    role="Healthcare Economics Analyst",
    goal="Evaluate economic impact, cost-benefit analysis, and market dynamics",
    backstory="""You are a healthcare economist who analyzes the financial impact
    of new technologies on healthcare systems, insurance models, hospital budgets,
    and patient outcomes. You understand ROI calculations for healthcare IT.""",
    verbose=True,
    llm=llm
)

# Agent 5: Strategic Content Synthesizer
# Specializes in combining multiple perspectives into cohesive strategic analysis
strategic_synthesizer = Agent(
    role="Strategic Content Synthesizer",
    goal="Integrate multi-domain insights into cohesive strategic analysis",
    backstory="""You are an expert strategic analyst who excels at synthesizing
    complex information from multiple domains. You create comprehensive reports
    that weave together technical, regulatory, economic, and domain-specific 
    insights into actionable strategic recommendations.""",
    verbose=True,
    llm=llm
)


# ============================================================================
# TASK CREATION FUNCTIONS - DEFINING WORK FOR AGENTS
# ============================================================================

"""
WHAT ARE TASKS?
Tasks are specific assignments given to agents. Each task includes:
- description: Detailed instructions for the agent
- agent: Which specialist should handle this task
- expected_output: What format/length we expect back

CONTEXT PASSING:
Later agents receive insights from earlier agents through the 'context' parameter.
This creates a collaborative workflow where each agent builds on previous work.
"""

def create_healthcare_task(topic, context=""):
    """
    Creates a task for the Healthcare Domain Expert.
    
    Args:
        topic (str): The research topic to analyze
        context (str): Insights from previous agents (empty for first agent)
    
    Returns:
        Task: CrewAI Task object for healthcare analysis
    """
    return Task(
        description=f"""Analyze "{topic}" from a healthcare domain perspective.
        
        {f"Context from previous analyses: {context}" if context else ""}
        
        Focus on:
        - Current healthcare challenges this addresses
        - Clinical workflow integration requirements
        - Impact on patient care and outcomes
        - Healthcare provider adoption barriers
        - Real-world implementation examples
        
        Provide domain-specific insights that only a healthcare expert would know.
        Target: 400-500 words with detailed analysis.""",
        agent=healthcare_expert,
        expected_output="Healthcare domain analysis with clinical insights (400-500 words)"
    )


def create_technical_task(topic, context=""):
    """
    Creates a task for the AI Technical Analyst.
    
    Args:
        topic (str): The research topic to analyze  
        context (str): Healthcare insights to build upon
    
    Returns:
        Task: CrewAI Task object for technical analysis
    """
    return Task(
        description=f"""Analyze the technical aspects of "{topic}".
        
        {f"Context from previous analyses: {context}" if context else ""}
        
        Examine:
        - AI/ML model requirements and validation needs
        - Data infrastructure and integration challenges
        - Scalability and performance considerations
        - Security and data privacy technical requirements
        - Implementation complexity and technical risks
        
        Focus on technical feasibility and engineering challenges.
        Target: 400-500 words with specific implementation details.""",
        agent=technical_analyst,
        expected_output="Technical feasibility analysis with implementation details (400-500 words)"
    )


def create_regulatory_task(topic, context=""):
    """
    Creates a task for the Healthcare Regulatory Specialist.
    
    Args:
        topic (str): The research topic to analyze
        context (str): Healthcare and technical insights to consider
    
    Returns:
        Task: CrewAI Task object for regulatory analysis
    """
    return Task(
        description=f"""Analyze regulatory implications of "{topic}".
        
        {f"Context from previous analyses: {context}" if context else ""}
        
        Cover:
        - FDA approval pathways and requirements
        - HIPAA and data privacy compliance
        - International regulatory considerations
        - Clinical trial requirements
        - Liability and legal risk factors
        
        Provide regulatory roadmap and compliance strategy.
        Target: 400-500 words with specific regulatory guidance.""",
        agent=regulatory_specialist,
        expected_output="Regulatory compliance analysis and approval strategy (400-500 words)"
    )


def create_economic_task(topic, context=""):
    """
    Creates a task for the Healthcare Economics Analyst.
    
    Args:
        topic (str): The research topic to analyze
        context (str): All previous insights to inform economic analysis
    
    Returns:
        Task: CrewAI Task object for economic analysis
    """
    return Task(
        description=f"""Analyze economic impact of "{topic}".
        
        {f"Context from previous analyses: {context}" if context else ""}
        
        Evaluate:
        - Cost-benefit analysis for healthcare systems
        - Impact on healthcare spending and insurance
        - ROI calculations for hospitals and providers
        - Market size and growth projections
        - Economic barriers to adoption
        
        Provide financial impact assessment and business case.
        Target: 400-500 words with financial projections.""",
        agent=economic_analyst,
        expected_output="Economic impact analysis with financial projections (400-500 words)"
    )


def create_synthesis_task(topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis):
    """
    Creates a task for the Strategic Content Synthesizer.
    This is the final step that combines all specialist insights.
    
    Args:
        topic (str): The research topic being analyzed
        healthcare_analysis (str): Complete output from healthcare expert
        technical_analysis (str): Complete output from technical analyst  
        regulatory_analysis (str): Complete output from regulatory specialist
        economic_analysis (str): Complete output from economic analyst
    
    Returns:
        Task: CrewAI Task object for strategic synthesis
    """
    return Task(
        description=f"""Create comprehensive strategic analysis of "{topic}" by synthesizing insights from domain, technical, regulatory, and economic analyses.
        
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
        
        Target: 1500-2000 words for C-suite executives and strategic decision makers.""",
        agent=strategic_synthesizer,
        expected_output="Comprehensive strategic analysis with integrated insights (1500-2000 words)"
    )


# ============================================================================
# MAIN WORKFLOW EXECUTION FUNCTION
# ============================================================================

def run_crewai_analysis(topic):
    """
    Executes the complete CrewAI multi-agent analysis workflow.
    
    WORKFLOW OVERVIEW:
    This function orchestrates a 5-phase sequential workflow:
    1. Healthcare Domain Expert analyzes the topic
    2. Technical Analyst builds on healthcare insights
    3. Regulatory Specialist considers both previous analyses
    4. Economic Analyst incorporates all three perspectives
    5. Strategic Synthesizer creates final integrated report
    
    WHY SEQUENTIAL?
    Each agent builds on previous work, creating increasingly comprehensive analysis.
    This mimics how expert teams collaborate in real consulting projects.
    
    Args:
        topic (str): The research topic to analyze (e.g., "AI-Powered Medical Diagnostics")
    
    Returns:
        dict: Comprehensive results including:
            - All individual agent analyses
            - Final synthesized report
            - Performance metrics
            - Agent completion tracking
    """
    
    print(f"🧩 Starting CrewAI analysis: {topic}")
    print("=" * 70)
    
    # Performance tracking
    start_time = time.time()
    individual_outputs = {}      # Stores each agent's analysis
    completed_agents = []        # Tracks which agents completed successfully
    
    try:
        # ----------------------------------------------------------------
        # PHASE 1: HEALTHCARE DOMAIN ANALYSIS
        # ----------------------------------------------------------------
        print("🏥 Phase 1/5: Healthcare Domain Expert")
        
        # Create the task for healthcare analysis (no context yet - first agent)
        healthcare_task = create_healthcare_task(topic)
        
        # Create a crew with just the healthcare expert
        healthcare_crew = Crew(
            agents=[healthcare_expert],
            tasks=[healthcare_task],
            process=Process.sequential,  # Only one agent, so sequential
            verbose=False                # Reduce output noise
        )
        
        # Execute the crew and get results
        healthcare_result = healthcare_crew.kickoff()
        
        # Extract the analysis text (CrewAI results have .raw attribute)
        healthcare_analysis = str(healthcare_result.raw) if hasattr(healthcare_result, 'raw') else str(healthcare_result)
        
        # Store results and track completion
        individual_outputs['healthcare'] = healthcare_analysis
        completed_agents.append('healthcare_expert')
        
        print(f"✅ Healthcare analysis completed ({len(healthcare_analysis.split())} words)")
        
        # ----------------------------------------------------------------
        # PHASE 2: TECHNICAL ANALYSIS (WITH HEALTHCARE CONTEXT)
        # ----------------------------------------------------------------
        print("🔧 Phase 2/5: AI Technical Analyst")
        
        # Create context from healthcare analysis (first 300 characters to avoid token limits)
        context = f"Healthcare perspective: {healthcare_analysis[:300]}..."
        
        # Create technical task with healthcare context
        technical_task = create_technical_task(topic, context)
        
        # Execute technical analysis
        technical_crew = Crew(
            agents=[technical_analyst],
            tasks=[technical_task],
            process=Process.sequential,
            verbose=False
        )
        
        technical_result = technical_crew.kickoff()
        technical_analysis = str(technical_result.raw) if hasattr(technical_result, 'raw') else str(technical_result)
        
        # Store results
        individual_outputs['technical'] = technical_analysis
        completed_agents.append('technical_analyst')
        
        print(f"✅ Technical analysis completed ({len(technical_analysis.split())} words)")
        
        # ----------------------------------------------------------------
        # PHASE 3: REGULATORY ANALYSIS (WITH ACCUMULATED CONTEXT)
        # ----------------------------------------------------------------
        print("⚖️ Phase 3/5: Healthcare Regulatory Specialist")
        
        # Build context from both previous analyses
        context = f"Healthcare: {healthcare_analysis[:200]}... Technical: {technical_analysis[:200]}..."
        
        regulatory_task = create_regulatory_task(topic, context)
        regulatory_crew = Crew(
            agents=[regulatory_specialist],
            tasks=[regulatory_task],
            process=Process.sequential,
            verbose=False
        )
        
        regulatory_result = regulatory_crew.kickoff()
        regulatory_analysis = str(regulatory_result.raw) if hasattr(regulatory_result, 'raw') else str(regulatory_result)
        
        individual_outputs['regulatory'] = regulatory_analysis
        completed_agents.append('regulatory_specialist')
        
        print(f"✅ Regulatory analysis completed ({len(regulatory_analysis.split())} words)")
        
        # ----------------------------------------------------------------
        # PHASE 4: ECONOMIC ANALYSIS (WITH ALL CONTEXT)
        # ----------------------------------------------------------------
        print("💰 Phase 4/5: Healthcare Economics Analyst")
        
        # Build comprehensive context from all three previous analyses
        context = f"Healthcare: {healthcare_analysis[:150]}... Technical: {technical_analysis[:150]}... Regulatory: {regulatory_analysis[:150]}..."
        
        economic_task = create_economic_task(topic, context)
        economic_crew = Crew(
            agents=[economic_analyst],
            tasks=[economic_task],
            process=Process.sequential,
            verbose=False
        )
        
        economic_result = economic_crew.kickoff()
        economic_analysis = str(economic_result.raw) if hasattr(economic_result, 'raw') else str(economic_result)
        
        individual_outputs['economic'] = economic_analysis
        completed_agents.append('economic_analyst')
        
        print(f"✅ Economic analysis completed ({len(economic_analysis.split())} words)")
        
        # ----------------------------------------------------------------
        # PHASE 5: STRATEGIC SYNTHESIS (COMBINING ALL INSIGHTS)
        # ----------------------------------------------------------------
        print("🎯 Phase 5/5: Strategic Content Synthesizer")
        
        # Create synthesis task with ALL previous analyses
        synthesis_task = create_synthesis_task(
            topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis
        )
        
        synthesis_crew = Crew(
            agents=[strategic_synthesizer],
            tasks=[synthesis_task],
            process=Process.sequential,
            verbose=False
        )
        
        synthesis_result = synthesis_crew.kickoff()
        final_synthesis = str(synthesis_result.raw) if hasattr(synthesis_result, 'raw') else str(synthesis_result)
        
        completed_agents.append('strategic_synthesizer')  # Track synthesis completion
        
        print(f"✅ Strategic synthesis completed ({len(final_synthesis.split())} words)")
        
        # ----------------------------------------------------------------
        # CALCULATE FINAL METRICS AND RETURN RESULTS
        # ----------------------------------------------------------------
        
        duration = time.time() - start_time
        
        # Calculate total words (individual analyses + synthesis, no double counting)
        individual_word_count = sum(len(output.split()) for output in individual_outputs.values())
        synthesis_word_count = len(final_synthesis.split())
        total_word_count = individual_word_count + synthesis_word_count
        
        # Create comprehensive results dictionary
        formatted_result = {
            "topic": topic,
            "framework": "crewai",
            "total_duration": duration,
            "total_words": total_word_count,
            "words_per_second": total_word_count / duration if duration > 0 else 0,
            "final_synthesis": final_synthesis,
            "individual_analyses": individual_outputs,
            "completed_agents": completed_agents,        # List of successfully completed agents
            "successful_agents": len(completed_agents),  # Count for compatibility
            "total_agents": 5,                          # Expected total (4 domain + 1 synthesis)
            "timestamp": datetime.now().isoformat(),
            "success": len(completed_agents) == 5       # True if all 5 agents completed
        }
        
        # Print summary
        print(f"\n⏱️ CrewAI analysis completed in {duration:.1f} seconds")
        print(f"📝 Generated {total_word_count:,} words ({total_word_count/duration:.1f} words/second)")
        print(f"✅ Completed agents: {len(completed_agents)}/5")
        print("=" * 70)
        
        return formatted_result
        
    except Exception as e:
        # Handle any errors during execution
        duration = time.time() - start_time
        print(f"❌ CrewAI analysis failed: {e}")
        
        return {
            "topic": topic,
            "framework": "crewai",
            "total_duration": duration,
            "total_words": 0,
            "words_per_second": 0,
            "final_synthesis": f"Analysis failed: {e}",
            "individual_analyses": {},
            "completed_agents": completed_agents,        # Include partial completion list
            "successful_agents": len(completed_agents),  # Count of successful agents
            "total_agents": 5,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }


# ============================================================================
# COMPATIBILITY FUNCTION
# ============================================================================

def run_complex_research(topic):
    """
    Legacy compatibility function for existing code.
    Simply calls the main analysis function with a different name.
    
    Args:
        topic (str): Research topic to analyze
    
    Returns:
        dict: Same results as run_crewai_analysis()
    """
    return run_crewai_analysis(topic)