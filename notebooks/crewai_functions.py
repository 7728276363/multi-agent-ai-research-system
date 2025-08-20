# CrewAI Multi-Agent System - BASELINE VERSION
# Save as: crewai_functions.py

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import time
from datetime import datetime

# Initialize LLM
print("🔧 Initializing LLM for workflows...")
llm = LLM(
    model="ollama/llama3.1:8b-instruct-q4_K_M",
    base_url="http://ollama:11434"
)

# AGENT DEFINITIONS (BASELINE)
# ========================================

# Agent 1: Healthcare Domain Expert
healthcare_expert = Agent(
    role="Healthcare Domain Expert",
    goal="Provide deep medical and healthcare industry insights",
    backstory="""You are a healthcare industry veteran with 15+ years experience in 
    medical technology adoption, clinical workflows, and healthcare regulations. 
    You understand how technology impacts patient care, hospital operations, and 
    medical decision-making processes.""",
    verbose=True,
    llm=llm
)

# Agent 2: AI Technical Analyst
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

# TASK TEMPLATES
# ===========================

def create_healthcare_task(topic, context=""):
    """Create healthcare domain analysis task"""
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
    """Create technical analysis task"""
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
    """Create regulatory analysis task"""
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
    """Create economic analysis task"""
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
    """Create strategic synthesis task"""
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

# WORKFLOW EXECUTION
# ===============================

def run_crewai_analysis(topic):
    """Run CrewAI multi-agent analysis"""
    print(f"🧩 Starting CrewAI analysis: {topic}")
    print("=" * 70)
    
    start_time = time.time()
    individual_outputs = {}
    
    try:
        # Phase 1: Healthcare Analysis
        print("🏥 Phase 1/5: Healthcare Domain Expert")
        healthcare_task = create_healthcare_task(topic)
        healthcare_crew = Crew(
            agents=[healthcare_expert],
            tasks=[healthcare_task],
            process=Process.sequential,
            verbose=False
        )
        healthcare_result = healthcare_crew.kickoff()
        healthcare_analysis = str(healthcare_result.raw) if hasattr(healthcare_result, 'raw') else str(healthcare_result)
        individual_outputs['healthcare'] = healthcare_analysis
        print(f"✅ Healthcare analysis completed ({len(healthcare_analysis.split())} words)")
        
        # Phase 2: Technical Analysis (with context)
        print("🔧 Phase 2/5: AI Technical Analyst")
        context = f"Healthcare perspective: {healthcare_analysis[:300]}..."
        technical_task = create_technical_task(topic, context)
        technical_crew = Crew(
            agents=[technical_analyst],
            tasks=[technical_task],
            process=Process.sequential,
            verbose=False
        )
        technical_result = technical_crew.kickoff()
        technical_analysis = str(technical_result.raw) if hasattr(technical_result, 'raw') else str(technical_result)
        individual_outputs['technical'] = technical_analysis
        print(f"✅ Technical analysis completed ({len(technical_analysis.split())} words)")
        
        # Phase 3: Regulatory Analysis (with context)
        print("⚖️ Phase 3/5: Healthcare Regulatory Specialist")
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
        print(f"✅ Regulatory analysis completed ({len(regulatory_analysis.split())} words)")
        
        # Phase 4: Economic Analysis (with context)
        print("💰 Phase 4/5: Healthcare Economics Analyst")
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
        print(f"✅ Economic analysis completed ({len(economic_analysis.split())} words)")
        
        # Phase 5: Strategic Synthesis (with all analyses)
        print("🎯 Phase 5/5: Strategic Content Synthesizer")
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
        # DO NOT add synthesis to individual_outputs - keep it separate to avoid duplication
        print(f"✅ Strategic synthesis completed ({len(final_synthesis.split())} words)")
        
        duration = time.time() - start_time
        # Calculate total words from individual analyses + synthesis (don't double count synthesis)
        individual_word_count = sum(len(output.split()) for output in individual_outputs.values())
        synthesis_word_count = len(final_synthesis.split())
        total_word_count = individual_word_count + synthesis_word_count
        
        # Format result
        formatted_result = {
            "topic": topic,
            "framework": "crewai",
            "total_duration": duration,
            "total_words": total_word_count,
            "words_per_second": total_word_count / duration if duration > 0 else 0,
            "final_synthesis": final_synthesis,
            "individual_analyses": individual_outputs,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }


        print("****************************************************")
        print("****************************************************")
        print("****************************************************")
        print("CrewAI Code")
        print("individual_analyses")
        print("****************************************************")
        print("****************************************************")
        print("****************************************************")




        
        print(f"\n⏱️ CrewAI analysis completed in {duration:.1f} seconds")
        print(f"📝 Generated {total_word_count:,} words ({total_word_count/duration:.1f} words/second)")
        print("=" * 70)
        
        return formatted_result
        
    except Exception as e:
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
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }

# Export function for compatibility
def run_complex_research(topic):
    """Compatibility function for existing code"""
    return run_crewai_analysis(topic)