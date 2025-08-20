# LangChain Multi-Agent System - MATCHED TO CREWAI BASELINE
# Save as: langchain_functions.py

from langchain_ollama import OllamaLLM
from datetime import datetime
import time
from typing import Dict, List, Any

# Initialize LLM
print("🔧 Initializing LangChain LLM for workflows...")
llm = OllamaLLM(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://ollama:11434"
)

# AGENT CLASS
# ========================

class Agent:
    """Standardized agent class matching CrewAI baseline"""
    
    def __init__(self, role: str, goal: str, backstory: str, llm):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.memory = []
        
    def analyze(self, task_description: str) -> Dict[str, Any]:
        """Perform analysis based on task description"""
        
        # Create prompt structure
        full_prompt = f"""You are a {self.role}.

Goal: {self.goal}

Background: {self.backstory}

Task: {task_description}

Provide your expert analysis focusing on your area of expertise. Be detailed, specific, and practical."""

        start_time = time.time()
        
        try:
            response = self.llm.invoke(full_prompt)
            duration = time.time() - start_time
            
            analysis_result = {
                "agent": self.role,
                "analysis": response,
                "duration": duration,
                "word_count": len(response.split()),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.memory.append(analysis_result)
            print(f"✅ {self.role} completed analysis in {duration:.1f}s ({len(response.split())} words)")
            return analysis_result
            
        except Exception as e:
            print(f"❌ {self.role} analysis failed: {e}")
            return {
                "agent": self.role,
                "analysis": f"Analysis failed: {e}",
                "duration": time.time() - start_time,
                "word_count": 0,
                "error": str(e),
                "success": False
            }

# AGENT DEFINITIONS (MATCHING CREWAI BASELINE)
# =========================================================

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

# TASK TEMPLATES (MATCHING CREWAI)
# =============================================

def create_healthcare_task(topic, context=""):
    """Create healthcare domain analysis task"""
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
    """Create technical analysis task"""
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
    """Create regulatory analysis task"""
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
    """Create economic analysis task"""
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
    """Create strategic synthesis task"""
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

# WORKFLOW EXECUTION
# ===============================

def run_langchain_analysis(topic):
    """Run LangChain multi-agent analysis matching CrewAI baseline"""
    print(f"🔗 Starting LangChain analysis: {topic}")
    print("=" * 70)
    
    start_time = time.time()
    individual_outputs = {}
    results = []
    
    try:
        # Phase 1: Healthcare Analysis
        print("🏥 Phase 1/5: Healthcare Domain Expert")
        healthcare_task = create_healthcare_task(topic)
        healthcare_result = healthcare_expert.analyze(healthcare_task)
        results.append(healthcare_result)
        
        if healthcare_result["success"]:
            healthcare_analysis = healthcare_result["analysis"]
            individual_outputs['healthcare'] = healthcare_analysis
        else:
            healthcare_analysis = ""
            individual_outputs['healthcare'] = f"Failed: {healthcare_result['analysis']}"
        
        # Phase 2: Technical Analysis (with context)
        print("🔧 Phase 2/5: AI Technical Analyst")
        context = f"Healthcare perspective: {healthcare_analysis[:300]}..." if healthcare_analysis else ""
        technical_task = create_technical_task(topic, context)
        technical_result = technical_analyst.analyze(technical_task)
        results.append(technical_result)
        
        if technical_result["success"]:
            technical_analysis = technical_result["analysis"]
            individual_outputs['technical'] = technical_analysis
        else:
            technical_analysis = ""
            individual_outputs['technical'] = f"Failed: {technical_result['analysis']}"
        
        # Phase 3: Regulatory Analysis (with context)
        print("⚖️ Phase 3/5: Healthcare Regulatory Specialist")
        context = f"Healthcare: {healthcare_analysis[:200]}... Technical: {technical_analysis[:200]}..."
        regulatory_task = create_regulatory_task(topic, context)
        regulatory_result = regulatory_specialist.analyze(regulatory_task)
        results.append(regulatory_result)
        
        if regulatory_result["success"]:
            regulatory_analysis = regulatory_result["analysis"]
            individual_outputs['regulatory'] = regulatory_analysis
        else:
            regulatory_analysis = ""
            individual_outputs['regulatory'] = f"Failed: {regulatory_result['analysis']}"
        
        # Phase 4: Economic Analysis (with context)
        print("💰 Phase 4/5: Healthcare Economics Analyst")
        context = f"Healthcare: {healthcare_analysis[:150]}... Technical: {technical_analysis[:150]}... Regulatory: {regulatory_analysis[:150]}..."
        economic_task = create_economic_task(topic, context)
        economic_result = economic_analyst.analyze(economic_task)
        results.append(economic_result)
        
        if economic_result["success"]:
            economic_analysis = economic_result["analysis"]
            individual_outputs['economic'] = economic_analysis
        else:
            economic_analysis = ""
            individual_outputs['economic'] = f"Failed: {economic_result['analysis']}"
        
        # Phase 5: Strategic Synthesis (with all analyses)
        print("🎯 Phase 5/5: Strategic Content Synthesizer")
        synthesis_task = create_synthesis_task(
            topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis
        )
        synthesis_result = strategic_synthesizer.analyze(synthesis_task)
        results.append(synthesis_result)
        
        if synthesis_result["success"]:
            final_synthesis = synthesis_result["analysis"]
            # Don't add synthesis to individual_outputs to avoid duplication
        else:
            final_synthesis = f"Synthesis failed: {synthesis_result['analysis']}"
        
        # Calculate metrics (don't double count synthesis)
        total_duration = time.time() - start_time
        successful_agents = sum(1 for r in results if r.get('success', False))
        # Only count words from individual domain analyses + synthesis separately
        individual_words = sum(r.get('word_count', 0) for r in results[:-1] if r.get('success', False))  # Exclude synthesis result
        synthesis_words = len(final_synthesis.split()) if 'failed' not in final_synthesis.lower() else 0
        total_words = individual_words + synthesis_words
        
        workflow_result = {
            "topic": topic,
            "framework": "langchain",
            "total_duration": total_duration,
            "total_words": total_words,
            "words_per_second": total_words / total_duration if total_duration > 0 else 0,
            "successful_agents": successful_agents,
            "total_agents": len(results),
            "individual_analyses": individual_outputs,
            "final_synthesis": final_synthesis,
            "timestamp": datetime.now().isoformat(),
            "success": successful_agents >= 4  # Need at least 4 successful agents
        }
        
        print(f"\n⏱️ LangChain analysis completed in {total_duration:.1f} seconds")
        print(f"📝 Generated {total_words:,} words ({total_words/total_duration:.1f} words/second)")
        print(f"✅ {successful_agents}/{len(results)} agents completed successfully")
        print("=" * 70)
        
        return workflow_result
        
    except Exception as e:
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
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }

# Export function for compatibility
def run_comprehensive_analysis(topic):
    """Compatibility function for existing code"""
    return run_langchain_analysis(topic)