"""
LlamaIndex Multi-Agent System
===============================================================

This file implements a multi-agent AI research system using LlamaIndex framework.
LlamaIndex is primarily designed for building context-aware AI applications with 
sophisticated data integration and retrieval capabilities.

LLAMAINDEX VS OTHER FRAMEWORKS:
- CrewAI: Purpose-built agent collaboration with workflows
- LangChain: General LLM framework with manual orchestration
- LangGraph: Stateful graph-based workflows  
- LlamaIndex: Data-aware AI applications with advanced retrieval

WHAT IS LLAMAINDEX?
LlamaIndex excels at:
- Intelligent data ingestion and indexing
- Sophisticated retrieval-augmented generation (RAG)
- Context-aware query processing
- Integration with various data sources (docs, APIs, databases)

LLAMAINDEX UNIQUE STRENGTHS:
1. **Data Integration**: Native support for 100+ data sources
2. **Intelligent Retrieval**: Advanced strategies for finding relevant context
3. **Query Engines**: Sophisticated question-answering over data
4. **Agent Tools**: Function calling and tool integration
5. **Evaluation**: Built-in metrics for RAG quality assessment

FRAMEWORK ADAPTATION CHALLENGES:
LlamaIndex wasn't designed for multi-agent collaboration like CrewAI,
so this implementation:
- Uses LlamaIndex's tool-calling capabilities to create agent functions
- Manually orchestrates workflow (similar to LangChain approach)  
- Leverages LlamaIndex's LLM interfaces for consistency
- Maintains compatibility with standardized prompts and outputs

DIAGNOSTIC SYSTEM:
This implementation includes comprehensive diagnostics to handle:
- LlamaIndex installation and import issues
- Ollama connection problems
- Model availability and compatibility
- LLM functionality verification

"""

# ============================================================================
# IMPORTS AND COMPREHENSIVE DIAGNOSTICS
# ============================================================================

import requests          # For testing Ollama connectivity
import json             # For parsing API responses
import time             # For performance timing
from datetime import datetime  # For timestamps
from typing import Dict, List, Any  # Type hints

# Global diagnostic tracking
LLAMAINDEX_WORKING = False
DIAGNOSTIC_INFO = {}


# ============================================================================
# OLLAMA CONNECTION UTILITIES
# ============================================================================

def find_ollama_connection():
    """
    Locate and verify Ollama API connectivity.
    
    CONNECTIVITY STRATEGY:
    Tests multiple common Ollama endpoints in order of likelihood:
    1. Docker container network (primary for containerized deployments)
    2. Local installation endpoints
    3. Docker Desktop bridge networks
    
    Returns:
        str or None: Working Ollama base URL, or None if no connection found
    """
    possible_urls = [
        "http://ollama:11434",                    # Docker container (primary)
        "http://localhost:11434",                 # Local installation  
        "http://127.0.0.1:11434",                 # Alternative localhost
        "http://host.docker.internal:11434",      # Docker Desktop bridge
    ]
    
    for url in possible_urls:
        try:
            print(f"🔍 Testing Ollama connection: {url}")
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama available at: {url}")
                return url
        except Exception as e:
            print(f"❌ Connection failed for {url}: {e}")
    
    print("❌ No working Ollama connection found")
    return None


def get_available_models(base_url):
    """
    Retrieve list of available models from Ollama instance.
    
    Args:
        base_url (str): Ollama API base URL
    
    Returns:
        List[str]: Available model names, empty list if error
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"📋 Available models: {models}")
            return models
        return []
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        return []


def find_working_model(base_url):
    """
    Find a compatible model for the analysis workflow.
    
    MODEL SELECTION STRATEGY:
    1. Prioritize exact match with other frameworks for fair comparison
    2. Fall back to compatible model variants
    3. Use first available model as last resort
    
    Args:
        base_url (str): Ollama API base URL
    
    Returns:
        str or None: Working model name, or None if no models available
    """
    available_models = get_available_models(base_url)
    
    # Primary model (exact match with other frameworks)
    primary_model = "llama3.1:8b-instruct-q4_K_M"
    
    if primary_model in available_models:
        print(f"✅ Found exact model match (same as other frameworks): {primary_model}")
        return primary_model
    
    # Fallback model candidates (in order of preference)
    model_candidates = [
        "llama3.1:8b-instruct-fp16",
        "llama3.1:8b-instruct", 
        "llama3.1:8b",
        "llama3.1:latest",
        "llama3.1",
        "llama3:8b-instruct",
        "llama3:8b", 
        "llama3:latest",
        "llama3",
        "llama2:7b-chat",
        "llama2:7b",
        "llama2:latest"
    ]
    
    # Check exact matches first
    for candidate in model_candidates:
        if candidate in available_models:
            print(f"✅ Found compatible model: {candidate}")
            return candidate
    
    # Check partial matches (for version variations)
    for candidate in model_candidates:
        for available in available_models:
            if candidate.split(':')[0] in available:
                print(f"✅ Found compatible model: {available} (requested: {candidate})")
                return available
    
    # Last resort: use first available model
    if available_models:
        model = available_models[0]
        print(f"⚠️ Using first available model: {model}")
        return model
    
    print("❌ No models available")
    return None


# ============================================================================
# LLAMAINDEX IMPORT AND FUNCTIONALITY TESTING
# ============================================================================

def test_llamaindex_imports():
    """
    Test all required LlamaIndex components with detailed diagnostics.
    
    IMPORT TESTING STRATEGY:
    Tests each LlamaIndex component individually to identify specific issues:
    - Core framework functionality
    - Ollama LLM integration
    - Agent capabilities
    - Tool integration
    
    Returns:
        Dict[str, bool]: Success status for each component
    """
    results = {}
    
    # Test 1: Core LlamaIndex framework
    try:
        from llama_index.core import Settings
        results['core_available'] = True
        print("✅ llama_index.core imported successfully")
    except ImportError as e:
        results['core_available'] = False
        print(f"❌ llama_index.core failed: {e}")
        print("   Install with: pip install llama-index")
    
    # Test 2: Ollama LLM integration
    try:
        from llama_index.llms.ollama import Ollama
        results['ollama_llm_available'] = True
        print("✅ llama_index.llms.ollama imported successfully")
    except ImportError as e:
        results['ollama_llm_available'] = False
        print(f"❌ llama_index.llms.ollama failed: {e}")
        print("   Install with: pip install llama-index-llms-ollama")
    
    # Test 3: Agent functionality
    try:
        from llama_index.core.agent import ReActAgent
        results['agent_available'] = True
        print("✅ ReActAgent imported successfully")
    except ImportError as e:
        results['agent_available'] = False
        print(f"❌ ReActAgent failed: {e}")
    
    # Test 4: Tool integration
    try:
        from llama_index.core.tools import FunctionTool
        results['tools_available'] = True
        print("✅ FunctionTool imported successfully")
    except ImportError as e:
        results['tools_available'] = False
        print(f"❌ FunctionTool failed: {e}")
    
    return results


def test_llm_functionality(base_url, model_name):
    """
    Test LLM creation and basic functionality.
    
    FUNCTIONALITY TESTING:
    1. Create LlamaIndex LLM instance
    2. Execute simple test prompt
    3. Verify response quality and format
    4. Configure global settings for framework use
    
    Args:
        base_url (str): Ollama API endpoint
        model_name (str): Model to test
    
    Returns:
        Tuple[bool, object]: (success_status, llm_instance or None)
    """
    try:
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings
        
        print(f"🔧 Creating LLM: {model_name} at {base_url}")
        
        # Create LLM with same parameters as other frameworks
        llm = Ollama(
            model=model_name,
            base_url=base_url,
            request_timeout=600.0,    # Extended timeout for complex analyses
            temperature=0.1           # Low temperature for consistent results
        )
        
        print("✅ LLM object created successfully")
        
        # Test with simple prompt (matching other frameworks)
        print("🔄 Testing LLM response...")
        test_response = llm.complete("Say hello")
        
        response_text = str(test_response).strip()
        print(f"📝 LLM response: '{response_text}'")
        print(f"📏 Response length: {len(response_text)}")
        
        # Lenient success criteria - any meaningful response
        if len(response_text) > 0 and response_text.lower() != 'none':
            Settings.llm = llm  # Configure global LlamaIndex settings
            print("✅ LLM is working!")
            return True, llm
        else:
            print("❌ LLM returned empty or null response")
            return False, None
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        print(f"🔍 Full error: {traceback.format_exc()}")
        return False, None


def safe_import_test():
    """
    Comprehensive LlamaIndex setup verification.
    
    DIAGNOSTIC WORKFLOW:
    1. Test Ollama connectivity
    2. Verify model availability
    3. Test LlamaIndex imports
    4. Verify LLM functionality
    5. Update global status variables
    """
    global LLAMAINDEX_WORKING, DIAGNOSTIC_INFO
    
    print("🔍 Starting LlamaIndex diagnostic test...")
    
    # Reset diagnostics
    DIAGNOSTIC_INFO.clear()
    
    # Test 1: Ollama connection
    print("\n🔗 Testing Ollama connection...")
    ollama_base_url = find_ollama_connection()
    DIAGNOSTIC_INFO['ollama_available'] = ollama_base_url is not None
    DIAGNOSTIC_INFO['ollama_base_url'] = ollama_base_url
    
    if not ollama_base_url:
        print("❌ Cannot proceed without Ollama connection")
        DIAGNOSTIC_INFO['status'] = 'failed_ollama_connection'
        return
    
    # Test 2: Model availability
    print("\n🤖 Finding working model...")
    working_model = find_working_model(ollama_base_url)
    DIAGNOSTIC_INFO['model_available'] = working_model is not None
    DIAGNOSTIC_INFO['working_model'] = working_model
    
    if not working_model:
        print("❌ Cannot proceed without a working model")
        DIAGNOSTIC_INFO['status'] = 'failed_no_model'
        return
    
    # Test 3: LlamaIndex imports
    print("\n📦 Testing LlamaIndex imports...")
    import_results = test_llamaindex_imports()
    DIAGNOSTIC_INFO.update(import_results)
    
    # Verify all required imports succeeded
    required_imports = ['core_available', 'ollama_llm_available', 'agent_available', 'tools_available']
    all_imports_ok = all(import_results.get(key, False) for key in required_imports)
    
    if not all_imports_ok:
        print("❌ Required imports not available")
        missing = [key for key in required_imports if not import_results.get(key, False)]
        print(f"   Missing: {missing}")
        DIAGNOSTIC_INFO['status'] = 'failed_imports'
        DIAGNOSTIC_INFO['missing_imports'] = missing
        return
    
    # Test 4: LLM functionality (critical test)
    print("\n🧠 Testing LLM functionality...")
    llm_works, llm_instance = test_llm_functionality(ollama_base_url, working_model)
    DIAGNOSTIC_INFO['llm_working'] = llm_works
    
    if llm_works:
        LLAMAINDEX_WORKING = True
        DIAGNOSTIC_INFO['status'] = 'fully_working'
        print("\n🎉 LlamaIndex is fully functional!")
    else:
        DIAGNOSTIC_INFO['status'] = 'failed_llm_test'
        print("\n❌ LlamaIndex setup incomplete - LLM test failed")


# Run diagnostic test immediately upon import
safe_import_test()


# ============================================================================
# ANALYSIS TOOL FUNCTIONS - IDENTICAL TO CREWAI BASELINE
# ============================================================================

"""
These functions implement the same agent analysis capabilities as CrewAI,
but using LlamaIndex's tool-based approach. Each function represents
a specialized analysis tool that can be called by the workflow orchestrator.
"""

def healthcare_analysis_tool(topic: str, context: str = "") -> str:
    """
    Healthcare Domain Expert analysis tool.
    
    Implements identical analysis logic as CrewAI Healthcare Domain Expert.
    Uses LlamaIndex LLM interface with the same prompts for fair comparison.
    
    Args:
        topic (str): Research topic to analyze
        context (str): Context from previous analyses (empty for first agent)
    
    Returns:
        str: Healthcare domain analysis or error message
    """
    prompt = f"""You are a Healthcare Domain Expert.

Goal: Provide deep medical and healthcare industry insights

Background: You are a healthcare industry veteran with 15+ years experience in 
medical technology adoption, clinical workflows, and healthcare regulations. 
You understand how technology impacts patient care, hospital operations, and 
medical decision-making processes.

Task: Analyze "{topic}" from a healthcare domain perspective.

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
        if LLAMAINDEX_WORKING:
            from llama_index.core import Settings
            print(f"🏥 Healthcare analysis starting for: {topic[:50]}...")
            response = Settings.llm.complete(prompt)
            result = str(response).strip()
            
            if len(result) > 10 and result.lower() != 'none':
                print(f"✅ Healthcare analysis completed ({len(result.split())} words)")
                return result
            else:
                print(f"❌ Healthcare analysis returned invalid response: '{result}'")
                return f"Healthcare analysis failed: Invalid response from LLM"
        else:
            return f"❌ LlamaIndex not functional - healthcare analysis unavailable for: {topic}"
    except Exception as e:
        print(f"❌ Healthcare analysis error: {e}")
        return f"Healthcare analysis failed: {str(e)}"


def technical_analysis_tool(topic: str, context: str = "") -> str:
    """AI Technical Analyst analysis tool - identical to CrewAI baseline."""
    prompt = f"""You are an AI Technical Analyst.

Goal: Analyze technical feasibility, architecture, and implementation challenges

Background: You are a senior AI engineer specializing in healthcare AI systems.
You understand machine learning model validation, data pipelines, integration
challenges, and technical requirements for medical-grade AI systems.

Task: Analyze the technical aspects of "{topic}".

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
        if LLAMAINDEX_WORKING:
            from llama_index.core import Settings
            print(f"🔧 Technical analysis starting for: {topic[:50]}...")
            response = Settings.llm.complete(prompt)
            result = str(response).strip()
            
            if len(result) > 10 and result.lower() != 'none':
                print(f"✅ Technical analysis completed ({len(result.split())} words)")
                return result
            else:
                print(f"❌ Technical analysis returned invalid response: '{result}'")
                return f"Technical analysis failed: Invalid response from LLM"
        else:
            return f"❌ LlamaIndex not functional - technical analysis unavailable for: {topic}"
    except Exception as e:
        print(f"❌ Technical analysis error: {e}")
        return f"Technical analysis failed: {str(e)}"


def regulatory_analysis_tool(topic: str, context: str = "") -> str:
    """Healthcare Regulatory Specialist analysis tool - identical to CrewAI baseline."""
    prompt = f"""You are a Healthcare Regulatory Specialist.

Goal: Analyze regulatory compliance, approval processes, and legal implications

Background: You are a regulatory affairs expert with deep knowledge of FDA
approval processes, HIPAA compliance, international medical device regulations,
and healthcare data privacy requirements.

Task: Analyze regulatory implications of "{topic}".

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
        if LLAMAINDEX_WORKING:
            from llama_index.core import Settings
            print(f"⚖️ Regulatory analysis starting for: {topic[:50]}...")
            response = Settings.llm.complete(prompt)
            result = str(response).strip()
            
            if len(result) > 10 and result.lower() != 'none':
                print(f"✅ Regulatory analysis completed ({len(result.split())} words)")
                return result
            else:
                print(f"❌ Regulatory analysis returned invalid response: '{result}'")
                return f"Regulatory analysis failed: Invalid response from LLM"
        else:
            return f"❌ LlamaIndex not functional - regulatory analysis unavailable for: {topic}"
    except Exception as e:
        print(f"❌ Regulatory analysis error: {e}")
        return f"Regulatory analysis failed: {str(e)}"


def economic_analysis_tool(topic: str, context: str = "") -> str:
    """Healthcare Economics Analyst analysis tool - identical to CrewAI baseline."""
    prompt = f"""You are a Healthcare Economics Analyst.

Goal: Evaluate economic impact, cost-benefit analysis, and market dynamics

Background: You are a healthcare economist who analyzes the financial impact
of new technologies on healthcare systems, insurance models, hospital budgets,
and patient outcomes. You understand ROI calculations for healthcare IT.

Task: Analyze economic impact of "{topic}".

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
        if LLAMAINDEX_WORKING:
            from llama_index.core import Settings
            print(f"💰 Economic analysis starting for: {topic[:50]}...")
            response = Settings.llm.complete(prompt)
            result = str(response).strip()
            
            if len(result) > 10 and result.lower() != 'none':
                print(f"✅ Economic analysis completed ({len(result.split())} words)")
                return result
            else:
                print(f"❌ Economic analysis returned invalid response: '{result}'")
                return f"Economic analysis failed: Invalid response from LLM"
        else:
            return f"❌ LlamaIndex not functional - economic analysis unavailable for: {topic}"
    except Exception as e:
        print(f"❌ Economic analysis error: {e}")
        return f"Economic analysis failed: {str(e)}"


def extract_key_insights(analysis_text, max_length):
    """
    Extract key insights from analysis while preserving important details.
    
    Args:
        analysis_text (str): Full analysis text
        max_length (int): Maximum length for extracted insights
    
    Returns:
        str: Compressed insights preserving key information
    """
    if len(analysis_text) <= max_length:
        return analysis_text
    
    # Find sentence boundaries and extract first portion that fits
    sentences = analysis_text.split('. ')
    result = ""
    for sentence in sentences:
        if len(result + sentence + '. ') <= max_length:
            result += sentence + '. '
        else:
            break
    
    return result.strip() if result else analysis_text[:max_length] + "..."


def synthesis_tool(topic: str, healthcare_analysis: str, technical_analysis: str, regulatory_analysis: str, economic_analysis: str) -> str:
    # Compress context but keep key insights to save tokens for longer output
    healthcare_key = extract_key_insights(healthcare_analysis, 150)
    technical_key = extract_key_insights(technical_analysis, 150)
    regulatory_key = extract_key_insights(regulatory_analysis, 150)
    economic_key = extract_key_insights(economic_analysis, 150)
    
    prompt = f"""You are a Strategic Content Synthesizer creating an EXECUTIVE-LEVEL COMPREHENSIVE REPORT.

CRITICAL REQUIREMENT: Your response must be 1500-2000 words. This is NOT a summary - it's a detailed executive report that expands significantly beyond the input analyses.

Previous Domain Insights to Build Upon:
• Healthcare: {healthcare_key}
• Technical: {technical_key}
• Regulatory: {regulatory_key}
• Economic: {economic_key}

Your Task: Create a comprehensive strategic analysis of "{topic}" that synthesizes and EXPANDS on these insights.

MANDATORY STRUCTURE (with word targets):
1. EXECUTIVE SUMMARY (300-400 words)
   - Strategic significance and market opportunity
   - Key success factors and critical decisions
   - Investment thesis and recommended actions

2. INTEGRATED MARKET ANALYSIS (400-500 words)
   - How healthcare needs drive technical requirements
   - Regulatory environment impact on implementation
   - Economic factors shaping market dynamics
   - Competitive landscape and positioning

3. STRATEGIC RECOMMENDATIONS (300-400 words)
   - For healthcare providers and systems
   - For technology companies and investors
   - For regulatory bodies and policymakers
   - For implementation teams and stakeholders

4. IMPLEMENTATION ROADMAP (200-300 words)
   - Phase 1: Foundation and pilot programs
   - Phase 2: Scale and optimization
   - Phase 3: Market expansion and evolution
   - Timeline, milestones, and success metrics

5. RISK ASSESSMENT & MITIGATION (200-300 words)
   - Technical and operational risks
   - Regulatory and compliance challenges
   - Market and competitive threats
   - Mitigation strategies and contingency plans

6. FUTURE OUTLOOK (200-300 words)
   - 3-5 year market evolution scenarios
   - Emerging opportunities and disruptions
   - Strategic implications for stakeholders
   - Long-term value creation potential

WRITING GUIDELINES:
- Use executive-level language appropriate for C-suite decision makers
- Include specific examples, data points, and concrete recommendations
- Each section should be substantially detailed and comprehensive
- Think like a McKinsey or BCG consultant writing for a Fortune 500 CEO
- Your total response should be 1500-2000 words - significantly longer than typical AI responses

Begin writing your comprehensive executive report now:"""
    
    try:
        if LLAMAINDEX_WORKING:
            from llama_index.core import Settings
            print(f"🎯 Enhanced synthesis starting for: {topic[:50]}...")
            response = Settings.llm.complete(prompt)
            result = str(response).strip()
            
            if len(result) > 10 and result.lower() != 'none':
                print(f"✅ Enhanced synthesis completed ({len(result.split())} words)")
                return result
            else:
                print(f"❌ Synthesis returned invalid response: '{result}'")
                return f"Synthesis failed: Invalid response from LLM"
        else:
            return f"❌ LlamaIndex not functional - synthesis unavailable for: {topic}"
    except Exception as e:
        print(f"❌ Enhanced synthesis error: {e}")
        return f"Synthesis failed: {str(e)}"


# ============================================================================
# WORKFLOW RUNNER IMPLEMENTATION
# ============================================================================

if LLAMAINDEX_WORKING:
    print("✅ Creating working LlamaIndex implementation")
    
    class LlamaIndexRunner:
        """
        Working LlamaIndex implementation matching CrewAI baseline workflow.
        
        DESIGN PHILOSOPHY:
        This class orchestrates the same 5-phase workflow as other frameworks
        but uses LlamaIndex's tool-based approach rather than agent objects.
        
        WORKFLOW ORCHESTRATION:
        1. Initialize with diagnostic logging
        2. Execute agents sequentially with context passing
        3. Track completions and performance metrics
        4. Format results to match standardized output format
        """
        
        def __init__(self):
            """Initialize the LlamaIndex workflow runner."""
            print("🔧 Initializing LlamaIndexRunner...")
            self.initialization_log = [
                "✅ LlamaIndex runner initialized successfully",
                f"✅ Using model: {DIAGNOSTIC_INFO.get('working_model', 'unknown')}",
                f"✅ Ollama URL: {DIAGNOSTIC_INFO.get('ollama_base_url', 'unknown')}"
            ]
        
        def run_agent_analysis(self, agent_name: str, topic: str, context: str = "") -> Dict[str, Any]:
            """
            Execute individual agent analysis using appropriate tool.
            
            Args:
                agent_name (str): Which agent to run
                topic (str): Research topic
                context (str): Context from previous agents or custom task
            
            Returns:
                Dict: Analysis results with performance metrics
            """
            print(f"🔄 Running {agent_name} analysis...")
            start_time = time.time()
            
            try:
                # Route to appropriate analysis tool
                if agent_name == "healthcare":
                    analysis_text = healthcare_analysis_tool(topic, context)
                elif agent_name == "technical":
                    analysis_text = technical_analysis_tool(topic, context)
                elif agent_name == "regulatory":
                    analysis_text = regulatory_analysis_tool(topic, context)
                elif agent_name == "economic":
                    analysis_text = economic_analysis_tool(topic, context)
                elif agent_name == "synthesizer":
                    # Parse context for synthesis (contains all previous analyses)
                    parts = context.split("|||") if "|||" in context else ["", "", "", ""]
                    healthcare_analysis = parts[0] if len(parts) > 0 else ""
                    technical_analysis = parts[1] if len(parts) > 1 else ""
                    regulatory_analysis = parts[2] if len(parts) > 2 else ""
                    economic_analysis = parts[3] if len(parts) > 3 else ""
                    
                    analysis_text = synthesis_tool(
                        topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis
                    )
                elif agent_name == "synthesizer_stage1" or agent_name == "synthesizer_stage2":
                    # For two-stage synthesis, context contains the full task description
                    if LLAMAINDEX_WORKING:
                        from llama_index.core import Settings
                        response = Settings.llm.complete(context)
                        analysis_text = str(response).strip()
                    else:
                        analysis_text = f"❌ LlamaIndex not functional - {agent_name} unavailable"
                else:
                    analysis_text = f"Unknown agent type: {agent_name}"
                
                duration = time.time() - start_time
                word_count = len(analysis_text.split())
                success = not analysis_text.startswith("❌") and "failed" not in analysis_text.lower()
                
                print(f"{'✅' if success else '❌'} {agent_name} analysis completed in {duration:.1f}s ({word_count} words)")
                
                return {
                    "agent": agent_name,
                    "analysis": analysis_text,
                    "success": success,
                    "duration": duration,
                    "word_count": word_count
                }
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ {agent_name} analysis failed: {e}")
                
                return {
                    "agent": agent_name,
                    "analysis": f"Analysis failed: {str(e)}",
                    "success": False,
                    "duration": duration,
                    "word_count": 0,
                    "error": str(e)
                }
        
        def run_comprehensive_analysis(self, topic: str) -> Dict[str, Any]:
            """
            Execute complete multi-agent analysis workflow.
            
            WORKFLOW EXECUTION:
            This method orchestrates the same 5-phase sequential workflow
            as CrewAI, LangChain, and LangGraph for fair comparison:
            
            1. Healthcare Domain Expert → Clinical insights
            2. Technical Analyst → Builds on healthcare context  
            3. Regulatory Specialist → Considers both previous analyses
            4. Economic Analyst → Incorporates all three perspectives
            5. Strategic Synthesizer → Creates final integrated report
            
            CONTEXT PASSING STRATEGY:
            Each agent receives truncated context from previous agents to:
            - Provide relevant background information
            - Avoid LLM token limit issues
            - Maintain workflow continuity and quality
            
            Args:
                topic (str): Research topic to analyze
            
            Returns:
                Dict: Comprehensive results matching other frameworks
            """
            print(f"🦙 Starting LlamaIndex analysis: {topic}")
            print("=" * 70)
            
            start_time = time.time()
            individual_analyses = {}
            results = []
            analysis_log = []
            
            # Log initialization and workflow start
            analysis_log.extend(self.initialization_log)
            analysis_log.append(f"🔄 Starting standardized analysis matching CrewAI baseline")
            
            # ----------------------------------------------------------------
            # PHASE 1: HEALTHCARE DOMAIN ANALYSIS
            # ----------------------------------------------------------------
            print("🏥 Phase 1/5: Healthcare Domain Expert")
            analysis_log.append("🏥 Phase 1/5: Healthcare Domain Expert")
            healthcare_result = self.run_agent_analysis("healthcare", topic)
            results.append(healthcare_result)
            
            if healthcare_result["success"]:
                healthcare_analysis = healthcare_result["analysis"]
                individual_analyses['healthcare'] = healthcare_analysis
            else:
                healthcare_analysis = ""
                individual_analyses['healthcare'] = f"Failed: {healthcare_result['analysis']}"
            
            # ----------------------------------------------------------------
            # PHASE 2: TECHNICAL ANALYSIS (WITH HEALTHCARE CONTEXT)
            # ----------------------------------------------------------------
            print("🔧 Phase 2/5: AI Technical Analyst")
            analysis_log.append("🔧 Phase 2/5: AI Technical Analyst")
            context = f"Healthcare perspective: {healthcare_analysis[:300]}..." if healthcare_analysis else ""
            technical_result = self.run_agent_analysis("technical", topic, context)
            results.append(technical_result)
            
            if technical_result["success"]:
                technical_analysis = technical_result["analysis"]
                individual_analyses['technical'] = technical_analysis
            else:
                technical_analysis = ""
                individual_analyses['technical'] = f"Failed: {technical_result['analysis']}"
            
            # ----------------------------------------------------------------
            # PHASE 3: REGULATORY ANALYSIS (WITH ACCUMULATED CONTEXT)
            # ----------------------------------------------------------------
            print("⚖️ Phase 3/5: Healthcare Regulatory Specialist")
            analysis_log.append("⚖️ Phase 3/5: Healthcare Regulatory Specialist")
            context = f"Healthcare: {healthcare_analysis[:200]}... Technical: {technical_analysis[:200]}..."
            regulatory_result = self.run_agent_analysis("regulatory", topic, context)
            results.append(regulatory_result)
            
            if regulatory_result["success"]:
                regulatory_analysis = regulatory_result["analysis"]
                individual_analyses['regulatory'] = regulatory_analysis
            else:
                regulatory_analysis = ""
                individual_analyses['regulatory'] = f"Failed: {regulatory_result['analysis']}"
            
            # ----------------------------------------------------------------
            # PHASE 4: ECONOMIC ANALYSIS (WITH ALL DOMAIN CONTEXT)
            # ----------------------------------------------------------------
            print("💰 Phase 4/5: Healthcare Economics Analyst")
            analysis_log.append("💰 Phase 4/5: Healthcare Economics Analyst")
            context = f"Healthcare: {healthcare_analysis[:150]}... Technical: {technical_analysis[:150]}... Regulatory: {regulatory_analysis[:150]}..."
            economic_result = self.run_agent_analysis("economic", topic, context)
            results.append(economic_result)
            
            if economic_result["success"]:
                economic_analysis = economic_result["analysis"]
                individual_analyses['economic'] = economic_analysis
            else:
                economic_analysis = ""
                individual_analyses['economic'] = f"Failed: {economic_result['analysis']}"
            
            # ----------------------------------------------------------------
            # PHASE 5: STRATEGIC SYNTHESIS (COMBINING ALL INSIGHTS)
            # ----------------------------------------------------------------
            print("🎯 Phase 5/5: Strategic Content Synthesizer")
            analysis_log.append("🎯 Phase 5/5: Strategic Content Synthesizer")
            
            # Combine all analyses for synthesis context
            synthesis_context = f"{healthcare_analysis}|||{technical_analysis}|||{regulatory_analysis}|||{economic_analysis}"
            synthesis_result = self.run_agent_analysis("synthesizer", topic, synthesis_context)
            results.append(synthesis_result)
            
            if synthesis_result["success"]:
                final_synthesis = synthesis_result["analysis"]
            else:
                final_synthesis = f"Synthesis failed: {synthesis_result['analysis']}"
            
            # ----------------------------------------------------------------
            # CALCULATE FINAL METRICS AND FORMAT RESULTS
            # ----------------------------------------------------------------
            total_duration = time.time() - start_time
            
            # Count successful domain agents (first 4 results) + synthesis success
            successful_domain_agents = sum(1 for r in results[:-1] if r["success"])  # First 4 agents
            synthesis_success = synthesis_result["success"]
            successful_agents = successful_domain_agents + (1 if synthesis_success else 0)
            
            # Calculate words: individual analyses + synthesis (no double counting)
            individual_words = sum(r.get("word_count", 0) for r in results[:-1] if r["success"])
            synthesis_words = len(final_synthesis.split()) if 'failed' not in final_synthesis.lower() else 0
            total_words = individual_words + synthesis_words
            
            analysis_log.append(f"🎉 Analysis complete! Total time: {total_duration:.1f}s")
            
            # Print workflow summary
            print(f"\n⏱️ LlamaIndex analysis completed in {total_duration:.1f} seconds")
            print(f"📝 Generated {total_words:,} words ({total_words/total_duration:.1f} words/second)")
            print(f"✅ {successful_agents}/{len(results)} agents completed successfully")
            print("=" * 70)
            
            # Return standardized results format (matching other frameworks)
            return {
                "topic": topic,
                "framework": "llamaindex",
                "total_duration": total_duration,
                "total_words": total_words,
                "words_per_second": total_words / total_duration if total_duration > 0 else 0,
                "successful_agents": successful_agents,
                "total_agents": len(results),
                "individual_analyses": individual_analyses,
                "final_synthesis": final_synthesis,
                "timestamp": datetime.now().isoformat(),
                "diagnostic_info": DIAGNOSTIC_INFO,
                "initialization_log": self.initialization_log,
                "analysis_log": analysis_log,
                "success": successful_agents >= 4,
                # Add completed_agents list for Streamlit compatibility
                "completed_agents": [r["agent"] for r in results if r["success"]]
            }
    
    # Create the working runner instance
    agent_runner = LlamaIndexRunner()

else:
    print("❌ Creating fallback LlamaIndex implementation")
    
    class FallbackLlamaIndexRunner:
        """
        Fallback implementation when LlamaIndex is not functional.
        
        This class provides graceful degradation when LlamaIndex
        cannot be properly initialized due to missing dependencies,
        connection issues, or other setup problems.
        """
        
        def run_comprehensive_analysis(self, topic: str) -> Dict[str, Any]:
            """
            Return diagnostic information instead of analysis when LlamaIndex fails.
            
            Args:
                topic (str): Research topic (for compatibility)
            
            Returns:
                Dict: Diagnostic information explaining the failure
            """
            return {
                "topic": topic,
                "framework": "llamaindex",
                "total_duration": 0,
                "total_words": 0,
                "words_per_second": 0,
                "successful_agents": 0,
                "total_agents": 5,
                "individual_analyses": {
                    "healthcare": f"LlamaIndex not functional - Status: {DIAGNOSTIC_INFO.get('status', 'unknown')}",
                    "technical": f"LlamaIndex not functional - Check setup",
                    "regulatory": f"LlamaIndex not functional - Check setup", 
                    "economic": f"LlamaIndex not functional - Check setup"
                },
                "final_synthesis": f"LlamaIndex framework not functional. Status: {DIAGNOSTIC_INFO.get('status', 'unknown')}. Check diagnostic info for details.",
                "timestamp": datetime.now().isoformat(),
                "diagnostic_info": DIAGNOSTIC_INFO,
                "success": False
            }
    
    # Create the fallback runner instance
    agent_runner = FallbackLlamaIndexRunner()


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_llamaindex_analysis(topic: str):
    """
    Main LlamaIndex analysis function - matches other frameworks.
    
    This function provides the primary interface for executing LlamaIndex
    multi-agent analysis, with comprehensive debug logging for troubleshooting.
    
    EXECUTION FLOW:
    1. Log debug information about LlamaIndex status
    2. Execute the appropriate runner (working or fallback)
    3. Add debug information to results for diagnostics
    4. Return standardized results matching other frameworks
    
    Args:
        topic (str): Research topic to analyze
    
    Returns:
        Dict: Analysis results with debug information
    """
    debug_log = []
    
    def log_debug(message):
        """Log debug message to both console and return list."""
        print(message)
        debug_log.append(message)
    
    log_debug(f"🦙 Starting LlamaIndex analysis for: {topic}")
    log_debug(f"🔍 LlamaIndex working status: {LLAMAINDEX_WORKING}")
    log_debug(f"🔍 Agent runner type: {type(agent_runner).__name__}")
    log_debug(f"🔍 Diagnostic status: {DIAGNOSTIC_INFO.get('status', 'unknown')}")
    
    try:
        # Execute analysis using appropriate runner
        result = agent_runner.run_comprehensive_analysis(topic)
        log_debug(f"🔍 Analysis completed. Success rate: {result.get('successful_agents', 0)}/{result.get('total_agents', 5)}")
        
        # Add debug log to result for Streamlit display
        result['debug_log'] = debug_log
        return result
        
    except Exception as e:
        log_debug(f"❌ LlamaIndex analysis completely failed: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        log_debug(f"   Traceback: {traceback_str}")
        
        # Return comprehensive error information
        return {
            "topic": topic,
            "framework": "llamaindex",
            "total_duration": 0,
            "total_words": 0,
            "words_per_second": 0,
            "successful_agents": 0,
            "total_agents": 5,
            "individual_analyses": {
                "healthcare": f"Complete analysis failure: {str(e)}",
                "technical": f"Complete analysis failure: {str(e)}",
                "regulatory": f"Complete analysis failure: {str(e)}",
                "economic": f"Complete analysis failure: {str(e)}"
            },
            "final_synthesis": f"LlamaIndex analysis completely failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "diagnostic_info": DIAGNOSTIC_INFO,
            "debug_log": debug_log,
            "traceback": traceback_str,
            "success": False
        }


# ============================================================================
# UTILITY FUNCTIONS FOR RESULTS DISPLAY AND SAVING
# ============================================================================

def display_llamaindex_results(result: Dict[str, Any]):
    """
    Display LlamaIndex results in a formatted way.
    
    Args:
        result (Dict): Analysis results from run_llamaindex_analysis()
    """
    print("\n" + "=" * 60)
    print("🦙 LLAMAINDEX RESULTS")
    print("=" * 60)
    
    # Show diagnostic info
    if 'diagnostic_info' in result:
        print("DIAGNOSTIC STATUS:")
        for key, value in result['diagnostic_info'].items():
            status = "✅" if value else "❌"
            print(f"  {key}: {status}")
        print()
    
    print(f"Topic: {result['topic']}")
    print(f"Framework: LlamaIndex (Standardized)")
    print(f"Success Rate: {result['successful_agents']}/{result['total_agents']}")
    print(f"Duration: {result['total_duration']:.1f}s")
    print(f"Words: {result['total_words']:,}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    # Show individual analyses
    individual_analyses = result.get('individual_analyses', {})
    if individual_analyses:
        print("\nINDIVIDUAL ANALYSES:")
        for domain, analysis in individual_analyses.items():
            if domain != 'synthesis':
                print(f"\n{domain.upper()}:")
                preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
                print(preview)
    
    print("\nFINAL SYNTHESIS:")
    synthesis = result.get('final_synthesis', 'No synthesis available')
    preview = synthesis[:300] + "..." if len(synthesis) > 300 else synthesis
    print(preview)


def save_llamaindex_result(result: Dict[str, Any], filename: str = None):
    """
    Save LlamaIndex results to a text file.
    
    Args:
        result (Dict): Analysis results
        filename (str, optional): Custom filename, auto-generated if None
    
    Returns:
        str or None: Filepath if successful, None if failed
    """
    import os
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_topic = result['topic'].lower().replace(' ', '_').replace(':', '').replace(',', '')[:50]
        filename = f"llamaindex_{safe_topic}_{timestamp}.txt"
    
    try:
        os.makedirs('/app/projects', exist_ok=True)
        filepath = f'/app/projects/{filename}'
        
        with open(filepath, 'w') as f:
            f.write("LLAMAINDEX ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Diagnostic info
            if 'diagnostic_info' in result:
                f.write("DIAGNOSTIC STATUS:\n")
                for key, value in result['diagnostic_info'].items():
                    status = "PASS" if value else "FAIL"
                    f.write(f"{key}: {status}\n")
                f.write("\n")
            
            f.write(f"Topic: {result['topic']}\n")
            f.write(f"Framework: LlamaIndex (Standardized to match CrewAI baseline)\n")
            f.write(f"Generated: {result['timestamp']}\n")
            f.write(f"Duration: {result['total_duration']:.1f} seconds\n")
            f.write(f"Success Rate: {result['successful_agents']}/{result['total_agents']}\n")
            f.write(f"Total Words: {result['total_words']:,}\n")
            
            if result.get('error'):
                f.write(f"Error: {result['error']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("FINAL SYNTHESIS:\n")
            f.write("-" * 20 + "\n")
            f.write(result.get('final_synthesis', 'No synthesis available'))
            
            # Individual analyses
            individual_analyses = result.get('individual_analyses', {})
            if individual_analyses:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("INDIVIDUAL ANALYSES:\n")
                f.write("=" * 30 + "\n")
                
                for domain, analysis in individual_analyses.items():
                    if domain != 'synthesis':
                        f.write(f"\n{domain.upper()}:\n")
                        f.write("-" * (len(domain) + 1) + "\n")
                        f.write(f"{analysis}\n")
        
        print(f"✅ LlamaIndex report saved to: {filename}")
        return filepath
        
    except Exception as e:
        print(f"❌ Failed to save LlamaIndex report: {e}")
        return None


# ============================================================================
# MODULE STATUS AND COMPATIBILITY
# ============================================================================

# Display module status
if LLAMAINDEX_WORKING:
    print("✅ LlamaIndex module ready and working")
else:
    print("❌ LlamaIndex module loaded in fallback mode")
    print(f"   Status: {DIAGNOSTIC_INFO.get('status', 'unknown')}")
    print("   Check diagnostic info for details on what's not working")

print(f"🦙 LlamaIndex diagnostic info: {DIAGNOSTIC_INFO}")


# ============================================================================
# COMPATIBILITY FUNCTION
# ============================================================================

def run_comprehensive_analysis(topic):
    """
    Legacy compatibility function for existing code.
    
    Args:
        topic (str): Research topic to analyze
    
    Returns:
        dict: Same results as run_llamaindex_analysis()
    """
    return run_llamaindex_analysis(topic)