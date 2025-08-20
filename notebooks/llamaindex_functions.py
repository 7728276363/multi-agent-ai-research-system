# LlamaIndex Multi-Agent System - MATCHED TO CREWAI BASELINE
# Save as: llamaindex_functions.py

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Global variables to track what's working
LLAMAINDEX_WORKING = False
DIAGNOSTIC_INFO = {}

def safe_import_test():
    """Test imports safely without breaking Streamlit"""
    global LLAMAINDEX_WORKING, DIAGNOSTIC_INFO
    
    try:
        # Test Ollama connectivity
        try:
            response = requests.get("http://ollama:11434/api/tags", timeout=5)
            ollama_available = response.status_code == 200
        except:
            ollama_available = False
        
        DIAGNOSTIC_INFO['ollama_available'] = ollama_available
        
        # Test LlamaIndex imports
        try:
            from llama_index.core import Settings
            core_available = True
        except ImportError:
            core_available = False
        
        try:
            from llama_index.core.agent import ReActAgent
            agent_available = True
        except ImportError:
            agent_available = False
            
        try:
            from llama_index.core.tools import FunctionTool
            tools_available = True
        except ImportError:
            tools_available = False
            
        try:
            from llama_index.llms.ollama import Ollama
            ollama_llm_available = True
        except ImportError:
            ollama_llm_available = False
        
        DIAGNOSTIC_INFO.update({
            'core_available': core_available,
            'agent_available': agent_available,
            'tools_available': tools_available,
            'ollama_llm_available': ollama_llm_available
        })
        
        # Only proceed if we have everything needed
        if all([ollama_available, core_available, agent_available, tools_available, ollama_llm_available]):
            # Test LLM creation
            try:
                from llama_index.core import Settings
                from llama_index.llms.ollama import Ollama

                llm = Ollama(
                    model="llama3.1:8b-instruct-q4_K_M",
                    base_url="http://ollama:11434",
                    request_timeout=600.0
                )
                
                # Quick test
                test_response = llm.complete("Hello")
                if len(str(test_response)) > 0:
                    Settings.llm = llm
                    LLAMAINDEX_WORKING = True
                    DIAGNOSTIC_INFO['llm_working'] = True
                    print("✅ LlamaIndex fully working")
                else:
                    DIAGNOSTIC_INFO['llm_working'] = False
                    print("❌ LlamaIndex LLM test failed")
                    
            except Exception as e:
                DIAGNOSTIC_INFO['llm_working'] = False
                DIAGNOSTIC_INFO['llm_error'] = str(e)
                print(f"❌ LlamaIndex LLM setup failed: {e}")
        else:
            print("❌ LlamaIndex missing required components")
            
    except Exception as e:
        print(f"❌ LlamaIndex import test failed: {e}")
        DIAGNOSTIC_INFO['import_error'] = str(e)

# Run the safe import test
safe_import_test()

# ANALYSIS FUNCTIONS (MATCHING CREWAI BASELINE)
# ===========================================================

def healthcare_analysis_tool(topic: str, context: str = "") -> str:
    """Healthcare Domain Expert analysis - matching CrewAI baseline exactly"""
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
            response = Settings.llm.complete(prompt)
            return str(response)
        else:
            return f"LlamaIndex not functional - healthcare analysis unavailable for: {topic}"
    except Exception as e:
        return f"Healthcare analysis failed: {str(e)}"

def technical_analysis_tool(topic: str, context: str = "") -> str:
    """AI Technical Analyst analysis - matching CrewAI baseline exactly"""
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
            response = Settings.llm.complete(prompt)
            return str(response)
        else:
            return f"LlamaIndex not functional - technical analysis unavailable for: {topic}"
    except Exception as e:
        return f"Technical analysis failed: {str(e)}"

def regulatory_analysis_tool(topic: str, context: str = "") -> str:
    """Healthcare Regulatory Specialist analysis - matching CrewAI baseline exactly"""
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
            response = Settings.llm.complete(prompt)
            return str(response)
        else:
            return f"LlamaIndex not functional - regulatory analysis unavailable for: {topic}"
    except Exception as e:
        return f"Regulatory analysis failed: {str(e)}"

def economic_analysis_tool(topic: str, context: str = "") -> str:
    """Healthcare Economics Analyst analysis - matching CrewAI baseline exactly"""
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
            response = Settings.llm.complete(prompt)
            return str(response)
        else:
            return f"LlamaIndex not functional - economic analysis unavailable for: {topic}"
    except Exception as e:
        return f"Economic analysis failed: {str(e)}"

def synthesis_tool(topic: str, healthcare_analysis: str, technical_analysis: str, regulatory_analysis: str, economic_analysis: str) -> str:
    """Strategic Content Synthesizer - matching CrewAI baseline exactly"""
    prompt = f"""You are a Strategic Content Synthesizer.

Goal: Integrate multi-domain insights into cohesive strategic analysis

Background: You are an expert strategic analyst who excels at synthesizing
complex information from multiple domains. You create comprehensive reports
that weave together technical, regulatory, economic, and domain-specific 
insights into actionable strategic recommendations.

Task: Create comprehensive strategic analysis of "{topic}" by synthesizing insights from domain, technical, regulatory, and economic analyses.

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
        if LLAMAINDEX_WORKING:
            from llama_index.core import Settings
            response = Settings.llm.complete(prompt)
            return str(response)
        else:
            return f"LlamaIndex not functional - synthesis unavailable for: {topic}"
    except Exception as e:
        return f"Synthesis failed: {str(e)}"

# Create the appropriate implementation based on what's working
if LLAMAINDEX_WORKING:
    print("✅ Creating working LlamaIndex implementation")
    
    # Import what we need
    from llama_index.core import Settings
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    
    # Create tools
    healthcare_tool = FunctionTool.from_defaults(
        fn=lambda topic, context="": healthcare_analysis_tool(topic, context), 
        name="healthcare_analysis"
    )
    technical_tool = FunctionTool.from_defaults(
        fn=lambda topic, context="": technical_analysis_tool(topic, context), 
        name="technical_analysis"
    )
    regulatory_tool = FunctionTool.from_defaults(
        fn=lambda topic, context="": regulatory_analysis_tool(topic, context), 
        name="regulatory_analysis"
    )
    economic_tool = FunctionTool.from_defaults(
        fn=lambda topic, context="": economic_analysis_tool(topic, context), 
        name="economic_analysis"
    )
    
    class LlamaIndexRunner:
        """Standardized LlamaIndex implementation matching CrewAI baseline"""
        
        def __init__(self):
            print("🔧 Initializing LlamaIndexRunner...")
            self.initialization_log = []
            
            def log_init(message):
                print(message)
                self.initialization_log.append(message)
            
            log_init("✅ LlamaIndex runner initialized")
        
        def run_agent_analysis(self, agent_name: str, topic: str, context: str = "") -> Dict[str, Any]:
            """Run agent analysis matching CrewAI workflow"""
            print(f"🔍 Running {agent_name} analysis...")
            start_time = time.time()
            
            try:
                if agent_name == "healthcare":
                    analysis_text = healthcare_analysis_tool(topic, context)
                elif agent_name == "technical":
                    analysis_text = technical_analysis_tool(topic, context)
                elif agent_name == "regulatory":
                    analysis_text = regulatory_analysis_tool(topic, context)
                elif agent_name == "economic":
                    analysis_text = economic_analysis_tool(topic, context)
                elif agent_name == "synthesizer":
                    # Extract individual analyses from context
                    parts = context.split("|||") if "|||" in context else ["", "", "", ""]
                    healthcare_analysis = parts[0] if len(parts) > 0 else ""
                    technical_analysis = parts[1] if len(parts) > 1 else ""
                    regulatory_analysis = parts[2] if len(parts) > 2 else ""
                    economic_analysis = parts[3] if len(parts) > 3 else ""
                    
                    analysis_text = synthesis_tool(
                        topic, healthcare_analysis, technical_analysis, regulatory_analysis, economic_analysis
                    )
                else:
                    analysis_text = f"Unknown agent type: {agent_name}"
                
                duration = time.time() - start_time
                word_count = len(analysis_text.split())
                
                print(f"✅ {agent_name} analysis completed in {duration:.1f}s ({word_count} words)")
                
                return {
                    "agent": agent_name,
                    "analysis": analysis_text,
                    "success": True,
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
            """Run comprehensive analysis matching CrewAI baseline exactly"""
            print(f"🦙 Starting LlamaIndex analysis: {topic}")
            print("=" * 70)
            
            start_time = time.time()
            individual_analyses = {}
            results = []
            analysis_log = []
            
            analysis_log.extend(self.initialization_log)
            analysis_log.append(f"🔍 Starting standardized analysis matching CrewAI baseline")
            
            # Phase 1: Healthcare Analysis
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
            
            # Phase 2: Technical Analysis (with context)
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
            
            # Phase 3: Regulatory Analysis (with context)
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
            
            # Phase 4: Economic Analysis (with context)
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
            
            # Phase 5: Strategic Synthesis (with all analyses)
            print("🎯 Phase 5/5: Strategic Content Synthesizer")
            analysis_log.append("🎯 Phase 5/5: Strategic Content Synthesizer")
            # Pass all analyses as structured context for synthesis
            synthesis_context = f"{healthcare_analysis}|||{technical_analysis}|||{regulatory_analysis}|||{economic_analysis}"
            synthesis_result = self.run_agent_analysis("synthesizer", topic, synthesis_context)
            results.append(synthesis_result)
            
            if synthesis_result["success"]:
                final_synthesis = synthesis_result["analysis"]
                # Don't add synthesis to individual_analyses to avoid duplication
            else:
                final_synthesis = f"Synthesis failed: {synthesis_result['analysis']}"
            
            total_duration = time.time() - start_time
            successful_agents = sum(1 for r in results if r["success"])
            # Calculate words from individual analyses + synthesis separately
            individual_words = sum(r.get("word_count", 0) for r in results[:-1] if r["success"])  # Exclude synthesis
            synthesis_words = len(final_synthesis.split()) if 'failed' not in final_synthesis.lower() else 0
            total_words = individual_words + synthesis_words
            
            analysis_log.append(f"🎉 Analysis complete! Total time: {total_duration:.1f}s")
            
            print(f"\n⏱️ LlamaIndex analysis completed in {total_duration:.1f} seconds")
            print(f"📝 Generated {total_words:,} words ({total_words/total_duration:.1f} words/second)")
            print(f"✅ {successful_agents}/{len(results)} agents completed successfully")
            print("=" * 70)
            
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
                "success": successful_agents >= 4
            }
    
    agent_runner = LlamaIndexRunner()

else:
    print("❌ Creating fallback LlamaIndex implementation")
    
    class FallbackLlamaIndexRunner:
        """Fallback when LlamaIndex doesn't work"""
        
        def run_comprehensive_analysis(self, topic: str) -> Dict[str, Any]:
            return {
                "topic": topic,
                "framework": "llamaindex",
                "total_duration": 0,
                "total_words": 0,
                "words_per_second": 0,
                "successful_agents": 0,
                "total_agents": 5,
                "individual_analyses": {
                    "healthcare": "LlamaIndex not functional - check setup",
                    "technical": "LlamaIndex not functional - check setup",
                    "regulatory": "LlamaIndex not functional - check setup",
                    "economic": "LlamaIndex not functional - check setup"
                },
                "final_synthesis": "LlamaIndex framework not functional",
                "timestamp": datetime.now().isoformat(),
                "diagnostic_info": DIAGNOSTIC_INFO,
                "success": False
            }
    
    agent_runner = FallbackLlamaIndexRunner()

# EXECUTION FUNCTIONS
# =================================

def run_llamaindex_analysis_main(topic: str):
    """Main LlamaIndex analysis function - no recursion"""
    debug_log = []
    
    def log_debug(message):
        """Log debug message to both console and return list"""
        print(message)
        debug_log.append(message)
    
    log_debug(f"🦙 Starting LlamaIndex analysis for: {topic}")
    log_debug(f"🔍 LlamaIndex working status: {LLAMAINDEX_WORKING}")
    log_debug(f"🔍 Agent runner type: {type(agent_runner).__name__}")
    
    try:
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

# Export functions for compatibility - FIXED VERSION NO RECURSION
def run_llamaindex_analysis(topic: str):
    """External interface function - calls the main implementation"""
    return run_llamaindex_analysis_main(topic)

def display_llamaindex_results(result: Dict[str, Any]):
    """Display LlamaIndex results"""
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
    """Save LlamaIndex results"""
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

# Module status
if LLAMAINDEX_WORKING:
    print("✅ LlamaIndex module ready and working")
else:
    print("❌ LlamaIndex module loaded in fallback mode")
    print("   Check diagnostic info for details on what's not working")

print(f"🦙 LlamaIndex diagnostic info: {DIAGNOSTIC_INFO}")

# REMOVED THE DUPLICATE FUNCTION DEFINITION THAT WAS CAUSING RECURSION