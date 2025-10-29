from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent # type: ignore[reportDeprecated]
import time
from typing import List, Dict, Any
import warnings
import os

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix for langchain version compatibility issue
import langchain
if not hasattr(langchain, 'verbose'):
    langchain.verbose = False
if not hasattr(langchain, 'debug'):
    langchain.debug = False
if not hasattr(langchain, 'llm_cache'):
    langchain.llm_cache = None

# ==================== Simple/Atomic Tools ====================

@tool
def search_web(query: str) -> str:
    """Search the web for information. Simple atomic operation.
    
    Args:
        query: Search query string
    """
    print(f"  🔧 [SIMPLE TOOL] search_web(query='{query}')")
    time.sleep(0.3)
    
    # Simulate web search results based on query
    results = f"""
Search Results for "{query}":

1. Agent Tool Call Speculation - Research Paper (2024)
   A novel approach to optimizing agent tool usage through speculative execution.
   Key concepts: draft models, target models, tool verification.

2. Speculative Decoding for LLM Agents
   Applying speculative decoding principles to agent tool calling for improved efficiency.
   
3. Tool Call Optimization in AI Agents
   Strategies for reducing latency in multi-tool agent workflows.
"""
    return results


@tool
def extract_key_points(text: str) -> str:
    """Extract key points from text. Simple atomic operation.
    
    Args:
        text: Text to extract key points from
    """
    print(f"  🔧 [SIMPLE TOOL] extract_key_points(text='{text[:50]}...')")
    time.sleep(0.2)
    
    # Extract key information from the input
    key_points = """
Key Points Extracted:
• Agent tool call speculation uses draft and target models
• Draft model: uses simple/atomic tools with more calls
• Target model: uses complex/composite tools with fewer calls
• Similar to speculative decoding in LLM inference
• Goal: verify if draft's tool sequence matches target's complex tool
• Improves efficiency through parallel execution and verification
"""
    return key_points


@tool
def summarize_text(text: str) -> str:
    """Summarize text concisely. Simple atomic operation.
    
    Args:
        text: Text to summarize
    """
    print(f"  🔧 [SIMPLE TOOL] summarize_text(text='{text[:50]}...')")
    time.sleep(0.2)
    
    # Create a summary
    summary = """
Summary:
Agent tool call speculation is a technique inspired by speculative decoding. It uses two models:
a draft model that breaks tasks into simple tool calls and a target model that uses complex
composite tools. The system verifies if the draft's sequence is equivalent to the target's
single complex call, enabling more efficient execution while maintaining accuracy.
"""
    return summary


@tool
def verify_facts(claim: str) -> str:
    """Verify factual claims. Simple atomic operation.
    
    Args:
        claim: Claim to verify
    """
    print(f"  🔧 [SIMPLE TOOL] verify_facts(claim='{claim[:50]}...')")
    time.sleep(0.2)
    
    # Simulate fact verification
    verification = """
Fact Verification:
✓ Claim verified against multiple sources
✓ Concept is consistent with speculative decoding literature
✓ Draft-target model approach is theoretically sound
✓ Tool composition equivalence is mathematically verifiable
Confidence: High (85%)
"""
    return verification


@tool
def calculate(expression: str) -> str:
    """Perform calculations. Simple atomic operation.
    
    Args:
        expression: Mathematical expression
    """
    print(f"  🔧 [SIMPLE TOOL] calculate(expression='{expression}')")
    try:
        result = eval(expression)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Calculation Error: {e}"


# ==================== Complex/Composite Tools ====================

@tool
def deep_research(topic: str) -> str:
    """Perform deep research on a topic. Complex operation that internally:
    1. Searches multiple sources
    2. Extracts key information
    3. Verifies facts
    4. Synthesizes comprehensive summary
    
    This is equivalent to calling: search_web → extract_key_points → verify_facts → summarize_text
    
    Args:
        topic: Research topic
    """
    print(f"  🔧 [COMPLEX TOOL] deep_research(topic='{topic}')")
    print(f"      → Internally performs: search → extract → verify → summarize")
    time.sleep(0.8)  # Simulates longer execution time
    
    # Simulate comprehensive research output
    report = f"""
Deep Research Report: {topic}
"""
    return report


@tool
def analyze_and_report(data: str) -> str:
    """Analyze data and generate comprehensive report. Complex operation that:
    1. Extracts key metrics
    2. Performs calculations
    3. Summarizes findings
    
    Equivalent to: extract_key_points → calculate → summarize_text
    
    Args:
        data: Data to analyze
    """
    print(f"  🔧 [COMPLEX TOOL] analyze_and_report(data='{data[:50]}...')")
    print(f"      → Internally performs: extract → calculate → summarize")
    time.sleep(0.6)
    
    # Simulate analysis report
    report = """
Analysis Report:

KEY METRICS EXTRACTED:
• Tool call count: Draft=4, Target=1 (75% reduction)
• Execution time: Draft=4.21s, Target=2.27s (46% faster)
• Efficiency ratio: 1.86x improvement
• Verification success rate: 100%

CALCULATIONS:
• Speedup factor: 4.21 / 2.27 = 1.86x
• Tool call reduction: (4-1)/4 = 75%
• Time savings: (4.21-2.27)/4.21 = 46.1%

SUMMARY:
The target model's complex tool approach demonstrates significant efficiency gains
over the draft model's simple tool sequence. The 1.86x speedup with maintained
accuracy validates the agent tool call speculation hypothesis. The approach shows
promise for production agent systems requiring optimal performance.
"""
    return report


# ==================== Agent Creation ====================

def create_draft_agent():
    """Draft model: Only has access to simple/atomic tools"""
    simple_tools = [
        search_web,
        extract_key_points,
        summarize_text,
        verify_facts,
        calculate,
    ]
    
    print(f"\n📋 Draft Model Tools (Simple): {[t.name for t in simple_tools]}")
    
    # Initialize LLM with Gemini
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Create agent using LangGraph
    agent = create_react_agent(model, simple_tools)
    
    return agent


def create_target_agent():
    """Target model: Has access to ALL tools (simple + complex)"""
    all_tools = [
        # Simple tools
        search_web,
        extract_key_points,
        summarize_text,
        verify_facts,
        calculate,
        # Complex tools
        deep_research,
        analyze_and_report,
    ]
    
    print(f"\n📋 Target Model Tools (All): {[t.name for t in all_tools]}")
    
    # Initialize LLM with Gemini
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Create agent using LangGraph
    agent = create_react_agent(model, all_tools)
    
    return agent


# ==================== Test Function ====================

def extract_tool_calls(result: Dict[str, Any]) -> List[str]:
    """Extract tool call names from agent result"""
    tool_calls = []
    messages = result.get("messages", [])
    
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc['name'])
    
    return tool_calls


def test_speculative_agent():
    """Test case: Research task that can use either approach"""
    
    print("\n" + "="*80)
    print("SPECULATIVE AGENT EXPERIMENT")
    print("="*80)
    print("\nTest Query: 'Do deep research on agent tool call speculation'")
    print("\nHypothesis:")
    print("  - Draft Model: Will use multiple simple tools (search → extract → verify → summarize)")
    print("  - Target Model: Will use single complex tool (deep_research)")
    print("="*80)
    
    query = "Do deep research on agent tool call speculation"
    
    # ==================== Draft Model ====================
    print("\n" + "="*80)
    print("🎯 DRAFT MODEL (Simple Tools Only)")
    print("="*80)
    
    draft_agent = create_draft_agent()
    
    print(f"\n⏱️  Starting Draft Model execution...")
    draft_start = time.time()
    
    try:
        # Add system message to guide draft model to use multiple tools
        from langchain_core.messages import SystemMessage
        draft_system_msg = SystemMessage(content="""You are a thorough research assistant with access to ONLY simple, atomic tools.

IMPORTANT: To complete this research task, you MUST use tools in this exact sequence:
1. First, call search_web to find information
2. Then, call extract_key_points on the search results
3. Then, call verify_facts on the key points
4. Finally, call summarize_text to create a comprehensive summary

Do NOT skip any steps. Use ALL four tools in sequence before giving your final answer.""")
        
        draft_result = draft_agent.invoke({"messages": [draft_system_msg, ("user", query)]})
        draft_time = time.time() - draft_start
        draft_tools = extract_tool_calls(draft_result)
        
        print(f"\n✅ Draft Model completed in {draft_time:.2f}s")
        print(f"📊 Tool Calls: {' → '.join(draft_tools)}")
        print(f"📝 Tool Count: {len(draft_tools)}")
        
        # Show detailed AIMessage with tool_calls
        print(f"\n{'='*80}")
        print(f"📋 DETAILED TOOL CALL TRACE:")
        print(f"{'='*80}")
        messages = draft_result.get("messages", [])
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            print(f"\n[Message {i+1}] {msg_type}")
            
            if msg_type == "AIMessage":
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"  🤖 AI decided to call {len(msg.tool_calls)} tool(s):")
                    for j, tc in enumerate(msg.tool_calls, 1):
                        print(f"    Tool Call #{j}:")
                        print(f"      - Tool Name: {tc.get('name', 'N/A')}")
                        print(f"      - Tool ID: {tc.get('id', 'N/A')}")
                        print(f"      - Parameters: {tc.get('args', {})}")
                elif hasattr(msg, 'content') and msg.content:
                    print(f"  💬 AI Response: {msg.content[:100]}...")
            
            elif msg_type == "ToolMessage":
                if hasattr(msg, 'name'):
                    print(f"  🔧 Tool: {msg.name}")
                    print(f"  📤 Output: {msg.content[:100]}...")
        
        # Extract final answer
        final_msg = messages[-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(f"\n{'='*80}")
            print(f"💬 Final Answer:")
            print(f"{'='*80}")
            print(final_msg.content)
        
    except Exception as e:
        print(f"\n❌ Draft Model Error: {e}")
        import traceback
        traceback.print_exc()
        draft_tools = []
        draft_time = 0
    
    # ==================== Target Model ====================
    print("\n" + "="*80)
    print("🎯 TARGET MODEL (All Tools Including Complex)")
    print("="*80)
    
    target_agent = create_target_agent()
    
    print(f"\n⏱️  Starting Target Model execution...")
    target_start = time.time()
    
    try:
        # Add system message to guide target model to use complex tools
        from langchain_core.messages import SystemMessage
        target_system_msg = SystemMessage(content="""You are an efficient assistant with access to both simple and complex tools.

IMPORTANT: You have a 'deep_research' tool that is specifically designed for research tasks.
The deep_research tool internally performs search, extract, verify, and summarize in ONE efficient call.
For this research task, USE the deep_research tool - it's much more efficient than calling multiple simple tools.""")
        
        target_result = target_agent.invoke({"messages": [target_system_msg, ("user", query)]})
        target_time = time.time() - target_start
        target_tools = extract_tool_calls(target_result)
        
        print(f"\n✅ Target Model completed in {target_time:.2f}s")
        print(f"📊 Tool Calls: {' → '.join(target_tools)}")
        print(f"📝 Tool Count: {len(target_tools)}")
        
        # Extract final answer
        messages = target_result.get("messages", [])
        final_msg = messages[-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(f"\n💬 Final Answer:\n{final_msg.content}")
        
    except Exception as e:
        print(f"\n❌ Target Model Error: {e}")
        import traceback
        traceback.print_exc()
        target_tools = []
        target_time = 0


# ==================== Main ====================

if __name__ == "__main__":
    try:
        test_speculative_agent()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()