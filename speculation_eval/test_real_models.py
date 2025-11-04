#!/usr/bin/env python3
"""
Real Model Test - Using Gemini models for draft and target

Tests tool selection equivalence with actual LLM models:
- Draft Model: gemini-2.0-flash-lite (has access to ALL tools)
- Target Model: gemini-2.5-pro (has access to ALL tools)

Both models have the same tools. We test if they choose equivalent tools
for the same query, using the target model as ground truth.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
import time
import json
import os
from typing import List, Dict, Any
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix for langchain version compatibility
import langchain
if not hasattr(langchain, 'verbose'):
    langchain.verbose = False
if not hasattr(langchain, 'debug'):
    langchain.debug = False
if not hasattr(langchain, 'llm_cache'):
    langchain.llm_cache = None

# Import our evaluation system
from speculation_checker import speculation_checker
from utils import parse_tool_calls_from_agent_output


# ==================== Simple/Atomic Tools ====================

@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query string
    """
    time.sleep(0.1)
    return f"Search results for '{query}': Found information about AI agents, tool calling, and speculation techniques."


@tool
def extract_key_points(text: str) -> str:
    """Extract key points from text.

    Args:
        text: Text to extract key points from
    """
    time.sleep(0.1)
    return "Key points: 1) Draft models use simple tools, 2) Target models use complex tools, 3) Verification via AST"


@tool
def verify_facts(claim: str) -> str:
    """Verify factual claims.

    Args:
        claim: Claim to verify
    """
    time.sleep(0.1)
    return f"Verified: The claim '{claim}' is supported by evidence."


@tool
def summarize_text(text: str) -> str:
    """Summarize text concisely.

    Args:
        text: Text to summarize
    """
    time.sleep(0.1)
    return "Summary: Agent tool call speculation uses draft and target models to optimize tool usage."


@tool
def calculate(expression: str) -> str:
    """Perform calculations.

    Args:
        expression: Mathematical expression to evaluate
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City or location name
    """
    time.sleep(0.1)
    import random
    temps = [18, 20, 22, 24, 26]
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]
    return f"Weather in {location}: {random.choice(conditions)}, {random.choice(temps)}°C"


@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight.

    Args:
        origin: Departure city
        destination: Arrival city
        date: Travel date
    """
    time.sleep(0.1)
    return f"Flight booked from {origin} to {destination} on {date}"


# ==================== Complex/Composite Tools ====================

@tool
def deep_research(topic: str) -> str:
    """Perform comprehensive research on a topic.

    This is equivalent to calling: search_web → extract_key_points → verify_facts → summarize_text

    Args:
        topic: Research topic
    """
    time.sleep(0.2)
    return f"Comprehensive research report on '{topic}': Includes search results, key points, verified facts, and summary."


@tool
def analyze_and_report(data: str) -> str:
    """Analyze data and generate report.

    Equivalent to: extract_key_points → calculate → summarize_text

    Args:
        data: Data to analyze
    """
    time.sleep(0.2)
    return f"Analysis report for '{data}': Extracted metrics, performed calculations, generated summary."


@tool
def plan_trip(destination: str, date: str) -> str:
    """Plan a complete trip including weather check, flight booking, and itinerary.

    Equivalent to: get_weather → book_flight → summarize_text

    Args:
        destination: Destination city
        date: Travel date
    """
    time.sleep(0.2)
    return f"Trip planned to {destination} on {date}: Checked weather, booked flight, created itinerary."


# ==================== Agent Creation ====================

def create_draft_agent():
    """Create draft agent with ALL tools (same as target)"""
    all_tools = [
        search_web,
        extract_key_points,
        verify_facts,
        summarize_text,
        calculate,
        get_weather,
        book_flight,
        deep_research,
        analyze_and_report,
        plan_trip,
    ]

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    agent = create_react_agent(model, all_tools)
    return agent


def create_target_agent():
    """Create target agent with all tools (simple + complex)"""
    all_tools = [
        search_web,
        extract_key_points,
        verify_facts,
        summarize_text,
        calculate,
        get_weather,
        book_flight,
        deep_research,
        analyze_and_report,
        plan_trip,
    ]

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    agent = create_react_agent(model, all_tools)
    return agent


# ==================== Test Cases ====================

TEST_QUERIES = [
    {
        "id": "calculation_test",
        "query": "What is 127 multiplied by 34?",
        "expected_target_tool": "calculate",
        "expected_draft_sequence": ["calculate"],
        "description": "Simple calculation - should just use calculate"
    },
    {
        "id": "weather_test",
        "query": "What's the weather like in Tokyo?",
        "expected_target_tool": "get_weather",
        "expected_draft_sequence": ["get_weather"],
        "description": "Simple weather query"
    },
    {
        "id": "research_test",
        "query": "Research the history of artificial intelligence and provide a comprehensive summary",
        "expected_target_tool": "deep_research",
        "expected_draft_sequence": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
        "description": "Research task that should use deep_research or sequence of simple tools"
    },
    {
        "id": "trip_planning_test",
        "query": "I want to travel to Paris next month. Check the weather, book a flight from New York, and create a summary",
        "expected_target_tool": "plan_trip",
        "expected_draft_sequence": ["get_weather", "book_flight", "summarize_text"],
        "description": "Trip planning that could use plan_trip or sequence"
    },
    {
        "id": "data_analysis_test",
        "query": "Analyze this data: accuracy 95%, latency 120ms, throughput 1000 req/s. Calculate the efficiency and summarize findings",
        "expected_target_tool": "analyze_and_report",
        "expected_draft_sequence": ["extract_key_points", "calculate", "summarize_text"],
        "description": "Data analysis task"
    },
]


# ==================== Composition Mappings ====================

COMPOSITION_MAPPINGS = {
    "calculate": {
        "components": ["calculate"],
        "parameter_mapping": {},
        "allow_reordering": False,
        "allow_extra_tools": False
    },
    "get_weather": {
        "components": ["get_weather"],
        "parameter_mapping": {},
        "allow_reordering": False,
        "allow_extra_tools": False
    },
    "deep_research": {
        "components": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
        "parameter_mapping": {"topic": "query"},
        "allow_reordering": False,
        "allow_extra_tools": False
    },
    "plan_trip": {
        "components": ["get_weather", "book_flight", "summarize_text"],
        "parameter_mapping": {"destination": "location", "date": "date"},
        "allow_reordering": False,
        "allow_extra_tools": False
    },
    "analyze_and_report": {
        "components": ["extract_key_points", "calculate", "summarize_text"],
        "parameter_mapping": {"data": "text"},
        "allow_reordering": False,
        "allow_extra_tools": False
    }
}


# ==================== Test Execution ====================

def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from agent result"""
    tool_calls = []
    messages = result.get("messages", [])

    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.get('name', ''),
                    "args": tc.get('args', {}),
                    "id": tc.get('id', '')
                })

    return tool_calls


def run_test(test_case: Dict[str, Any], draft_agent, target_agent):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_case['id']}")
    print(f"Query: {test_case['query']}")
    print(f"{'='*80}")

    # Draft model - same tools as target, less guidance on which to prefer
    draft_system_msg = SystemMessage(content="""You are a helpful assistant. Use the available tools to complete the user's request efficiently.

You have access to both simple and complex tools:
- Simple tools: search_web, extract_key_points, verify_facts, summarize_text, calculate, get_weather, book_flight
- Complex tools: deep_research, analyze_and_report, plan_trip

Choose the most appropriate tools for the task.""")

    print("\n🔹 Running DRAFT model (gemini-2.0-flash-lite with ALL tools)...")
    draft_start = time.time()
    draft_result = draft_agent.invoke({"messages": [draft_system_msg, ("user", test_case["query"])]})
    draft_time = time.time() - draft_start
    draft_tools = extract_tool_calls(draft_result)

    print(f"   ⏱️  Time: {draft_time:.2f}s")
    print(f"   🔧 Tools called: {[t['name'] for t in draft_tools]}")

    # Target model with similar prompt (considered ground truth)
    target_system_msg = SystemMessage(content="""You are an efficient assistant with both simple and complex tools.

You have access to both simple and complex tools:
- Simple tools: search_web, extract_key_points, verify_facts, summarize_text, calculate, get_weather, book_flight
- Complex tools: deep_research, analyze_and_report, plan_trip

Complex tools are more efficient as they combine multiple operations:
- deep_research: Performs comprehensive research (search + extract + verify + summarize)
- analyze_and_report: Analyzes data and generates reports (extract + calculate + summarize)
- plan_trip: Plans complete trips (weather + booking + summary)

For complex tasks, prefer using the composite tools.""")

    print("\n🔹 Running TARGET model (gemini-2.5-pro with ALL tools - GROUND TRUTH)...")
    target_start = time.time()
    target_result = target_agent.invoke({"messages": [target_system_msg, ("user", test_case["query"])]})
    target_time = time.time() - target_start
    target_tools = extract_tool_calls(target_result)

    print(f"   ⏱️  Time: {target_time:.2f}s")
    print(f"   🔧 Tools called: {[t['name'] for t in target_tools]}")

    # Get first target tool (ground truth)
    target_tool = target_tools[0] if target_tools else {"name": "none", "args": {}}

    # Run AST validation to check if draft's tool choice is equivalent to target's
    # Equivalence means either:
    # 1. Exact match: draft calls same tool as target
    # 2. Composition match: draft's tool sequence composes into target's complex tool
    print("\n🔹 Running AST VALIDATION (checking if draft ≈ target)...")
    validation_result = speculation_checker(
        draft_result=draft_tools,
        target_result=target_tool,
        composition_mapping=COMPOSITION_MAPPINGS,
        check_params=True,
        check_semantics=True
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS: Did draft choose equivalent tool(s) as target?")
    print(f"{'='*80}")
    print(f"✓ Overall Valid: {validation_result['valid']}")
    print(f"✓ Parameter Equivalent: {validation_result['param_equivalent']}")
    print(f"✓ Semantic Equivalent: {validation_result['semantic_equivalent']}")

    if validation_result['error']:
        print(f"\n❌ Draft's tool choice NOT equivalent to target:")
        for error in validation_result['error']:
            print(f"   - {error}")
    else:
        print(f"\n✅ Draft's tool choice IS equivalent to target (ground truth)!")

    return {
        "test_id": test_case['id'],
        "query": test_case['query'],
        "draft_tools": [t['name'] for t in draft_tools],
        "target_tool": target_tool['name'],
        "draft_time": draft_time,
        "target_time": target_time,
        "validation": validation_result,
        "expected_match": test_case.get('expected_target_tool') == target_tool['name']
    }


def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("REAL MODEL TEST - TOOL SELECTION EQUIVALENCE")
    print("="*80)
    print(f"Draft Model: gemini-2.0-flash-lite (ALL tools)")
    print(f"Target Model: gemini-2.5-pro (ALL tools - GROUND TRUTH)")
    print(f"\nGoal: Test if draft model chooses equivalent tools as target")
    print("="*80)

    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY not set!")
        print("Please set it in your environment:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    # Create agents
    print("\n📋 Creating agents...")
    draft_agent = create_draft_agent()
    target_agent = create_target_agent()
    print("✅ Agents created successfully")

    # Run all tests
    results = []
    for test_case in TEST_QUERIES:
        try:
            result = run_test(test_case, draft_agent, target_agent)
            results.append(result)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY: How often did draft match target (ground truth)?")
    print("="*80)

    total = len(results)
    valid = sum(1 for r in results if r['validation']['valid'])
    param_equiv = sum(1 for r in results if r['validation']['param_equivalent'])
    semantic_equiv = sum(1 for r in results if r['validation']['semantic_equivalent'])
    expected_match = sum(1 for r in results if r['expected_match'])

    print(f"\nTotal Tests: {total}")
    print(f"Draft ≈ Target (Both checks passed): {valid}/{total} ({valid/total*100:.0f}%)")
    print(f"Parameter Equivalent: {param_equiv}/{total} ({param_equiv/total*100:.0f}%)")
    print(f"Semantic Equivalent: {semantic_equiv}/{total} ({semantic_equiv/total*100:.0f}%)")
    print(f"Target used expected tool: {expected_match}/{total} ({expected_match/total*100:.0f}%)")

    print(f"\n📊 Key Insight: Draft model chose equivalent tools {valid}/{total} times")

    # Save results
    output_file = "real_model_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {output_file}")

    print("\n" + "="*80)
    print("✅ REAL MODEL TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
