#!/usr/bin/env python3
"""
Real Model Test - Using Gemini models with real Exa and Tavily tools

Tests tool selection equivalence with actual LLM models and real search tools:
- Draft Model: gemini-2.0-flash-lite (faster, cheaper)
- Target Model: gemini-2.5-pro (more capable, expensive)

Both models have access to the same real tools (Exa, Tavily, etc.).
Tests if draft model chooses similar tools as target, using embedding fallback for verification.
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
from acceptance_metrics import calculate_acceptance_rates, format_acceptance_report

# Import tools registry
from tools_registry import get_all_available_tools


# ==================== Agent Creation ====================

def create_draft_agent(tools: List):
    """Create draft agent (faster model)"""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    agent = create_react_agent(model, tools)
    return agent


def create_target_agent(tools: List):
    """Create target agent (more capable model)"""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    agent = create_react_agent(model, tools)
    return agent


# ==================== Test Cases ====================

TEST_QUERIES = [
    {
        "id": "search_test_1",
        "query": "What are the latest developments in quantum computing?",
        "description": "Web search query - should use Exa or Tavily"
    },
    {
        "id": "search_test_2",
        "query": "Find recent news about artificial intelligence safety",
        "description": "News search query"
    },
    {
        "id": "calculation_test",
        "query": "Calculate 127 multiplied by 34",
        "description": "Simple calculation"
    },
    {
        "id": "research_test",
        "query": "Research the impact of climate change on ocean temperatures and summarize the findings",
        "description": "Research task requiring search + summarization"
    },
    {
        "id": "comparison_test",
        "query": "Compare the features of React and Vue.js frameworks",
        "description": "Comparison requiring multiple searches"
    },
]


# ==================== Composition Mappings ====================

def get_composition_mappings(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate composition mappings for all tools seen in results.
    Each tool maps to itself for basic equivalence checking.
    """
    mappings = {}

    # Get unique tool names from tool calls
    tool_names = set()
    for tc in tool_calls:
        tool_names.add(tc.get("name", ""))

    # Create self-mapping for each tool
    for tool_name in tool_names:
        if tool_name:
            mappings[tool_name] = {
                "components": [tool_name],
                "parameter_mapping": {},
                "allow_reordering": False,
                "allow_extra_tools": False
            }

    # Add standard tool mappings (in case they're used)
    standard_tools = [
        "exa_search", "tavily_search_results_json",
        "arxiv_search", "asknews", "brave_search",
        "duckduckgo_search", "google_serper", "searxng_search", "you_search",
        "calculator_tool", "extract_info_tool", "summarize_tool"
    ]

    for tool_name in standard_tools:
        if tool_name not in mappings:
            mappings[tool_name] = {
                "components": [tool_name],
                "parameter_mapping": {},
                "allow_reordering": False,
                "allow_extra_tools": False
            }

    return mappings


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


def run_test(test_case: Dict[str, Any], draft_agent, target_agent, use_fallback: bool = False):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_case['id']}")
    print(f"Query: {test_case['query']}")
    print(f"{'='*80}")

    system_msg = SystemMessage(content="""You are a helpful assistant. Use the available tools to answer the user's question accurately and efficiently.""")

    print(f"\n🔹 Running DRAFT model (gemini-2.0-flash-lite)...")
    draft_start = time.time()
    try:
        draft_result = draft_agent.invoke({"messages": [system_msg, ("user", test_case["query"])]})
        draft_time = time.time() - draft_start
        draft_tools = extract_tool_calls(draft_result)
        print(f"   ⏱️  Time: {draft_time:.2f}s")
        print(f"   🔧 Tools called: {[t['name'] for t in draft_tools]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    print(f"\n🔹 Running TARGET model (gemini-2.5-pro)...")
    target_start = time.time()
    try:
        target_result = target_agent.invoke({"messages": [system_msg, ("user", test_case["query"])]})
        target_time = time.time() - target_start
        target_tools = extract_tool_calls(target_result)
        print(f"   ⏱️  Time: {target_time:.2f}s")
        print(f"   🔧 Tools called: {[t['name'] for t in target_tools]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    # Get first target tool (for comparison)
    target_tool = target_tools[0] if target_tools else {"name": "none", "args": {}}

    # Generate composition mappings based on actual tools used
    composition_mappings = get_composition_mappings(draft_tools + target_tools)

    # Run verification
    print(f"\n🔹 Running VERIFICATION (fallback={'enabled' if use_fallback else 'disabled'})...")
    validation_result = speculation_checker(
        draft_result=draft_tools,
        target_result=target_tool,
        composition_mapping=composition_mappings,
        check_params=True,
        check_semantics=True,
        use_embedding_fallback=use_fallback,
        embedding_threshold=0.5,
        embedding_method="gemini"
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"✓ Valid: {validation_result['valid']}")
    print(f"✓ Verified By: {validation_result['verified_by']}")
    print(f"✓ Parameter Equivalent: {validation_result['param_equivalent']}")
    print(f"✓ Semantic Equivalent: {validation_result['semantic_equivalent']}")

    if validation_result.get('details', {}).get('embedding_check'):
        emb = validation_result['details']['embedding_check']
        if 'similarity_score' in emb:
            print(f"✓ Embedding Similarity: {emb['similarity_score']:.3f}")

    if validation_result['error']:
        print(f"\n⚠️  Errors:")
        for error in validation_result['error'][:3]:  # Show first 3 errors
            print(f"   - {error}")

    return {
        "test_id": test_case['id'],
        "query": test_case['query'],
        "draft_tools": [t['name'] for t in draft_tools],
        "target_tool": target_tool['name'],
        "draft_time": draft_time,
        "target_time": target_time,
        "validation": validation_result,
        "verified_by": validation_result['verified_by']
    }


def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("REAL MODEL TEST - WITH REAL EXA AND TAVILY TOOLS")
    print("="*80)
    print(f"Draft Model: gemini-2.0-flash-lite")
    print(f"Target Model: gemini-2.5-pro")
    print(f"Tools: Real Exa, Tavily, and utility tools")
    print("="*80)

    # Check API keys
    missing_keys = []
    if not os.environ.get("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.environ.get("EXA_API_KEY"):
        missing_keys.append("EXA_API_KEY (optional)")
    if not os.environ.get("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY (optional)")

    if "GOOGLE_API_KEY" in missing_keys:
        print("\n❌ ERROR: GOOGLE_API_KEY not set!")
        print("Please set it in your environment:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    if len(missing_keys) > 1:
        print(f"\n⚠️  Warning: Missing API keys: {', '.join(missing_keys[1:])}")
        print("Some tools may not be available.")

    # Get tools using registry
    tools = get_all_available_tools(verbose=True)

    if len(tools) == 0:
        print("❌ No tools available! Cannot run tests.")
        return

    # Create agents
    print("\n📋 Creating agents...")
    draft_agent = create_draft_agent(tools)
    target_agent = create_target_agent(tools)
    print("✅ Agents created successfully")

    # Run tests in two modes: strict only, and with fallback
    print("\n" + "="*80)
    print("MODE 1: STRICT AST VERIFICATION ONLY")
    print("="*80)

    strict_results = []
    for test_case in TEST_QUERIES:
        try:
            result = run_test(test_case, draft_agent, target_agent, use_fallback=False)
            if result:
                strict_results.append(result)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\n❌ Test failed: {e}")

    print("\n" + "="*80)
    print("MODE 2: WITH EMBEDDING FALLBACK")
    print("="*80)

    fallback_results = []
    for test_case in TEST_QUERIES:
        try:
            result = run_test(test_case, draft_agent, target_agent, use_fallback=True)
            if result:
                fallback_results.append(result)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\n❌ Test failed: {e}")

    # Calculate and display acceptance rates
    print("\n" + "="*80)
    print("ACCEPTANCE RATE COMPARISON")
    print("="*80)

    print("\nMODE 1 - STRICT ONLY:")
    strict_metrics = calculate_acceptance_rates([r['validation'] for r in strict_results])
    print(f"  Passed: {strict_metrics['strict_only_passed']}/{strict_metrics['total_cases']} ({strict_metrics['strict_only_rate']:.1f}%)")

    print("\nMODE 2 - WITH FALLBACK:")
    fallback_metrics = calculate_acceptance_rates([r['validation'] for r in fallback_results])
    print(f"  Passed: {fallback_metrics['with_fallback_passed']}/{fallback_metrics['total_cases']} ({fallback_metrics['with_fallback_rate']:.1f}%)")
    print(f"  Fallback Used: {fallback_metrics['fallback_used']} times ({fallback_metrics['fallback_usage_rate']:.1f}%)")
    print(f"  Improvement: +{fallback_metrics['improvement']:.1f} percentage points")

    # Save results
    output = {
        "strict_mode": {
            "results": strict_results,
            "metrics": strict_metrics
        },
        "fallback_mode": {
            "results": fallback_results,
            "metrics": fallback_metrics
        },
        "comparison": {
            "improvement": fallback_metrics['improvement'],
            "fallback_effectiveness": fallback_metrics['fallback_usage_rate']
        }
    }

    output_file = "real_model_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
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
