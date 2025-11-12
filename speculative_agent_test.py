import sys
import os
sys.path.insert(0, 'speculation_eval')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
import time
import json
from typing import List, Dict, Any
import warnings
import pandas as pd
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix for langchain version compatibility
import langchain
if not hasattr(langchain, 'verbose'):
    langchain.verbose = False
if not hasattr(langchain, 'debug'):
    langchain.debug = False
if not hasattr(langchain, 'llm_cache'):
    langchain.llm_cache = None

# Import speculation evaluation system
# NOTE: Old complex speculation_checker removed - use speculative_pipeline.py instead
# which has simple AST + embedding verification (no composition mapping)
try:
    from speculation_eval.acceptance_metrics import calculate_acceptance_rates
    from speculation_eval.tools_registry import get_all_available_tools
    EVAL_AVAILABLE = False  # Disable old verification - use new speculative_pipeline.py
except ImportError:
    EVAL_AVAILABLE = False
    print("Warning: Could not import speculation_eval modules")


# ==================== Create Agents ====================

def create_draft_agent(tools):
    """Draft agent - OpenAI GPT-5-nano"""
    model = ChatOpenAI(
        model="gpt-5-nano",
        reasoning_effort="low",
        # API key from environment variable OPENAI_API_KEY
    )
    return create_react_agent(model, tools)


def create_target_agent(tools):
    """Target agent - Gemini 2.5 Flash Lite"""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    return create_react_agent(model, tools)


# ==================== Load Test Queries ====================

def load_test_queries(
    dataset: str = "browsecomp",
    n_samples: int = 100
) -> List[str]:
    """
    Load random queries from the specified dataset.

    Args:
        dataset: Either "browsecomp" or "hotpot"
        n_samples: Number of random samples to load

    Returns:
        List of query strings
    """
    # Dataset configurations
    DATASETS = {
        "browsecomp": {
            "path": "browsecomp_decrypted.csv",
            "column": "problem"
        },
        "hotpot": {
            "path": "hotpot_dev_distractor.csv",
            "column": "question"
        }
    }

    if dataset not in DATASETS:
        print(f"❌ Error: Unknown dataset '{dataset}'. Choose 'browsecomp' or 'hotpot'")
        dataset = "browsecomp"

    config = DATASETS[dataset]
    csv_path = config["path"]
    column_name = config["column"]

    try:
        df = pd.read_csv(csv_path)
        if column_name not in df.columns:
            raise ValueError(f"CSV file must have a '{column_name}' column. Found columns: {df.columns.tolist()}")

        # Sample random queries
        sample_size = min(n_samples, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)  # Fixed seed for reproducibility
        queries = sampled_df[column_name].tolist()

        print(f"✅ Loaded {len(queries)} random queries from {dataset.upper()} dataset ({csv_path})")
        return queries
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_path}")
        print("Using fallback queries instead.")
        return [
            "What are the latest developments in artificial intelligence?",
            "Find information about climate change effects",
            "Search for recent breakthroughs in quantum computing",
        ]
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        print("Using fallback queries instead.")
        return [
            "What are the latest developments in artificial intelligence?",
            "Find information about climate change effects",
            "Search for recent breakthroughs in quantum computing",
        ]


# Configure dataset here: "browsecomp" or "hotpot"
DATASET = os.environ.get("DATASET", "browsecomp")  # Can be set via environment variable
TEST_QUERIES = load_test_queries(dataset=DATASET, n_samples=100)


# ==================== Test Execution ====================

def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from agent result"""
    tool_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.get('name', ''),
                    "args": tc.get('args', {}),
                    "id": tc.get('id', '')
                })
    return tool_calls


def run_test(query: str, draft_agent, target_agent, use_fallback: bool, verbose: bool = False):
    """Run a single test"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Fallback: {'ENABLED' if use_fallback else 'DISABLED'}")
        print(f"{'='*80}")

    system_msg = SystemMessage(content="You are a helpful assistant. Use tools to answer accurately.")

    # Draft model
    if verbose:
        print("Running draft (gpt-5-nano)...")
    try:
        draft_result = draft_agent.invoke({"messages": [system_msg, ("user", query)]})
        draft_tools = extract_tool_calls(draft_result)
        if verbose:
            print(f"  Draft tools: {[t['name'] for t in draft_tools]}")
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None

    # Target model
    if verbose:
        print("Running target (gemini-2.5-flash-lite)...")
    try:
        target_result = target_agent.invoke({"messages": [system_msg, ("user", query)]})
        target_tools = extract_tool_calls(target_result)
        if verbose:
            print(f"  Target tools: {[t['name'] for t in target_tools]}")
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None

    if not target_tools:
        if verbose:
            print("  No target tools to compare")
        return None

    # Verify with timing (simple AST + embedding fallback, no composition mapping)
    if EVAL_AVAILABLE:
        # Measure verification time
        verification_start = time.time()
        validation = speculation_checker(
            draft_result=draft_tools,
            target_result=target_tools[0] if target_tools else {},
            composition_mapping={},  # Not used - direct comparison only
            check_params=False,  # Skip complex parameter mapping
            check_semantics=False,  # Skip complex semantic checks
            use_embedding_fallback=use_fallback,
            embedding_threshold=0.5,
            embedding_method="gemini",
            verbose_embedding=verbose
        )
        verification_time = time.time() - verification_start

        if verbose:
            print(f"\n  ✓ Valid: {validation['valid']}")
            print(f"  ✓ Verified by: {validation['verified_by']}")
            print(f"  ✓ Verification time: {verification_time*1000:.2f}ms")

            if validation.get('details', {}).get('embedding_check'):
                emb = validation['details']['embedding_check']
                if 'similarity_score' in emb:
                    print(f"  ✓ Embedding similarity: {emb['similarity_score']:.4f} (threshold: {emb.get('threshold', 0.5)})")

        return {
            "query": query,
            "draft_tools": draft_tools,
            "target_tools": target_tools,
            "validation": validation,
            "verified_by": validation['verified_by'],
            "verification_time": verification_time,
            "draft_model": "gpt-5-nano",
            "target_model": "gemini-2.5-flash-lite"
        }

    return None


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SPECULATIVE AGENT TEST - ACCEPTANCE RATE COMPARISON")
    print(f"Dataset: {DATASET.upper()} ({len(TEST_QUERIES)} queries)")
    print("="*80)

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set!")
        return

    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not set!")
        return

    if not EVAL_AVAILABLE:
        print("❌ ERROR: speculation_eval module not available!")
        return

    # Get tools using registry
    tools = get_all_available_tools(verbose=True)
    if not tools:
        print("❌ No tools available!")
        return

    # Create agents
    print("\nCreating agents...")
    draft_agent = create_draft_agent(tools)
    target_agent = create_target_agent(tools)

    # Run tests once with fallback enabled
    print("\n" + "="*80)
    print("RUNNING TESTS WITH FALLBACK VERIFICATION")
    print("="*80)
    results = []
    for query in tqdm(TEST_QUERIES, desc="Processing queries", unit="query"):
        result = run_test(query, draft_agent, target_agent, use_fallback=True, verbose=False)
        if result:
            results.append(result)
        time.sleep(0.5)

    # Analyze results - count by verification method
    print("\n" + "="*80)
    print("ACCEPTANCE RATE ANALYSIS")
    print("="*80)

    total = len(results)
    strict_only_passed = sum(1 for r in results if r['verified_by'] == 'strict_ast')
    fallback_passed = sum(1 for r in results if r['verified_by'] == 'embedding_fallback')
    both_failed = sum(1 for r in results if r['verified_by'] == 'none')

    # Acceptance rates
    without_fallback = strict_only_passed
    with_fallback = strict_only_passed + fallback_passed

    without_fallback_rate = (without_fallback / total * 100) if total > 0 else 0
    with_fallback_rate = (with_fallback / total * 100) if total > 0 else 0
    improvement = with_fallback_rate - without_fallback_rate

    print(f"\nTotal queries processed: {total}")
    print(f"\nVerification breakdown:")
    print(f"  ✅ Passed strict AST:        {strict_only_passed:3d} ({strict_only_passed/total*100:.1f}%)")
    print(f"  🔄 Required fallback:        {fallback_passed:3d} ({fallback_passed/total*100:.1f}%)")
    print(f"  ❌ Failed both:              {both_failed:3d} ({both_failed/total*100:.1f}%)")

    print(f"\nAcceptance rates:")
    print(f"  Without fallback: {without_fallback:3d}/{total} passed ({without_fallback_rate:.1f}%)")
    print(f"  With fallback:    {with_fallback:3d}/{total} passed ({with_fallback_rate:.1f}%)")
    print(f"  Improvement:      +{improvement:.1f} percentage points")
    print(f"\nFallback usage: {fallback_passed} times ({fallback_passed/total*100:.1f}% of queries)")

    # Verification overhead analysis
    verification_times = [r['verification_time'] for r in results]
    total_verification_time = sum(verification_times)
    avg_verification_time = total_verification_time / total if total > 0 else 0

    # Break down by verification method
    strict_times = [r['verification_time'] for r in results if r['verified_by'] == 'strict_ast']
    fallback_times = [r['verification_time'] for r in results if r['verified_by'] == 'embedding_fallback']

    avg_strict_time = sum(strict_times) / len(strict_times) if strict_times else 0
    avg_fallback_time = sum(fallback_times) / len(fallback_times) if fallback_times else 0

    print(f"\n{'='*80}")
    print("VERIFICATION OVERHEAD ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal verification time: {total_verification_time:.2f}s")
    print(f"Average per query:       {avg_verification_time*1000:.2f}ms")
    print(f"\nBreakdown by method:")
    print(f"  Strict AST only:       {avg_strict_time*1000:.2f}ms average ({len(strict_times)} queries)")
    print(f"  With fallback:         {avg_fallback_time*1000:.2f}ms average ({len(fallback_times)} queries)")
    if fallback_times and strict_times:
        overhead = avg_fallback_time - avg_strict_time
        print(f"  Fallback overhead:     +{overhead*1000:.2f}ms ({overhead/avg_strict_time*100:.1f}% increase)")

    # Save detailed results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"detailed_results_{DATASET}_{timestamp}.json")

    # Prepare results for JSON serialization
    detailed_results = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": DATASET,
            "total_queries": total,
            "draft_model": "gpt-5-nano",
            "target_model": "gemini-2.5-flash-lite",
            "embedding_method": "gemini-embedding-001",
            "embedding_threshold": 0.5
        },
        "summary": {
            "strict_only_passed": strict_only_passed,
            "fallback_passed": fallback_passed,
            "both_failed": both_failed,
            "without_fallback_rate": without_fallback_rate,
            "with_fallback_rate": with_fallback_rate,
            "improvement": improvement,
            "total_verification_time": total_verification_time,
            "avg_verification_time": avg_verification_time,
            "avg_strict_time": avg_strict_time,
            "avg_fallback_time": avg_fallback_time
        },
        "queries": results
    }

    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_file}")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
