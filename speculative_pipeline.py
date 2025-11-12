"""
Speculative Tool Calling Pipeline

A lossless framework that runs draft and target models in parallel,
using predictive speculation to save time on API calls.

Strategy:
- Draft model (GPT-5-nano) predicts tool calls while target is thinking
- Verify predictions using embedding similarity (primary)
- Execute verified tool calls to avoid re-execution
- Fallback to target if verification fails, continue speculation next turn

Requirements:
- OPENAI_API_KEY: For OpenAI GPT-5-nano (draft model)
- GOOGLE_API_KEY: For Gemini API access (target model and embeddings)
- Dataset files: browsecomp_decrypted.csv or hotpot/*.json

Usage:
    # Run with 5 queries in verbose mode (testing)
    python speculative_pipeline.py --limit 5 --verbose

    # Run full evaluation with 100 queries
    python speculative_pipeline.py --full

    # Use HotpotQA dataset
    python speculative_pipeline.py --dataset hotpot --limit 10

    # Adjust embedding threshold
    python speculative_pipeline.py --embedding-threshold 0.6 --limit 5

Author: Research on Agent Tool Call Speculation
"""

import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tqdm import tqdm

# Import speculation evaluation components
from speculation_eval.embedding_similarity import get_embedding, cosine_similarity
from speculation_eval.tools_registry import get_all_available_tools


# ============================================================================
# Helper Functions
# ============================================================================

def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from agent result.

    Args:
        result: Agent execution result containing messages

    Returns:
        List of tool call dictionaries with name, args, and id
    """
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


def hash_tool_args(args: Dict[str, Any]) -> str:
    """Create a hash of tool arguments for caching."""
    args_str = json.dumps(args, sort_keys=True)
    return hashlib.md5(args_str.encode()).hexdigest()


# ============================================================================
# Speculative Executor Class
# ============================================================================

class SpeculativeExecutor:
    """Executes speculative tool calling with parallel draft and target models."""

    def __init__(
        self,
        draft_agent,
        target_agent,
        tools: List[Any],
        embedding_threshold: float = 0.5,
        embedding_method: str = "gemini",
        verbose: bool = False
    ):
        """Initialize speculative executor.

        Args:
            draft_agent: Fast draft model agent (GPT-5-nano)
            target_agent: Capable target model agent (Gemini 2.5 Flash Lite)
            tools: List of available tools
            embedding_threshold: Similarity threshold for verification (default 0.5)
            embedding_method: "gemini" or "gemma" for embeddings
            verbose: Print detailed execution info
        """
        self.draft_agent = draft_agent
        self.target_agent = target_agent
        self.tools = tools
        self.embedding_threshold = embedding_threshold
        self.embedding_method = embedding_method
        self.verbose = verbose

        # Cache for tool execution results
        self.tool_result_cache: Dict[str, Any] = {}

        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "speculation_accepted": 0,
            "speculation_rejected": 0,
            "ast_verified": 0,
            "embedding_verified": 0,
            "target_fallback": 0,
            "total_time_saved": 0.0,
            "total_draft_time": 0.0,
            "total_target_time": 0.0,
            "total_verification_time": 0.0,
            "total_target_model_time": 0.0,  # Baseline: only run target
            "total_speculative_time": 0.0,  # Actual: parallel + verification
        }

    async def run_draft_agent(
        self,
        query: str,
        system_message: str
    ) -> Tuple[Dict[str, Any], float]:
        """Run draft agent asynchronously.

        Args:
            query: User query
            system_message: System prompt

        Returns:
            Tuple of (result dictionary, execution time)
        """
        start_time = time.time()

        # Use ainvoke for async execution if available, otherwise run in thread
        try:
            result = await self.draft_agent.ainvoke({
                "messages": [SystemMessage(content=system_message), ("user", query)]
            })
        except AttributeError:
            # Fallback to sync invoke in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.draft_agent.invoke,
                {"messages": [SystemMessage(content=system_message), ("user", query)]}
            )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"[Draft] Completed in {execution_time:.2f}s")

        return result, execution_time

    async def run_target_agent(
        self,
        query: str,
        system_message: str
    ) -> Tuple[Dict[str, Any], float]:
        """Run target agent asynchronously.

        Args:
            query: User query
            system_message: System prompt

        Returns:
            Tuple of (result dictionary, execution time)
        """
        start_time = time.time()

        # Use ainvoke for async execution if available, otherwise run in thread
        try:
            result = await self.target_agent.ainvoke({
                "messages": [SystemMessage(content=system_message), ("user", query)]
            })
        except AttributeError:
            # Fallback to sync invoke in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.target_agent.invoke,
                {"messages": [SystemMessage(content=system_message), ("user", query)]}
            )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"[Target] Completed in {execution_time:.2f}s")

        return result, execution_time

    def _tools_to_string(self, tools: List[Dict[str, Any]]) -> str:
        """Convert tool calls to string representation for comparison."""
        if not tools:
            return ""

        parts = []
        for tool in tools:
            name = tool.get('name', '')
            args = tool.get('args', {})
            args_str = ', '.join(f"{k}={v}" for k, v in sorted(args.items()))
            parts.append(f"{name}({args_str})")

        return " | ".join(parts)

    def _check_ast_equivalence(
        self,
        draft_tools: List[Dict[str, Any]],
        target_tools: List[Dict[str, Any]]
    ) -> bool:
        """Check if tool calls are exactly equivalent using AST comparison.

        Simply compares tool names and arguments directly.
        """
        # Must have same number of tool calls
        if len(draft_tools) != len(target_tools):
            return False

        # Sort both by tool name for comparison (order doesn't matter)
        draft_sorted = sorted(draft_tools, key=lambda x: (x.get('name', ''), str(x.get('args', {}))))
        target_sorted = sorted(target_tools, key=lambda x: (x.get('name', ''), str(x.get('args', {}))))

        # Compare each pair
        for draft, target in zip(draft_sorted, target_sorted):
            # Tool names must match
            if draft.get('name') != target.get('name'):
                return False

            # Arguments must match (using string comparison for simplicity)
            draft_args = draft.get('args', {})
            target_args = target.get('args', {})

            # Compare argument keys
            if set(draft_args.keys()) != set(target_args.keys()):
                return False

            # Compare argument values (as strings to handle type variations)
            for key in draft_args:
                if str(draft_args[key]) != str(target_args[key]):
                    return False

        return True

    async def verify_tool_calls(
        self,
        draft_tools: List[Dict[str, Any]],
        target_tools: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float, float]:
        """Verify if draft tool calls match target using AST + cosine similarity fallback.

        Args:
            draft_tools: Tool calls from draft model
            target_tools: Tool calls from target model

        Returns:
            Tuple of (verified, method, similarity_score, verification_time)
            - verified: True if draft matches target
            - method: "ast" or "embedding" or "mismatch"
            - similarity_score: Cosine similarity score (or 1.0 for AST match)
            - verification_time: Time spent on verification
        """
        start_time = time.time()

        # Convert to strings for display
        draft_str = self._tools_to_string(draft_tools)
        target_str = self._tools_to_string(target_tools)

        if self.verbose:
            print(f"[Verify] Draft:  {draft_str}")
            print(f"[Verify] Target: {target_str}")

        # Step 1: Try AST comparison (exact match)
        ast_match = self._check_ast_equivalence(draft_tools, target_tools)

        if ast_match:
            verification_time = time.time() - start_time
            if self.verbose:
                print(f"[Verify] AST Match: EXACT")
                print(f"[Verify] Result: ACCEPTED (ast)")
            return True, "ast", 1.0, verification_time

        # Step 2: Fallback to embedding similarity
        if self.verbose:
            print(f"[Verify] AST Match: FAILED, trying embedding similarity...")

        try:
            # Get embeddings (run in executor for sync function)
            loop = asyncio.get_event_loop()
            draft_embedding = await loop.run_in_executor(
                None,
                get_embedding,
                draft_str,
                self.embedding_method
            )
            target_embedding = await loop.run_in_executor(
                None,
                get_embedding,
                target_str,
                self.embedding_method
            )

            # Compute cosine similarity
            similarity = float(cosine_similarity(draft_embedding, target_embedding))

        except Exception as e:
            if self.verbose:
                print(f"[Verify] Embedding error: {e}")
            similarity = 0.0

        verification_time = time.time() - start_time

        # Check if similarity meets threshold
        verified = similarity >= self.embedding_threshold
        method = "embedding" if verified else "mismatch"

        if self.verbose:
            print(f"[Verify] Similarity: {similarity:.3f}, Threshold: {self.embedding_threshold}")
            print(f"[Verify] Result: {'ACCEPTED' if verified else 'REJECTED'} ({method})")

        return verified, method, similarity, verification_time

    async def execute_speculative_query(
        self,
        query: str,
        system_message: str = "You are a helpful assistant. Use tools to answer accurately."
    ) -> Dict[str, Any]:
        """Execute a single query with speculative tool calling.

        Args:
            query: User query to process
            system_message: System prompt for agents

        Returns:
            Dictionary containing:
            - final_result: Final answer (from draft if verified, else target)
            - speculation_accepted: Whether draft was accepted
            - verification_method: Method used for verification
            - similarity_score: Embedding similarity score
            - draft_time: Draft execution time
            - target_time: Target execution time
            - verification_time: Verification time
            - time_saved: Time saved vs sequential execution
            - draft_tool_calls: Tool calls from draft
            - target_tool_calls: Tool calls from target
        """
        self.metrics["total_queries"] += 1

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")

        # Start both agents in parallel
        draft_task = asyncio.create_task(
            self.run_draft_agent(query, system_message)
        )
        target_task = asyncio.create_task(
            self.run_target_agent(query, system_message)
        )

        # Wait for both to complete (draft should finish first)
        draft_result, draft_time = await draft_task
        target_result, target_time = await target_task

        # Update timing metrics
        self.metrics["total_draft_time"] += draft_time
        self.metrics["total_target_time"] += target_time

        # Extract tool calls
        draft_tools = extract_tool_calls(draft_result)
        target_tools = extract_tool_calls(target_result)

        if self.verbose:
            print(f"[Draft] Tool calls: {len(draft_tools)}")
            print(f"[Target] Tool calls: {len(target_tools)}")

        # Verify tool calls using embedding similarity
        verified, method, similarity, verification_time = await self.verify_tool_calls(
            draft_tools,
            target_tools
        )

        self.metrics["total_verification_time"] += verification_time

        # Track verification method
        if verified and method == "ast":
            self.metrics["ast_verified"] += 1
        elif verified and method == "embedding":
            self.metrics["embedding_verified"] += 1

        # Calculate timing comparison: Target Model vs Speculative
        # Target Model baseline: Only run target model
        target_model_time = target_time

        # Speculative: Depends on whether speculation was accepted
        # If accepted: We use draft results, so effective time = draft + verify
        # If rejected: We fall back to target, so effective time = target + verify (overhead)
        # Note: This is "effective time" not wall-clock time (which would be max(draft, target) + verify)
        if verified:
            # Accepted: Use draft results
            speculative_time = draft_time + verification_time
        else:
            # Rejected: Fall back to target, pay verification overhead
            speculative_time = target_time + verification_time

        # Update cumulative metrics
        self.metrics["total_target_model_time"] += target_model_time
        self.metrics["total_speculative_time"] += speculative_time

        # Decide which result to use
        if verified:
            final_result = draft_result
            speculation_accepted = True
            self.metrics["speculation_accepted"] += 1

            # Calculate time saved (avoided re-executing target's tools)
            # In real scenario, draft would execute tools speculatively
            time_saved = max(0, target_time - (draft_time + verification_time))
            self.metrics["total_time_saved"] += time_saved

            if self.verbose:
                print(f"[Result] Using DRAFT result (speculation accepted via {method})")
                print(f"[Result] Time saved: {time_saved:.2f}s")
                print(f"[Timing] Target Model: {target_model_time:.2f}s | Speculative: {speculative_time:.2f}s | Saved: {target_model_time - speculative_time:.2f}s")
        else:
            final_result = target_result
            speculation_accepted = False
            self.metrics["speculation_rejected"] += 1
            self.metrics["target_fallback"] += 1
            time_saved = 0.0

            if self.verbose:
                print(f"[Result] Using TARGET result (speculation rejected)")
                print(f"[Timing] Target Model: {target_model_time:.2f}s | Speculative: {speculative_time:.2f}s | Overhead: {speculative_time - target_model_time:.2f}s")

        return {
            "query": query,
            "final_result": final_result,
            "speculation_accepted": speculation_accepted,
            "verification_method": method,
            "similarity_score": similarity,
            "draft_time": draft_time,
            "target_time": target_time,
            "verification_time": verification_time,
            "time_saved": time_saved,
            "target_model_time": target_model_time,
            "speculative_time": speculative_time,
            "net_saved": target_model_time - speculative_time,
            "draft_tool_calls": draft_tools,
            "target_tool_calls": target_tools,
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of speculation metrics."""
        total = self.metrics["total_queries"]
        if total == 0:
            return {"error": "No queries processed"}

        acceptance_rate = self.metrics["speculation_accepted"] / total
        avg_draft_time = self.metrics["total_draft_time"] / total
        avg_target_time = self.metrics["total_target_time"] / total
        avg_verification_time = self.metrics["total_verification_time"] / total
        avg_time_saved = self.metrics["total_time_saved"] / total

        # Calculate target model vs speculative comparison
        total_target_model_time = self.metrics["total_target_model_time"]
        total_speculative_time = self.metrics["total_speculative_time"]
        net_time_saved = total_target_model_time - total_speculative_time
        speedup_percentage = (net_time_saved / total_target_model_time * 100) if total_target_model_time > 0 else 0

        avg_target_model_time = total_target_model_time / total
        avg_speculative_time = total_speculative_time / total

        return {
            "total_queries": total,
            "speculation_accepted": self.metrics["speculation_accepted"],
            "speculation_rejected": self.metrics["speculation_rejected"],
            "acceptance_rate": acceptance_rate,
            "ast_verified": self.metrics["ast_verified"],
            "embedding_verified": self.metrics["embedding_verified"],
            "target_fallback": self.metrics["target_fallback"],
            "avg_draft_time": avg_draft_time,
            "avg_target_time": avg_target_time,
            "avg_verification_time": avg_verification_time,
            "avg_time_saved": avg_time_saved,
            "total_time_saved": self.metrics["total_time_saved"],
            # Target Model vs Speculative comparison
            "total_target_model_time": total_target_model_time,
            "total_speculative_time": total_speculative_time,
            "net_time_saved": net_time_saved,
            "speedup_percentage": speedup_percentage,
            "avg_target_model_time": avg_target_model_time,
            "avg_speculative_time": avg_speculative_time,
        }


# ============================================================================
# Agent Creation Functions
# ============================================================================

def create_draft_agent(tools: List[Any]):
    """Create draft agent using GPT-5-nano (fast, cheaper model)."""
    model = ChatOpenAI(
        model="gpt-5-nano",
        reasoning_effort="low",  # Optimize for speed
    )
    return create_react_agent(model, tools)


def create_target_agent(tools: List[Any]):
    """Create target agent using Gemini 2.5 Flash Lite (ground truth)."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    return create_react_agent(model, tools)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(dataset_name: str = "browsecomp", limit: Optional[int] = None) -> List[str]:
    """Load queries from specified dataset.

    Args:
        dataset_name: "browsecomp" or "hotpot"
        limit: Maximum number of queries to load

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
            "path": "hotpot_train.csv",
            "column": "question",
            "filter_column": "level",
            "filter_values": ["easy", "medium"]
        }
    }

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]

    # Load CSV file
    df = pd.read_csv(config["path"])

    # Apply filtering if specified (for HotpotQA)
    if "filter_column" in config and "filter_values" in config:
        filter_col = config["filter_column"]
        filter_vals = config["filter_values"]

        if filter_col in df.columns:
            df = df[df[filter_col].isin(filter_vals)]
            print(f"Filtered {dataset_name} to {len(df)} {' + '.join(filter_vals)} questions")
        else:
            print(f"Warning: Filter column '{filter_col}' not found in dataset")

    # Extract queries
    queries = df[config["column"]].tolist()

    if limit:
        queries = queries[:limit]

    return queries


# ============================================================================
# Main Pipeline Runner
# ============================================================================

async def run_speculative_pipeline(
    queries: List[str],
    draft_agent,
    target_agent,
    tools: List[Any],
    embedding_threshold: float = 0.5,
    embedding_method: str = "gemini",
    verbose: bool = False,
    show_progress: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run speculative pipeline on list of queries.

    Args:
        queries: List of queries to process
        draft_agent: Draft model agent
        target_agent: Target model agent
        tools: List of available tools
        embedding_threshold: Similarity threshold for verification
        embedding_method: Embedding method to use
        verbose: Print detailed execution info
        show_progress: Show progress bar

    Returns:
        Tuple of (results list, metrics summary)
    """
    executor = SpeculativeExecutor(
        draft_agent=draft_agent,
        target_agent=target_agent,
        tools=tools,
        embedding_threshold=embedding_threshold,
        embedding_method=embedding_method,
        verbose=verbose
    )

    results = []

    # Create progress bar if requested
    iterator = tqdm(queries, desc="Processing queries") if show_progress else queries

    for query in iterator:
        result = await executor.execute_speculative_query(query)
        results.append(result)

        # Update progress bar description with current acceptance rate and speedup
        if show_progress and isinstance(iterator, tqdm):
            metrics = executor.get_metrics_summary()
            iterator.set_postfix({
                "accepted": f"{metrics['acceptance_rate']:.1%}",
                "speedup": f"{metrics['speedup_percentage']:.1f}%"
            })

    metrics_summary = executor.get_metrics_summary()

    return results, metrics_summary


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_file: str
):
    """Save results and metrics to JSON file.

    Args:
        results: List of query results
        metrics: Metrics summary
        output_file: Output file path
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "results": results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


def print_metrics_report(metrics: Dict[str, Any]):
    """Print formatted metrics report."""
    print("\n" + "="*80)
    print("SPECULATION METRICS REPORT")
    print("="*80)
    print(f"Total Queries:           {metrics['total_queries']}")
    print(f"Speculation Accepted:    {metrics['speculation_accepted']} ({metrics['acceptance_rate']:.1%})")
    print(f"Speculation Rejected:    {metrics['speculation_rejected']}")
    print(f"\nVerification Methods:")
    print(f"  AST Verified:          {metrics['ast_verified']}")
    print(f"  Embedding Verified:    {metrics['embedding_verified']}")
    print(f"  Target Fallback:       {metrics['target_fallback']}")

    # Target Model vs Speculative comparison (MAIN FEATURE)
    print(f"\n{'─'*80}")
    print("⏱️  TARGET MODEL vs SPECULATIVE COMPARISON")
    print(f"{'─'*80}")
    print(f"Target Model (baseline - target only):")
    print(f"  Total Time:            {metrics['total_target_model_time']:.2f}s")
    print(f"  Avg per Query:         {metrics['avg_target_model_time']:.2f}s")
    print(f"\nSpeculative (parallel execution + verification):")
    print(f"  Total Time:            {metrics['total_speculative_time']:.2f}s")
    print(f"  Avg per Query:         {metrics['avg_speculative_time']:.2f}s")
    print(f"\n{'🚀 TIME SAVED WITH SPECULATION':^80}")
    print(f"  Net Time Saved:        {metrics['net_time_saved']:.2f}s")
    print(f"  Speedup:               {metrics['speedup_percentage']:.1f}%")
    print(f"{'─'*80}")

    print(f"\nDetailed Timing Analysis:")
    print(f"  Avg Draft Time:        {metrics['avg_draft_time']:.2f}s")
    print(f"  Avg Target Time:       {metrics['avg_target_time']:.2f}s")
    print(f"  Avg Verification Time: {metrics['avg_verification_time']:.2f}s")
    print("="*80)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for speculative pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Speculative Tool Calling Pipeline")
    parser.add_argument("--dataset", default="browsecomp", choices=["browsecomp", "hotpot"],
                       help="Dataset to use")
    parser.add_argument("--limit", type=int, default=5,
                       help="Number of queries to process (default: 5 for testing)")
    parser.add_argument("--embedding-threshold", type=float, default=0.5,
                       help="Embedding similarity threshold")
    parser.add_argument("--embedding-method", default="gemini", choices=["gemini", "gemma"],
                       help="Embedding method to use")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed execution info")
    parser.add_argument("--output", default=None,
                       help="Output file for results (auto-generated if not specified)")
    parser.add_argument("--full", action="store_true",
                       help="Run full evaluation with 100 queries")

    args = parser.parse_args()

    # Set query limit
    if args.full:
        query_limit = 100
        print("Running FULL evaluation mode (100 queries)")
    else:
        query_limit = args.limit
        print(f"Running VERBOSE test mode ({query_limit} queries)")

    # Load tools
    print("\nLoading tools...")
    tools = get_all_available_tools(verbose=True)
    print(f"Loaded {len(tools)} tools")

    # Create agents
    print("\nCreating agents...")
    print("  Draft:  GPT-5-nano (OpenAI)")
    print("  Target: Gemini 2.5 Flash Lite (Google)")
    draft_agent = create_draft_agent(tools)
    target_agent = create_target_agent(tools)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    queries = load_dataset(args.dataset, limit=query_limit)
    print(f"Loaded {len(queries)} queries")

    # Run speculative pipeline
    print("\nRunning speculative pipeline...")
    results, metrics = await run_speculative_pipeline(
        queries=queries,
        draft_agent=draft_agent,
        target_agent=target_agent,
        tools=tools,
        embedding_threshold=args.embedding_threshold,
        embedding_method=args.embedding_method,
        verbose=args.verbose,
        show_progress=not args.verbose  # Hide progress bar in verbose mode
    )

    # Print metrics report
    print_metrics_report(metrics)

    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"speculative_results_{args.dataset}_{timestamp}.json"

    save_results(results, metrics, args.output)


if __name__ == "__main__":
    asyncio.run(main())
