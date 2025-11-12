#!/usr/bin/env python3
"""
Results Analyzer - Analyze saved test results

This script helps you explore and filter the detailed results saved by the test scripts.
You can filter by verification method, analyze specific queries, and export subsets.
"""

import json
import os
import sys
from typing import List, Dict, Any
import argparse


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def filter_by_method(results: List[Dict], method: str) -> List[Dict]:
    """
    Filter results by verification method.

    Args:
        results: List of query results
        method: 'strict_ast', 'embedding_fallback', or 'none'

    Returns:
        Filtered list of results
    """
    return [r for r in results if r['verified_by'] == method]


def print_summary(data: Dict[str, Any]):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    metadata = data['metadata']
    summary = data['summary']

    print(f"\nMetadata:")
    print(f"  Timestamp:     {metadata['timestamp']}")
    print(f"  Dataset:       {metadata['dataset']}")
    print(f"  Total queries: {metadata['total_queries']}")
    print(f"  Draft model:   {metadata['draft_model']}")
    print(f"  Target model:  {metadata['target_model']}")

    print(f"\nAcceptance Rates:")
    print(f"  Strict only:   {summary['strict_only_passed']}/{metadata['total_queries']} ({summary['without_fallback_rate']:.1f}%)")
    print(f"  With fallback: {summary['strict_only_passed'] + summary['fallback_passed']}/{metadata['total_queries']} ({summary['with_fallback_rate']:.1f}%)")
    print(f"  Failed both:   {summary['both_failed']}/{metadata['total_queries']}")

    print(f"\nVerification Timing:")
    print(f"  Total time:     {summary['total_verification_time']:.2f}s")
    print(f"  Average:        {summary['avg_verification_time']*1000:.2f}ms per query")
    print(f"  Strict AST:     {summary['avg_strict_time']*1000:.2f}ms average")
    print(f"  With fallback:  {summary['avg_fallback_time']*1000:.2f}ms average")


def print_query_details(query_result: Dict[str, Any], index: int):
    """Print detailed information for a single query."""
    print(f"\n{'='*80}")
    print(f"Query {index + 1}")
    print(f"{'='*80}")

    print(f"\nQuery: {query_result['query'][:200]}...")

    print(f"\nDraft tools ({query_result['draft_model']}):")
    for tool in query_result['draft_tools']:
        print(f"  - {tool['name']}: {tool.get('args', {})}")

    print(f"\nTarget tools ({query_result['target_model']}):")
    for tool in query_result['target_tools']:
        print(f"  - {tool['name']}: {tool.get('args', {})}")

    validation = query_result['validation']
    print(f"\nVerification:")
    print(f"  Valid:          {validation['valid']}")
    print(f"  Verified by:    {validation['verified_by']}")
    print(f"  Time:           {query_result['verification_time']*1000:.2f}ms")

    if validation.get('details', {}).get('parameter_check'):
        param = validation['details']['parameter_check']
        print(f"  Parameter check: {'PASS' if param['valid'] else 'FAIL'}")
        if param.get('errors'):
            for err in param['errors']:
                print(f"    - {err}")

    if validation.get('details', {}).get('semantic_check'):
        sem = validation['details']['semantic_check']
        print(f"  Semantic check:  {'PASS' if sem['valid'] else 'FAIL'}")
        if sem.get('errors'):
            for err in sem['errors']:
                print(f"    - {err}")

    if validation.get('details', {}).get('embedding_check'):
        emb = validation['details']['embedding_check']
        print(f"  Embedding check: {'PASS' if emb['valid'] else 'FAIL'}")
        if 'similarity_score' in emb:
            print(f"    Similarity: {emb['similarity_score']:.4f} (threshold: {emb.get('threshold', 0.5)})")


def export_filtered(results: List[Dict], output_file: str):
    """Export filtered results to a new JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Exported {len(results)} results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze speculation verification results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--filter", choices=['strict_ast', 'embedding_fallback', 'none'],
                       help="Filter by verification method")
    parser.add_argument("--query", type=int, help="Show details for specific query (1-based index)")
    parser.add_argument("--list", action='store_true', help="List all queries with their verification status")
    parser.add_argument("--export", help="Export filtered results to specified file")

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ Error: File not found: {args.results_file}")
        return

    # Load results
    data = load_results(args.results_file)
    queries = data['queries']

    # Print summary
    print_summary(data)

    # Apply filter if specified
    if args.filter:
        queries = filter_by_method(queries, args.filter)
        print(f"\n🔍 Filtered to {len(queries)} queries with method: {args.filter}")

    # List queries
    if args.list:
        print(f"\n{'='*80}")
        print("QUERY LIST")
        print(f"{'='*80}")
        for i, q in enumerate(queries):
            status_icon = {
                'strict_ast': '✅',
                'embedding_fallback': '🔄',
                'none': '❌'
            }[q['verified_by']]
            print(f"{i+1:3d}. {status_icon} {q['verified_by']:20s} {q['verification_time']*1000:6.1f}ms | {q['query'][:60]}...")

    # Show specific query
    if args.query is not None:
        if 0 < args.query <= len(queries):
            print_query_details(queries[args.query - 1], args.query - 1)
        else:
            print(f"\n❌ Error: Query index {args.query} out of range (1-{len(queries)})")

    # Export filtered results
    if args.export:
        export_filtered(queries, args.export)

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
