"""
Acceptance Metrics Module - Tracks and analyzes acceptance rates.

This module calculates acceptance rates for draft model verification:
1. Strict AST only: How many pass with strict parameter + semantic checks
2. With embedding fallback: How many pass when fallback is enabled
3. Fallback usage: How often the embedding fallback is actually used

This helps assess the impact of the embedding fallback feature.
"""

from typing import List, Dict, Any, Tuple
import json
from pathlib import Path


def calculate_acceptance_rates(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate acceptance rates from evaluation results.

    Args:
        results: List of evaluation results from speculation_checker

    Returns:
        Dictionary with acceptance metrics:
        - total_cases: Total number of test cases
        - strict_only_passed: Number passing strict AST checks only
        - with_fallback_passed: Number passing with embedding fallback
        - fallback_used: Number of cases where fallback was used
        - strict_only_rate: Percentage passing strict checks only
        - with_fallback_rate: Percentage passing with fallback enabled
        - fallback_usage_rate: Percentage where fallback was used
        - improvement: Improvement in acceptance rate (percentage points)
    """
    total = len(results)

    if total == 0:
        return {
            "total_cases": 0,
            "strict_only_passed": 0,
            "with_fallback_passed": 0,
            "fallback_used": 0,
            "strict_only_rate": 0.0,
            "with_fallback_rate": 0.0,
            "fallback_usage_rate": 0.0,
            "improvement": 0.0
        }

    # Count cases by verification method
    strict_ast_count = sum(1 for r in results if r.get("verified_by") == "strict_ast")
    embedding_fallback_count = sum(1 for r in results if r.get("verified_by") == "embedding_fallback")
    failed_count = sum(1 for r in results if r.get("verified_by") == "none")

    # Calculate rates
    strict_only_passed = strict_ast_count
    with_fallback_passed = strict_ast_count + embedding_fallback_count
    fallback_used = embedding_fallback_count

    strict_only_rate = (strict_only_passed / total) * 100 if total > 0 else 0.0
    with_fallback_rate = (with_fallback_passed / total) * 100 if total > 0 else 0.0
    fallback_usage_rate = (fallback_used / total) * 100 if total > 0 else 0.0
    improvement = with_fallback_rate - strict_only_rate

    return {
        "total_cases": total,
        "strict_only_passed": strict_only_passed,
        "with_fallback_passed": with_fallback_passed,
        "fallback_used": fallback_used,
        "failed": failed_count,
        "strict_only_rate": strict_only_rate,
        "with_fallback_rate": with_fallback_rate,
        "fallback_usage_rate": fallback_usage_rate,
        "improvement": improvement
    }


def format_acceptance_report(metrics: Dict[str, Any]) -> str:
    """
    Format acceptance metrics into a readable report.

    Args:
        metrics: Dictionary from calculate_acceptance_rates

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("ACCEPTANCE RATE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total Test Cases: {metrics['total_cases']}")
    report.append("")
    report.append("VERIFICATION RESULTS:")
    report.append(f"  Passed with Strict AST Only:     {metrics['strict_only_passed']:3d} ({metrics['strict_only_rate']:.1f}%)")
    report.append(f"  Passed with Embedding Fallback:  {metrics['fallback_used']:3d}")
    report.append(f"  Total Passed (with fallback):    {metrics['with_fallback_passed']:3d} ({metrics['with_fallback_rate']:.1f}%)")
    report.append(f"  Failed Both Methods:              {metrics['failed']:3d}")
    report.append("")
    report.append("KEY METRICS:")
    report.append(f"  Strict AST Only Rate:        {metrics['strict_only_rate']:.1f}%")
    report.append(f"  With Fallback Enabled Rate:  {metrics['with_fallback_rate']:.1f}%")
    report.append(f"  Improvement:                 +{metrics['improvement']:.1f} percentage points")
    report.append(f"  Fallback Usage Rate:         {metrics['fallback_usage_rate']:.1f}% of all cases")
    report.append("")

    if metrics['improvement'] > 0:
        report.append(f"✅ Embedding fallback improved acceptance rate by {metrics['improvement']:.1f} percentage points!")
    elif metrics['improvement'] == 0:
        report.append("➖ Embedding fallback did not change acceptance rate.")
    else:
        report.append("⚠️  Warning: Acceptance rate decreased (unexpected)")

    report.append("=" * 80)

    return "\n".join(report)


def compare_verification_modes(
    strict_results: List[Dict[str, Any]],
    fallback_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare results from two verification modes side by side.

    Args:
        strict_results: Results with strict AST only (use_embedding_fallback=False)
        fallback_results: Results with fallback enabled (use_embedding_fallback=True)

    Returns:
        Comparison dictionary with metrics from both modes
    """
    strict_metrics = calculate_acceptance_rates(strict_results)
    fallback_metrics = calculate_acceptance_rates(fallback_results)

    return {
        "strict_mode": strict_metrics,
        "fallback_mode": fallback_metrics,
        "comparison": {
            "acceptance_improvement": fallback_metrics["with_fallback_rate"] - strict_metrics["strict_only_rate"],
            "additional_cases_passed": fallback_metrics["with_fallback_passed"] - strict_metrics["strict_only_passed"],
            "fallback_saved_cases": fallback_metrics["fallback_used"]
        }
    }


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """
    Format comparison between strict and fallback modes.

    Args:
        comparison: Dictionary from compare_verification_modes

    Returns:
        Formatted comparison report
    """
    strict = comparison["strict_mode"]
    fallback = comparison["fallback_mode"]
    comp = comparison["comparison"]

    report = []
    report.append("=" * 80)
    report.append("VERIFICATION MODE COMPARISON")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total Test Cases: {strict['total_cases']}")
    report.append("")
    report.append("MODE 1: STRICT AST ONLY")
    report.append(f"  Passed:  {strict['strict_only_passed']:3d} / {strict['total_cases']} ({strict['strict_only_rate']:.1f}%)")
    report.append(f"  Failed:  {strict['failed']:3d}")
    report.append("")
    report.append("MODE 2: WITH EMBEDDING FALLBACK")
    report.append(f"  Passed:  {fallback['with_fallback_passed']:3d} / {fallback['total_cases']} ({fallback['with_fallback_rate']:.1f}%)")
    report.append(f"  Failed:  {fallback['failed']:3d}")
    report.append(f"  Fallback Used: {fallback['fallback_used']:3d} ({fallback['fallback_usage_rate']:.1f}%)")
    report.append("")
    report.append("IMPACT ANALYSIS:")
    report.append(f"  Additional Cases Passed:     +{comp['additional_cases_passed']}")
    report.append(f"  Acceptance Rate Improvement:  {comp['acceptance_improvement']:.1f} percentage points")
    report.append(f"  Fallback Effectiveness:       {comp['fallback_saved_cases']} cases saved by fallback")
    report.append("")

    if comp['acceptance_improvement'] > 0:
        report.append(f"✅ Embedding fallback is effective! Improved acceptance by {comp['acceptance_improvement']:.1f}pp")
    else:
        report.append("➖ Embedding fallback did not improve acceptance rate")

    report.append("=" * 80)

    return "\n".join(report)


def analyze_verification_methods(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze which verification methods were used across all results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with breakdown by verification method
    """
    method_counts = {
        "strict_ast": 0,
        "embedding_fallback": 0,
        "none": 0
    }

    method_examples = {
        "strict_ast": [],
        "embedding_fallback": [],
        "none": []
    }

    for result in results:
        verified_by = result.get("verified_by", "none")
        method_counts[verified_by] = method_counts.get(verified_by, 0) + 1

        # Store first few examples
        if len(method_examples[verified_by]) < 3:
            method_examples[verified_by].append({
                "test_id": result.get("test_id", "unknown"),
                "draft_sequence": result.get("draft_sequence", []),
                "target_tool": result.get("target_tool", "unknown"),
                "similarity_score": result.get("details", {}).get("embedding_check", {}).get("similarity_score")
            })

    return {
        "counts": method_counts,
        "examples": method_examples,
        "total": len(results)
    }


def save_metrics_report(
    metrics: Dict[str, Any],
    output_path: str,
    include_detailed: bool = True
):
    """
    Save acceptance metrics to a JSON file.

    Args:
        metrics: Metrics dictionary to save
        output_path: Path to save the report
        include_detailed: Whether to include detailed results
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    # Example usage with mock data
    print("Testing Acceptance Metrics Module")
    print("=" * 80)

    # Mock evaluation results
    mock_results = [
        {"verified_by": "strict_ast", "test_id": "test_1", "valid": True},
        {"verified_by": "strict_ast", "test_id": "test_2", "valid": True},
        {"verified_by": "embedding_fallback", "test_id": "test_3", "valid": True},
        {"verified_by": "embedding_fallback", "test_id": "test_4", "valid": True},
        {"verified_by": "none", "test_id": "test_5", "valid": False},
        {"verified_by": "strict_ast", "test_id": "test_6", "valid": True},
        {"verified_by": "embedding_fallback", "test_id": "test_7", "valid": True},
        {"verified_by": "none", "test_id": "test_8", "valid": False},
    ]

    # Calculate metrics
    metrics = calculate_acceptance_rates(mock_results)
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # Format report
    print("\n" + format_acceptance_report(metrics))

    # Analyze verification methods
    analysis = analyze_verification_methods(mock_results)
    print("\nVerification Method Breakdown:")
    print(f"  Strict AST: {analysis['counts']['strict_ast']}")
    print(f"  Embedding Fallback: {analysis['counts']['embedding_fallback']}")
    print(f"  Failed: {analysis['counts']['none']}")

    print("\n" + "=" * 80)
    print("Test complete!")
