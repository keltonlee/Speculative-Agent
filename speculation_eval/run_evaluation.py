#!/usr/bin/env python3
"""
Main entry point for running speculation evaluations.

This script provides a CLI for evaluating draft model tool call sequences
against target model complex tool calls.

Usage:
    python run_evaluation.py --model-results results.json
    python run_evaluation.py --model-results results.json --output eval.json
    python run_evaluation.py --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from eval_runner import SpeculationEvaluator, run_evaluation
from utils import (
    load_composition_mappings,
    parse_tool_calls_from_agent_output,
    create_sample_test_data
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate speculation: draft tool sequences vs target complex tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from model results file
  python run_evaluation.py --model-results results/model_output.json

  # Specify output path
  python run_evaluation.py --model-results results.json --output eval_results.json

  # Use custom composition mappings
  python run_evaluation.py --model-results results.json --mappings custom_mappings.json

  # Only check parameter equivalence
  python run_evaluation.py --model-results results.json --no-semantic-check

  # Create sample test data
  python run_evaluation.py --create-sample-data

  # Generate model results template
  python run_evaluation.py --generate-template
        """
    )

    parser.add_argument(
        "--model-results",
        type=str,
        help="Path to model results JSON file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save evaluation results (default: results/evaluation_results.json)"
    )

    parser.add_argument(
        "--mappings",
        type=str,
        help="Path to custom composition mappings JSON (default: data/composition_mappings.json)"
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth JSON file"
    )

    parser.add_argument(
        "--no-param-check",
        action="store_true",
        help="Skip parameter equivalence check"
    )

    parser.add_argument(
        "--no-semantic-check",
        action="store_true",
        help="Skip semantic equivalence check"
    )

    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample test data files in data/ directory"
    )

    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Generate a template for model results JSON"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )

    return parser.parse_args()


def generate_model_results_template():
    """Generate a template for model results JSON."""
    template = {
        "speculation_0": {
            "draft_result": [
                {
                    "name": "search_web",
                    "args": {"query": "agent tool call speculation"},
                    "id": "call_001"
                },
                {
                    "name": "extract_key_points",
                    "args": {"text": "Search results text here..."},
                    "id": "call_002"
                },
                {
                    "name": "verify_facts",
                    "args": {"claim": "Key points here..."},
                    "id": "call_003"
                },
                {
                    "name": "summarize_text",
                    "args": {"text": "Verified facts here..."},
                    "id": "call_004"
                }
            ],
            "target_result": {
                "name": "deep_research",
                "args": {"topic": "agent tool call speculation"},
                "id": "call_target_001"
            }
        },
        "speculation_1": {
            "draft_result": [
                {
                    "name": "extract_key_points",
                    "args": {"text": "performance data"},
                    "id": "call_101"
                },
                {
                    "name": "calculate",
                    "args": {"expression": "4.21 / 2.27"},
                    "id": "call_102"
                },
                {
                    "name": "summarize_text",
                    "args": {"text": "analysis results"},
                    "id": "call_103"
                }
            ],
            "target_result": {
                "name": "analyze_and_report",
                "args": {"data": "performance data"},
                "id": "call_target_101"
            }
        }
    }

    output_path = "model_results_template.json"
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"✅ Template generated: {output_path}")
    print("\nEdit this file with your actual model results, then run:")
    print(f"  python run_evaluation.py --model-results {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle special commands
    if args.create_sample_data:
        print("Creating sample test data...")
        create_sample_test_data()
        return 0

    if args.generate_template:
        generate_model_results_template()
        return 0

    # Validate required arguments
    if not args.model_results:
        print("Error: --model-results is required")
        print("Run with --help for usage information")
        return 1

    # Validate file exists
    if not Path(args.model_results).exists():
        print(f"Error: Model results file not found: {args.model_results}")
        return 1

    # Set defaults
    output_path = args.output or "results/evaluation_results.json"
    check_params = not args.no_param_check
    check_semantics = not args.no_semantic_check

    # Run evaluation
    try:
        if not args.quiet:
            print(f"Running speculation evaluation...")
            print(f"  Model results: {args.model_results}")
            print(f"  Output: {output_path}")
            print(f"  Parameter check: {check_params}")
            print(f"  Semantic check: {check_semantics}")
            print()

        results = run_evaluation(
            model_results_path=args.model_results,
            composition_mappings_path=args.mappings,
            ground_truth_path=args.ground_truth,
            output_path=output_path,
            check_params=check_params,
            check_semantics=check_semantics,
            print_report=not args.quiet
        )

        if not args.quiet:
            print(f"\n✅ Evaluation complete!")
            print(f"   Evaluated: {len(results)} test cases")
            print(f"   Results saved to: {output_path}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
