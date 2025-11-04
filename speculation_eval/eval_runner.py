"""
Evaluation Runner - Orchestrates speculation evaluation across test cases.

This module runs the evaluation workflow:
1. Load test cases and composition mappings
2. Load model results (draft and target)
3. Run speculation checker on each test case
4. Aggregate results and generate reports
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path

# Support both package and direct execution
try:
    from .speculation_checker import speculation_checker
    from .utils import (
        load_test_cases,
        load_composition_mappings,
        load_ground_truth,
        load_model_results,
        load_json_file,
        save_evaluation_results,
        format_evaluation_summary,
        parse_tool_calls_from_agent_output
    )
except ImportError:
    from speculation_checker import speculation_checker
    from utils import (
        load_test_cases,
        load_composition_mappings,
        load_ground_truth,
        load_model_results,
        load_json_file,
        save_evaluation_results,
        format_evaluation_summary,
        parse_tool_calls_from_agent_output
    )


class SpeculationEvaluator:
    """Main evaluator class for speculation evaluation."""

    def __init__(
        self,
        composition_mappings: Optional[Dict[str, Any]] = None,
        check_params: bool = True,
        check_semantics: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            composition_mappings: Tool composition definitions. If None, loads from file.
            check_params: Whether to check parameter equivalence
            check_semantics: Whether to check semantic equivalence
        """
        self.check_params = check_params
        self.check_semantics = check_semantics

        # Load composition mappings
        if composition_mappings is None:
            try:
                self.composition_mappings = load_composition_mappings()
            except FileNotFoundError:
                print("Warning: No composition mappings found. Using empty dict.")
                self.composition_mappings = {}
        else:
            self.composition_mappings = composition_mappings

    def evaluate_single_case(
        self,
        test_id: str,
        draft_result: List[Dict[str, Any]],
        target_result: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case.

        Args:
            test_id: Unique identifier for test case
            draft_result: List of tool calls from draft model
            target_result: Single tool call from target model
            ground_truth: Optional ground truth data for reference

        Returns:
            Evaluation result dictionary
        """
        # Run speculation checker
        result = speculation_checker(
            draft_result=draft_result,
            target_result=target_result,
            composition_mapping=self.composition_mappings,
            check_params=self.check_params,
            check_semantics=self.check_semantics
        )

        # Add test metadata
        result["test_id"] = test_id
        result["draft_tool_count"] = len(draft_result)
        result["target_tool"] = target_result.get("name", "unknown")

        # Add draft tool sequence
        result["draft_sequence"] = [t.get("name", "") for t in draft_result]

        # Add ground truth reference if provided
        if ground_truth:
            result["ground_truth"] = ground_truth

        return result

    def evaluate_batch(
        self,
        model_results: Dict[str, Dict[str, Any]],
        ground_truth_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of test cases.

        Args:
            model_results: Dictionary mapping test IDs to results
                          {"test_id": {"draft_result": [...], "target_result": {...}}}
            ground_truth_data: Optional ground truth data

        Returns:
            List of evaluation results
        """
        results = []

        for test_id, test_data in model_results.items():
            draft_result = test_data.get("draft_result", [])
            target_result = test_data.get("target_result", {})

            # Get ground truth for this test if available
            gt = None
            if ground_truth_data and test_id in ground_truth_data:
                gt = ground_truth_data[test_id]

            # Evaluate
            result = self.evaluate_single_case(
                test_id=test_id,
                draft_result=draft_result,
                target_result=target_result,
                ground_truth=gt
            )

            results.append(result)

        return results

    def evaluate_from_files(
        self,
        model_results_path: str,
        ground_truth_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation from result files.

        Args:
            model_results_path: Path to model results JSON file
            ground_truth_path: Optional path to ground truth JSON file
            output_path: Optional path to save evaluation results

        Returns:
            List of evaluation results
        """
        print(f"Loading model results from: {model_results_path}")
        model_results = load_model_results(model_results_path)

        # Load ground truth if provided
        ground_truth_data = None
        if ground_truth_path:
            print(f"Loading ground truth from: {ground_truth_path}")
            ground_truth_data = load_json_file(ground_truth_path)

        # Run evaluation
        print(f"\nEvaluating {len(model_results)} test cases...")
        results = self.evaluate_batch(model_results, ground_truth_data)

        # Save results if output path provided
        if output_path:
            save_evaluation_results(results, output_path)

        return results

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate evaluation report.

        Args:
            results: List of evaluation results

        Returns:
            Formatted report string
        """
        return format_evaluation_summary(results)


def run_evaluation(
    model_results_path: str,
    composition_mappings_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    output_path: Optional[str] = None,
    check_params: bool = True,
    check_semantics: bool = True,
    print_report: bool = True
) -> List[Dict[str, Any]]:
    """
    Convenience function to run complete evaluation.

    Args:
        model_results_path: Path to model results JSON
        composition_mappings_path: Optional path to composition mappings
        ground_truth_path: Optional path to ground truth
        output_path: Optional path to save results
        check_params: Whether to check parameter equivalence
        check_semantics: Whether to check semantic equivalence
        print_report: Whether to print summary report

    Returns:
        List of evaluation results
    """
    # Load composition mappings if path provided
    comp_mappings = None
    if composition_mappings_path:
        with open(composition_mappings_path, 'r') as f:
            comp_mappings = json.load(f)

    # Create evaluator
    evaluator = SpeculationEvaluator(
        composition_mappings=comp_mappings,
        check_params=check_params,
        check_semantics=check_semantics
    )

    # Run evaluation
    results = evaluator.evaluate_from_files(
        model_results_path=model_results_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path
    )

    # Print report
    if print_report:
        report = evaluator.generate_report(results)
        print("\n" + report)

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_runner.py <model_results_path> [output_path]")
        print("\nExample:")
        print("  python eval_runner.py results/model_results.json results/eval_results.json")
        sys.exit(1)

    model_results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/evaluation_results.json"

    # Run evaluation
    results = run_evaluation(
        model_results_path=model_results_path,
        output_path=output_path,
        print_report=True
    )

    print(f"\n✅ Evaluation complete! {len(results)} test cases evaluated.")
