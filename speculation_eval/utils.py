"""
Utility functions for speculation evaluation system.

Handles data loading, formatting, and result processing.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / "data"


def load_json_file(filepath: str) -> Any:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def load_test_cases(filename: str = "test_cases.json") -> List[Dict[str, Any]]:
    """
    Load test cases from data directory.

    Expected format:
    [
        {
            "id": "speculation_0",
            "question": "Research agent tool call speculation",
            "draft_tools": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
            "target_tool": "deep_research",
            "description": "Test if draft sequence matches target complex tool"
        },
        ...
    ]

    Args:
        filename: Name of test cases file

    Returns:
        List of test case dictionaries
    """
    filepath = get_data_dir() / filename
    return load_json_file(str(filepath))


def load_composition_mappings(filename: str = "composition_mappings.json") -> Dict[str, Any]:
    """
    Load composition mappings that define how simple tools compose to complex tools.

    Expected format:
    {
        "deep_research": {
            "components": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
            "parameter_mapping": {
                "topic": "query"
            },
            "allow_reordering": false,
            "allow_extra_tools": false,
            "description": "Deep research = search + extract + verify + summarize"
        },
        ...
    }

    Args:
        filename: Name of composition mappings file

    Returns:
        Dictionary of composition mappings
    """
    filepath = get_data_dir() / filename
    return load_json_file(str(filepath))


def load_ground_truth(filename: str = "ground_truth.json") -> Dict[str, Any]:
    """
    Load ground truth data for test cases.

    Expected format:
    {
        "speculation_0": {
            "draft_expected": [
                {"name": "search_web", "args": {"query": "agent tool call speculation"}},
                {"name": "extract_key_points", "args": {"text": "..."}},
                {"name": "verify_facts", "args": {"claim": "..."}},
                {"name": "summarize_text", "args": {"text": "..."}}
            ],
            "target_expected": {
                "name": "deep_research",
                "args": {"topic": "agent tool call speculation"}
            }
        },
        ...
    }

    Args:
        filename: Name of ground truth file

    Returns:
        Dictionary mapping test IDs to expected results
    """
    filepath = get_data_dir() / filename
    return load_json_file(str(filepath))


def load_model_results(filepath: str) -> Dict[str, Any]:
    """
    Load model results from a JSON file.

    Expected format:
    {
        "speculation_0": {
            "draft_result": [...],
            "target_result": {...}
        },
        ...
    }

    Args:
        filepath: Path to model results file

    Returns:
        Dictionary of model results
    """
    return load_json_file(filepath)


def save_evaluation_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of evaluation result dictionaries
        output_path: Path to save results
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Evaluation results saved to: {output_path}")


def format_evaluation_summary(results: List[Dict[str, Any]]) -> str:
    """
    Format evaluation results into a human-readable summary.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        Formatted summary string
    """
    total = len(results)
    valid = sum(1 for r in results if r.get("valid", False))
    param_equivalent = sum(1 for r in results if r.get("param_equivalent", False))
    semantic_equivalent = sum(1 for r in results if r.get("semantic_equivalent", False))

    summary = []
    summary.append("=" * 80)
    summary.append("SPECULATION EVALUATION SUMMARY")
    summary.append("=" * 80)
    summary.append(f"Total Test Cases: {total}")
    summary.append(f"Overall Valid: {valid}/{total} ({valid/total*100:.1f}%)")
    summary.append(f"Parameter Equivalent: {param_equivalent}/{total} ({param_equivalent/total*100:.1f}%)")
    summary.append(f"Semantic Equivalent: {semantic_equivalent}/{total} ({semantic_equivalent/total*100:.1f}%)")
    summary.append("=" * 80)

    # Breakdown by error type
    error_types = {}
    for r in results:
        if not r.get("valid", False):
            error_type = r.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

    if error_types:
        summary.append("\nError Breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            summary.append(f"  {error_type}: {count}")

    # Failed cases detail
    failed = [r for r in results if not r.get("valid", False)]
    if failed:
        summary.append(f"\n{len(failed)} Failed Test Cases:")
        for r in failed[:10]:  # Show first 10
            test_id = r.get("test_id", "unknown")
            errors = r.get("error", [])
            summary.append(f"\n  [{test_id}]")
            for error in errors[:3]:  # Show first 3 errors
                summary.append(f"    - {error}")

    summary.append("=" * 80)

    return "\n".join(summary)


def parse_tool_calls_from_agent_output(agent_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LangChain agent output format.

    Extracts tool calls from the messages structure returned by
    create_react_agent.invoke()

    Args:
        agent_output: Agent result from LangChain (with "messages" key)

    Returns:
        List of tool call dictionaries
    """
    tool_calls = []
    messages = agent_output.get("messages", [])

    for msg in messages:
        # Check if message has tool_calls attribute
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.get('name', ''),
                    "args": tc.get('args', {}),
                    "id": tc.get('id', '')
                })

    return tool_calls


def create_sample_test_data():
    """
    Create sample test data files if they don't exist.
    Useful for initial setup and testing.
    """
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)

    # Sample test cases
    test_cases = [
        {
            "id": "speculation_0",
            "question": "Do deep research on agent tool call speculation",
            "draft_tools": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
            "target_tool": "deep_research",
            "description": "Research task using simple vs complex tools"
        },
        {
            "id": "speculation_1",
            "question": "Analyze the performance data and create a report",
            "draft_tools": ["extract_key_points", "calculate", "summarize_text"],
            "target_tool": "analyze_and_report",
            "description": "Analysis task using computation tools"
        }
    ]

    # Sample composition mappings
    composition_mappings = {
        "deep_research": {
            "components": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
            "parameter_mapping": {
                "topic": "query"
            },
            "allow_reordering": False,
            "allow_extra_tools": False,
            "description": "Comprehensive research = search + extract + verify + summarize"
        },
        "analyze_and_report": {
            "components": ["extract_key_points", "calculate", "summarize_text"],
            "parameter_mapping": {
                "data": "text"
            },
            "allow_reordering": False,
            "allow_extra_tools": False,
            "description": "Analysis and reporting = extract + calculate + summarize"
        }
    }

    # Sample ground truth
    ground_truth = {
        "speculation_0": {
            "draft_expected": [
                {"name": "search_web", "args": {"query": "agent tool call speculation"}},
                {"name": "extract_key_points", "args": {"text": "Search Results..."}},
                {"name": "verify_facts", "args": {"claim": "Key Points..."}},
                {"name": "summarize_text", "args": {"text": "Verified Facts..."}}
            ],
            "target_expected": {
                "name": "deep_research",
                "args": {"topic": "agent tool call speculation"}
            }
        },
        "speculation_1": {
            "draft_expected": [
                {"name": "extract_key_points", "args": {"text": "performance data"}},
                {"name": "calculate", "args": {"expression": "speedup calculation"}},
                {"name": "summarize_text", "args": {"text": "analysis results"}}
            ],
            "target_expected": {
                "name": "analyze_and_report",
                "args": {"data": "performance data"}
            }
        }
    }

    # Write files
    files_to_create = [
        ("test_cases.json", test_cases),
        ("composition_mappings.json", composition_mappings),
        ("ground_truth.json", ground_truth)
    ]

    for filename, data in files_to_create:
        filepath = data_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Created: {filepath}")
        else:
            print(f"Already exists: {filepath}")


if __name__ == "__main__":
    # Create sample data when run directly
    print("Creating sample test data...")
    create_sample_test_data()
    print("\n✅ Sample data created successfully!")
