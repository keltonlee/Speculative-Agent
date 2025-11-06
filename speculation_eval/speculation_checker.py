"""
Speculation Checker - Core AST-level comparison logic

This module implements the verification logic for tool call speculation by comparing:
1. Draft model's sequence of simple tool calls
2. Target model's single complex tool call

The checker validates both parameter equivalence and semantic equivalence.
With the new embedding fallback feature, it can also use semantic similarity
when strict AST comparison fails.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Literal
import json

# Add BFCL to path to reuse AST utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bfcl'))

try:
    from bfcl_eval.eval_checker.ast_eval.ast_checker import (
        ast_checker,
        is_empty_output,
        type_checker,
    )
    BFCL_AVAILABLE = True
except ImportError:
    BFCL_AVAILABLE = False

# Import embedding similarity module
try:
    from .embedding_similarity import check_embedding_equivalence
    EMBEDDING_AVAILABLE = True
except ImportError:
    try:
        from embedding_similarity import check_embedding_equivalence
        EMBEDDING_AVAILABLE = True
    except ImportError:
        EMBEDDING_AVAILABLE = False
        print("Warning: embedding_similarity module not available")


def check_parameter_equivalence(
    draft_sequence: List[Dict[str, Any]],
    target_call: Dict[str, Any],
    composition_mapping: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if parameters from draft tool sequence collectively match target tool parameters.

    Args:
        draft_sequence: List of simple tool calls from draft model
                       [{"name": "search_web", "args": {"query": "..."}}, ...]
        target_call: Single complex tool call from target model
                    {"name": "deep_research", "args": {"topic": "..."}}
        composition_mapping: Defines how simple tools map to complex tool
                           {"deep_research": {
                               "components": ["search_web", "extract_key_points", ...],
                               "parameter_mapping": {"topic": "query"}
                           }}

    Returns:
        (is_equivalent, errors): Tuple of boolean and list of error messages
    """
    errors = []

    # Extract composition info
    target_name = target_call.get("name", "")
    if target_name not in composition_mapping:
        errors.append(f"No composition mapping found for target tool: {target_name}")
        return False, errors

    mapping = composition_mapping[target_name]
    expected_components = mapping.get("components", [])
    param_mapping = mapping.get("parameter_mapping", {})

    # Collect all parameters from draft sequence
    draft_params = {}
    draft_tool_names = []

    for call in draft_sequence:
        tool_name = call.get("name", "")
        draft_tool_names.append(tool_name)
        args = call.get("args", {})

        # Accumulate all parameters
        for param_name, param_value in args.items():
            if param_name in draft_params:
                # Handle multiple occurrences - store as list
                if not isinstance(draft_params[param_name], list):
                    draft_params[param_name] = [draft_params[param_name]]
                draft_params[param_name].append(param_value)
            else:
                draft_params[param_name] = param_value

    # Check target parameters
    target_params = target_call.get("args", {})

    # Verify each target parameter has an equivalent in draft
    for target_param, target_value in target_params.items():
        # Check if there's a mapping defined
        if target_param in param_mapping:
            draft_param = param_mapping[target_param]
        else:
            draft_param = target_param  # Assume same name

        # Check if draft has this parameter
        if draft_param not in draft_params:
            errors.append(
                f"Target parameter '{target_param}' (mapped to '{draft_param}') "
                f"not found in draft sequence"
            )
            continue

        # Compare values
        draft_value = draft_params[draft_param]

        # Normalize for comparison
        if not values_match(draft_value, target_value):
            errors.append(
                f"Parameter mismatch: {target_param}={target_value} (target) vs "
                f"{draft_param}={draft_value} (draft)"
            )

    is_equivalent = len(errors) == 0
    return is_equivalent, errors


def check_semantic_equivalence(
    draft_sequence: List[Dict[str, Any]],
    target_call: Dict[str, Any],
    composition_mapping: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if draft tool sequence follows the correct composition pattern.

    Validates that:
    1. The sequence contains the expected component tools
    2. Tools appear in approximately the right order (with some flexibility)
    3. No unexpected tools are present

    Args:
        draft_sequence: List of simple tool calls
        target_call: Single complex tool call
        composition_mapping: Tool equivalence definitions

    Returns:
        (is_equivalent, errors): Tuple of boolean and list of error messages
    """
    errors = []

    target_name = target_call.get("name", "")
    if target_name not in composition_mapping:
        errors.append(f"No composition mapping found for: {target_name}")
        return False, errors

    mapping = composition_mapping[target_name]
    expected_components = mapping.get("components", [])
    allow_reordering = mapping.get("allow_reordering", False)
    allow_extra_tools = mapping.get("allow_extra_tools", False)

    # Extract tool names from draft sequence
    draft_tools = [call.get("name", "") for call in draft_sequence]

    # Check if all expected components are present
    missing_tools = []
    for expected_tool in expected_components:
        if expected_tool not in draft_tools:
            missing_tools.append(expected_tool)

    if missing_tools:
        errors.append(
            f"Missing expected tools: {missing_tools}. "
            f"Expected: {expected_components}, Got: {draft_tools}"
        )

    # Check for unexpected tools (if not allowed)
    if not allow_extra_tools:
        unexpected_tools = []
        for draft_tool in draft_tools:
            if draft_tool not in expected_components:
                unexpected_tools.append(draft_tool)

        if unexpected_tools:
            errors.append(
                f"Unexpected tools in sequence: {unexpected_tools}"
            )

    # Check order (if reordering not allowed)
    if not allow_reordering and not missing_tools:
        # Create a filtered list of draft tools that are in expected components
        relevant_draft_tools = [t for t in draft_tools if t in expected_components]

        # Check if they appear in the expected order
        if not is_subsequence(expected_components, relevant_draft_tools):
            errors.append(
                f"Tools not in expected order. "
                f"Expected order: {expected_components}, "
                f"Got order: {relevant_draft_tools}"
            )

    is_equivalent = len(errors) == 0
    return is_equivalent, errors


def speculation_checker(
    draft_result: List[Dict[str, Any]],
    target_result: Dict[str, Any],
    composition_mapping: Dict[str, Any],
    check_params: bool = True,
    check_semantics: bool = True,
    use_embedding_fallback: bool = False,
    embedding_threshold: float = 0.5,
    embedding_method: Literal["gemini", "gemma"] = "gemini",
    verbose_embedding: bool = False
) -> Dict[str, Any]:
    """
    Main checker function - validates draft sequence against target call.

    Args:
        draft_result: List of tool calls from draft model
        target_result: Single tool call from target model
        composition_mapping: Tool equivalence definitions
        check_params: Whether to check parameter equivalence
        check_semantics: Whether to check semantic equivalence
        use_embedding_fallback: If True, use embedding similarity when strict checks fail
        embedding_threshold: Similarity threshold for embedding fallback (default 0.5)
        embedding_method: Embedding method to use ("gemini" or "gemma")
        verbose_embedding: If True, print detailed embedding information

    Returns:
        Dictionary with:
        - valid: bool - Whether the check passed
        - error: List[str] - Error messages
        - error_type: str - Category of error
        - param_equivalent: bool - Parameter check result
        - semantic_equivalent: bool - Semantic check result
        - verified_by: str - Which method verified the result (strict_ast, embedding_fallback, none)
        - embedding_details: Dict - Details from embedding check (if used)
    """
    result = {
        "valid": False,
        "error": [],
        "error_type": "",
        "param_equivalent": False,
        "semantic_equivalent": False,
        "verified_by": "none",
        "details": {}
    }

    # Validate inputs
    if not draft_result or not isinstance(draft_result, list):
        result["error"].append("Invalid draft result: must be non-empty list")
        result["error_type"] = "speculation:invalid_draft_format"
        return result

    if not target_result or not isinstance(target_result, dict):
        result["error"].append("Invalid target result: must be dict")
        result["error_type"] = "speculation:invalid_target_format"
        return result

    # Run parameter equivalence check
    if check_params:
        param_valid, param_errors = check_parameter_equivalence(
            draft_result, target_result, composition_mapping
        )
        result["param_equivalent"] = param_valid
        result["details"]["parameter_check"] = {
            "valid": param_valid,
            "errors": param_errors
        }
        if not param_valid:
            result["error"].extend([f"[Param] {e}" for e in param_errors])
    else:
        result["param_equivalent"] = True  # Skip check

    # Run semantic equivalence check
    if check_semantics:
        semantic_valid, semantic_errors = check_semantic_equivalence(
            draft_result, target_result, composition_mapping
        )
        result["semantic_equivalent"] = semantic_valid
        result["details"]["semantic_check"] = {
            "valid": semantic_valid,
            "errors": semantic_errors
        }
        if not semantic_valid:
            result["error"].extend([f"[Semantic] {e}" for e in semantic_errors])
    else:
        result["semantic_equivalent"] = True  # Skip check

    # Overall validity from strict checks
    strict_valid = result["param_equivalent"] and result["semantic_equivalent"]
    result["valid"] = strict_valid

    # If strict checks passed, mark as verified by strict AST
    if strict_valid:
        result["verified_by"] = "strict_ast"
    else:
        # Strict checks failed - try embedding fallback if enabled
        if use_embedding_fallback:
            if not EMBEDDING_AVAILABLE:
                result["error"].append("[Embedding] Embedding module not available")
                result["details"]["embedding_check"] = {
                    "valid": False,
                    "errors": ["Embedding module not available"]
                }
            else:
                # Run embedding similarity check
                if verbose_embedding:
                    print(f"\n{'='*80}")
                    print("ATTEMPTING EMBEDDING FALLBACK")
                    print(f"{'='*80}")
                    print("Strict AST checks failed. Trying embedding-based similarity...")

                try:
                    embedding_valid, embedding_errors, embedding_details = check_embedding_equivalence(
                        draft_result,
                        target_result,
                        threshold=embedding_threshold,
                        method=embedding_method,
                        verbose=verbose_embedding
                    )

                    result["details"]["embedding_check"] = {
                        "valid": embedding_valid,
                        "errors": embedding_errors,
                        "similarity_score": embedding_details.get("similarity_score", 0.0),
                        "threshold": embedding_threshold,
                        "method": embedding_method,
                        "draft_string": embedding_details.get("draft_string", ""),
                        "target_string": embedding_details.get("target_string", "")
                    }

                    if embedding_valid:
                        # Embedding fallback succeeded!
                        result["valid"] = True
                        result["verified_by"] = "embedding_fallback"
                        result["error"] = []  # Clear previous errors since we passed with fallback
                        result["error_type"] = ""
                    else:
                        # Embedding fallback also failed
                        result["error"].extend([f"[Embedding] {e}" for e in embedding_errors])

                except Exception as e:
                    result["error"].append(f"[Embedding] Fallback check failed: {e}")
                    result["details"]["embedding_check"] = {
                        "valid": False,
                        "errors": [str(e)]
                    }

    # Set error type if still not valid
    if not result["valid"]:
        if not result["param_equivalent"] and not result["semantic_equivalent"]:
            result["error_type"] = "speculation:both_checks_failed"
        elif not result["param_equivalent"]:
            result["error_type"] = "speculation:parameter_mismatch"
        else:
            result["error_type"] = "speculation:semantic_mismatch"

    return result


# ==================== Helper Functions ====================

def values_match(value1: Any, value2: Any) -> bool:
    """
    Compare two values with normalization.

    Handles:
    - String comparison (case-insensitive, whitespace-normalized)
    - Numeric comparison (int/float equivalence)
    - List comparison
    - Dict comparison
    """
    # Handle None
    if value1 is None and value2 is None:
        return True
    if value1 is None or value2 is None:
        return False

    # Handle strings
    if isinstance(value1, str) and isinstance(value2, str):
        return normalize_string(value1) == normalize_string(value2)

    # Handle numbers (int/float equivalence)
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return abs(value1 - value2) < 1e-9

    # Handle lists (including list vs single value)
    if isinstance(value1, list):
        if isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            return all(values_match(v1, v2) for v1, v2 in zip(value1, value2))
        else:
            # Single value vs list - check if value in list
            return len(value1) == 1 and values_match(value1[0], value2)
    elif isinstance(value2, list):
        return len(value2) == 1 and values_match(value1, value2[0])

    # Handle dicts
    if isinstance(value1, dict) and isinstance(value2, dict):
        if set(value1.keys()) != set(value2.keys()):
            return False
        return all(values_match(value1[k], value2[k]) for k in value1.keys())

    # Direct comparison
    return value1 == value2


def normalize_string(s: str) -> str:
    """Normalize string for comparison (lowercase, strip whitespace)."""
    return s.lower().strip()


def is_subsequence(expected: List[str], actual: List[str]) -> bool:
    """
    Check if 'expected' is a subsequence of 'actual'.

    All items in 'expected' must appear in 'actual' in the same relative order,
    but 'actual' may have additional items in between.
    """
    if not expected:
        return True

    expected_idx = 0
    for item in actual:
        if item == expected[expected_idx]:
            expected_idx += 1
            if expected_idx == len(expected):
                return True

    return expected_idx == len(expected)
