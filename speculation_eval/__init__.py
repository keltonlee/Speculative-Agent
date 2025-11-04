"""
Speculation Evaluation System

This module provides evaluation tools for comparing draft model tool call sequences
with target model complex tool calls using AST-level verification.

Core concept:
- Draft Model: Uses simple, atomic tools (e.g., search_web, extract_key_points)
- Target Model: Uses complex, composite tools (e.g., deep_research)
- Evaluation: Verify if draft sequence is equivalent to target call

Key components:
- speculation_checker: Core AST-level comparison logic
- eval_runner: Orchestrates evaluation across test cases
- utils: Helper functions for data loading and formatting
"""

__version__ = "0.1.0"

from .speculation_checker import speculation_checker, check_parameter_equivalence, check_semantic_equivalence
from .utils import load_test_cases, load_composition_mappings, load_ground_truth

__all__ = [
    "speculation_checker",
    "check_parameter_equivalence",
    "check_semantic_equivalence",
    "load_test_cases",
    "load_composition_mappings",
    "load_ground_truth",
]
