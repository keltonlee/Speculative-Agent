#!/usr/bin/env python3
"""Test the programmatic API of speculation evaluation."""

from speculation_checker import speculation_checker
from eval_runner import SpeculationEvaluator

# Test 1: Direct checker call
print("=" * 80)
print("TEST 1: Direct speculation_checker() call")
print("=" * 80)

draft_result = [
    {"name": "search_web", "args": {"query": "test"}},
    {"name": "extract_key_points", "args": {"text": "results"}},
    {"name": "verify_facts", "args": {"claim": "points"}},
    {"name": "summarize_text", "args": {"text": "facts"}}
]

target_result = {
    "name": "deep_research",
    "args": {"topic": "test"}
}

composition_mapping = {
    "deep_research": {
        "components": ["search_web", "extract_key_points", "verify_facts", "summarize_text"],
        "parameter_mapping": {"topic": "query"}
    }
}

result = speculation_checker(draft_result, target_result, composition_mapping)
print(f"Valid: {result['valid']}")
print(f"Parameter Equivalent: {result['param_equivalent']}")
print(f"Semantic Equivalent: {result['semantic_equivalent']}")
print(f"Errors: {result['error']}")
print()

# Test 2: Using SpeculationEvaluator
print("=" * 80)
print("TEST 2: SpeculationEvaluator class")
print("=" * 80)

evaluator = SpeculationEvaluator(
    composition_mappings=composition_mapping,
    check_params=True,
    check_semantics=True
)

test_result = evaluator.evaluate_single_case(
    test_id="api_test",
    draft_result=draft_result,
    target_result=target_result
)

print(f"Test ID: {test_result['test_id']}")
print(f"Valid: {test_result['valid']}")
print(f"Draft Tool Count: {test_result['draft_tool_count']}")
print(f"Target Tool: {test_result['target_tool']}")
print(f"Draft Sequence: {test_result['draft_sequence']}")
print()

# Test 3: Batch evaluation
print("=" * 80)
print("TEST 3: Batch evaluation")
print("=" * 80)

batch_results = {
    "test_1": {
        "draft_result": draft_result,
        "target_result": target_result
    },
    "test_2": {
        "draft_result": [
            {"name": "search_web", "args": {"query": "test"}},
            {"name": "extract_key_points", "args": {"text": "results"}}
        ],
        "target_result": target_result
    }
}

results = evaluator.evaluate_batch(batch_results)
print(f"Evaluated {len(results)} test cases")
print(f"Passed: {sum(1 for r in results if r['valid'])}")
print(f"Failed: {sum(1 for r in results if not r['valid'])}")
print()

# Test 4: Generate report
print("=" * 80)
print("TEST 4: Report generation")
print("=" * 80)

report = evaluator.generate_report(results)
print(report)

print("\n" + "=" * 80)
print("✅ ALL API TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
