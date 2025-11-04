#!/usr/bin/env python3
"""
Test code correctness - edge cases and potential bugs
"""

from speculation_checker import (
    speculation_checker,
    check_parameter_equivalence,
    check_semantic_equivalence,
    values_match
)
from eval_runner import SpeculationEvaluator

print("=" * 80)
print("CODE CORRECTNESS TESTS")
print("=" * 80)

# Test 1: Empty draft_result (should be caught)
print("\nTest 1: Empty draft result")
try:
    result = speculation_checker(
        draft_result=[],  # Empty list
        target_result={"name": "test", "args": {}},
        composition_mapping={"test": {"components": []}}
    )
    if result.get("error_type") == "speculation:invalid_draft_format":
        print("✅ PASS - Empty draft result correctly rejected")
    else:
        print(f"❌ FAIL - Empty draft accepted. Result: {result}")
except Exception as e:
    print(f"❌ FAIL - Exception: {e}")

# Test 2: Empty target_result (should be caught)
print("\nTest 2: Empty target result")
try:
    result = speculation_checker(
        draft_result=[{"name": "test", "args": {}}],
        target_result={},  # Empty dict
        composition_mapping={"": {"components": []}}
    )
    # Empty dict should work - it's a valid (though minimal) target
    print(f"ℹ️  Empty dict result: {result.get('valid')}")
except Exception as e:
    print(f"❌ FAIL - Exception: {e}")

# Test 3: None values
print("\nTest 3: None values handling")
try:
    result = speculation_checker(
        draft_result=None,  # None
        target_result={"name": "test", "args": {}},
        composition_mapping={}
    )
    if result.get("error_type") == "speculation:invalid_draft_format":
        print("✅ PASS - None draft result correctly rejected")
    else:
        print(f"❌ FAIL - None accepted: {result}")
except Exception as e:
    print(f"❌ FAIL - Exception: {e}")

# Test 4: Values matching with different types
print("\nTest 4: Value matching edge cases")
tests = [
    (None, None, True, "None vs None"),
    (None, "test", False, "None vs string"),
    ("Test", "test", True, "Case insensitive"),
    ("  test  ", "test", True, "Whitespace"),
    (5, 5.0, True, "Int vs float"),
    (5.0000001, 5.0, True, "Float tolerance"),
    ([1, 2], [1, 2], True, "Lists equal"),
    ([1], 1, True, "Single item list vs value"),
    (1, [1], True, "Value vs single item list"),
    ([1, 2], [2, 1], False, "List order matters"),
    ({"a": 1}, {"a": 1}, True, "Dicts equal"),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True, "Dict order doesn't matter"),
]

all_passed = True
for val1, val2, expected, desc in tests:
    result = values_match(val1, val2)
    if result == expected:
        print(f"  ✅ {desc}")
    else:
        print(f"  ❌ {desc}: expected {expected}, got {result}")
        all_passed = False

if all_passed:
    print("✅ PASS - All value matching tests passed")
else:
    print("❌ FAIL - Some value matching tests failed")

# Test 5: Parameter accumulation with duplicates
print("\nTest 5: Multiple parameters with same name")
composition = {
    "test_tool": {
        "components": ["tool_a", "tool_b"],
        "parameter_mapping": {}
    }
}

draft = [
    {"name": "tool_a", "args": {"param": "value1"}},
    {"name": "tool_b", "args": {"param": "value2"}}
]

target = {"name": "test_tool", "args": {"param": "value1"}}

is_equiv, errors = check_parameter_equivalence(draft, target, composition)
print(f"  Result: {is_equiv}")
print(f"  Errors: {errors}")
# This should fail because draft has param=["value1", "value2"] but target expects param="value1"
if not is_equiv:
    print("  ✅ PASS - Correctly detected parameter duplication issue")
else:
    print("  ❌ FAIL - Should have detected parameter mismatch")

# Test 6: Order checking
print("\nTest 6: Tool order validation")
composition = {
    "ordered_tool": {
        "components": ["a", "b", "c"],
        "parameter_mapping": {},
        "allow_reordering": False
    }
}

# Wrong order
draft = [
    {"name": "c", "args": {}},
    {"name": "a", "args": {}},
    {"name": "b", "args": {}}
]

target = {"name": "ordered_tool", "args": {}}

is_equiv, errors = check_semantic_equivalence(draft, target, composition)
if not is_equiv and "not in expected order" in str(errors):
    print("  ✅ PASS - Correctly detected wrong order")
else:
    print(f"  ❌ FAIL - Should detect wrong order. Errors: {errors}")

# Test 7: Evaluator with no composition mappings
print("\nTest 7: Evaluator initialization without mappings")
try:
    evaluator = SpeculationEvaluator(composition_mappings=None)
    print("  ✅ PASS - Evaluator created with None mappings")
except Exception as e:
    print(f"  ❌ FAIL - Exception: {e}")

# Test 8: Missing composition mapping
print("\nTest 8: Missing composition mapping for target tool")
result = speculation_checker(
    draft_result=[{"name": "tool", "args": {}}],
    target_result={"name": "unknown_tool", "args": {}},
    composition_mapping={}
)

if "No composition mapping found" in str(result.get("error")):
    print("  ✅ PASS - Missing mapping detected")
else:
    print(f"  ❌ FAIL - Should detect missing mapping: {result}")

print("\n" + "=" * 80)
print("✅ CODE CORRECTNESS TESTS COMPLETE")
print("=" * 80)
