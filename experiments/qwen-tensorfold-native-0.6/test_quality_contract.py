"""Integrity checks for the public bounded quality fixtures and graders."""
import hashlib
import json
from pathlib import Path
import unittest
from collections import Counter

from coding_cases import CASES
from quality_contract import coding_prompt, coding_source, grade_reasoning, protocol
from quality_fixtures import make, reasoning_fixtures, request_seed

ROOT = Path(__file__).resolve().parent


class QualityContractTests(unittest.TestCase):
    def test_reasoning_fixture_sets_match_count_and_distinctness_contract(self):
        fixtures = reasoning_fixtures()
        self.assertEqual(len(fixtures), 200)
        self.assertEqual(len({prompt for _, prompt, _ in fixtures}), 195)
        self.assertEqual(Counter(category for category, _, _ in fixtures),
                         {"arithmetic": 50, "code-semantics": 50, "state": 50, "graph": 50})
        self.assertEqual(fixtures[:100], make(937))
        self.assertEqual(fixtures[100:], make(1459))
        self.assertEqual(request_seed(61000, 0), 61000)
        self.assertEqual(request_seed(72000, 199), 72199)

    def test_coding_cases_are_exactly_twenty_and_prompt_policy_is_pinned(self):
        self.assertEqual(len(CASES), 20)
        self.assertTrue(coding_prompt(CASES[0][0]).startswith("Implement a pure Python function solve(x). "))
        self.assertTrue(coding_prompt(CASES[0][0]).endswith("Output only Python source, with no Markdown or explanation."))

    def test_reasoning_grade_rejects_boolean_integer_and_accepts_optional_fence(self):
        self.assertTrue(grade_reasoning('{"answer":7}', 7)["format_compliant"])
        self.assertTrue(grade_reasoning('```json\n{"answer":7}\n```', 7)["logical_answer_correct"])
        self.assertFalse(grade_reasoning('{"answer":true}', 1)["logical_answer_correct"])
        self.assertFalse(grade_reasoning('{"answer":7,"extra":0}', 7)["format_compliant"])

    def test_coding_normalization_removes_at_most_one_documented_fence(self):
        self.assertEqual(coding_source('```python\nprint(1)\n```'), "print(1)")
        self.assertEqual(coding_source('```\nprint(1)\n```'), "print(1)")

    def test_protocol_json_matches_executable_contract(self):
        expected = protocol()
        actual = json.loads((ROOT / "quality-protocol.json").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(actual["request"]["clients_per_group"], 4)
        self.assertIn("C4", actual["request"]["concurrency"])
        canonical = json.dumps(reasoning_fixtures(), ensure_ascii=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), actual["reasoning_fixture_sha256"])
