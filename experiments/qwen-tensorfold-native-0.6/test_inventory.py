#!/usr/bin/env python3
"""Synthetic tests for the public inventory verifier; no image or GPU required."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("verify_runtime.py")
ROOT = SCRIPT.parent


class InventoryVerifierTests(unittest.TestCase):
    def test_real_inventory_pins_encode_only_recorded_package_deltas(self):
        def inventory(name):
            return {row["name"]: row["version"] for row in
                    json.loads((ROOT / f"expected-{name}-inventory.json").read_text())}

        base, primary, grammar = inventory("base"), inventory("primary"), inventory("grammar")

        def parse(path):
            return {re.split(r"==", line, maxsplit=1)[0]: line.split("==", 1)[1]
                    for line in path.read_text().splitlines() if line and not line.startswith("#")}

        self.assertEqual(len(base), 216)
        self.assertEqual(len(primary), 217)
        self.assertEqual(len(grammar), 222)
        self.assertEqual({name: (base.get(name), primary.get(name))
                          for name in base.keys() | primary.keys()
                          if base.get(name) != primary.get(name)},
                         {"tensorfold": (None, "0.6.0")})
        self.assertEqual(parse(ROOT / "runtime-constraints.txt"), base)
        self.assertEqual(parse(ROOT / "grammar-constraints.txt"),
                         {name: version for name, version in grammar.items() if name != "tensorfold"})
        delta = {name: version for name, version in grammar.items()
                 if name != "tensorfold" and primary.get(name) != version}
        requirements = parse(ROOT / "grammar-requirements.txt")
        self.assertEqual(requirements, delta)

    def test_accepts_pinned_runtime_and_reports_only_tensorfold_addition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, runtime, constraints, expected, expected_base, receipt = (root / name for name in
                ("base.json", "runtime.json", "constraints.txt", "expected.json", "expected-base.json", "receipt.json"))
            base.write_text(json.dumps([{"name": "alpha-package", "version": "1.2.3"}]))
            runtime.write_text(json.dumps([
                {"name": "alpha-package", "version": "1.2.3"},
                {"name": "tensorfold", "version": "0.6.0"},
            ]))
            constraints.write_text("alpha_package==1.2.3\n")
            expected.write_text(json.dumps([
                {"name": "alpha-package", "version": "1.2.3"},
                {"name": "tensorfold", "version": "0.6.0"},
            ]))
            expected_base.write_text(json.dumps([{"name": "alpha-package", "version": "1.2.3"}]))
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--base", str(base), "--runtime", str(runtime),
                "--constraints", str(constraints), "--expected", str(expected), "--expected-base", str(expected_base),
                "--revision", "c" * 40, "--output", str(receipt),
            ], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(receipt.read_text())
            self.assertEqual(data["result"], "passed")
            self.assertEqual(data["base_to_tested_runtime_delta"]["tensorfold"],
                             {"base": None, "tested_runtime": "0.6.0"})

    def test_rejects_base_package_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, runtime, constraints, expected, expected_base = (root / name for name in
                ("base.json", "runtime.json", "constraints.txt", "expected.json", "expected-base.json"))
            base.write_text(json.dumps([{"name": "alpha-package", "version": "1.2.2"}]))
            runtime.write_text(json.dumps([
                {"name": "alpha-package", "version": "1.2.2"},
                {"name": "tensorfold", "version": "0.6.0"},
            ]))
            constraints.write_text("alpha-package==1.2.3\n")
            expected.write_text(json.dumps([
                {"name": "alpha-package", "version": "1.2.3"},
                {"name": "tensorfold", "version": "0.6.0"},
            ]))
            expected_base.write_text(json.dumps([{"name": "alpha-package", "version": "1.2.2"}]))
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--base", str(base), "--runtime", str(runtime),
                "--constraints", str(constraints), "--expected", str(expected), "--expected-base", str(expected_base),
                "--revision", "c" * 40,
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("built package inventory differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
