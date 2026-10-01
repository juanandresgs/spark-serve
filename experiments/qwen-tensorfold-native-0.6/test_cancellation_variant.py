"""Static integrity checks for the optional cancellation source derivative."""
import hashlib
import json
from pathlib import Path
import unittest
from build_cancellation import valid_cpu_test_result

ROOT = Path(__file__).resolve().parent


class CancellationVariantTests(unittest.TestCase):
    def test_patch_hash_and_upstream_pin_are_consistent(self):
        manifest = json.loads((ROOT / "cancellation/patch-manifest.json").read_text())
        pins = json.loads((ROOT / "pins.json").read_text())
        patch_hash = hashlib.sha256((ROOT / "cancellation/prefill-cancellation.patch").read_bytes()).hexdigest()
        self.assertEqual(patch_hash, manifest["patch_sha256"])
        self.assertEqual(manifest["upstream_revision"], pins["runtime_revision"])
        self.assertEqual(set(manifest["files"]), {
            "src/tensorfold/cuda/scheduler.py",
            "src/tensorfold/families/qwen4_exp/cuda/multi.py",
        })

    def test_patch_is_separate_and_public_receipts_have_no_private_parent_pin(self):
        manifest = json.loads((ROOT / "cancellation/patch-manifest.json").read_text())
        self.assertNotIn("parent_image", manifest)
        self.assertIn("separate", manifest["scope"].lower())
        self.assertNotIn("/Users/", manifest["scope"])
        self.assertNotIn("/Users/", (ROOT / "build_cancellation.py").read_text())

    def test_source_kit_manifest_matches_all_packaged_files(self):
        manifest = json.loads((ROOT / "MANIFEST.json").read_text())
        self.assertFalse(manifest["public_build_gpu_qualified"])
        for rel, expected in manifest["files"].items():
            actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, rel)

    def test_cpu_gate_requires_six_successful_tests_and_no_skips(self):
        good = {"tests_run": 6, "failures": 0, "errors": 0, "skipped": 0, "successful": True}
        self.assertTrue(valid_cpu_test_result(good))
        for change in ({"tests_run": 5}, {"failures": 1}, {"errors": 1},
                       {"skipped": 1}, {"successful": False}):
            self.assertFalse(valid_cpu_test_result(good | change), change)


if __name__ == "__main__":
    unittest.main()
