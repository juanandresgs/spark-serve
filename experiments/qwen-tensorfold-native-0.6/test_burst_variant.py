"""Integrity checks for the separate cancellation-plus-burst-4 derivative."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BurstVariantTests(unittest.TestCase):
    def test_patch_is_chained_to_exact_cancellation_source(self):
        cancel = json.loads((ROOT / "cancellation/patch-manifest.json").read_text())
        burst = json.loads((ROOT / "decode-burst/patch-manifest.json").read_text())
        self.assertEqual(burst["parent_cancellation_patch_sha256"], cancel["patch_sha256"])
        self.assertEqual(burst["upstream_commit"], cancel["upstream_revision"])
        self.assertEqual(sha(ROOT / "decode-burst/decode-burst.patch"), burst["patch_sha256"])
        self.assertEqual(set(burst["files"]), {"src/tensorfold/families/qwen4_exp/cuda/multi.py"})
        for path, details in burst["files"].items():
            self.assertEqual(details["before_sha256"], cancel["files"][path]["after_sha256"])

    def test_derivative_uses_local_parent_and_only_selects_burst_four(self):
        dockerfile = (ROOT / "Dockerfile.burst").read_text()
        builder = (ROOT / "build_burst.py").read_text()
        self.assertIn("ARG PARENT_IMAGE", dockerfile)
        self.assertIn("--parent-cancellation-image-id", builder)
        self.assertIn("TF_FLASH_DECODE_BURST=4", dockerfile)
        self.assertIn("EXPECTED_TESTS = 14", builder)
        self.assertIn("--network=none", builder)
        for text in (dockerfile, builder):
            self.assertNotIn("/" + "Users" + "/", text)
            self.assertNotIn("/" + "home/" + "user" + "j", text)
        ignored = (ROOT / ".dockerignore").read_text()
        for asset in ("!Dockerfile.burst", "!build_burst.py", "!decode-burst/decode-burst.patch",
                      "!decode-burst/patch-manifest.json", "!decode-burst/test_decode_burst.py"):
            self.assertIn(asset, ignored)

    def test_public_build_receipt_is_allowlisted(self):
        receipt = json.loads((ROOT / "receipts/qwen-native-burst4-build-rank2-20261001-v1.json").read_text())
        self.assertEqual(receipt["source_archive_sha256"], "5923a65316d56aebb2e517bc1e3abe8e0bbd1c60ec5bcfe39f03bbe4df53673c")
        self.assertEqual(receipt["cpu_test_result"], {
            "tests_run": 14, "failures": 0, "errors": 0, "skipped": 0, "successful": True})
        text = (ROOT / "receipts/qwen-native-burst4-build-rank2-20261001-v1.json").read_text()
        self.assertNotIn("/" + "home/" + "user" + "j", text)
        self.assertNotIn("/" + "Users" + "/", text)
        self.assertNotIn("spark-" + "".join(map(chr, (101, 57, 57, 99))), text)

    def test_base_cancellation_derivative_remains_burst_one(self):
        dockerfile = (ROOT / "Dockerfile.cancellation").read_text()
        self.assertNotIn("TF_FLASH_DECODE_BURST=4", dockerfile)
        patch = (ROOT / "decode-burst/decode-burst.patch").read_text()
        self.assertIn('os.environ.get("TF_FLASH_DECODE_BURST", "1")', patch)
