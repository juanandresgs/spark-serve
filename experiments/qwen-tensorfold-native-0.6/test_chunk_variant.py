"""Integrity checks for the separate burst-4 plus 1024-row derivative."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChunkVariantTests(unittest.TestCase):
    def test_patch_is_chained_to_exact_burst4_source(self):
        burst = json.loads((ROOT / "decode-burst/patch-manifest.json").read_text())
        chunk = json.loads((ROOT / "scheduling/chunk/patch-manifest.json").read_text())
        self.assertEqual(chunk["parent_burst_patch_sha256"], burst["patch_sha256"])
        self.assertEqual(chunk["upstream_commit"], burst["upstream_commit"])
        self.assertEqual(sha(ROOT / "scheduling/chunk/prefill-rows.patch"), chunk["patch_sha256"])
        self.assertEqual(set(chunk["files"]), {"src/tensorfold/families/qwen4_exp/cuda/multi.py"})
        for path, details in chunk["files"].items():
            self.assertEqual(details["before_sha256"], burst["files"][path]["after_sha256"])

    def test_builder_binds_burst4_parent_and_sets_only_chunk_derivative(self):
        dockerfile = (ROOT / "Dockerfile.chunk").read_text()
        builder = (ROOT / "build_chunk.py").read_text()
        self.assertIn("ARG PARENT_IMAGE", dockerfile)
        self.assertIn("--parent-burst-image-id", builder)
        self.assertIn("TF_FLASH_DECODE_BURST=4", dockerfile)
        self.assertIn("TF_FLASH_PREFILL_ROWS=1024", dockerfile)
        self.assertIn('entry.startswith("TF_FLASH_PREFILL_ROWS=")', builder)
        self.assertIn("EXPECTED_TESTS = 19", builder)
        self.assertIn("--network=none", builder)
        self.assertIn("RootFS", builder)
        self.assertNotIn("/" + "Users" + "/", dockerfile + builder)
        self.assertNotIn("/" + "home/" + "user" + "j", dockerfile + builder)

    def test_dockerignore_includes_only_chunk_runtime_inputs(self):
        ignored = (ROOT / ".dockerignore").read_text()
        for asset in ("!Dockerfile.chunk", "!build_chunk.py", "!scheduling/chunk/prefill-rows.patch",
                      "!scheduling/chunk/patch-manifest.json", "!scheduling/chunk/test_prefill_rows.py"):
            self.assertIn(asset, ignored)
        for asset in ("!benchmark_speed.py", "!quality_fixtures.py", "!quality_contract.py"):
            self.assertNotIn(asset, ignored)


if __name__ == "__main__":
    unittest.main()
