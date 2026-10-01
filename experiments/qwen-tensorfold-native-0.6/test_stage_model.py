"""Local digest and path tests for the model stager; no Hub or weights needed."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from stage_model import check


class ModelStagerTests(unittest.TestCase):
    def test_checks_lfs_sha256_and_git_blob_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lfs_body = b"model shard fixture"
            git_body = b"config fixture"
            (root / "weights.safetensors").write_bytes(lfs_body)
            (root / "config.json").write_bytes(git_body)
            siblings = [
                {"rfilename": "weights.safetensors", "size": len(lfs_body),
                 "lfs": {"sha256": hashlib.sha256(lfs_body).hexdigest()}},
                {"rfilename": "config.json", "size": len(git_body),
                 "blobId": hashlib.sha1(f"blob {len(git_body)}\0".encode() + git_body).hexdigest()},
            ]
            self.assertEqual(check(root, siblings), (2, len(lfs_body) + len(git_body)))

    def test_rejects_tampered_size_or_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            body = b"fixture"
            (root / "weight.bin").write_bytes(body)
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                check(root, [{"rfilename": "weight.bin", "size": len(body) + 1,
                              "lfs": {"sha256": hashlib.sha256(body).hexdigest()}}])
            with self.assertRaisesRegex(ValueError, "Digest mismatch"):
                check(root, [{"rfilename": "weight.bin", "size": len(body),
                              "lfs": {"sha256": "0" * 64}}])

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                check(Path(temp), [{"rfilename": "../outside", "size": 0,
                                    "lfs": {"sha256": hashlib.sha256(b"").hexdigest()}}])

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            target = Path(outside) / "payload"
            target.write_bytes(b"fixture")
            (root / "escape.bin").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "outside the snapshot"):
                check(root, [{"rfilename": "escape.bin", "size": len(b"fixture"),
                              "lfs": {"sha256": hashlib.sha256(b"fixture").hexdigest()}}])


if __name__ == "__main__":
    unittest.main()
