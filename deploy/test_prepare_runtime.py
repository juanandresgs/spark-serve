"""Source-provenance regression checks for portable runtime normalization."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from prepare_runtime import inventory, normalize


PIN = 'a' * 40


class RuntimeProvenanceChecks(unittest.TestCase):
    def prepared(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / 'start.sh').write_text('pinned launcher fixture')
        (root / '.env').write_text('SAFE=1\n')
        (root / 'ablit').mkdir()
        (root / 'ablit/fetch_transplant.py').write_text('print("fixture")\n')
        return temporary, root, inventory(root)

    def test_modified_support_file_is_rejected(self):
        temporary, root, authenticated = self.prepared()
        self.addCleanup(temporary.cleanup)
        (root / '.env').write_text('UNSAFE=1\n')
        with self.assertRaisesRegex(ValueError, 'pinned staging inventory'):
            normalize(root, 'glm-adaptive', PIN, authenticated)

    def test_added_support_file_is_rejected(self):
        temporary, root, authenticated = self.prepared()
        self.addCleanup(temporary.cleanup)
        (root / 'unexpected.py').write_text('print("unexpected")\n')
        with self.assertRaisesRegex(ValueError, 'pinned staging inventory'):
            normalize(root, 'glm-adaptive', PIN, authenticated)

    def test_symlinked_support_file_is_rejected(self):
        temporary, root, authenticated = self.prepared()
        self.addCleanup(temporary.cleanup)
        (root / 'linked.env').symlink_to(root / '.env')
        with self.assertRaisesRegex(ValueError, 'Unsafe prepared source path'):
            normalize(root, 'glm-adaptive', PIN, authenticated)

    def test_inventory_records_every_support_file(self):
        temporary, root, authenticated = self.prepared()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(authenticated, {
            '.env': hashlib.sha256(b'SAFE=1\n').hexdigest(),
            'ablit/fetch_transplant.py': hashlib.sha256(b'print("fixture")\n').hexdigest(),
            'start.sh': hashlib.sha256(b'pinned launcher fixture').hexdigest(),
        })


if __name__ == '__main__':
    unittest.main()
