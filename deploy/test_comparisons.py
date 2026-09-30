"""Exercise real evidence resolution, generated drift detection and CLI discovery."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from spark_serve.comparisons import check_page, load, options, render
from spark_serve.config import ConfigError

ROOT = Path(__file__).resolve().parents[1]


class ComparisonChecks(unittest.TestCase):
    def test_page_and_choices(self):
        check_page(ROOT)
        for name, expected in [('qwen', 'qwen38-flash-affine4-1spark'),
                               ('glm', 'glm53-flash-adaptive-2spark')]:
            model, = options(ROOT, name)
            self.assertEqual([o['id'] for o in model['options'] if o['recommended']], [expected])
            self.assertGreater(len(model['options']), 1)

    def test_cli_json_and_invalid_model(self):
        command = [sys.executable, '-m', 'spark_serve', '--json', 'recipes', 'options', '--model']
        result = subprocess.run(command + ['qwen'], cwd=ROOT, check=True, capture_output=True, text=True)
        model, = json.loads(result.stdout)
        self.assertEqual(model['id'], 'qwen')
        self.assertTrue(model['options'][0]['recommended'])
        bad = subprocess.run(command + ['unknown'], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)

    def test_evidence_changes_page_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'comparisons', root / 'comparisons')
            # Copy just published evidence/configuration and experiment guides.
            for folder in ['recipes', 'experiments']:
                shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy(ROOT / 'README.md', root / 'README.md')
            check_page(root)
            path = root / 'recipes/glm53-flash-adaptive-2spark/results.json'
            data = json.loads(path.read_text())
            data['observations']['matched_pair_b_output_tokens_per_second']['adaptive_full_graph']['prose_c1'] = 12.34
            path.write_text(json.dumps(data))
            self.assertIn('12.34 tokens/s', render(root))
            with self.assertRaisesRegex(ConfigError, 'stale'):
                check_page(root)
            del data['observations']
            path.write_text(json.dumps(data))
            with self.assertRaises(ConfigError):
                load(root)

    def test_recommendation_and_public_boundary_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'comparisons', root / 'comparisons')
            path = root / 'comparisons/models.json'
            data = json.loads(path.read_text())
            data['models'][0]['recommended'] = 'missing-recipe'
            path.write_text(json.dumps(data))
            with self.assertRaises(ConfigError):
                load(root)
            # Resolve the actual files but supply a mutated data file.
            shutil.copytree(ROOT / 'recipes', root / 'recipes')
            shutil.copytree(ROOT / 'experiments', root / 'experiments')
            data = json.loads((ROOT / 'comparisons/models.json').read_text())
            data['tables']['public']['rows'][0]['boundary'] = ''
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ConfigError, 'boundary'):
                load(root)


if __name__ == '__main__':
    unittest.main()
