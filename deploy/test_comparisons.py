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
from spark_serve.comparison_charts import chart_data, outputs as chart_outputs
import xml.etree.ElementTree as ET

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
            for folder in ['recipes', 'experiments', 'evidence']:
                shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy(ROOT / 'README.md', root / 'README.md')
            check_page(root)
            record = json.loads((root / 'evidence/runs/import-20260930-glm-0-1-v1.json').read_text())
            path = root / record['measurements'][0]['source']['file']
            data = json.loads(path.read_text())
            data['observations']['matched_pair_b_output_tokens_per_second']['adaptive_full_graph']['prose_c1'] = 12.34
            path.write_text(json.dumps(data))
            # Imported receipts are pinned: changing history is an integrity error, not a new result.
            with self.assertRaisesRegex(ConfigError, 'Source receipt changed'):
                render(root)
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
            shutil.copytree(ROOT / 'evidence', root / 'evidence')
            data = json.loads((ROOT / 'comparisons/models.json').read_text())
            data['tables']['public']['rows'][0]['boundary'] = ''
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ConfigError, 'boundary'):
                load(root)

    def test_charts_follow_evidence_and_detect_chart_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ['comparisons', 'recipes', 'experiments', 'evidence']:
                shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy(ROOT / 'README.md', root / 'README.md')
            chart = root / 'comparisons/charts/qwen-throughput.svg'
            chart.write_text(chart.read_text().replace('92.64', '999.99'))
            with self.assertRaisesRegex(ConfigError, 'Chart is stale'):
                check_page(root)
            record = json.loads((root / 'evidence/runs/import-20260930-qwen-0-1-v1.json').read_text())
            path = root / record['measurements'][0]['source']['file']
            data = json.loads(path.read_text())
            data['reliability-affine-packaged-speed']['prose']['median_aggregate_tps'] = 123.45
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ConfigError, 'Source receipt changed'):
                chart_outputs(root, load(root))

    def test_chart_structure_and_boundaries(self):
        data = load(ROOT)
        charts = chart_data(ROOT, data)
        self.assertEqual(len(charts), 4)
        for chart in charts:
            self.assertTrue(all(p['table'] != 'public' for p in chart['panels']))
            parsed = ET.fromstring(chart_outputs(ROOT, data)[chart['file']])
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            self.assertEqual(parsed.attrib['role'], 'img')
            self.assertTrue(parsed.find('svg:title', ns).text)
            self.assertTrue(parsed.find('svg:desc', ns).text)
            # All evidence bars share a true zero origin. No truncated bars.
            bars = [r for r in parsed.findall('.//svg:rect', ns) if r.attrib.get('height') == '25']
            self.assertEqual(len(bars), 2 * len(chart['panels']))
            self.assertTrue(all(r.attrib['x'] == '24' for r in bars))
        tail = next(c for c in charts if c['file'].endswith('qwen-tails.svg'))
        self.assertEqual([round(v, 2) for v in tail['panels'][1]['values']], [25.06, 243.56])

    def test_chart_selection_rejects_unmatched_evidence_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'comparisons', root / 'comparisons')
            path = root / 'comparisons/charts.json'
            spec = json.loads(path.read_text())
            spec['charts'][0]['panels'][0]['table'] = 'public'
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ConfigError, 'matched local'):
                chart_data(root, load(ROOT))
            spec['charts'][0]['file'] = 'comparisons/charts/../../escape.svg'
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ConfigError, 'SVG in comparisons/charts'):
                chart_data(root, load(ROOT))


if __name__ == '__main__':
    unittest.main()
