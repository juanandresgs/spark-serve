"""Exercise real evidence resolution, generated drift detection and CLI discovery."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from spark_serve.comparisons import cell, check_page, load, options, render
from spark_serve.config import ConfigError
from spark_serve.comparison_charts import chart_data, outputs as chart_outputs, svg
from spark_serve import glm_production
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class ComparisonChecks(unittest.TestCase):
    def test_page_and_choices(self):
        check_page(ROOT)
        for name, expected in [('qwen', 'qwen-tensorfold-native-exl3'),
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

    def test_public_reference_has_separate_local_decode_proxy(self):
        data = load(ROOT)
        row, = [r for r in data['tables']['public']['rows']
                if r['metric'] == 'Qwen C1 prose decode']
        self.assertIn('local', row)
        self.assertNotIn('local_pending', row)
        rendered = render(ROOT)
        self.assertIn('62.4 tokens/s', rendered)
        self.assertIn('51.99 tokens/s', rendered)
        self.assertIn('These timing definitions and source details differ', rendered)

    def test_charts_follow_evidence_and_detect_chart_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ['comparisons', 'recipes', 'experiments', 'evidence']:
                shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy(ROOT / 'README.md', root / 'README.md')
            chart = root / 'comparisons/charts/qwen-throughput.svg'
            chart.write_text(chart.read_text().replace('92.64', '999.99'))
            with self.assertRaisesRegex(ConfigError, 'stale'):
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
        self.assertEqual(len(charts), 6)
        for chart in charts:
            self.assertTrue(all(p['table'] != 'public' for p in chart['panels']))
            parsed = ET.fromstring(chart_outputs(ROOT, data)[chart['file']])
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            self.assertEqual(parsed.attrib['role'], 'img')
            self.assertTrue(parsed.find('svg:title', ns).text)
            self.assertTrue(parsed.find('svg:desc', ns).text)
            # All evidence bars share a true zero origin. No truncated bars.
            bars = [r for r in parsed.findall('.//svg:rect', ns) if r.attrib.get('height') == '25']
            self.assertEqual(len(bars), len(chart['series']) * len(chart['panels']))
            self.assertTrue(all(r.attrib['x'] == '24' for r in bars))
        tail = next(c for c in charts if c['file'].endswith('qwen-tails.svg'))
        self.assertEqual([round(v, 2) for v in tail['panels'][1]['values']], [25.06, 243.56])

    def test_glm_production_projection_drives_chart_and_table(self):
        data = glm_production.load(ROOT)
        candidate = data['arms']['candidate']
        control = data['arms']['control']
        self.assertEqual(candidate['strict_passes'], 8)
        self.assertEqual(control['strict_passes'], 7)
        self.assertEqual(glm_production.row(candidate, 'long-prime')['cached_tokens'], 0)
        self.assertEqual(glm_production.row(candidate, 'long-continuation')['cached_tokens'], 841604)
        self.assertEqual(glm_production.row(control, 'long-continuation')['cached_tokens'], 0)
        self.assertIn('5.047 s', glm_production.summary(data))
        parsed = ET.fromstring(glm_production.chart_output(data))
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        self.assertEqual(parsed.attrib['role'], 'img')
        self.assertEqual(len([r for r in parsed.findall('.//svg:rect', ns)
                              if r.attrib.get('height') == '25']), 6)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ['comparisons', 'recipes', 'experiments', 'evidence']:
                shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy(ROOT / 'README.md', root / 'README.md')
            path = root / glm_production.DATA
            projection = json.loads(path.read_text())
            projection['arms']['candidate']['rows'][0]['ttft_seconds'] += 1
            path.write_text(json.dumps(projection))
            with self.assertRaisesRegex(ConfigError, 'stale'):
                check_page(root)

    def test_four_arm_chart_and_explained_null_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / 'recipes').mkdir()
            (root / 'comparisons').mkdir()
            (root / 'evidence/runs').mkdir(parents=True)
            values = [111.0, None, 222.0, 333.0]
            missing_reason = ('Failed qualification: typed-tool gate did not pass; request produced no valid '
                              'typed arguments and could not enter the timed cohort.')
            cells = []
            for index, number in enumerate(values):
                relative = f'evidence/runs/arm-{index}.json'
                (root / relative).write_text(json.dumps({'measurements': [{'value': number}]}))
                cells.append({'refs': [{'file': relative, 'path': ['measurements', 0, 'value']}],
                              'format': '{0:.2f}',
                              **({'null_text': missing_reason} if number is None else {})})
            self.assertEqual(cell(root, {'refs': cells[1]['refs'], 'format': '{0:.2f}',
                                         'null_text': missing_reason}), missing_reason)
            with self.assertRaisesRegex(ValueError, 'nonempty explanation'):
                cell(root, {'refs': cells[1]['refs'], 'format': '{0:.2f}'})

            names = ['Legacy cooperative EXL3', 'Native EXL3 overlap scheduling arm with long label',
                     'Native EXL3 whole-pass scheduling arm', 'Retained Affine4']
            chart = {
                'file': 'comparisons/charts/synthetic.svg',
                'title': 'Synthetic renderer fixture', 'subtitle': 'No benchmark results',
                'description': 'Synthetic layout coverage only.', 'series': names,
                'footer': 'Synthetic values are not published evidence.',
                'panels': [{'table': 'overnight', 'metric': 'C1 group throughput',
                            'label': 'Synthetic test', 'note': 'Layout fixture', 'unit': 'tokens/s'}]
            }
            (root / 'comparisons/charts.json').write_text(json.dumps({'charts': [chart]}))
            data = {'tables': {'overnight': {'kind': 'matched-local', 'rows': [
                {'metric': 'C1 group throughput', 'cells': cells}]}}}
            mapped, = chart_data(root, data)
            self.assertEqual(mapped['panels'][0]['values'], values)
            self.assertEqual(mapped['panels'][0]['null_labels'][1], cells[1]['null_text'])
            rendered = svg(mapped)
        parsed = ET.fromstring(rendered)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        self.assertEqual(int(parsed.attrib['width']), 600)
        rects = [r for r in parsed.findall('.//svg:rect', ns) if r.attrib.get('height') == '25']
        self.assertEqual(len(rects), 3)
        self.assertEqual(len({r.attrib['fill'] for r in rects}), 3)
        visible_text = ' '.join(n.text or '' for n in parsed.findall('.//svg:text', ns))
        self.assertIn(names[1], visible_text)
        self.assertIn(missing_reason, visible_text)

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
