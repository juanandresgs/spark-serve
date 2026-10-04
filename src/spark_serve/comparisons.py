"""Focused recipe choices and evidence-backed Markdown tables; no live operations."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import re
from pathlib import Path

from spark_serve.config import ConfigError
from spark_serve import evidence
from spark_serve.recipes import asset, read_recipe, source_root
from spark_serve.comparison_charts import outputs as chart_outputs
from spark_serve import glm_production


def value(root, reference, allow_null=False):
    result = json.loads(asset(root, reference['file']).read_text())
    for key in reference['path']:
        result = result[key]
    if result is None:
        if allow_null:
            return None
        raise ValueError('Null comparison values need a nonempty explanation')
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError('Comparison evidence must resolve to a finite number')
    return result


def cell(root, spec):
    if '%' in spec['format']:
        raise ValueError('Percentage cells require the evidence.percent_change comparability gate')
    null_text = spec.get('null_text')
    if null_text is not None and (not isinstance(null_text, str) or not null_text.strip()):
        raise ValueError('Null comparison values need a nonempty explanation')
    for ref in spec['refs']:
        if not ref['file'].startswith('evidence/runs/') or ref['path'][0] != 'measurements' or ref['path'][-1] != 'value':
            raise ValueError('Comparison values must reference structured run measurements')
    values = [value(root, ref, allow_null=null_text is not None) for ref in spec['refs']]
    if any(item is None for item in values):
        if null_text is None:
            raise ValueError('Null comparison values need a nonempty explanation')
        return null_text
    return spec['format'].format(*values)


def load(root):
    root = root.resolve()
    try:
        data = json.loads(asset(root, 'comparisons/models.json').read_text())
        if data['schema_version'] != 1:
            raise ValueError('Unknown comparison schema')
        records = evidence.load(root, schema_check=False)
        ids = [m['id'] for m in data['models']]
        if len(ids) != len(set(ids)) or set(ids) != {'qwen', 'glm'}:
            raise ValueError('Expected unique Qwen and GLM model choices')
        for model in data['models']:
            decision = records[model['recommendation_record']]
            selected = records[decision['selected']['id']]
            if (decision['kind'] != 'recommendation' or decision['review']['status'] != 'reviewed'
                    or selected['catalog_id'] != model['recommended']
                    or decision['hardware_nodes'] != model['sparks']
                    or decision['model'] != model['id']):
                raise ValueError('Model choice disagrees with reviewed recommendation')
            options = model['options']
            option_ids = [o['id'] for o in options]
            if len(set(option_ids)) != len(option_ids) or model['recommended'] not in option_ids:
                raise ValueError('Each model needs unique options and one recommendation')
            if type(model['sparks']) is not int or model['sparks'] < 1:
                raise ValueError('Invalid Spark count')
            for option in options:
                asset(root, option['guide'])
                if 'recipe' in option:
                    recipe = read_recipe(root, option['recipe'])
                    if recipe['guide'] != option['guide']:
                        raise ValueError('Option guide differs from recipe guide')
                    if option['id'] == model['recommended'] and recipe['status'] != 'recommended-with-caveats':
                        raise ValueError('Recommended option must have recommended recipe status')
                elif option['id'] == model['recommended']:
                    raise ValueError('Recommended option requires a catalog recipe')
        for key, table in data['tables'].items():
            if not table['scope']:
                raise ValueError('Comparison needs its qualification scope')
            if key == 'public':
                sources = json.loads(asset(root, table['sources_file']).read_text())['sources']
                if table['kind'] != 'unmatched-public' or len(sources) != 2:
                    raise ValueError('Public comparison requires attribution and unmatched status')
                for row in table['rows']:
                    if not row['boundary']:
                        raise ValueError('Public row needs a comparison boundary')
                    cell(root, row['public'])
                    if 'local' in row:
                        cell(root, row['local'])
                    elif not isinstance(row.get('local_pending'), str) or not row['local_pending'].strip():
                        raise ValueError('Public row needs a local result or an explicit pending explanation')
            else:
                if table['kind'] != 'matched-local':
                    raise ValueError('Unknown local comparison kind')
                for row in table['rows']:
                    if len(row['cells']) != len(table['columns']) - 1:
                        raise ValueError('Comparison row has the wrong number of columns')
                    for spec in row['cells']:
                        cell(root, spec)
        return data
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ConfigError(f'Invalid comparison data: {exc}') from exc


def options(root, model_id=None):
    data = load(root)
    return [dict(model, options=[dict(option, recommended=option['id'] == model['recommended'])
                                for option in model['options']])
            for model in data['models'] if model_id is None or model['id'] == model_id]


def table(headers, rows):
    def line(cells):
        return '| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in cells) + ' |'
    return '\n'.join([line(headers), line(['---'] * len(headers)), *(line(row) for row in rows)])


def render(root):
    root = root.resolve()
    data = load(root)
    choices = []
    for model in data['models']:
        chosen = next(o for o in model['options'] if o['id'] == model['recommended'])
        hardware = f"{model['sparks']} Spark" + ('s' if model['sparks'] > 1 else '')
        choices.append([hardware, f"[**{model['name']} · {chosen['label']}**]({chosen['guide']})"])
    rendered = {'choices': table(['Your hardware', 'Recommended recipe · build and run'], choices)}
    production = glm_production.load(root)
    rendered['glm-production-chart'] = ('![GLM retention sequence, cold first-output, and cached '
                                        'follow-up comparison; numeric equivalent below]('
                                        + glm_production.CHART + ')')
    rendered['glm-production-summary'] = glm_production.summary(production)
    rendered['glm-production-reduction'] = f"{production['comparison']['observed_elapsed_reduction_percent']:.3f}%"
    rendered['glm-prior-summary'] = glm_production.prior_summary(production)
    for name in ('qwen-final-v2-throughput', 'qwen-restored-throughput', 'qwen-throughput', 'glm-throughput', 'qwen-waiting', 'qwen-tails'):
        rendered[name] = f'![{name.replace("-", " ")} comparison; numeric equivalent in the table below](comparisons/charts/{name}.svg)'
    def row(table_id, metric):
        matches = [r for r in data['tables'][table_id]['rows'] if r['metric'] == metric]
        if len(matches) != 1:
            raise ConfigError('Headline metric must resolve to exactly one comparison row')
        return matches[0]
    def pair(table_id, metric):
        return [cell(root, c) for c in row(table_id, metric)['cells']]
    cold = pair('qwen', 'Cold 253,843-token prompt, complete JSON response ↓')
    mixed = pair('qwen', 'Short-request p95 during mixed long/short traffic ↓')
    rendered['gains'] = (f"In the September 29 test, Affine4 completed the fresh long prompt in **{cold[1]}**, "
                         f"versus **{cold[0]}** for EXL3. Mixed-traffic short-request p95 was "
                         f"**{mixed[1]}** for Affine4 and **{mixed[0]}** for EXL3.")
    prose_c1 = pair('glm', 'Single-request prose')
    prose_c8 = pair('glm', 'Eight-request aggregate prose')
    rendered['glm-gains'] = (f"Recorded prose output rates were **{prose_c1[1]}** with adaptive drafting "
                             f"versus **{prose_c1[0]}** with fixed drafts at one request; "
                             f"**{prose_c8[1]}** versus **{prose_c8[0]}** across eight requests.")
    rendered['alternatives'] = '\n'.join(f"- [{model['name']} · {option['label']}]({option['guide']}): {option['reason']}"
                                         for model in data['models'] for option in model['options']
                                         if option['id'] != model['recommended'])
    for key, spec in data['tables'].items():
        rows = []
        for row in spec['rows']:
            if spec['kind'] == 'unmatched-public':
                local = cell(root, row['local']) if 'local' in row else row['local_pending']
                rows.append([row['metric'], cell(root, row['public']), local, row['boundary']])
            else:
                rows.append([row['metric'], *(cell(root, c) for c in row['cells'])])
        rendered[key] = table(spec['columns'], rows) + '\n\n' + spec['scope']
        if key == 'public':
            rendered[key] += '\n\nSources: ' + ', '.join(f'[Pinned {name} benchmark README]({url})' for name, url in zip(('Qwen', 'GLM'), json.loads(asset(root, spec['sources_file']).read_text())['sources'])) + '.'
    template = asset(root, 'comparisons/README.template.md').read_text()
    for key, content in rendered.items():
        marker = '{{' + key + '}}'
        if template.count(marker) != 1:
            raise ConfigError(f'Expected exactly one template marker: {marker}')
        template = template.replace(marker, content)
    if re.search(r'\{\{\w+\}\}', template):
        raise ConfigError('Unknown comparison template marker')
    return '<!-- Generated by spark_serve.comparisons; edit comparisons/README.template.md and models.json. -->\n' + template


def check_page(root):
    root = root.resolve()
    if (root / 'README.md').read_text() != render(root):
        raise ConfigError('README tables are stale: run PYTHONPATH=src python3 -m spark_serve.comparisons')
    expected = chart_outputs(root, load(root))
    expected[glm_production.CHART] = glm_production.chart_output(glm_production.load(root))
    for rel, content in expected.items():
        path = root / rel
        if not path.is_file() or path.read_text() != content:
            raise ConfigError(f'Chart is stale: {rel}; run PYTHONPATH=src python3 -m spark_serve.comparisons')
    if {str(p.relative_to(root)) for p in (root / 'comparisons/charts').glob('*.svg')} != set(expected):
        raise ConfigError('Unexpected generated chart; remove obsolete SVGs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = source_root(args.source)
    if args.check:
        check_page(root)
        print('Comparison data and generated page are consistent')
    else:
        generated = {'README.md': render(root), **chart_outputs(root, load(root)),
                     glm_production.CHART: glm_production.chart_output(glm_production.load(root))}
        before = {rel: (root / rel).read_bytes() if (root / rel).exists() else None for rel in generated}
        for rel, content in generated.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            try:
                current = target.read_bytes() if target.exists() else None
                if current != before[rel]:
                    raise ConfigError(f'{rel} changed during generation; retry from the current files')
                temporary.chmod(target.stat().st_mode & 0o777 if target.exists() else 0o644)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        print('Updated README.md and charts from comparison data and evidence')


if __name__ == '__main__':
    main()
