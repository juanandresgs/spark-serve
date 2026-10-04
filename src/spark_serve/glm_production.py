"""Render the dated GLM production screen from its sanitized numeric projection."""
from __future__ import annotations

import json
import math

from spark_serve.comparison_charts import svg
from spark_serve.config import ConfigError
from spark_serve.recipes import asset

DATA = 'comparisons/glm-production-20261004.json'
CHART = 'comparisons/charts/glm-retention-20261004.svg'


def load(root):
    data = json.loads(asset(root, DATA).read_text())
    if data.get('schema_version') != 1 or set(data.get('arms', {})) != {'candidate', 'control'}:
        raise ConfigError('Invalid GLM production projection')
    if data['configuration']['active_requests_per_arm'] != 1:
        raise ConfigError('GLM sequence chart requires one active request per arm')
    for arm in data['arms'].values():
        rows = arm['rows']
        if len(rows) != 10 or len({r['task'] for r in rows}) != 10:
            raise ConfigError('GLM projection must preserve every sequence task')
        if sum(r['output_tokens'] for r in rows) != arm['total_output_tokens']:
            raise ConfigError('GLM output total disagrees with rows')
        if sum(r['strict_pass'] for r in rows) != arm['strict_passes']:
            raise ConfigError('GLM strict score disagrees with rows')
        if not math.isclose(arm['total_output_tokens'] / arm['sequence_seconds'],
                            arm['aggregate_output_tokens_per_second'], rel_tol=1e-5):
            raise ConfigError('GLM aggregate rate disagrees with wall time')
    a, b = data['arms']['candidate'], data['arms']['control']
    reduction = 100 * (b['sequence_seconds'] - a['sequence_seconds']) / b['sequence_seconds']
    if not math.isclose(reduction, data['comparison']['observed_elapsed_reduction_percent'], abs_tol=.001):
        raise ConfigError('GLM elapsed reduction disagrees with receipt projection')
    if data['comparison']['equally_successful_completed_workloads'] is not False:
        raise ConfigError('GLM quality limitation must be explicit')
    prior = data['prior_interleave_screen']['arms']
    if set(prior) != {'v18', 'interleave12'}:
        raise ConfigError('GLM prior screen requires V18 and interleave arms')
    for arm in prior.values():
        for cohort in ('cold_c8', 'warm_c16'):
            result = arm[cohort]
            if not math.isclose(result['output_tokens'] / result['wall_seconds'],
                                result['aggregate_output_tokens_per_second'], rel_tol=1e-4):
                raise ConfigError('GLM prior aggregate rate disagrees with wall time')
    return data


def row(arm, task):
    return next(r for r in arm['rows'] if r['task'] == task)


def chart_output(data):
    control, candidate = data['arms']['control'], data['arms']['candidate']
    def panel(label, note, unit, numbers):
        return {'label': label, 'note': note, 'unit': unit, 'values': numbers,
                'null_labels': ['', '']}
    chart = {
        'title': 'GLM retention: one ten-request sequence per pair',
        'subtitle': 'October 4, 2026 · separate two-Spark pairs · one active request per arm',
        'description': 'Observed cross-pair screening. Lower elapsed and first-output times are better. '
                       'The arms differ in output length and task success; no cold-prefill gain is established.',
        'series': ['r4 + LRU control', 'r4 + recompute retention'],
        'panels': [
            panel('Full sequence elapsed', 'Same 8192-token cap, unequal completed work', 'seconds',
                  [control['sequence_seconds'], candidate['sequence_seconds']]),
            panel('Cold 841K request: first output', 'Zero cached tokens on both arms', 'seconds',
                  [row(control, 'long-prime')['ttft_seconds'], row(candidate, 'long-prime')['ttft_seconds']]),
            panel('Long follow-up: complete response', 'Control cache miss; candidate retained prefix', 'seconds',
                  [row(control, 'long-continuation')['complete_seconds'],
                   row(candidate, 'long-continuation')['complete_seconds']]),
        ],
        'footer': 'One screening sequence, no hardware swap. Each arm failed both long strict-JSON checks.',
    }
    return svg(chart)


def summary(data):
    control, candidate = data['arms']['control'], data['arms']['candidate']
    cold_b, cold_a = row(control, 'long-prime'), row(candidate, 'long-prime')
    follow_b, follow_a = row(control, 'long-continuation'), row(candidate, 'long-continuation')
    def pair(left, right, unit='', precision=3):
        return f'{left:,.{precision}f}{unit} | {right:,.{precision}f}{unit}'
    return '\n'.join([
        '| Observed measure | r4 + LRU control | r4 + retention candidate |',
        '| --- | ---: | ---: |',
        '| Ten-request sequence elapsed | ' + pair(control['sequence_seconds'], candidate['sequence_seconds'], ' s') + ' |',
        '| Sequence output tokens | ' + f"{control['total_output_tokens']:,} | {candidate['total_output_tokens']:,}" + ' |',
        '| Sequence output tokens / full wall second | ' + pair(control['aggregate_output_tokens_per_second'], candidate['aggregate_output_tokens_per_second'], ' tok/s') + ' |',
        '| Strict task passes | ' + f"{control['strict_passes']}/10 | {candidate['strict_passes']}/10" + ' |',
        '| Cold 841K first output; cached input | ' +
            f"{cold_b['ttft_seconds']:.3f} s; {cold_b['cached_tokens']:,} | {cold_a['ttft_seconds']:.3f} s; {cold_a['cached_tokens']:,} |",
        '| Long follow-up complete; cached input | ' +
            f"{follow_b['complete_seconds']:.3f} s; {follow_b['cached_tokens']:,} / {follow_b['input_tokens']:,} | " +
            f"{follow_a['complete_seconds']:.3f} s; {follow_a['cached_tokens']:,} / {follow_a['input_tokens']:,} |",
    ])


def prior_summary(data):
    prior = data['prior_interleave_screen']['arms']
    v18, r4 = prior['v18'], prior['interleave12']
    cold_v, cold_r = v18['cold_c8'], r4['cold_c8']
    warm_v, warm_r = v18['warm_c16'], r4['warm_c16']
    long_v, long_r = cold_v['near_limit'], cold_r['near_limit']
    return '\n'.join([
        '| Production-pair screen | V18 baseline | r4 / interleave12 |',
        '| --- | ---: | ---: |',
        f"| Cold C8 full cohort elapsed | {cold_v['wall_seconds']:,.3f} s | {cold_r['wall_seconds']:,.3f} s |",
        f"| Cold C8 output / elapsed | {cold_v['output_tokens']:,} / {cold_v['aggregate_output_tokens_per_second']:.3f} tok/s | {cold_r['output_tokens']:,} / {cold_r['aggregate_output_tokens_per_second']:.3f} tok/s |",
        f"| Cold C8 strict passes | {cold_v['strict_passes']}/{cold_v['requests']} | {cold_r['strict_passes']}/{cold_r['requests']} |",
        f"| 841K code request first output / complete | {long_v['ttft_seconds']:,.3f} / {long_v['complete_seconds']:,.3f} s | {long_r['ttft_seconds']:,.3f} / {long_r['complete_seconds']:,.3f} s |",
        f"| Warm C16 full cohort elapsed | {warm_v['wall_seconds']:,.3f} s | {warm_r['wall_seconds']:,.3f} s |",
        f"| Warm C16 output / elapsed | {warm_v['output_tokens']:,} / {warm_v['aggregate_output_tokens_per_second']:.3f} tok/s | {warm_r['output_tokens']:,} / {warm_r['aggregate_output_tokens_per_second']:.3f} tok/s |",
        f"| Warm C16 strict passes | {warm_v['strict_passes']}/{warm_v['requests']} | {warm_r['strict_passes']}/{warm_r['requests']} |",
    ])
