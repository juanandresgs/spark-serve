"""Deterministic, dependency-free SVG charts over the comparison table references."""
from __future__ import annotations

import json
import math
import re
import textwrap
from html import escape

from spark_serve.config import ConfigError
from spark_serve.recipes import asset


def chart_data(root, data):
    root = root.resolve()
    # Import lazily: the page renderer uses this module too.
    from spark_serve.comparisons import value
    spec = json.loads(asset(root, 'comparisons/charts.json').read_text())
    charts = []
    for chart in spec['charts']:
        if not re.fullmatch(r'comparisons/charts/[a-z0-9-]+\.svg', chart['file']):
            raise ConfigError('Chart must be an SVG in comparisons/charts')
        panels = []
        for panel in chart['panels']:
            table = data['tables'][panel['table']]
            if table['kind'] != 'matched-local':
                raise ConfigError('Charts require matched local evidence')
            matches = [row for row in table['rows'] if row['metric'] == panel['metric']]
            if len(matches) != 1:
                raise ConfigError('Chart metric must resolve to exactly one comparison row')
            row, = matches
            values = [value(root, c['refs'][panel.get('ref_index', 0)]) for c in row['cells']]
            if len(values) != len(chart['series']) or any(v < 0 for v in values):
                raise ConfigError('Chart requires nonnegative values and matching series')
            panels.append(dict(panel, values=values))
        charts.append(dict(chart, panels=panels))
    paths = [c['file'] for c in charts]
    if len(set(paths)) != len(paths):
        raise ConfigError('Duplicate chart output')
    return charts


def svg(chart):
    """Each workload has its own zero-based axis; labels never rely on color."""
    width = 600
    height = 164 + 248 * len(chart['panels'])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(chart["title"])}</title>',
             f'<desc id="desc">{escape(chart["description"])} ' + escape('; '.join(
                 p['label'] + ': ' + ', '.join(f'{s} {v:.3f} {p["unit"]}' for s, v in zip(chart['series'], p['values']))
                 for p in chart['panels'])) + '</desc>',
             '<rect width="100%" height="100%" rx="12" fill="#ffffff"/>',
             '<g font-family="Arial, Helvetica, sans-serif" fill="#172c3d">']

    def text(x, y, content, size=20, **attrs):
        attributes = ' '.join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{escape(content)}</text>')

    text(24, 36, chart['title'], 26, font_weight='bold')
    for j, line in enumerate(textwrap.wrap(chart['subtitle'], 57)):
        text(24, 66 + j * 24, line, 20)
    for i, panel in enumerate(chart['panels']):
        y = 124 + 248 * i
        text(24, y, panel['label'], 22, font_weight='bold')
        text(24, y + 24, panel['note'], 18)
        left, span = 24, 470
        maximum = max(panel['values'])
        # Rounded limits remain explicit. Every bar starts at zero.
        scale = 10 ** math.floor(math.log10(maximum)) if maximum else 1
        limit = math.ceil(maximum / scale) * scale or 1
        for j, (series, number) in enumerate(zip(chart['series'], panel['values'])):
            by = y + 52 + j * 60
            text(left, by, series, 20)
            parts.append(f'<rect x="{left}" y="{by + 8}" width="{number / limit * span:.3f}" height="25" rx="3" fill="{["#63778b", "#007f79"][j]}"/>')
            precision = 3 if maximum < 10 else 2
            text(left + number / limit * span + 8, by + 28, f'{number:.{precision}f}', 20, font_weight='bold')
        axis_y = y + 164
        parts.append(f'<path d="M {left} {axis_y} H {left + span}" stroke="#687c8c"/>')
        for fraction in (0, .5, 1):
            x = left + fraction * span
            parts.append(f'<path d="M {x} {axis_y} v 5" stroke="#687c8c"/>')
            text(x, axis_y + 26, f'{limit * fraction:g}', 20, text_anchor='middle')
        text(left + span / 2, axis_y + 49, panel['unit'], 18, text_anchor='middle')
    lines = textwrap.wrap(chart['footer'], 58)
    for j, line in enumerate(lines):
        text(24, height - 38 + j * 24, line, 18)
    parts.append('</g></svg>\n')
    return '\n'.join(parts)


def outputs(root, data):
    return {c['file']: svg(c) for c in chart_data(root, data)}
