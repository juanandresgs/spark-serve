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
        if not 1 <= len(chart['series']) <= 4:
            raise ConfigError('Charts support one to four explicitly labeled series')
        panels = []
        for panel in chart['panels']:
            table = data['tables'][panel['table']]
            if table['kind'] != 'matched-local':
                raise ConfigError('Charts require matched local evidence')
            matches = [row for row in table['rows'] if row['metric'] == panel['metric']]
            if len(matches) != 1:
                raise ConfigError('Chart metric must resolve to exactly one comparison row')
            row, = matches
            ref_index = panel.get('ref_index', 0)
            values = [value(root, c['refs'][ref_index], allow_null='null_text' in c) for c in row['cells']]
            if len(values) != len(chart['series']) or any(v is not None and v < 0 for v in values):
                raise ConfigError('Chart requires nonnegative values or explained missing values, and matching series')
            null_labels = [c.get('null_text', '') if v is None else '' for c, v in zip(row['cells'], values)]
            if any(v is None and not label.strip() for v, label in zip(values, null_labels)):
                raise ConfigError('Chart null values need a per-arm explanation')
            panels.append(dict(panel, values=values, null_labels=null_labels))
        charts.append(dict(chart, panels=panels))
    paths = [c['file'] for c in charts]
    if len(set(paths)) != len(paths):
        raise ConfigError('Duplicate chart output')
    return charts


def svg(chart):
    """Each workload has its own zero-based axis; labels never rely on color."""
    colors = ['#63778b', '#007f79', '#8c5b9b', '#bd5b36']
    width = 600
    height = 0  # Recomputed from wrapped labels and series below.
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(chart["title"])}</title>',
             f'<desc id="desc">{escape(chart["description"])} ' + escape('; '.join(
                 p['label'] + ': ' + ', '.join(
                     f'{s} {v:.3f} {p["unit"]}' if v is not None else f'{s} {p["null_labels"][i]}'
                     for i, (s, v) in enumerate(zip(chart['series'], p['values'])))
                 for p in chart['panels'])) + '</desc>',
             '<rect width="100%" height="100%" rx="12" fill="#ffffff"/>',
             '<g font-family="Arial, Helvetica, sans-serif" fill="#172c3d">']

    def text(x, y, content, size=20, **attrs):
        attributes = ' '.join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{escape(content)}</text>')

    title_lines = textwrap.wrap(chart['title'], 38, break_long_words=True) or ['']
    for j, line in enumerate(title_lines):
        text(24, 36 + j * 28, line, 26, font_weight='bold')
    subtitle_lines = textwrap.wrap(chart['subtitle'], 52, break_long_words=True) or ['']
    subtitle_top = 36 + 28 * len(title_lines) + 2
    for j, line in enumerate(subtitle_lines):
        text(24, subtitle_top + j * 24, line, 20)
    panel_top = max(124, subtitle_top + 24 * len(subtitle_lines) + 18)
    for panel in chart['panels']:
        y = panel_top
        label_lines = textwrap.wrap(panel['label'], 42, break_long_words=True) or ['']
        for j, line in enumerate(label_lines):
            text(24, y + j * 24, line, 22, font_weight='bold')
        note_lines = textwrap.wrap(panel['note'], 52, break_long_words=True) or ['']
        note_top = y + 24 * len(label_lines) + 2
        for j, line in enumerate(note_lines):
            text(24, note_top + j * 22, line, 18)
        left, span = 24, 470
        numeric_values = [v for v in panel['values'] if v is not None]
        maximum = max(numeric_values, default=0)
        # Rounded limits remain explicit. Every bar starts at zero.
        scale = 10 ** math.floor(math.log10(maximum)) if maximum else 1
        limit = math.ceil(maximum / scale) * scale or 1
        row_top = note_top + 22 * len(note_lines) + 8
        for j, (series, number) in enumerate(zip(chart['series'], panel['values'])):
            series_lines = textwrap.wrap(series, 34, break_long_words=True, break_on_hyphens=False) or ['']
            for line_number, line in enumerate(series_lines):
                text(left, row_top + 20 * line_number, line, 20)
            value_y = row_top + 20 * len(series_lines) + 4
            if number is None:
                reason_lines = textwrap.wrap(panel['null_labels'][j], 46, break_long_words=True) or ['No valid result']
                for line_number, line in enumerate(reason_lines):
                    text(left + 8, value_y + 20 * line_number, line, 18)
                row_top = value_y + max(28, 20 * len(reason_lines)) + 14
            else:
                parts.append(f'<rect x="{left}" y="{value_y - 16}" width="{number / limit * span:.3f}" height="25" rx="3" fill="{colors[j]}"/>')
                precision = 3 if maximum < 10 else 2
                text(left + number / limit * span + 8, value_y + 4, f'{number:.{precision}f}', 20, font_weight='bold')
                row_top = value_y + 38
        axis_y = row_top + 8
        parts.append(f'<path d="M {left} {axis_y} H {left + span}" stroke="#687c8c"/>')
        for fraction in (0, .5, 1):
            x = left + fraction * span
            parts.append(f'<path d="M {x} {axis_y} v 5" stroke="#687c8c"/>')
            text(x, axis_y + 26, f'{limit * fraction:g}', 20, text_anchor='middle')
        text(left + span / 2, axis_y + 49, panel['unit'], 18, text_anchor='middle')
        panel_top = axis_y + 88
    lines = textwrap.wrap(chart['footer'], 58)
    footer_y = panel_top + 22
    height = footer_y + 20 + 24 * max(0, len(lines) - 1)
    for j, line in enumerate(lines):
        text(24, footer_y + j * 24, line, 18)
    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
    parts.append('</g></svg>\n')
    return '\n'.join(parts)


def outputs(root, data):
    return {c['file']: svg(c) for c in chart_data(root, data)}
