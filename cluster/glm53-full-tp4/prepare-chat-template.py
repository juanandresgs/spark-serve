#!/usr/bin/env python3
"""Apply the proven thinking-off fix to the pinned official GLM-5.3 template."""

from pathlib import Path
import sys


source_path = Path(sys.argv[1])
destination_path = Path(sys.argv[2])
source = source_path.read_text()
old = "    <|assistant|>{{- '<think>' -}}"
new = (
    "    <|assistant|>{%- if enable_thinking is defined and not enable_thinking -%}"
    "{{- '<think></think>' -}}{%- else -%}{{- '<think>' -}}{%- endif -%}"
)
if source.count(old) != 1:
    raise SystemExit("official template generation-prompt anchor drifted")
destination_path.write_text(source.replace(old, new))

