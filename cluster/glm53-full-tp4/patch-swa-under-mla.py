#!/usr/bin/env python3
"""Allow a DFlash2 sliding-window draft layer under an MLA target model."""

from pathlib import Path
import ast
import sys


path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/attention/attention.py"
)
source = path.read_text()
marker = "GLM53-FULL-SWA-UNDER-MLA"
if marker not in source:
    old = (
        "        if self.sliding_window is not None:\n"
        "            assert not vllm_config.model_config.use_mla, (\n"
        '                "MLA is not supported for slidingwindow"\n'
        "            )\n"
    )
    if source.count(old) != 1:
        raise SystemExit(f"SWA-under-MLA anchor count is {source.count(old)}, expected 1")
    new = (
        "        if self.sliding_window is not None:\n"
        f"            # {marker}: this non-MLA layer belongs to the DFlash2 drafter.\n"
    )
    source = source.replace(old, new)
ast.parse(source, filename=str(path))
path.write_text(source)

