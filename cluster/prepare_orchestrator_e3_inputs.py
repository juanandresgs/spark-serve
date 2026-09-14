#!/usr/bin/env python3
"""Prepare reviewed E3 build inputs offline; never build, install or deploy."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'cluster/agent-fleet-e3'


def patched(source, patch_name, expected_input, expected_output):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_input:
        raise ValueError(f'Baseline drift: {source}')
    with tempfile.TemporaryDirectory(prefix='e3-input-') as scratch:
        target = Path(scratch) / 'input.py'
        target.write_bytes(data)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-d', scratch],
                       input=(ASSETS / patch_name).read_bytes(), check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = target.read_bytes()
    if hashlib.sha256(result).hexdigest() != expected_output:
        raise ValueError(f'Patched output drift: {patch_name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--e2-integration', type=Path, required=True)
    parser.add_argument('--e2-selfcheck', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((ASSETS / 'provenance.json').read_text())
    actual = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != provenance['source_commit']:
        raise ValueError('Source commit differs from reviewed pin')
    inputs = {}
    for name, digest in provenance['source_files'].items():
        data = (args.source / 'overlay' / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Reviewed source drift: {name}')
        inputs[name] = data
    inputs['exl3.py'] = patched(args.e2_integration, 'exl3-e3-only.patch',
        'c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843',
        'aba22fcf3eb8beeff4eb61e127ad59e8e37a4a2a6f2a77bd67cbb5dff65e8f15')
    inputs['test_e2_regression.py'] = patched(args.e2_selfcheck, 'e2-regression.patch',
        '21af9e305adcfc0ad991ce82f5c7b4e23e8b59903e2a28b4956c1ef08065abe2',
        '6bf6b23cb7cd452ba7ebe807ba70f891c405251d5fbefe3bdb20145ffbafb4cc')
    for name in ('Dockerfile', 'selfcheck.py', 'e3-trial-check.py'):
        inputs[name] = (ASSETS / name).read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in inputs.items():
        (args.output / name).write_bytes(data)
    print(json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()}, indent=2))


if __name__ == '__main__':
    main()
