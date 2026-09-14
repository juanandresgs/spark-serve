#!/usr/bin/env python3
"""Build the pinned native resident-ready router, including its upstream UI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from build_images import source_copy


def build(root, source, output):
    pin = json.loads((root / 'sources.json').read_text())['sources']['llama-swap']['revision']
    source_copy(source, output, pin)
    patch = root / 'cluster/glm-throughput/router/ready-strategy.patch'
    expected = '82dfce2e148576c726f0acb0a83ea74d8a73b02d79d66001185388545ea320ff'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
        raise ValueError('Router patch drift')
    build_env = os.environ | {'GOCACHE': str(output / '.go-cache')}
    def run(argv, cwd=output, env=None):
        subprocess.run(argv, cwd=cwd, env=env or build_env, check=True)
    run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch.absolute())])
    version = subprocess.check_output(['go', 'version'], text=True)
    if 'go1.26.5 ' not in version:
        raise ValueError('Use the recorded Go 1.26.5 toolchain')
    run(['npm', 'ci', '--ignore-scripts', '--cache', str(output / '.npm-cache'), '--no-audit', '--no-fund'], output / 'ui-svelte')
    run(['npm', 'run', 'build'], output / 'ui-svelte')
    run(['go', 'test', '-race', './...'])
    binary = output / 'llama-swap-linux-arm64'
    environment = build_env | {'GOOS': 'linux', 'GOARCH': 'arm64', 'CGO_ENABLED': '0'}
    run(['go', 'build', '-trimpath', '-tags', 'embed_ui',
         '-ldflags=-X main.commit=60226b6-spark-serve -X main.version=v250-spark-serve -X main.date=2026-09-14',
         '-o', str(binary), '.'], env=environment)
    receipt = {'source': pin, 'patch_sha256': expected, 'target': 'linux/arm64', 'go': version.strip(),
               'tests': 'go test -race ./...', 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
               'private_guest_extension': False}
    binary.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.root.resolve(), a.source.resolve(), a.output.absolute()), indent=2))
