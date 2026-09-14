#!/usr/bin/env python3
"""Stage runtime source or pinned weights on one Spark; never start a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from build_images import module, source_copy
from prepare_runtime import normalize

MODELS = {
    'glm-adaptive': [('Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw', '25a44fdbf16862a46b7cc9921142c6c81350af2f'), ('incoai/GLM-5.3-Flash-DFlash2', 'dc77ff1c99eeb2df044ee3d4f0094eb033fee410')],
    'qwen-single': [('Mia-AiLab/Qwen3.8-Flash-Next-NVFP4', '925d7be6c14c6c9442ef83e8f05b5a3c39304f69')],
    'glm-full': [('Tech2wild/GLM-5.3-Int4-Int8Mix', '206507bbb047d8223964a0414cd83230c59428f9'), ('incoai/GLM-5.3-DFlash2', '425aa615ce320caac34400208b30808c8f14f76c')],
}


def runtime(root, sources, output, engine):
    if engine == 'glm-adaptive':
        rep = module(root / 'reproduce.py', 'stage_reproduce')
        rep.prepare_e2(root, sources / 'glm-e2', output)
        return normalize(output, engine)
    if engine == 'qwen-single':
        pins = json.loads((root / 'sources.json').read_text())
        source_copy(sources / engine, output, pins['sources'][engine]['revision'])
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str((root / 'cluster/agent-fleet/qwen-loopback.patch').absolute())], cwd=output, check=True)
        return normalize(output, engine)
    from prepare_full import prepare
    return prepare(root, sources, output)


def weights(engine, hf_home, runtime_root, execute=False):
    commands = [['hf', 'download', repo, '--revision', rev, '--cache-dir', str(hf_home / 'hub')]
                for repo, rev in MODELS[engine]]
    if not execute:
        return {'engine': engine, 'commands': commands, 'weights_in_release': False,
                'download_executed': False, 'donor_transplant_required': engine == 'glm-adaptive'}
    for command in commands:
        subprocess.run(command, check=True)
    targets = []
    for repo, rev in MODELS[engine]:
        snapshot = hf_home / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / rev
        config = snapshot / 'config.json'
        if not config.is_file():
            raise ValueError('Pinned config missing')
        index = snapshot / 'model.safetensors.index.json'
        names = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        if not names or any(not (snapshot / n).is_file() or (snapshot / n).stat().st_size < 1024 for n in names):
            raise ValueError('Pinned checkpoint has missing or empty shards')
        targets.append({'repository': repo, 'revision': rev, 'snapshot': str(snapshot),
                        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                        'files': {n: (snapshot / n).stat().st_size for n in sorted(names)}})
    receipt = {'engine': engine, 'targets': targets, 'hardware_qualified': False}
    if engine == 'glm-adaptive':
        env = os.environ | {'ABLIT_DONOR': 'dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4', 'ABLIT_DONOR_REVISION': 'e90ef415a2bf1274f848965a61144eca1e99eabf'}
        subprocess.run(['python3', str(runtime_root / 'ablit/fetch_transplant.py')], env=env, check=True)
        manifest = json.loads((runtime_root / 'ablit/transplant/MANIFEST.json').read_text())
        if manifest['donor_sha'] != env['ABLIT_DONOR_REVISION'] or set(manifest['layers']) != {str(x) for x in range(15, 46)}:
            raise ValueError('Donor transplant identity mismatch')
        for layer, item in manifest['layers'].items():
            with (runtime_root / f'ablit/transplant/L{layer}.bin').open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError('Donor tensor checksum mismatch')
        receipt['donor'] = manifest
    (runtime_root / 'artifact-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('runtime')
    a.add_argument('engine', choices=MODELS)
    a.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a.add_argument('--sources', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('weights')
    b.add_argument('engine', choices=MODELS)
    b.add_argument('--hf-home', type=Path, required=True)
    b.add_argument('--runtime', type=Path, required=True)
    b.add_argument('--execute', action='store_true')
    a = p.parse_args()
    result = runtime(a.root.resolve(), a.sources.resolve(), a.output.absolute(), a.engine) if a.action == 'runtime' else weights(a.engine, a.hf_home.absolute(), a.runtime.absolute(), a.execute)
    print(json.dumps(result, indent=2))
