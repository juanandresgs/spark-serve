#!/usr/bin/env python3
"""Verify staged source and pinned checkpoint inventories without loading weights."""
import hashlib
import json
from pathlib import Path
import sys
from stage import MODELS


def verify(runtime, engine):
    source = json.loads((runtime / 'runtime-source.json').read_text())
    if source['engine'] != engine:
        raise ValueError('Prepared runtime engine mismatch')
    for rel, sha in source['files'].items():
        p = runtime / rel
        if p.is_symlink() or not p.resolve().is_relative_to(runtime.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError('Runtime source drift: ' + rel)
    artifacts = json.loads((runtime / 'artifact-receipt.json').read_text())
    if artifacts['engine'] != engine or [(p['repository'], p['revision']) for p in artifacts['targets']] != MODELS[engine]:
        raise ValueError('Checkpoint identity mismatch')
    for p in artifacts['targets']:
        path = Path(p['snapshot'])
        if path.name != p['revision']:
            raise ValueError('Checkpoint path is not revision-pinned')
        if hashlib.sha256((path / 'config.json').read_bytes()).hexdigest() != p['config_sha256']:
            raise ValueError('Checkpoint configuration drift')
        if any(not (path / rel).is_file() or (path / rel).stat().st_size != size for rel, size in p['files'].items()):
            raise ValueError('Checkpoint shard inventory changed')
    if engine == 'glm-adaptive':
        donor = artifacts['donor']
        if donor['donor_sha'] != 'e90ef415a2bf1274f848965a61144eca1e99eabf' or set(donor['layers']) != {str(i) for i in range(15, 46)}:
            raise ValueError('Donor identity drift')
        for layer, entry in donor['layers'].items():
            with (runtime / f'ablit/transplant/L{layer}.bin').open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != entry['sha256']:
                    raise ValueError('Donor tensor drift')
    return {'source': 'verified', 'artifacts': 'verified', 'engine': engine}


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]), sys.argv[2])))
