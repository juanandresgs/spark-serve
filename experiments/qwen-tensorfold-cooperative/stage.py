#!/usr/bin/env python3
"""Download immutable model dependencies and verify every published file digest."""
import hashlib
import json
import os
import sys
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

import argparse
p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--verify-only', action='store_true')
a = p.parse_args()
ROOT = a.directory.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
MODELS = [('target', 'turboderp/Qwen3.8-Flash-Next-exl3', '69e33439ae950f17bcbe95c98f117d80f759ab6d')]
for name, repo, revision in MODELS:
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=60) as r:
        manifest = json.load(r)
    assert manifest['sha'] == revision
    (ROOT / (name + '-manifest.json')).write_text(json.dumps(manifest, indent=2))
    if '--verify-only' not in sys.argv:
        snapshot_download(repo, revision=revision, local_dir=ROOT/name, max_workers=4)
    receipt = []
    for item in manifest['siblings']:
        path = ROOT/name/item['rfilename']
        before = path.stat()
        assert before.st_size == item['size'], path
        h = hashlib.sha256() if item.get('lfs') else hashlib.sha1()
        if not item.get('lfs'):
            h.update(f'blob {before.st_size}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16*1024*1024), b''):
                h.update(chunk)
        expected = item['lfs']['sha256'] if item.get('lfs') else item['blobId']
        assert h.hexdigest() == expected and path.stat().st_mtime_ns == before.st_mtime_ns, path
        receipt.append(dict(name=item['rfilename'], size=before.st_size, mtime_ns=before.st_mtime_ns, digest=expected))
    out = ROOT/(name+'-verified.json')
    temporary = out.with_suffix('.next')
    temporary.write_text(json.dumps(dict(repository=repo, revision=revision, files=receipt), indent=2))
    os.replace(temporary, out)
    print(name, 'verified', len(receipt), flush=True)
