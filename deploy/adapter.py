#!/usr/bin/env python3
"""Site adapter invoked only by the existing spark-serve lifecycle broker."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import shlex


def run(node, argv, **kwargs):
    if node != 'local':
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=10', node, shlex.join(argv)]
    return subprocess.run(argv, text=True, capture_output=True, check=True, timeout=60, **kwargs).stdout.strip()


def owned_id(node, name, deployment, profile):
    ids = run(node, ['docker', 'container', 'ls', '-aq', '--filter', 'name=^/' + name + '$']).split()
    if not ids:
        return None
    if len(ids) != 1:
        raise ValueError('Ambiguous container name')
    value = json.loads(run(node, ['docker', 'container', 'inspect', ids[0]]))[0]
    labels = value['Config'].get('Labels') or {}
    if labels.get('dgx.spark-serve.deployment') != deployment or labels.get('dgx.spark-serve.profile') != profile:
        raise ValueError('Refusing a container owned by another deployment or profile')
    return ids[0]


def execute(manifest, action):
    profile, deployment = manifest['profile'], manifest['deployment']
    if os.environ.get('DGX_SPARK_SERVE_MANAGED') != '1' or any(
        os.environ.get(key) != value for key, value in [
            ('DGX_SPARK_SERVE_DEPLOYMENT_ID', deployment), ('DGX_SPARK_SERVE_PROFILE_ID', profile)]):
        raise ValueError('Use spark-serve managed lifecycle')
    for value in (profile, deployment):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
            raise ValueError('Invalid ownership identity')
    if action == 'stop':
        # Check every rank before stopping any; do not suppress remote failures.
        targets = [(n['host'], owned_id(n['host'], n['container'], deployment, profile)) for n in manifest['ranks']]
        for node, cid in reversed(targets):
            if cid:
                run(node, ['docker', 'container', 'rm', '-f', cid])
        if manifest['engine'] == 'glm-full':
            for n in manifest['ranks']:
                run(n['host'], ['sudo', '-n', '/usr/local/libexec/dgx-spark-serve-resource', 'cache-stop'])
        return
    for n in manifest['ranks']:
        if run(n['host'], ['docker', 'container', 'ls', '-aq', '--filter', 'name=^/' + n['container'] + '$']):
            raise ValueError('Container already exists; use managed stop first')
        if run(n['host'], ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']):
            raise ValueError('Another GPU workload is active on a required node')
        actual = run(n['host'], ['docker', 'image', 'inspect', '--format', '{{.Id}}', manifest['image']])
        if actual != n['image_id']:
            raise ValueError('Staged image differs from the site qualification input')
        if n.get('require_no_swap', True) and run(n['host'], ['swapon', '--noheadings']):
            raise ValueError('Swap is still enabled')
        run(n['host'], ['python3', str(Path(__file__).with_name('verify_artifacts.py')), n['runtime'], manifest['engine']])
    if manifest['engine'] == 'glm-full':
        checker = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
r=json.loads((root/'runtime-source.json').read_text())
for rel,sha in r['files'].items():
 p=root/rel
 assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, rel
print('verified')
'''
        for n in manifest['ranks']:
            run(n['host'], ['python3', '-c', checker, n['runtime']])
        for n in manifest['ranks']:
            run(n['host'], ['sudo', '-n', '/usr/local/libexec/dgx-spark-serve-resource', 'cache-start'])
        for n in reversed(manifest['ranks']):
            env = n['environment'] | {'DGX_SPARK_SERVE_DEPLOYMENT_ID': deployment,
                                       'DGX_SPARK_SERVE_PROFILE_ID': profile}
            run(n['host'], ['env', *[k + '=' + v for k, v in env.items()],
                            'bash', n['runtime'] + '/start.sh', str(n['rank']), 'dflash'])
        return
    root = Path(manifest['runtime'])
    receipt = json.loads((root / 'runtime-source.json').read_text())
    for rel, sha in receipt['files'].items():
        p = root / rel
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError('Prepared runtime source drift: ' + rel)
    environment = os.environ.copy()
    environment.update(manifest['environment'])
    # A retained trial override would silently replace the qualified adaptive policy.
    if manifest['engine'] == 'glm-adaptive':
        for n in manifest['ranks']:
            override = n['cache'] + '/glm-throughput-106.json'
            exists = run(n['host'], ['python3', '-c', 'import os,sys;print(int(os.path.lexists(sys.argv[1])))', override])
            if exists != '0':
                raise ValueError('Remove or explicitly review the retained adaptive trial override')
    os.execvpe('bash', ['bash', str(root / 'start.sh'), *manifest['arguments']], environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('action', choices=['launch', 'stop'])
    args = parser.parse_args()
    execute(json.loads(args.manifest.read_text()), args.action)
