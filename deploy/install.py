#!/usr/bin/env python3
"""Install a verified release into a new path; register services without starting models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys


def verified_files(root, manifest):
    for rel, sha in manifest.items():
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe file: ' + rel)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('File changed: ' + rel)


def copy_manifested(root, destination, files):
    verified_files(root, files)
    for rel in files:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)


def install(source, bundle, node, router, register=False):
    if os.geteuid() != 0:
        raise ValueError('Installation requires sudo; it does not start models')
    if sys.platform != 'linux':
        raise ValueError('Install on Linux Sparks')
    site = json.loads((bundle / 'site.json').read_text())
    if node not in site['nodes']:
        raise ValueError('Unknown logical node')
    release = Path(site['release_root'])
    if release.exists():
        raise ValueError('Select a new release path; preserve the previous installation')
    account = pwd.getpwnam(site['user'])
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    verified_files(source, manifest['files'])
    preparation = json.loads((bundle / 'preparation.json').read_text())
    verified_files(bundle, preparation['files'])
    if router is None or not router.is_file():
        raise ValueError('Supply the Linux ARM64 router built by build_router.py')
    router_receipt = json.loads(router.with_suffix('.json').read_text())
    if router_receipt['binary_sha256'] != hashlib.sha256(router.read_bytes()).hexdigest() or router_receipt['target'] != 'linux/arm64':
        raise ValueError('Router binary/receipt mismatch')
    release.mkdir(parents=True, exist_ok=False)
    copy_manifested(source, release, manifest['files'])
    shutil.copyfile(source / 'MANIFEST.json', release / 'MANIFEST.json')
    copy_manifested(bundle, release / 'site', preparation['files'])
    shutil.copyfile(bundle / 'preparation.json', release / 'site/preparation.json')
    (release / 'bin').mkdir(exist_ok=True)
    shutil.copyfile(router, release / 'bin/llama-swap')
    (release / 'bin/llama-swap').chmod(0o755)
    subprocess.run(['python3', '-m', 'venv', str(release / 'venv')], check=True)
    subprocess.run([str(release / 'venv/bin/pip'), 'install', '--no-cache-dir', str(release)], check=True)
    cli = str(release / 'venv/bin/spark-serve')
    config = str(release / 'site/cluster.json')
    subprocess.run([cli, '--config', config, 'validate'], check=True)
    # The gateway consumes this derived file directly; never rely on startup to render it.
    subprocess.run([cli, '--config', config, 'render-gateway', '--output', str(release / 'site/llama-swap.json')], check=True)
    for part in ('run', 'state'):
        path = Path(site['state_root']) / part
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, account.pw_uid, account.pw_gid)
    for field in ('runtime_root', 'cache_root', 'model_root', 'hf_home'):
        path = Path(site['nodes'][node][field])
        if not path.exists():
            path.mkdir(parents=True)
            os.chown(path, account.pw_uid, account.pw_gid)
    # Only runtime state is writable by the service account. Release source stays root-owned.
    units = {}
    if node == 'head':
        units['dgx-spark-serve.service'] = f'''[Unit]
Description=Spark model broker
After=network-online.target docker.service
Wants=network-online.target
[Service]
Type=simple
User={site['user']}
Group={account.pw_gid}
Environment=PATH={release}/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart={cli} --config {config} gateway
ExecStartPost={cli} --config {config} bootstrap
TimeoutStartSec=14400
TimeoutStopSec=900
KillMode=mixed
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
'''
    full = preparation['recipe'] == 'glm53-full-4spark'
    if full:
        units['dgx-spark-serve-cache.service'] = '''[Unit]
Description=Full GLM bounded-memory cache maintenance
[Service]
Type=simple
ExecStart=/bin/bash -c 'while true; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 60; done'
KillMode=control-group
'''
    devices = site['nodes'][node].get('swap_devices', [])
    for device in devices:
        if device != '/swap.img':
            raise ValueError('This release supports the standard /swap.img swap device; configure other swap separately')
    helper = '''#!/usr/bin/env python3
import os,subprocess,sys
assert os.geteuid()==0
a=sys.argv[1:]
allowed={('swap-off',): ['/sbin/swapoff','-a'], ('swap-on','/swap.img'): ['/sbin/swapon','/swap.img']}
'''
    if full:
        helper += "allowed.update({('cache-start',): ['systemctl','start','dgx-spark-serve-cache.service'], ('cache-stop',): ['systemctl','stop','dgx-spark-serve-cache.service']})\n"
    helper += "if tuple(a) not in allowed: raise SystemExit('Unsupported resource action')\nsubprocess.run(allowed[tuple(a)],check=True)\n"
    helper_path = Path('/usr/local/libexec/dgx-spark-serve-resource')
    actions = ['swap-off', 'swap-on /swap.img'] + (['cache-start', 'cache-stop'] if full else [])
    sudoers = ''.join(f'{site["user"]} ALL=(root) NOPASSWD: {helper_path} {action}\n' for action in actions)
    staged_system = release / 'system'
    staged_system.mkdir()
    (staged_system / 'resource-helper').write_text(helper)
    (staged_system / 'sudoers').write_text(sudoers)
    for name, text in units.items():
        (staged_system / name).write_text(text)
    subprocess.run(['visudo', '-cf', str(staged_system / 'sudoers')], check=True)
    if register:
        targets = {helper_path: (staged_system / 'resource-helper', 0o755),
                   Path('/etc/sudoers.d/dgx-spark-serve'): (staged_system / 'sudoers', 0o440)}
        targets.update({Path('/etc/systemd/system') / n: (staged_system / n, 0o644) for n in units})
        # Existing installations require a reviewed migration; do not overwrite live authority.
        for target, (src, _) in targets.items():
            if target.exists() and target.read_bytes() != src.read_bytes():
                raise ValueError('Existing installation differs; staged files retained for migration: ' + str(target))
        for target, (src, mode) in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as handle:
                    handle.write(src.read_bytes())
                target.chmod(mode)
        subprocess.run(['systemctl', 'daemon-reload'], check=True)
    receipt = {'release': str(release), 'node': node, 'services_registered': register,
               'models_started': False, 'hardware_qualified': False,
               'source_manifest_sha256': hashlib.sha256((source / 'MANIFEST.json').read_bytes()).hexdigest()}
    (release / 'installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--node', required=True)
    p.add_argument('--router', type=Path, required=True)
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    print(json.dumps(install(a.source, a.bundle, a.node, a.router, a.register), indent=2))
