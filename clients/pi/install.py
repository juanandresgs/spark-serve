#!/usr/bin/env python3
"""Preview/apply Pi recipe configuration; never installs packages or changes serving."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cluster/agent-fleet/pi'
ROUTES = {
    'agent-fleet-211': ('dgx-orchestrator', 262144),
    'glm53-flash-2spark': ('glm-5.3-flash-exl3-ablit', 262144),
    'glm53-full-4spark': ('glm-5.3-full-tp4', 307200),
    'qwen38-flash-2spark': ('qwen-worker-pair', 262144),
}


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_json(home, name):
    p = target(home, name)
    value = json.loads(p.read_text()) if p.exists() else {}
    if not isinstance(value, dict):
        raise ValueError(f'{name} must contain an object')
    return value


def plan(home, recipe, base_url, api_key_env=None):
    url = urlsplit(base_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment or not url.path.rstrip('/').endswith('/v1'):
        raise ValueError('base URL must be an HTTP(S) /v1 endpoint without embedded credentials, query or fragment')
    if api_key_env and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env):
        raise ValueError('API key option must name an environment variable, not contain a token')
    model, context = ROUTES[recipe]
    fleet = recipe == 'agent-fleet-211'
    models = read_json(home, 'models.json')
    provider = models.setdefault('providers', {}).setdefault('dgx-spark', {})
    provider.update({'api': 'openai-completions', 'baseUrl': base_url.rstrip('/')})
    if api_key_env:
        provider['apiKey'] = '${' + api_key_env + '}'
        provider['authHeader'] = True
    elif 'apiKey' not in provider:
        provider['apiKey'] = 'local-no-key'
    routes = [(model, context)] + ([(m, 262144) for m in ['qwen-workers', 'qwen-worker-a', 'qwen-worker-b']] if fleet else [])
    by_id = {m['id']: m for m in provider.get('models', [])}
    for mid, ctx in routes:
        by_id[mid] = {'id': mid, 'name': mid, 'reasoning': not mid.startswith('qwen'), 'input': ['text'], 'contextWindow': ctx, 'maxTokens': 8192,
                      'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}
    provider['models'] = list(by_id.values())
    settings = read_json(home, 'settings.json')
    settings.update({'defaultProvider': 'dgx-spark', 'defaultModel': model, 'defaultThinkingLevel': 'off' if model.startswith('qwen') else 'high'})
    extensions = settings.setdefault('extensions', [])
    if 'extensions/dgx-fleet.ts' not in extensions:
        extensions.append('extensions/dgx-fleet.ts')
    sub = read_json(home, 'subagents.json')
    sub.update(json.loads((SOURCE / 'subagents.json').read_text()))
    if not fleet:
        sub.update({'maxConcurrent': 1, 'maxConcurrentForeground': 1})
    updates = {'models.json': encoded(models), 'settings.json': encoded(settings), 'subagents.json': encoded(sub),
               'extensions/dgx-fleet.ts': (SOURCE / 'dgx-fleet.ts').read_bytes()}
    for p in (SOURCE / 'agents').glob('*.md'):
        text = p.read_text()
        if not fleet:
            text = re.sub(r'^model: .+$', 'model: dgx-spark/' + model, text, flags=re.M)
            text = re.sub(r'^thinking: .+$', 'thinking: ' + ('off' if model.startswith('qwen') else 'high'), text, flags=re.M)
            text = re.sub(r'^description: .+$', 'description: Bounded task on the selected shared model; no separate worker capacity.', text, flags=re.M)
        updates['agents/' + p.name] = text.encode()
    return updates


def target(home, name):
    p = home / name
    if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(home.resolve()):
        raise ValueError(f'Unsafe managed path: {name}')
    return p


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.recipe-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def apply(home, updates, expected=None):
    before = {name: (target(home, name).read_bytes() if target(home, name).exists() else None) for name in updates}
    if expected is not None and before != expected:
        raise ValueError('Concurrent edit before restore transaction')
    backups = home / 'fleet-backups'
    backups.mkdir(mode=0o700, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='recipe-', dir=backups))
    manifest = []
    for name, content in before.items():
        if content is not None:
            write_atomic(backup / name, content)
        manifest.append({'path': name, 'existed': content is not None, 'before_sha256': sha(content) if content is not None else None, 'after_sha256': sha(updates[name]) if updates[name] is not None else None})
    write_atomic(backup / 'manifest.json', encoded(manifest))
    applied = []
    try:
        for name, content in updates.items():
            p = target(home, name)
            if (p.read_bytes() if p.exists() else None) != before[name]:
                raise ValueError(f'Concurrent edit: {name}')
            if content is None:
                p.unlink(missing_ok=True)
            else:
                write_atomic(p, content)
            applied.append(name)
    except BaseException:
        for name in reversed(applied):
            p = target(home, name)
            if (p.read_bytes() if p.exists() else None) != updates[name]:
                continue  # Never overwrite someone else's intervening change.
            if before[name] is None:
                p.unlink()
            else:
                write_atomic(p, before[name])
        raise
    return {'backup': str(backup), 'updated': list(updates)}


def restore(home, backup):
    manifest = json.loads((backup / 'manifest.json').read_text())
    expected, updates = {}, {}
    for entry in manifest:
        name = entry['path']
        p = target(home, name)
        current = p.read_bytes() if p.exists() else None
        if (sha(current) if current is not None else None) != entry['after_sha256']:
            raise ValueError(f'Restore refused: current file changed: {name}')
        content = target(backup, name).read_bytes() if entry['existed'] else None
        if (sha(content) if content is not None else None) != entry['before_sha256']:
            raise ValueError('Backup checksum mismatch')
        expected[name], updates[name] = current, content
    result = apply(home, updates, expected=expected)

    return {'restored': str(backup), **result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent-dir', type=Path, default=Path.home()/'.pi/agent')
    p.add_argument('--recipe', choices=ROUTES)
    p.add_argument('--base-url')
    p.add_argument('--api-key-env')
    p.add_argument('--allow-experimental', action='store_true')
    p.add_argument('--apply', action='store_true', help='write files; default is preview')
    p.add_argument('--restore', type=Path, help='restore this installer backup; requires --apply')
    args = p.parse_args()
    try:
        home = args.agent_dir.expanduser().resolve()
        if args.restore:
            if not args.apply:
                p.error('--restore requires --apply')
        else:
            if not args.recipe or not args.base_url:
                p.error('--recipe and --base-url are required')
            if args.recipe == 'qwen38-flash-2spark' and not args.allow_experimental:
                p.error('Paired Qwen is unqualified on this ring; explicit --allow-experimental required')
        if not args.apply:
            changes = plan(home, args.recipe, args.base_url, args.api_key_env)
            print(json.dumps({'preview': True, 'recipe': args.recipe, 'files': list(changes), 'packages_installed': False, 'serving_changed': False}, indent=2))
            return
        home.mkdir(parents=True, exist_ok=True)
        with (home / '.spark-serve-recipe.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = restore(home, args.restore.resolve()) if args.restore else apply(home, plan(home, args.recipe, args.base_url, args.api_key_env))
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(1, f'Pi recipe setup failed: {exc}\n')


if __name__ == '__main__':
    main()
