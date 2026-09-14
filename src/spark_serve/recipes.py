"""Read-only recipe discovery and provenance checks for a source checkout."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from spark_serve.config import ConfigError

STATUSES = {'recommended-with-caveats', 'bounded-qualification', 'experimental'}


def source_root(value: Path | None = None) -> Path:
    root = (value or Path(__file__).resolve().parents[2]).resolve()
    if not (root / 'recipes').is_dir():
        raise ConfigError('Recipe assets require the source checkout; pass recipes --source /path/to/checkout')
    return root


def asset(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ConfigError(f'Invalid recipe asset: {name!r}')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ConfigError(f'Missing or escaping recipe asset: {name}')
    return path


def read_recipe(root: Path, name: str) -> dict[str, Any]:
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name):
        raise ConfigError('Invalid recipe name')
    try:
        value = json.loads(asset(root, f'recipes/{name}/recipe.json').read_text())
        if value['schema_version'] != 1 or value['id'] != name or value['status'] not in STATUSES:
            raise ValueError('invalid identity, version or status')
        if value['status'] == 'experimental' and not value.get('hold'):
            raise ValueError('experimental recipe requires a hold reason')
        if not value['profiles'] or not value['assets']:
            raise ValueError('missing source inputs')
        return value
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f'Invalid recipe {name}: {exc}') from exc


def catalog(root: Path) -> list[dict[str, Any]]:
    return [read_recipe(root, p.parent.name) for p in sorted((root / 'recipes').glob('*/recipe.json'))]


def check(root: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    failures = []
    try:
        profiles = [json.loads(asset(root, path).read_text()) for path in recipe['profiles']]
        pins = {p['id']: {k: p['model'][k] for k in ('repository', 'revision', 'context_tokens')} | {'image': p['runtime']['image']} for p in profiles}
        if pins != recipe['profile_pins']:
            failures.append('Profile model, image or context differs from recipe pins')
        if len(pins) != len(profiles):
            failures.append('Duplicate model profiles')
        cfg = recipe['configuration']
        if cfg['default_model'] not in pins or set(cfg['preload_models']) != set(pins):
            failures.append('Default/preload set differs from selected profiles')
        pool = cfg.get('worker_pool')
        if pool and (pool['id'] in pins or not pool['targets'] or not set(pool['targets']) <= set(pins)):
            failures.append('Invalid worker pool')
        if pool:
            claims = set()
            for p in profiles:
                current = set(p['resources']['claims'])
                if claims & current:
                    failures.append('Resident fleet profiles have conflicting resource claims')
                claims |= current
        for path, digest in recipe['assets'].items():
            actual = hashlib.sha256(asset(root, path).read_bytes()).hexdigest()
            if actual != digest:
                failures.append(f'Asset digest drift: {path}')
        for path in (recipe['guide'], recipe['evidence'], recipe['pi']['installer']):
            asset(root, path)
            if path not in recipe['assets']:
                failures.append(f'Unhashed required asset: {path}')
        evidence = json.loads(asset(root, recipe['evidence']).read_text())
        if evidence['recipe_id'] != recipe['id'] or evidence['status'] != recipe['status']:
            failures.append('Evidence identity/status does not match manifest')
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        failures.append(str(exc))
    return {'recipe': recipe['id'], 'valid': not failures, 'status': recipe['status'], 'failures': failures,
            'scope': 'source consistency only; not live hardware qualification'}


def add_parser(sub):
    parser = sub.add_parser('recipes', help='browse and check versioned recipes without a live cluster')
    parser.add_argument('--source', type=Path, help='source checkout containing recipes and launchers')
    actions = parser.add_subparsers(dest='recipe_action', required=True)
    actions.add_parser('list', help='list recipes and qualification status')
    show = actions.add_parser('show', help='show pinned configuration and guide path')
    show.add_argument('recipe')
    actions.add_parser('check', help='check all recipe source pins and evidence references')
    prep = actions.add_parser('prepare', help='write a new broker config bundle; never deploy')
    prep.add_argument('recipe')
    prep.add_argument('--site', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--allow-experimental', action='store_true')


def execute(args) -> int:
    root = source_root(args.source)
    if args.recipe_action == 'prepare':
        try:
            result = prepare(root, read_recipe(root, args.recipe), args.site, args.output, args.allow_experimental)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f'Recipe preparation failed: {exc}') from exc
        print(json.dumps(result, indent=2))
        return 0
    if args.recipe_action == 'show':
        print(json.dumps(read_recipe(root, args.recipe), indent=2))
        return 0
    entries = catalog(root)
    if args.recipe_action == 'check':
        results = [check(root, item) for item in entries]
        print(json.dumps({'valid': bool(results) and all(r['valid'] for r in results), 'recipes': results}, indent=2))
        return 0 if results and all(r['valid'] for r in results) else 1
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        for entry in entries:
            print(f"{entry['id']:<25} {entry['status']:<26} {entry['title']}")
        print('\nCatalog status is qualification evidence, not current deployment health.')
    return 0


def merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def replace_values(value, replacements):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_values(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: replace_values(v, replacements) for k, v in value.items()}
    return value


def prepare(root, recipe, site_path, output, allow_experimental=False):
    """Write a new, validated config directory. Never invoke a lifecycle command."""
    from spark_serve.config import load_catalog, render_gateway_config
    import os
    from dataclasses import replace
    import shutil
    import tempfile
    if recipe['status'] == 'experimental' and not allow_experimental:
        raise ConfigError('Recipe is experimental and held; --allow-experimental is required to prepare it')
    result = check(root, recipe)
    if not result['valid']:
        raise ConfigError('; '.join(result['failures']))
    site = json.loads(site_path.read_text())
    cluster = site['cluster']
    replacements = site.get('path_replacements', {})
    if not isinstance(replacements, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in replacements.items()):
        raise ConfigError('path_replacements must map nonempty strings to strings')
    overrides = site.get('profile_overrides', {})
    ids = set(recipe['profile_pins'])
    if set(overrides) - ids:
        raise ConfigError('Profile override does not belong to selected recipe')
    output = output.resolve()
    if output.exists():
        raise ConfigError('Output already exists; prepare a new directory to preserve existing files')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.recipe-', dir=output.parent))
    try:
        (stage / 'profiles').mkdir()
        for rel in recipe['profiles']:
            original = json.loads(asset(root, rel).read_text())
            profile = replace_values(json.loads(asset(root, rel).read_text()), replacements)
            original_id = original['id']
            merge(profile, overrides.get(original_id, {}))
            # Site paths may change; the qualified logical GPU placement may not.
            for key in ('topology', 'resources'):
                if profile[key] != original[key]:
                    raise ConfigError(f'Site overrides may not change pinned {key}')
            for key in ('launch', 'stop', 'backend'):
                if profile['runtime'].get(key, {}).get('node') != original['runtime'].get(key, {}).get('node'):
                    raise ConfigError('Site overrides may not change runtime node placement')
            if [a['node'] for a in profile['artifacts']] != [a['node'] for a in original['artifacts']]:
                raise ConfigError('Site overrides may not change artifact node placement')
            pin = recipe['profile_pins'][original_id]
            if profile['id'] != original_id or profile['model']['repository'] != pin['repository'] or profile['model']['revision'] != pin['revision'] or profile['runtime']['image'] != pin['image'] or profile['model']['context_tokens'] != pin['context_tokens']:
                raise ConfigError('Site overrides may not change pinned model identity, image or context')
            profile['operator']['exposure'] = 'production'
            profile['operator']['readiness'] = 'experimental'
            profile['operator'].setdefault('warnings', []).append('Prepared site configuration has not been qualified on its destination hardware.')
            (stage / 'profiles' / (original_id + '.json')).write_text(json.dumps(profile, indent=2)+'\n')
        cluster.pop('worker_pool', None)
        cluster.pop('model_routes', None)
        cluster.update(recipe['configuration'])
        cluster['gateway']['config_path'] = str(output / 'llama-swap.json')
        cluster['profiles_dir'] = str(stage / 'profiles')
        (stage / 'cluster.json').write_text(json.dumps(cluster, indent=2)+'\n')
        catalog_value = load_catalog(stage / 'cluster.json')
        rendered = render_gateway_config(replace(catalog_value, config_path=output / "cluster.json"))
        cluster['profiles_dir'] = str(output / 'profiles')
        (stage / 'cluster.json').write_text(json.dumps(cluster, indent=2)+'\n')
        (stage / 'llama-swap.json').write_text(json.dumps(rendered, indent=2)+'\n')
        receipt = {'recipe': recipe['id'], 'status': recipe['status'], 'deployment_performed': False,
                   'scope': 'Broker config only. Launch adapters, assets, service paths and fabric must be staged and checked for this site.',
                   'overridden_profiles': sorted(overrides), 'source_checks': result}
        (stage / 'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        # A sibling rename publishes the bundle at once; refuse an intervening target.
        if output.exists():
            raise ConfigError('Output appeared during preparation')
        from spark_serve.atomic import publish_exclusive
        publish_exclusive(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {'prepared': str(output), 'recipe': recipe['id'], 'deployed': False}
