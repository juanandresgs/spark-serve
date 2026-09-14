#!/usr/bin/env python3
"""Check release integrity and render every supported adapter at two offline sites."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from spark_serve import recipes
from spark_serve.config import load_catalog
from render_site import render, SUPPORTED


def check(root):
    root = root.resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for rel, sha in manifest['files'].items():
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe release path: ' + rel)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Release file changed: ' + rel)
    checked = []
    for path in sorted((root / 'recipes').glob('*/recipe.json')):
        recipe = recipes.read_recipe(root, path.parent.name)
        result = recipes.check(root, recipe)
        if not result['valid']:
            raise ValueError(str(result['failures']))
        checked.append(recipe['id'])
    bindings = []
    with tempfile.TemporaryDirectory() as temp:
        scratch = Path(temp).resolve()
        for variant in ('a', 'b'):
            site = json.loads((root / 'examples/deployment-site.json').read_text())
            site['id'] = 'offline-' + variant
            site['user'] = 'operator' + variant
            site['release_root'] = str(scratch / ('installed-' + variant))
            site['state_root'] = str(scratch / ('state-' + variant))
            for i, node in enumerate(site['nodes'].values()):
                node['images'] = {key: 'sha256:' + str(i + 1) * 64 for key in node['images']}
                for key in ('home', 'hf_home', 'runtime_root', 'cache_root', 'model_root'):
                    node[key] = node[key].replace('spark', site['user'])
                if node['host'] != 'local':
                    node['host'] = site['user'] + '@node-' + str(i)
                node['fabric']['ip'] = '203.0.113.' + str(i + (10 if variant == 'a' else 20))
            for name in sorted(SUPPORTED):
                output = scratch / (name + variant)
                render(root, name, site, output)
                config = json.loads((output / 'cluster.json').read_text())
                config['profiles_dir'] = str(output / 'profiles')
                validation = output / 'validation.json'
                validation.write_text(json.dumps(config))
                load_catalog(validation)
                bindings.append(name + ':' + variant)
    return {'passed': True, 'files_verified': len(manifest['files']),
            'recipes_checked': checked, 'site_bindings_checked': bindings,
            'hardware_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(check(args.root), indent=2))
