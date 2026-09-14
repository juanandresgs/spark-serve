#!/usr/bin/env python3
"""Prepare and execute image builds from public sources, with no model weights."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kwargs).stdout.strip()


def source_copy(source, output, revision):
    if run(['git', '-C', str(source), 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Source revision drift: ' + str(source))
    if run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no']):
        raise ValueError('Source has tracked modifications')
    output.mkdir(parents=True, exist_ok=False)
    for rel in filter(None, run(['git', '-C', str(source), 'ls-files', '-z']).split('\0')):
        p = source / rel
        if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()):
            raise ValueError('Unsafe source path')
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        dest.chmod(p.stat().st_mode & 0o777)


def prepare(root, sources, output, engine):
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    pins = json.loads((root / 'sources.json').read_text())
    stamp = hashlib.sha256((root / 'sources.json').read_bytes()).hexdigest()[:12]
    prefix = 'spark-serve/build-' + stamp
    commands = []
    def cmd(argv, cwd=None):
        commands.append({'argv': list(map(str, argv)), 'cwd': str(cwd or output)})
    if engine == 'glm-adaptive':
        rep = module(root / 'reproduce.py', 'image_reproduce')
        e2 = output / 'e2'
        receipt = rep.prepare_e2(root, sources / 'glm-e2', e2)
        tag_e2, tag_e3, tag_final = prefix + ':e2', prefix + ':e3', prefix + ':adaptive'
        argv = receipt['build_argv']
        argv[argv.index('--tag') + 1] = tag_e2
        cmd(['docker', 'pull', pins['image_reproduction']['E2_base']])
        cmd(argv)
        e3 = output / 'e3'
        rep.prepare_e3(root, sources / 'glm-e3', sources / 'glm-e2/overlay/exl3.py', sources / 'glm-e2/tests/test_exl3_overlay.py', e3)
        # The selected CUDA sources are compiled inside the exact E2 toolchain.
        cmd(['docker', 'run', '--rm', '--network=none', '-e', 'MAX_JOBS=1',
             '-v', str(e3) + ':/src:ro', '-v', str(e3) + ':/build', '--entrypoint', 'python3',
             tag_e2, '/src/build_exl3_fat_moe_ext.py', '--src', '/src', '--out', '/build', '--verbose'])
        dockerfile = (e3 / 'Dockerfile').read_text().replace('FROM dgx-local/glm53-exl3-e2:eb0469f-905c0293-c5d9c657', 'FROM ' + tag_e2)
        (e3 / 'Dockerfile').write_text(dockerfile)
        cmd(['docker', 'build', '--network=none', '-t', tag_e3, e3])
        adaptive = output / 'adaptive'
        for rel in ('Dockerfile', 'overlay/patch_adaptive_k.py', 'tests/test_adaptive_k_patch.py'):
            dest = adaptive / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / 'cluster/glm-throughput' / rel, dest)
        cmd(['docker', 'build', '--network=none', '--build-arg', 'BASE=' + tag_e3, '-t', tag_final, adaptive])
        tags = [tag_e2, tag_e3, tag_final]
    elif engine == 'glm-full':
        base_pin = pins['image_reproduction']['full_base']
        base = output / 'base'
        source_copy(sources / 'glm-full-base', base, pins['sources']['glm-full-base']['revision'])
        dockerfile = (base / 'Dockerfile').read_text()
        for old, new in [
            ('ARG VLLM_APPLY_PRESET_PRS=""', 'ARG VLLM_APPLY_PRESET_PRS="false"'),
            ('ARG DEEPGEMM_REF=nv_dev', 'ARG DEEPGEMM_REF=' + base_pin['deepgemm']),
            ('ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04', 'ARG CUDA_IMAGE=' + base_pin['cuda_base']),
        ]:
            if dockerfile.count(old) != 1:
                raise ValueError('Full base source anchor drift')
            dockerfile = dockerfile.replace(old, new)
        (base / 'Dockerfile').write_text(dockerfile)
        # Avoid host discovery and existing environment files in the upstream helper.
        (base / '.env').write_text('BUILD_JOBS=16\n')
        tag_base, tag_final = prefix + ':full-base', prefix + ':full'
        cmd(['bash', './build-and-copy.sh', '--use-wheels', '--vllm-ref', base_pin['vllm'],
             '--flashinfer-ref', base_pin['flashinfer'], '--gpu-arch', '12.1a', '--build-jobs', '16',
             '--tag', tag_base, '--config', str(base / '.env'), '--full-log'], base)
        ctx = output / 'full'
        ctx.mkdir()
        for name in ('glm-full-kernels', 'glm-full-port', 'glm-full-dcp'):
            source_copy(sources / name, output / name, pins['sources'][name]['revision'])
        kernels = output / 'glm-full-kernels'
        shutil.copytree(kernels / 'kernels', ctx / 'kernels')
        shutil.copytree(kernels / 'mods', ctx / 'mods')
        shutil.copytree(kernels / 'patches', ctx / 'kernel-patches')
        patches = ctx / 'patches'
        patches.mkdir()
        for name in ('patch_base_dflash2.py', 'patch_base_kv_dsa.py', 'verify_dflash2.py'):
            shutil.copyfile(output / 'glm-full-port/dflash2-port' / name, patches / name)
        kv = (output / 'glm-full-dcp/overlay/vllm/v1/core/kv_cache_utils.py').read_text()
        block = kv[kv.index('def _dsa_unpadded_drafter_specs('):kv.index('def get_kv_cache_config_from_groups(')]
        if 'def _get_kv_cache_groups_dsa_drafter(' not in block:
            raise ValueError('Drafter helper reconstruction failed')
        (patches / 'dsa_block.py').write_text(block)
        shutil.copyfile(root / 'cluster/glm53-full-tp4/patch-swa-under-mla.py', patches / 'patch-swa-under-mla.py')
        v = '/usr/local/lib/python3.12/dist-packages/vllm'
        donor = 'ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6'
        (ctx / 'Dockerfile').write_text(f'''FROM {donor} AS donor
FROM {tag_base}
COPY kernels/ /root/models/models15/glm-triton/
COPY mods/ /mods/
COPY kernel-patches/ /kernel-patches/
RUN bash /mods/glm52-sm12x-sparse/run.sh && bash /mods/glm52-b12x-sparse/run.sh && python3 /kernel-patches/fix-indexer-mtp-overhang.py && python3 -c "import b12x"
COPY --from=donor {v}/model_executor/models/qwen3_dflash2.py {v}/model_executor/models/qwen3_dflash2.py
COPY --from=donor {v}/v1/worker/gpu/spec_decode/dflash2/ {v}/v1/worker/gpu/spec_decode/dflash2/
COPY patches/ /opt/dflash2-port/
RUN python3 /opt/dflash2-port/patch_base_dflash2.py && python3 /opt/dflash2-port/patch_base_kv_dsa.py && python3 /opt/dflash2-port/patch-swa-under-mla.py && python3 /opt/dflash2-port/verify_dflash2.py
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
''')
        cmd(['docker', 'build', '-t', tag_final, ctx])
        tags = [tag_base, tag_final]
    else:
        raise ValueError('Unknown image engine')
    plan = {'engine': engine, 'commands': commands, 'images': tags, 'final_image': tag_final,
            'source_files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(output.rglob('*')) if p.is_file()},
            'build_executed': False, 'hardware_qualified': False}
    (output / 'build-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def execute(path):
    root = path.resolve().parent
    plan = json.loads(path.read_text())
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise ValueError('Execute image builds on an idle ARM64 Linux Spark')
    if run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']):
        raise ValueError('GPU compute is active; drain this Spark before building')
    for tag in plan['images']:
        if subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True).returncode == 0:
            raise ValueError('Refusing to replace an existing image tag: ' + tag)
    for rel, sha in plan['source_files'].items():
        p = root / rel
        if p.is_symlink() or not p.resolve().is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError('Build input drift: ' + rel)
    for i, c in enumerate(plan['commands']):
        print(f'Build step {i + 1}/{len(plan["commands"])}: {c["argv"][0]}', flush=True)
        with (root / f'build-{i + 1}.log').open('wb') as log:
            subprocess.run(c['argv'], cwd=c['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = {'build_executed': True, 'hardware_qualified': False,
               'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'images': {tag: run(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag]) for tag in plan['images']}}
    (root / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('engine', choices=['glm-adaptive', 'glm-full'])
    a.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a.add_argument('--sources', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('execute')
    b.add_argument('plan', type=Path)
    args = p.parse_args()
    result = execute(args.plan) if args.action == 'execute' else prepare(args.root, args.sources, args.output, args.engine)
    print(json.dumps(result, indent=2))
