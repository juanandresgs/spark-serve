#!/usr/bin/env python3
"""Bind reviewed recipes to a site, without deploying or changing model settings."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shutil
import tempfile

from spark_serve.config import load_catalog, render_gateway_config
from spark_serve import recipes

GRAPH = '1 2 3 4 5 6 8 9 10 12 15 16 18 20 21 24 25 30 32 35 40 48 56 64'
SUPPORTED = {'glm53-flash-adaptive-2spark', 'glm53-flash-adaptive-4spark', 'agent-fleet-211-adaptive', 'glm53-full-4spark', 'qwen38-flash-1spark'}


def safe(value, pattern, label):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError('Invalid ' + label)
    return value


def validate(site, selected):
    safe(site['id'], r'[A-Za-z0-9_.-]+', 'site ID')
    safe(site['user'], r'[a-z_][a-z0-9_-]*', 'service user')
    safe(site['release_root'], r'/[A-Za-z0-9_./-]+', 'release root')
    safe(site['state_root'], r'/[A-Za-z0-9_./-]+', 'state root')
    for path in [site['release_root'], site['state_root']]:
        if '..' in Path(path).parts:
            raise ValueError('Relative traversal is not allowed')
    if not set(selected) <= set(site['nodes']):
        raise ValueError('Site is missing required logical nodes')
    if site['nodes']['head']['host'] != 'local':
        raise ValueError('The broker head must be local')
    for name in selected:
        n = site['nodes'][name]
        safe(n['host'], r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', 'SSH host or alias')
        for field in ('home', 'hf_home', 'runtime_root', 'cache_root', 'model_root'):
            path = safe(n[field], r'/[A-Za-z0-9_./-]+', field)
            if '..' in Path(path).parts:
                raise ValueError('Relative traversal is not allowed')
        for image in n['images'].values():
            safe(image, r'sha256:[a-f0-9]{64}', 'built image ID')
        if 'fabric' in n:
            f = n['fabric']
            ipaddress.IPv4Address(f['ip'])
            for k in ('interface', 'hca'):
                safe(f[k], r'[A-Za-z0-9_.-]+', k)
            if type(f['gid']) is not int or not 0 <= f['gid'] <= 255:
                raise ValueError('GID must be an integer from 0 to 255')
        if 'ring' in n:
            f = n['ring']
            ipaddress.IPv4Address(f['ip'])
            safe(f['interface'], r'[A-Za-z0-9_.-]+', 'ring LAN interface')
            safe(f['hcas'], r'=[A-Za-z0-9_.-]+,[A-Za-z0-9_.-]+', 'ring HCAs')
            if type(f['gid']) is not int or not 0 <= f['gid'] <= 255:
                raise ValueError('Invalid ring GID')


def glm_environment(p, head, worker):
    env = {key: str(value) for key, value in {
        'HEAD_IP': head['fabric']['ip'], 'WORKER_IP': worker['fabric']['ip'],
        'HEAD_CX7_IF': head['fabric']['interface'], 'WORKER_CX7_IF': worker['fabric']['interface'],
        'HEAD_CX7_IB': head['fabric']['hca'], 'WORKER_CX7_IB': worker['fabric']['hca'],
        'HEAD_GID': head['fabric']['gid'], 'WORKER_GID': worker['fabric']['gid'],
        'WORKER_SSH': worker['host'], 'WORKER_HOME': worker['home'],
        'HF_HOME': head['hf_home'], 'SPARK_WORKER_HF_HOME': worker['hf_home'],
        'CACHE_ROOT': head['cache_root'] + '/glm-adaptive',
        'WORKER_VLLM_CACHE': worker['cache_root'] + '/glm-adaptive',
        'HOST_BIND': '127.0.0.1', 'PORT': p['runtime']['backend']['port'], 'MASTER_PORT': 29524,
        'TP': 2, 'NNODES': 2, 'IMAGE': p['runtime']['image'],
        'MODEL': p['model']['repository'], 'MODEL_FALLBACK': p['model']['repository'],
        'MODEL_REVISION': p['model']['revision'],
        'MODEL_CACHE_NAME': 'models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw',
        'DFLASH_MODEL': 'incoai/GLM-5.3-Flash-DFlash2',
        'DFLASH_REVISION': 'dc77ff1c99eeb2df044ee3d4f0094eb033fee410',
        'DFLASH_CACHE_NAME': 'models--incoai--GLM-5.3-Flash-DFlash2',
        'SERVED_MODEL_NAME': p['model']['served_name'],
        'MAX_MODEL_LEN': 850000, 'GPU_MEM_UTIL': '0.84', 'MAX_NUM_SEQS': 8,
        'MAX_NUM_BATCHED_TOKENS': 2048, 'QUANTIZATION': 'exl3', 'KV_CACHE_DTYPE': 'fp8',
        'ENFORCE_EAGER': 0, 'SPEC_METHOD': 'dflash', 'DFLASH_TOKENS': 7, 'DFLASH_DRAFT_TP': 2,
        'MTP_TOKENS': 2, 'EXL3_FUSED_MOE': 1, 'EXL3_MOE_ROW_TILE': 0, 'EXL3_TEMP_ROWS_FUSED': 128,
        'EXL3_FAT_SORTED': 0, 'EXL3_FAT_BATCHED': 0, 'EXL3_FAT_KERNEL': 1,
        'GLM53_INDEXER_WORKSPACE': 'rightsize', 'GLM53_SPINWAIT_MS': 'stock',
        'ABLIT': 1, 'ABLIT_METHOD': 'transplant', 'ABLIT_DIRECTION': 'dealign',
        'ABLIT_LAYERS': '15-45', 'ABLIT_ALPHA': '3.0', 'ABLIT_INCLUDE_MTP': 1,
        'LANGUAGE_MODEL_ONLY': 0, 'SKIP_MM_PROFILING': 1, 'USE_HOST_NCCL': 0,
        'GLM53_MIXED_PREFILL_CHUNK': 512,
        'CONTAINER_HEAD': 'spark-' + p['id'] + '-head', 'CONTAINER_WORKER': 'spark-' + p['id'] + '-worker',
        'SKIP_DOWNLOAD': 1, 'SKIP_SYNC': 1, 'SKIP_PULL': 1, 'SKIP_SHIP': 1, 'SKIP_BUILD': 1,
        'BUILD': 0, 'PULL': 0, 'REFRESH_WEIGHTS': 0, 'FORCE_SYNC': 0,
        'READY_TIMEOUT': 1800, 'TAIL': 0, 'HF_HUB_OFFLINE': 1,
        'EXTRA_ARGS': '--scheduling-policy priority --kv-cache-memory-bytes=11811160064 --cudagraph-capture-sizes ' + GRAPH + ' --cudagraph-metrics',
    }.items()}
    return env


def qwen_environment(p, head):
    return {k: str(v) for k, v in {
        'HF_HOME': head['hf_home'], 'TP1_CONTAINER_NAME': 'spark-' + p['id'],
        'SPARK_MODEL_REVISION': p['model']['revision'],
        'TP1_MODEL_ID': p['model']['repository'],
        'IMAGE': p['runtime']['image'], 'HOST_BIND': '127.0.0.1', 'PORT': p['runtime']['backend']['port'],
        'SERVED_MODEL_NAME': p['model']['served_name'], 'ABLIT': 0, 'YARN': 0,
        'MAX_MODEL_LEN': 262144, 'KV_TARGET_GIB': 20, 'HOST_RESERVE_GIB': 26, 'HOST_SLACK_GIB': 5,
        'KV_CACHE_DTYPE': 'fp8', 'MAMBA_SSM_CACHE_DTYPE': 'bfloat16', 'MAX_NUM_SEQS': 8,
        'MAX_NUM_BATCHED_TOKENS': 2048, 'MTP_NUM_SPECULATIVE_TOKENS': 3,
        'CUDAGRAPH_CAPTURE_SIZES': 'auto', 'COMPILATION_MODE': 0,
        'EXTRA_VLLM_ARGS': '--scheduling-policy priority', 'HF_TOKEN': '', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': 1,
    }.items()}


def render(root, recipe_id, site, output):
    root = root.resolve()
    if recipe_id not in SUPPORTED:
        raise ValueError('This recipe does not have a supported portable adapter')
    recipe = recipes.read_recipe(root, recipe_id)
    result = recipes.check(root, recipe)
    if not result['valid']:
        raise ValueError(str(result['failures']))
    selected = recipe['management_nodes']
    validate(site, selected)
    output = output.absolute()
    if output.exists():
        raise ValueError('Select a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.site-', dir=output.parent))
    try:
        release = site['release_root']
        config_root = release + '/site'
        cluster = deepcopy(site['cluster'])
        cluster.update(schema_version=1, cluster_id=site['id'], service_user=site['user'],
                       cli_path=release + '/venv/bin/spark-serve', profiles_dir=str(stage / 'profiles'),
                       runtime_dir=site['state_root'] + '/run', state_dir=site['state_root'] + '/state')
        cluster.pop('worker_pool', None)
        cluster.pop('model_routes', None)
        cluster.update(recipe['configuration'])
        cluster['gateway'].update(binary=release + '/bin/llama-swap', config_path=config_root + '/llama-swap.json')
        cluster['nodes'] = {k: {'host': site['nodes'][k]['host'], 'user': site['user'],
                              'model_root': site['nodes'][k]['model_root'],
                              'swap_devices': site['nodes'][k].get('swap_devices', [])} for k in selected}
        (stage / 'profiles').mkdir()
        (stage / 'adapters').mkdir()
        for rel in recipe['profiles']:
            p = json.loads((root / rel).read_text())
            engine = ('glm-full' if p['id'] == 'glm-5.3-full-tp4' else
                      'glm-adaptive' if 'GLM-5.3-Flash-EXL3' in p['model']['repository'] else 'qwen-single')
            if engine in site.get('image_references', {}):
                p['runtime']['image'] = safe(site['image_references'][engine], r'[A-Za-z0-9_./:@-]+', 'rebuilt image reference')
            nodes = p['topology']['nodes']
            launch_node = p['runtime']['launch']['node']
            head = site['nodes'][launch_node]
            if engine == 'glm-adaptive':
                other = next(n for n in nodes if n != launch_node)
                worker = site['nodes'][other]
                env = glm_environment(p, head, worker)
                ranks = [{'host': 'local', 'container': env['CONTAINER_HEAD'], 'image_id': head['images'][engine], 'cache': env['CACHE_ROOT'], 'runtime': head['runtime_root'] + '/' + engine},
                         {'host': worker['host'], 'container': env['CONTAINER_WORKER'], 'image_id': worker['images'][engine], 'cache': env['WORKER_VLLM_CACHE'], 'runtime': worker['runtime_root'] + '/' + engine}]
            elif engine == 'qwen-single':
                env = qwen_environment(p, head)
                env['EXTRA_DOCKER_ARGS'] = '-e VLLM_USE_V2_MODEL_RUNNER=1 --label dgx.spark-serve.deployment=' + site['id'] + ' --label dgx.spark-serve.profile=' + p['id']
                ranks = [{'host': 'local', 'container': env['TP1_CONTAINER_NAME'], 'image_id': head['images'][engine], 'runtime': head['runtime_root'] + '/' + engine}]
            else:
                env = {}
                ranks = []
                for index, name in enumerate(nodes):
                    n = site['nodes'][name]
                    rt = n['runtime_root'] + '/glm-full'
                    model = n['hf_home'] + '/hub/models--Tech2wild--GLM-5.3-Int4-Int8Mix/snapshots/' + p['model']['revision']
                    draft = n['hf_home'] + '/hub/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c'
                    ranks.append({'host': n['host'], 'container': 'spark-' + p['id'], 'image_id': n['images'][engine],
                                  'runtime': rt, 'rank': index, 'require_no_swap': False,
                                  'environment': {k: str(v) for k, v in {
                                      'SPARK_CONTAINER': 'spark-' + p['id'], 'SPARK_HOST_IP': n['ring']['ip'],
                                      'SPARK_HEAD_IP': head['ring']['ip'], 'SPARK_LAN_IF': n['ring']['interface'],
                                      'SPARK_HCAS': n['ring']['hcas'], 'SPARK_GID': n['ring']['gid'],
                                      'SPARK_WEIGHTS': model, 'SPARK_DRAFT': draft, 'SPARK_CACHE': n['cache_root'] + '/glm-full',
                                      'SPARK_TEMPLATE': rt + '/chat_template.jinja', 'KERNELS_DIR': rt + '/glm-triton',
                                      'DCP_DIR': rt + '/glm-dcp', 'DCP_IMAGE': p['runtime']['image'], 'DCP_SIZE': 4,
                                      'MAXLEN': 307200, 'MAXSEQS': 12, 'MAXBATCHED': 2048, 'KVBYTES': 6000000000,
                                      'KVTIER': 0, 'DFLASH_K': 7, 'NCCL_HOTPLUG': 0, 'PORT': p['runtime']['backend']['port'],
                                      'MASTER_PORT': 29541,
                                  }.items()}})
            runtime = head['runtime_root'] + '/' + engine
            manifest = {'engine': engine, 'profile': p['id'], 'deployment': site['id'],
                        'runtime': runtime, 'image': p['runtime']['image'], 'ranks': ranks,
                        'environment': env, 'arguments': ['start'] if engine == 'glm-adaptive' else []}
            write(stage / 'adapters' / (p['id'] + '.json'), manifest)
            for action in ('launch', 'stop'):
                p['runtime'][action]['argv'] = ['python3', release + '/deploy/adapter.py', config_root + '/adapters/' + p['id'] + '.json', action]
                p['runtime'][action]['node'] = launch_node
                p['runtime'][action].pop('environment', None)
            p['artifacts'] = [{'node': n, 'kind': 'docker_image', 'value': p['runtime']['image']} for n in nodes]
            p['artifacts'] += [{'node': launch_node, 'kind': 'file', 'value': runtime + '/runtime-source.json'}]
            for n in nodes:
                hf = site['nodes'][n]['hf_home'] + '/hub/models--' + p['model']['repository'].replace('/', '--')
                p['artifacts'].append({'node': n, 'kind': 'file', 'value': hf + '/snapshots/' + p['model']['revision'] + '/config.json'})
                p['artifacts'].append({'node': n, 'kind': 'file', 'value': site['nodes'][n]['runtime_root'] + '/' + engine + '/artifact-receipt.json'})
            p['operator'] = {'readiness': 'experimental', 'exposure': 'production', 'use_cases': ['managed model serving'],
                             'warnings': ['New site binding requires deployment acceptance. Performance evidence applies to the pinned settings and source.']}
            p['description'] = 'Pinned serving settings with explicit site bindings.'
            write(stage / 'profiles' / (p['id'] + '.json'), p)
        write(stage / 'cluster.json', cluster)
        catalog = load_catalog(stage / 'cluster.json')
        gateway = render_gateway_config(replace(catalog, config_path=Path(config_root + '/cluster.json')))
        cluster['profiles_dir'] = config_root + '/profiles'
        write(stage / 'cluster.json', cluster)
        write(stage / 'llama-swap.json', gateway)
        write(stage / 'site.json', site)
        write(stage / 'preparation.json', {'recipe': recipe_id, 'deployed': False, 'hardware_qualified': False,
              'source_checks': result, 'destination': config_root,
              'files': {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()}})
        # Reuse the exporter's exclusive, atomic directory publication primitive.
        from spark_serve.atomic import publish_exclusive
        publish_exclusive(stage, output)
        return {'prepared': str(output), 'deployed': False, 'recipe': recipe_id}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--recipe', required=True)
    p.add_argument('--site', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render(a.root, a.recipe, json.loads(a.site.read_text()), a.output), indent=2))
