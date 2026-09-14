#!/usr/bin/env python3
"""Prepare the pinned full-GLM runtime and site-independent rank launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from build_images import source_copy
from prepare_runtime import replace_once


def prepare(root, sources, output):
    pins = json.loads((root / 'sources.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    for name in ('glm-full-dcp', 'glm-full-kernels'):
        source_copy(sources / name, output / 'sources' / name, pins['sources'][name]['revision'])
    dcp = output / 'sources/glm-full-dcp'
    kernel = output / 'sources/glm-full-kernels'
    shutil.copytree(kernel / 'kernels', output / 'glm-triton')
    # Resolve by expected content digest, never by a possibly duplicate basename.
    overlay = list((dcp / 'overlay').rglob('*.py'))
    for line in (dcp / 'stage/SHA256SUMS').read_text().splitlines():
        sha, rel = line.split()
        matches = [p for p in overlay if hashlib.sha256(p.read_bytes()).hexdigest() == sha]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous DCP overlay: ' + rel)
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0], dest)
    text = (dcp / 'launch-glm53big-dcp.sh').read_text()
    text = replace_once(text, 'set -uo pipefail', 'set -euo pipefail')
    begin = text.index('# RANK ORDER MUST MATCH THE PHYSICAL RING')
    end = text.index('\nesac', begin) + len('\nesac')
    text = text[:begin] + '''# Rank placement is supplied by the site manifest.
case "$NODE_RANK" in 0) HEADLESS=0 ;; 1|2|3) HEADLESS=1 ;; *) exit 2 ;; esac
HOST_IP="${SPARK_HOST_IP:?Site rank IP required}"
''' + text[end:]
    replacements = [
        ('NAME="vllm_glm53big"', 'NAME="${SPARK_CONTAINER:?Container identity required}"'),
        ('PORT=8000', 'PORT="${PORT:-8895}"'),
        ('MASTER_PORT=29541', 'MASTER_PORT="${MASTER_PORT:-29541}"'),
        ('LAN_IF="enP7s7"', 'LAN_IF="${SPARK_LAN_IF:?Site interface required}"'),
        ('HEAD_IP="192.168.1.228"', 'HEAD_IP="${SPARK_HEAD_IP:?Site head IP required}"'),
        ('WEIGHTS=/var/tmp/models/GLM-5.3-Int4-Int8Mix', 'WEIGHTS="${SPARK_WEIGHTS:?Pinned weight path required}"'),
        ('KVTIER="${KVTIER:-1}"', 'KVTIER="${KVTIER:-0}"'),
        ('[ "${DRYRUN:-0}" = 1 ] || docker rm -f "$NAME" 2>/dev/null', '# Container replacement belongs to the broker.'),
        ('run_docker run -d --name "$NAME"', 'run_docker run -d --label "dgx.spark-serve.deployment=$DGX_SPARK_SERVE_DEPLOYMENT_ID" --label "dgx.spark-serve.profile=$DGX_SPARK_SERVE_PROFILE_ID" --name "$NAME"'),
        ('    --async-scheduling \\', '    --async-scheduling --scheduling-policy priority \\'),
        ('/var/tmp/models/GLM-5.3-DFlash2-draft', '${SPARK_DRAFT}'),
        ('-v /var/tmp/models:/cache/huggingface', '-v "${SPARK_CACHE}:/cache/huggingface"'),
        ('-v "$WEIGHTS:/models/glm-5.3:ro"', '-v "$WEIGHTS:/models/glm-5.3:ro" -v "${SPARK_TEMPLATE}:/opt/glm53/chat_template.jinja:ro"'),
        ("-e NCCL_IB_HCA='=roceP2p1s0f0,roceP2p1s0f1'", '-e "NCCL_IB_HCA=${SPARK_HCAS:?Site ring HCAs required}"'),
        ('-e NCCL_IB_GID_INDEX=3', '-e "NCCL_IB_GID_INDEX=${SPARK_GID:?Site RoCE GID required}"'),
        ('--served-model-name glm-5.3 --host 0.0.0.0', '--served-model-name glm-5.3 --host 127.0.0.1'),
        ('    --default-chat-template-kwargs', '    --chat-template /opt/glm53/chat_template.jinja \\\n    --default-chat-template-kwargs'),
    ]
    for old, new in replacements:
        if old == '/var/tmp/models/GLM-5.3-DFlash2-draft':
            if text.count(old) != 3:
                raise ValueError('Draft mount anchor drift')
            text = text.replace(old, new)
        else:
            text = replace_once(text, old, new)
    (output / 'start.sh').write_text(text)
    (output / 'start.sh').chmod(0o755)
    template = output / 'chat_template.official.jinja'
    url = 'https://huggingface.co/zai-org/GLM-5.3/resolve/aca966e4e02791568aa6a4ced368624b3d897f42/chat_template.jinja'
    template.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    subprocess.run(['python3', str(root / 'cluster/glm53-full-tp4/prepare-chat-template.py'), str(template), str(output / 'chat_template.jinja')], check=True)
    if hashlib.sha256((output / 'chat_template.jinja').read_bytes()).hexdigest() != '59ef6e1e3e592d1a7ab4581719f502c81e06c8ef66d7749ebb4f003feaa804eb':
        raise ValueError('Transformed template drift')
    subprocess.run(['bash', '-n', str(output / 'start.sh')], check=True)
    receipt = {'engine': 'glm-full', 'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}, 'hardware_qualified': False}
    (output / 'runtime-source.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.root, a.sources, a.output), indent=2))
