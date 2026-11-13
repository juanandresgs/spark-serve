#!/usr/bin/env python3
"""Prepare portable launch source; never download weights or touch a running engine."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Launcher source anchor drift: ' + old[:80])
    return text.replace(old, new, 1)


def inventory(output):
    root = output.resolve()
    files = {}
    for path in sorted(output.rglob('*')):
        relative = path.relative_to(output)
        if path.is_symlink() or '.git' in relative.parts:
            raise ValueError('Unsafe prepared source path: ' + str(relative))
        if path.is_dir():
            continue
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe prepared source path: ' + str(relative))
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def normalize(output, engine, source_revision, source_files):
    if not re.fullmatch(r'[0-9a-f]{40}', source_revision):
        raise ValueError('Prepared source revision is not an immutable pin')
    source_files = dict(source_files)
    if not source_files or inventory(output) != source_files:
        raise ValueError('Prepared source differs from the pinned staging inventory')
    path = output / 'start.sh'
    raw = path.read_bytes()
    text = raw.decode()
    if engine == 'glm-adaptive':
        if hashlib.sha256(raw).hexdigest() != 'c7bc7d5857c0383aceeae7fd633a440abf0d3ad80bcdd06572e155df2fc0a4bb':
            raise ValueError('Expected the pinned E2 integration patch first')
        anchor = 'set -a\n# shellcheck disable=SC1091\nsource "$_env_file"\nset +a'
        text = replace_once(text, anchor, '_orch_mixed="${GLM53_MIXED_PREFILL_CHUNK:-512}"\n' + anchor + '\nexport GLM53_MIXED_PREFILL_CHUNK="$_orch_mixed"')
        text = replace_once(text, 'WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"', 'WORKER_CACHE_DIR="${SPARK_WORKER_HF_HOME:?Worker cache path is required}"')
        # A Docker name collision must fail, even if a second controller races us.
        # Managed stop resolves immutable IDs and checks ownership in adapter.py.
        text = replace_once(text, '    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true\n    worker_ssh "docker rm -f \'$CONTAINER_WORKER\'" >/dev/null 2>&1 || true', '    # Container replacement belongs to the spark-serve broker.')
        text = replace_once(text, '    worker_ssh "docker run -d --name \'$CONTAINER_WORKER\' ', '    worker_ssh "docker run -d --label dgx.spark-serve.deployment=$DGX_SPARK_SERVE_DEPLOYMENT_ID --label dgx.spark-serve.profile=$DGX_SPARK_SERVE_PROFILE_ID --name \'$CONTAINER_WORKER\' ')
        text = replace_once(text, '    docker run -d --name "$CONTAINER_HEAD" ', '    docker run -d --label "dgx.spark-serve.deployment=$DGX_SPARK_SERVE_DEPLOYMENT_ID" --label "dgx.spark-serve.profile=$DGX_SPARK_SERVE_PROFILE_ID" --name "$CONTAINER_HEAD" ')
    elif engine == 'qwen-single':
        # The loopback patch must be applied before this function.
        if hashlib.sha256(raw).hexdigest() != '167685607453b22e162a0e5be4786bbc17d2032ce1680b1bfb8f091d2ccb20f1':
            raise ValueError('Expected the pinned Qwen loopback patch first')
        text = replace_once(text, 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true', '# Container replacement belongs to the spark-serve broker.')
        text = replace_once(text, '    python3 - "$1" <<\'PY\'', '    python3 - "$1" "${SPARK_MODEL_REVISION:?Pinned model revision is required}" <<\'PY\'')
        text = replace_once(text, 'main = (repo / "refs" / "main").read_text().strip() if (repo / "refs" / "main").is_file() else ""', 'main = sys.argv[2]')
        text = replace_once(text, 'complete_snaps = []', 'raise SystemExit("Pinned checkpoint is incomplete; run the weight staging step")\ncomplete_snaps = []')
        shutil.copyfile(output / '.env.sample', output / '.env')
    else:
        raise ValueError('Unsupported engine')
    path.write_text(text)
    path.chmod(0o755)
    files = inventory(output)
    transformed = {'start.sh'} | ({'.env'} if engine == 'qwen-single' else set())
    if set(files) != set(source_files) | (transformed - {'start.sh'}) or any(
            files[name] != digest for name, digest in source_files.items() if name not in transformed):
        raise ValueError('Prepared source changed during normalization')
    receipt = {'engine': engine, 'source_revision': source_revision,
               'source_files': source_files, 'files': files, 'hardware_qualified': False,
               'changes': 'Site cache path and managed ownership only; serving geometry comes from the pinned profile.'}
    (output / 'runtime-source.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
