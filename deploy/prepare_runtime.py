#!/usr/bin/env python3
"""Prepare portable launch source; never download weights or touch a running engine."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Launcher source anchor drift: ' + old[:80])
    return text.replace(old, new, 1)


def prepare(source, output, engine):
    source, output = source.resolve(), output.absolute()
    if output.exists():
        raise ValueError('Use a new output directory')
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    if not any(names):
        raise ValueError('Source checkout is empty')
    for name in filter(None, names):
        src = source / name
        if src.is_symlink() or not src.resolve().is_relative_to(source):
            raise ValueError('Unsafe source path')
        dst = output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        dst.chmod(src.stat().st_mode & 0o777)
    return normalize(output, engine)


def normalize(output, engine):
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
    files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file() and '.git' not in p.parts and p.name != 'runtime-source.json'}
    receipt = {'engine': engine, 'files': files, 'hardware_qualified': False,
               'changes': 'Site cache path and managed ownership only; serving geometry comes from the pinned profile.'}
    (output / 'runtime-source.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('engine', choices=['glm-adaptive', 'qwen-single'])
    p.add_argument('--prepared-source', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(normalize(a.prepared_source, a.engine), indent=2))
