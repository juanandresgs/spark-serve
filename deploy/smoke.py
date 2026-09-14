#!/usr/bin/env python3
"""Record API and optional managed restart acceptance for one prepared profile."""
import argparse
import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import zlib
import urllib.request

from spark_serve.config import load_catalog
from spark_serve.gateway import json_request, _api_key
from spark_serve.system import run_command


def red_image():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0))
    raw += chunk(b'IDAT', zlib.compress((b'\x00' + b'\xff\x00\x00' * 64) * 64)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


def check_api(catalog, profile, rows=None):
    if rows is None:
        rows = []
    def request(name, messages, extra=None):
        payload = {'model': profile['id'], 'messages': messages, 'temperature': 0, 'max_tokens': 96,
                   'chat_template_kwargs': {'enable_thinking': False}} | (extra or {})
        start = time.monotonic()
        result = json_request(catalog, 'POST', '/v1/chat/completions', payload=payload, timeout=300)
        message = result['choices'][0]['message']
        rows.append({'check': name, 'seconds': round(time.monotonic() - start, 3), 'usage': result.get('usage'), 'passed': False})
        return message
    models = json_request(catalog, 'GET', '/v1/models')
    if profile['id'] not in {p['id'] for p in models['data']}:
        raise ValueError('Profile is absent from the gateway model catalog')
    payload = {'model': profile['id'], 'messages': [{'role': 'user', 'content': 'Reply with exactly the word READY.'}],
               'temperature': 0, 'max_tokens': 96, 'stream': True, 'stream_options': {'include_usage': True},
               'chat_template_kwargs': {'enable_thinking': False}}
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('SPARK_SERVE_API_KEY') or _api_key(catalog)
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(catalog.cluster['gateway']['base_url'].rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers)
    content, usage, done = [], None, False
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            value = json.loads(data)
            if value.get('usage'):
                usage = value['usage']
            for choice in value.get('choices', []):
                content.append(choice.get('delta', {}).get('content') or '')
    if not done or ''.join(content).strip() != 'READY':
        raise ValueError('Text smoke did not return READY')
    rows.append({'check': 'streamed-text', 'seconds': round(time.monotonic() - start, 3), 'usage': usage, 'passed': True})
    if profile['serving']['tools']:
        answer = request('tool', [{'role': 'user', 'content': 'Call record_value once with value 37. Do not answer in prose.'}],
                         {'tools': [{'type': 'function', 'function': {'name': 'record_value', 'description': 'Record the integer provided by the user.', 'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value'], 'additionalProperties': False}}}], 'tool_choice': 'auto'})
        calls = answer.get('tool_calls') or []
        if len(calls) != 1 or calls[0]['function']['name'] != 'record_value' or json.loads(calls[0]['function']['arguments']) != {'value': 37}:
            raise ValueError('Tool smoke failed')
        rows[-1]['passed'] = True
    if 'image' in profile['model'].get('input_modalities', []):
        answer = request('image', [{'role': 'user', 'content': [{'type': 'text', 'text': 'What solid color is this image? Reply with only the color name.'}, {'type': 'image_url', 'image_url': {'url': red_image()}}]}])
        if 'red' not in (answer.get('content') or '').lower():
            raise ValueError('Image smoke failed')
        rows[-1]['passed'] = True
    return rows


def smoke(config, model, output, cycle=False):
    if output.exists():
        raise ValueError('Use a new receipt path')
    catalog = load_catalog(config)
    profile = catalog.resolve(model)
    receipt = {'model': profile['id'], 'context_tokens': profile['model']['context_tokens'],
               'model_revision': profile['model']['revision'], 'cycle_requested': cycle,
               'passed': False, 'checks': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        check_api(catalog, profile, receipt['checks'])
        if cycle:
            cli = [catalog.cluster['cli_path'], '--config', str(config)]
            subprocess.run([*cli, 'stop', profile['id']], check=True)
            adapter_path = Path(profile['runtime']['launch']['argv'][2])
            manifest = json.loads(adapter_path.read_text())
            # Query relative to the adapter's launch node, including its SSH peer.
            launch_node = catalog.cluster['nodes'][profile['runtime']['launch']['node']]
            for rank in manifest['ranks']:
                argv = ['docker', 'container', 'ls', '-aq', '--filter', 'name=^/' + rank['container'] + '$']
                if rank['host'] != 'local':
                    import shlex
                    argv = ['ssh', '-o', 'BatchMode=yes', rank['host'], shlex.join(argv)]
                result = run_command(launch_node, argv)
                if result.stdout.strip():
                    raise ValueError('A rank container remains after managed stop')
            receipt['checks'].append({'check': 'all-ranks-stopped', 'passed': True})
            subprocess.run([*cli, 'serve', profile['id']], check=True)
            check_api(catalog, profile, receipt['checks'])
            receipt['checks'].append({'check': 'managed-restart', 'passed': True})
        receipt['passed'] = True
    except Exception as exc:
        receipt['failure'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycle', action='store_true', help='Stop and restart this profile after the first API checks; drain callers first')
    a = p.parse_args()
    print(json.dumps(smoke(a.config, a.model, a.output, a.cycle), indent=2))
