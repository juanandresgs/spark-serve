"""Versioned evidence records. No network, GPU, or model lifecycle operations."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def fingerprint(record):
    return hashlib.sha256(canonical({k: v for k, v in record.items() if k != 'fingerprint'})).hexdigest()


def seal(record):
    return dict(record, fingerprint=fingerprint(record))


def safe_file(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe evidence path: ' + relative)
    return path


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def resolve(root, reference):
    path = safe_file(root, reference['file'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Source receipt changed: ' + reference['file'])
    value = read(path)
    for key in reference['path']:
        value = value[key]
    return value


def validate_record(record, root, schema_check=True):
    # Optional tooling dependency, deliberately not a serving runtime dependency.
    kind = record.get('kind')
    if kind not in {'recipe', 'run', 'recommendation'}:
        raise ValueError('Unknown evidence kind')
    if schema_check:
        from jsonschema import Draft202012Validator
        schema = read(Path(root) / 'evidence/schemas' / f'{kind}-v1.schema.json')
        Draft202012Validator(schema).validate(record)
    if record['fingerprint'] != fingerprint(record):
        raise ValueError('Record content fingerprint mismatch: ' + record['id'])
    if kind == 'run':
        if len({m['id'] for m in record['measurements']}) != len(record['measurements']):
            raise ValueError('Duplicate measurement ID')
        expected = {'ttft':'s', 'response_latency':'s', 'decode_rate_proxy':'tokens/s',
                    'end_to_end_throughput':'tokens/s', 'reported_throughput':'tokens/s',
                    'correct':'count', 'cases':'count', 'reported_prompt_tokens':'tokens'}
        for m in record['measurements']:
            if m['unit'] != expected[m['metric']]:
                raise ValueError('Metric unit mismatch')
            if m['value'] is not None and not math.isfinite(m['value']):
                raise ValueError('Nonfinite measurement')
            if m['source'] is not None:
                original = resolve(root, m['source'])
                if isinstance(original, bool) or original != m['value']:
                    raise ValueError('Measurement differs from source')
        for source in record['sources']:
            resolve(root, source)
        if record['observations']:
            observations = record['observations']
            accounting = record['accounting']
            if accounting['requests_attempted'] != len(observations):
                raise ValueError('Observation count differs from request accounting')
            for field in ['prompt_tokens', 'completion_tokens', 'cached_tokens']:
                values = [o[field] for o in observations]
                expected_sum = sum(values) if all(v is not None for v in values) else None
                if accounting[field] != expected_sum:
                    raise ValueError('Token accounting differs from observations: ' + field)
        if record['origin'] == 'external_report':
            if not record['external_sources'] or record['identity']['relationship'] != 'external_report':
                raise ValueError('External report needs pinned attribution')
            if any('/blob/' + source['revision'] + '/' not in source['url'] for source in record['external_sources']):
                raise ValueError('External source URL must be pinned to its recorded commit')
            if any(q['status'] == 'passed' for q in record['qualification'].values()):
                raise ValueError('External report cannot qualify our recipe')
        if record['origin'] == 'local' and (record['identity']['relationship'] != 'attested_exact' or
                any(record['identity'][k] is None for k in ['image_digest','runtime_revision','model_revision'])):
            raise ValueError('New local runs need an explicit actual-identity attestation')
    return record


def load(root, schema_check=True):
    records = {}
    for kind in ['recipe', 'run', 'recommendation']:
        for path in sorted((Path(root) / 'evidence' / (kind + 's')).glob('*.json')):
            r = validate_record(read(safe_file(root, str(path.relative_to(root)))), root, schema_check)
            if r['id'] in records:
                raise ValueError('Duplicate record ID: ' + r['id'])
            if r['id'] != path.stem:
                raise ValueError('Filename must equal record ID')
            records[r['id']] = r
    def linked(ref, kind):
        target = records.get(ref['id'])
        if target is None or target['kind'] != kind or target['fingerprint'] != ref['fingerprint']:
            raise ValueError('Missing or mismatched ' + kind + ' reference: ' + ref['id'])
        return target
    for r in records.values():
        if r['kind'] == 'run':
            linked(r['recipe'], 'recipe')
        elif r['kind'] == 'recommendation':
            selected = linked(r['selected'], 'recipe')
            if selected['model'] != r['model'] or selected['hardware']['nodes'] != r['hardware_nodes']:
                raise ValueError('Recommendation model/hardware mismatch')
            runs = [linked(ref, 'run') for ref in r['evidence_runs']]
            if not any(run['recipe'] == r['selected'] for run in runs):
                raise ValueError('Recommendation has no selected-recipe evidence')
    return records


def comparison_reasons(left, lm, right, rm):
    """Fail closed. Complete-recipe comparisons may differ in recipe settings only."""
    reasons = []
    if left['origin'] == 'external_report' or right['origin'] == 'external_report':
        reasons.append('external reports are unmatched')
    for key in ['metric', 'unit', 'definition', 'aggregation', 'samples', 'sample_unit', 'population', 'conditions']:
        if lm[key] != rm[key]:
            reasons.append('different ' + key)
    for m in [lm, rm]:
        c = m['conditions']
        if m['samples'] is None or m['value'] is None or c['fixture_sha256'] is None or c['concurrency'] is None:
            reasons.append('missing sample, fixture, concurrency or value metadata')
        if c['cache'] in {'unknown', 'uncontrolled'} or not c['sampling'] or not c['request']:
            reasons.append('unknown cache or request settings')
        if m['metric'] == 'reported_throughput' or m['aggregation'] == 'reported_unknown':
            reasons.append('unknown measurement definition')
    for r in [left, right]:
        if (r['identity']['relationship'] != 'attested_exact' or r['environment']['unknowns']
                or any(r['identity'][k] is None for k in ['image_digest','runtime_revision','model_revision'])):
            reasons.append('incomplete actual identity/environment')
        if r['failures']:
            reasons.append('run contains failures')
    if left['environment'] != right['environment'] or left['workload_version'] != right['workload_version']:
        reasons.append('different environment/workload version')
    return sorted(set(reasons))


def percent_change(left, lm, right, rm):
    reasons = comparison_reasons(left, lm, right, rm)
    if reasons:
        raise ValueError('Incomparable evidence: ' + '; '.join(reasons))
    if lm['value'] == 0:
        raise ValueError('Zero baseline')
    return 100 * (rm['value'] / lm['value'] - 1)


def write_bytes_once(target, body):
    """Publish complete bytes atomically without replacing any existing name."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink()
    return target


def write_immutable(record, directory, root):
    """Atomic create-only publication. Concurrent writers cannot replace an existing ID."""
    record = seal(record)
    validate_record(record, root)
    target = Path(directory) / (record['id'] + '.json')
    return write_bytes_once(target, json.dumps(record, indent=2, allow_nan=False).encode() + b'\n')


def check_immutable(root, base):
    """CI compares history, not only a mutable checksum stored beside the records."""
    result = subprocess.run(['git', '-C', str(root), 'diff', '--name-status', base, '--',
                             'evidence/recipes', 'evidence/runs', 'evidence/recommendations', 'evidence/sources', 'evidence/schemas'],
                            check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if not line.startswith('A\t'):
            raise ValueError('Evidence history is append-only: ' + line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--immutable-base')
    args = parser.parse_args()
    records = load(args.root)
    if args.immutable_base:
        check_immutable(args.root, args.immutable_base)
    print(f'Validated {len(records)} linked evidence records')

if __name__ == '__main__':
    main()
