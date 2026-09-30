"""Site-owned schema decoding at the pinned TensorFold CUDA parser boundary.

The runtime owns envelope parsing; this adapter decodes JSON representations only
when an explicit parameter schema accepts the resulting value. Never execute tools.
"""
import copy
import json
import math
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable

def no_remote(uri):
    raise Unresolvable(ref=uri)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite JSON constant')
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('non-finite JSON number')
        if isinstance(item, dict):
            for child in item.values(): finite(child)
        if isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def decode(value, schema, validator):
    if not isinstance(value, str) or not isinstance(schema, dict):
        return value
    # Preserve strings whenever accepted, including unions, refs and untyped fields.
    scoped = validator.evolve(schema=schema)
    if scoped.is_valid(value):
        return value
    candidates = []
    # XML values are text, not necessarily JSON: Qwen also emits Python booleans.
    # Only these two exact spellings are recognized, and schema validation below
    # still decides whether conversion is allowed. String fields returned above.
    if value.strip() in ('True', 'False'):
        candidates.append(value.strip() == 'True')
    try:
        candidates.append(strict_json(value))
    except (ValueError, TypeError):
        # Some templates/model outputs encode an object as JSON-string contents.
        # Decode exactly one JSON string layer, never replace slashes heuristically.
        if value.lstrip().startswith(('{\\"', '[{\\"')):
            try:
                candidates.append(strict_json(strict_json('"' + value + '"')))
            except (ValueError, TypeError):
                pass
    for candidate in candidates:
        if scoped.is_valid(candidate):
            return candidate
    return value  # Invalid model output remains invalid for the client to reject.


def adapt_calls(calls, tools):
    if not calls:
        return calls
    specs = {}
    for tool in tools:
        function = tool.get('function', tool)
        if isinstance(function, dict) and isinstance(function.get('parameters'), dict):
            specs[function.get('name')] = function['parameters']
    result = copy.deepcopy(calls)
    for call in result:
        function = call.get('function', {})
        schema = specs.get(function.get('name'))
        if schema is None:
            continue
        try:
            args = strict_json(function['arguments'])
            if not isinstance(args, dict):
                continue
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema, registry=Registry(retrieve=no_remote))
            properties = schema.get('properties', {})
            for name, value in args.items():
                if name in properties:
                    args[name] = decode(value, properties[name], validator)
            function['arguments'] = json.dumps(args, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        except (ValueError, TypeError, KeyError, SchemaError, Unresolvable):
            continue
    return result


def install():
    import tensorfold.cuda.server as server
    original = server.parse_tool_calls
    if getattr(original, '_spark_schema_adapter', False):
        raise RuntimeError('adapter already installed')
    def parse(text, tools, *, max_calls=None):
        content, calls = original(text, tools, max_calls=max_calls)
        return content, adapt_calls(calls, tools)
    parse._spark_schema_adapter = True
    server.parse_tool_calls = parse
