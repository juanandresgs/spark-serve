"""Opt-in, request-local JSON validation and bounded reasoning continuation.

Independently authored at App.prepare/run boundaries; no decoder math changes.
JSON replies are buffered until validated, not grammar-constrained. Ordinary
requests take the original path. A thinking cap closes the reasoning segment
and resumes the same token prefix once, within the caller's total allowance.
"""
import dataclasses
import hashlib
import json
import re
from jsonschema import Draft202012Validator
from referencing import Registry
from tool_adapter import strict_json, no_remote


def contract(body):
    if not isinstance(body, dict):
        raise ValueError('request body must be an object')
    fmt = body.get('response_format')
    schema = None
    if fmt is not None:
        if not isinstance(fmt, dict):
            raise ValueError('response_format must be an object')
        kind = fmt.get('type', 'text')
        if kind == 'json_object':
            schema = {'type': 'object'}
        elif kind == 'json_schema':
            spec = fmt.get('json_schema', {})
            if not isinstance(spec, dict) or not isinstance(spec.get('schema'), dict):
                raise ValueError('json_schema.schema must be an object')
            schema = spec['schema']
        elif kind != 'text':
            raise ValueError('unsupported response_format')
    if schema is not None:
        Draft202012Validator.check_schema(schema)
        # Reject external references before inference; local $defs remain usable.
        def refs(x):
            if isinstance(x, dict):
                for k,v in x.items():
                    if k in ('$ref', '$dynamicRef') and (not isinstance(v,str) or not v.startswith('#')):
                        raise ValueError('external schema references are not supported')
                    refs(v)
            elif isinstance(x,list):
                for v in x: refs(v)
        refs(schema)
    policy = body.get('spark_reliability', {})
    if not isinstance(policy, dict) or set(policy) - {'thinking_budget','answer_reserve'}:
        raise ValueError('unsupported spark_reliability policy')
    budget = policy.get('thinking_budget', 0)
    reserve = policy.get('answer_reserve', 1024)
    if type(budget) is not int or budget < 0 or type(reserve) is not int or reserve < 32:
        raise ValueError('thinking_budget must be nonnegative and answer_reserve at least 32')
    if (budget or schema is not None) and body.get('tools'):
        raise ValueError('reliability policy with tools is not qualified; use ordinary typed tools')
    return schema, budget, reserve


def normalize(content, schema, finish):
    if finish != 'stop':
        raise ValueError('structured output did not finish; refusing a partial object')
    raw = content.strip()
    fenced = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```',raw,re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    value = strict_json(candidate)
    validator = Draft202012Validator(schema, registry=Registry(retrieve=no_remote))
    validator.validate(value)
    # Only an entire JSON fence is removed; no content repairs or coercions.
    return candidate.strip() if fenced else content, bool(fenced)


def combined_stats(first, second):
    """Preserve original-input cache accounting and report both generation phases."""
    result = dict(first)
    for name in ('prefill_s', 'decode_s', 'rounds'):
        result[name] = first.get(name, 0) + second.get(name, 0)
    if 'min_rows' in first and 'min_rows' in second:
        result['min_rows'] = min(first['min_rows'], second['min_rows'])
    return result


def install():
    import tensorfold.cuda.server as server
    original_prepare, original_run = server.App.prepare, server.App.run
    if getattr(original_run,'_spark_reliability',False):
        raise RuntimeError('reliability already installed')

    def prepare(self, body, chat):
        try:
            schema,budget,reserve=contract(body)
            prepared=original_prepare(self,body,chat)
            if (schema is not None or budget) and not chat:
                raise ValueError('reliability policy requires chat completions')
            boundary_size = len(self.tok.encode('\n</think>\n\n', add_special_tokens=False).ids) if budget else 0
            if budget and (not prepared.thinking or budget+reserve+boundary_size>prepared.max_tokens):
                raise ValueError('thinking budget requires thinking and space for answer reserve plus boundary')
            return prepared
        except (ValueError,KeyError,TypeError) as exc:
            raise server.RequestError(str(exc)) from None
        except Exception as exc:
            if exc.__class__.__module__.startswith(('jsonschema','referencing')):
                raise server.RequestError(str(exc)) from None
            raise

    def run(self,body,chat,emit,*,prepared=None,cancelled=None):
        schema,budget,reserve=contract(body)
        if schema is None and not budget:
            return original_run(self,body,chat,emit,prepared=prepared,cancelled=cancelled)
        prepared=prepared if prepared is not None else self.prepare(body,chat)
        work=dict(body,return_token_ids=True)
        if budget and work.get('seed') is None:
            from tensorfold.engine.exact_sampling import seed_for
            work['seed']=seed_for(prepared.prompt)
        first=dataclasses.replace(prepared,max_tokens=budget) if budget else prepared
        # Buffer opt-in replies: no invalid JSON fragments can escape through SSE.
        result=original_run(self,work,chat,lambda delta: True,prepared=first,cancelled=cancelled)
        meta={'version':1,'buffered':True,'thinking_budget':budget,'forced_exit':False,'normalized_fence':False}
        if budget and result['finish']=='length':
            ids=result['stats']['token_ids']
            decoded=self.tok.decode(ids,skip_special_tokens=False)
            needs_boundary='</think>' not in decoded
            boundary=self.tok.encode('\n</think>\n\n',add_special_tokens=False).ids if needs_boundary else []
            remaining=prepared.max_tokens-len(ids)-len(boundary)
            if remaining < reserve:
                raise server.RequestError('insufficient remaining answer budget')
            follow=dataclasses.replace(prepared,prompt=prepared.prompt+ids+boundary,max_tokens=remaining,thinking=False)
            tail=original_run(self,work,chat,lambda delta: True,prepared=follow,cancelled=cancelled)
            meta.update(forced_exit=needs_boundary,injected_tokens=len(boundary),continuation=True,
                        continuation_prompt_tokens=tail['prompt_tokens'],first_phase_tokens=len(ids))
            meta['phase_stats']=[{k:v for k,v in phase['stats'].items() if k != 'token_ids'} for phase in (result,tail)]
            stats=combined_stats(result['stats'],tail['stats'])
            all_ids=ids+boundary+tail['stats']['token_ids']
            result={**tail,'content':result['content']+tail['content'],'reasoning':result['reasoning'],
                    'prompt_tokens':len(prepared.prompt),
                    'completion_tokens':result['completion_tokens']+tail['completion_tokens'],
                    'stats':{**stats,'token_ids':all_ids,'token_sha':server.token_sha(all_ids)}}
        raw=result['content']
        meta['raw_content_sha256']=hashlib.sha256(raw.encode()).hexdigest()
        if schema is not None:
            try:
                result['content'],meta['normalized_fence']=normalize(raw,schema,result['finish'])
            except Exception as exc:
                raise server.RequestError('structured output validation failed: '+str(exc)) from None
            meta['schema_validated']=True
        result['stats']['spark_reliability']=meta
        if not body.get('return_token_ids'):
            result['stats'].pop('token_ids',None)
        if cancelled is not None and cancelled():
            raise server.RequestCancelled('client left during buffered reply')
        delta={k:v for k,v in [('reasoning_content',result['reasoning']),('content',result['content'])] if v}
        if delta and not emit(delta):
            raise server.RequestCancelled('client left during buffered reply')
        result['final']={}
        return result
    run._spark_reliability=True
    server.App.prepare,server.App.run=prepare,run
