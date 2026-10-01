"""Portable prompts and graders for the bounded synthetic quality samples."""
import json
import hashlib
import re

REASONING_SUFFIX = " Return only a JSON object with one integer field named answer."
CODING_PREFIX = "Implement a pure Python function solve(x). "
CODING_SUFFIX = " Use only the standard library. Output only Python source, with no Markdown or explanation."


def reasoning_prompt(prompt):
    return prompt + REASONING_SUFFIX


def coding_prompt(prompt):
    return CODING_PREFIX + prompt + CODING_SUFFIX


def integer_answer(value, expected):
    return (
        type(value) is dict
        and set(value) == {"answer"}
        and type(value["answer"]) is int
        and value["answer"] == expected
    )


def grade_reasoning(output, expected):
    try:
        raw = json.loads(output)
        format_compliant = type(raw) is dict and set(raw) == {"answer"} and type(raw["answer"]) is int
    except (ValueError, TypeError):
        raw = None
        format_compliant = False
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", output.strip(), re.S)
    normalized = match.group(1) if match else output
    try:
        logical_correct = integer_answer(json.loads(normalized), expected)
    except (ValueError, TypeError):
        logical_correct = False
    return {
        "format_compliant": format_compliant,
        "logical_answer_correct": logical_correct,
        "markdown_fence_removed": bool(match),
    }


def coding_source(output):
    """Apply the published single optional Python-fence normalization."""
    code = output.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", code, re.S)
    return match.group(1) if match else code


def coding_prompt_seed(base_seed, index):
    if base_seed not in (PRIMARY_PROMPT_SEED, SECONDARY_PROMPT_SEED):
        raise ValueError("unknown published coding prompt-seed set")
    if type(index) is not int or not 0 <= index < 20:
        raise ValueError("coding case index must be from 0 through 19")
    return base_seed + index


PRIMARY_PROMPT_SEED = 61000
SECONDARY_PROMPT_SEED = 72000


def protocol():
    from quality_fixtures import reasoning_fixtures
    from coding_cases import CASES

    fixture_bytes = json.dumps(
        reasoning_fixtures(), ensure_ascii=True, separators=(",", ":")
    ).encode()
    coding_bytes = json.dumps(CASES, ensure_ascii=True, separators=(",", ":")).encode()
    return {
        "reasoning_fixture_seeds": [937, 1459],
        "reasoning_fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "coding_cases_sha256": hashlib.sha256(coding_bytes).hexdigest(),
        "reasoning_prompt_seed_bases": [PRIMARY_PROMPT_SEED, SECONDARY_PROMPT_SEED],
        "reasoning_prompts_per_seed_set": 200,
        "reasoning_unique_prompts": 195,
        "coding_cases": 20,
        "coding_prompt_seed_bases": [PRIMARY_PROMPT_SEED, SECONDARY_PROMPT_SEED],
        "request": {
            "clients_per_group": 4,
            "concurrency": "C4: four concurrent client requests in each reasoning or coding latency group",
            "stream": True,
            "stream_usage": True,
            "prompt_seed": "base_seed + fixture_index",
        },
        "generation": {
            "reasoning_effort_top_level": "medium",
            "reasoning_effort_chat_template": "medium",
            "enable_thinking": True,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
        "reasoning_grade": "Only a JSON object with exactly one integer key named answer is format-compliant; logical answer accepts an optional json fence, with booleans rejected as integers.",
        "coding_grade": "Remove one optional Python fence, execute solve(x) against every trusted input/expected pair in an isolated CPU sandbox, require exit code 0 and CHECKS_PASSED.",
        "completion_gate": "finish_reason must be stop and the request must have no transport/API error.",
        "scope": "Synthetic exact-answer reasoning and 20 executable coding cases; not a broad model-quality benchmark.",
        "runner_included": False,
    }
