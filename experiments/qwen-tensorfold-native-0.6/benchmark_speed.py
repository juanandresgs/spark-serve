#!/usr/bin/env python3
"""Replay the portable cell-warmup-v2 C1/C4 fixed-cap throughput protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import time
from urllib.parse import urlsplit

import speed_protocol
from curate_speed import curate
from transport import request

ROOT = Path(__file__).resolve().parent
IDENTITY_KEYS = {
    "schema_version", "image_digest", "runtime_repository", "runtime_revision",
    "runtime_version", "model_repository", "model_revision", "model_branch",
    "profile_sha256", "configuration_sha256", "settings",
}
SETTING_KEYS = {
    "context_tokens", "parallel_slots", "max_output_tokens", "kv_cache",
    "mtp_tokens", "prefill_chunk_size", "decode_share", "reasoning_effort",
    "thinking", "cache_policy",
}
INTEGER_SETTINGS = {"context_tokens", "parallel_slots", "max_output_tokens", "mtp_tokens", "prefill_chunk_size"}
ENUM_SETTINGS = {
    "kv_cache": {"bfloat16", "bf16", "fp8", "int8", "auto", "none", "unknown"},
    "reasoning_effort": {"none", "low", "medium", "high", "xhigh", "unknown"},
    "thinking": {"on", "off", "unknown"},
    "cache_policy": {"default", "prefix", "disabled", "unknown"},
}
HEX64 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_identity(path):
    raw = path.read_bytes()
    identity = json.loads(raw)
    if not isinstance(identity, dict) or set(identity) - IDENTITY_KEYS:
        raise ValueError("identity has unknown or unsupported fields")
    required = {"image_digest", "runtime_version", "runtime_revision",
                "model_repository", "model_revision", "settings"}
    if not required <= set(identity):
        raise ValueError("identity is missing required image/runtime/model/settings fields")
    if not isinstance(identity["image_digest"], str) or not HEX64.fullmatch(identity["image_digest"]):
        raise ValueError("image_digest must be a sha256 digest or local image ID")
    if not isinstance(identity["runtime_revision"], str) or not HEX40.fullmatch(identity["runtime_revision"]):
        raise ValueError("runtime_revision must be a 40-character commit")
    if not isinstance(identity["model_revision"], str) or not HEX40.fullmatch(identity["model_revision"]):
        raise ValueError("model_revision must be a 40-character commit")
    for field in ("runtime_version", "model_repository"):
        if not isinstance(identity[field], str) or not identity[field] or len(identity[field]) > 200:
            raise ValueError(f"{field} must be a short nonempty string")
    for field in ("runtime_repository", "model_branch"):
        if field in identity and (not isinstance(identity[field], str) or len(identity[field]) > 200):
            raise ValueError(f"{field} must be a short string")
    if "runtime_repository" in identity and not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", identity["runtime_repository"]):
        raise ValueError("runtime_repository must be a public GitHub repository URL")
    if not REPOSITORY.fullmatch(identity["model_repository"]):
        raise ValueError("model_repository must use owner/repository form")
    if "model_branch" in identity and not re.fullmatch(r"[A-Za-z0-9._/-]+", identity["model_branch"]):
        raise ValueError("model_branch must be a short public branch name")
    for field in ("profile_sha256", "configuration_sha256"):
        if field in identity and (not isinstance(identity[field], str) or not HEX64.fullmatch(identity[field])):
            raise ValueError(f"{field} must be a SHA-256 hex digest")
    settings = identity["settings"]
    if not isinstance(settings, dict) or set(settings) - SETTING_KEYS:
        raise ValueError("settings may contain only the documented public recipe keys")
    if not {"context_tokens", "parallel_slots", "max_output_tokens"} <= set(settings):
        raise ValueError("settings must identify context, parallel slots, and configured output limit")
    for key, value in settings.items():
        if key in INTEGER_SETTINGS:
            minimum = 0 if key == "mtp_tokens" else 1
            if type(value) is not int or value < minimum:
                qualifier = "nonnegative" if minimum == 0 else "positive"
                raise ValueError(f"settings.{key} must be a {qualifier} integer")
        elif key == "decode_share":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("settings.decode_share must be between 0 and 1")
        elif key in ENUM_SETTINGS:
            if not isinstance(value, str) or value not in ENUM_SETTINGS[key]:
                raise ValueError(f"settings.{key} must be one of the documented public values")
    return identity, sha_bytes(raw)


def sanitize(result, clients, phase, rep=None):
    usage = result.get("usage") or {}
    error = result.get("error")
    known_errors = ("HTTPError", "URLError", "TimeoutError", "ConnectionResetError",
                    "IncompleteRead", "RemoteDisconnected", "JSONDecodeError")
    error_text = str(error) if error is not None else ""
    error_class = next((name for name in known_errors
                        if error_text.startswith(name) or error_text.startswith("<" + name)),
                       None if error is None else "APIorStreamError")
    stats = result.get("runtime_stats") or {}
    cache = {}
    if isinstance(stats, dict):
        for key, value in stats.items():
            key_lower = str(key).lower()
            if "cache" in key_lower and isinstance(value, (int, float, bool, type(None))):
                cache[str(key)[:80]] = value
    text = result.get("output", "")
    reasons = result.get("reasoning", "")
    return {
        "phase": phase,
        "rep": rep,
        "clients": clients,
        "kind": result.get("kind"),
        "fixture": result.get("fixture"),
        "max_tokens": result.get("max_tokens"),
        "request_sha256": result.get("request_sha256"),
        "prompt_sha256": result.get("prompt_sha256"),
        "output_sha256": result.get("output_sha256"),
        "visible_output_nonempty": bool(text.strip()),
        "reasoning_characters": len(reasons),
        "http_status": result.get("http_status"),
        "error_class": error_class,
        "gate": bool(result.get("gate")),
        "finish_reason": result.get("finish"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "seconds": result.get("seconds"),
        "first_output_event_seconds": result.get("ttft"),
        "visible_answer_seconds": result.get("visible_ttft"),
        "first_chunk_characters": result.get("first_chunk_chars"),
        "output_event_count": result.get("output_chunks"),
        "maximum_output_event_gap_seconds": result.get("max_output_gap"),
        "p95_output_event_gap_seconds": result.get("p95_output_gap"),
        "decode_tokens_per_second_proxy": result.get("decode_tps"),
        "cache_observations": cache,
    }


def ineligible_reasons(rows, cap, exact=False, expected_count=4):
    reasons = []
    if len(rows) != expected_count:
        reasons.append(f"expected {expected_count} requests; received {len(rows)}")
    for index, row in enumerate(rows):
        prefix = f"request {index}"
        if row.get("error_class") is not None:
            reasons.append(prefix + " transport/API error")
        if not row.get("gate") or not row.get("visible_output_nonempty"):
            reasons.append(prefix + " missing required visible-output gate")
        if row.get("reasoning_characters") != 0:
            reasons.append(prefix + " emitted reasoning while thinking was disabled")
        if type(row.get("completion_tokens")) is not int or row["completion_tokens"] <= 0:
            reasons.append(prefix + " missing/invalid completion token usage")
        if type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] < 0:
            reasons.append(prefix + " missing/invalid prompt token usage")
        finish = row.get("finish_reason")
        if finish not in ({"stop"} if exact else {"stop", "length"}):
            reasons.append(prefix + " disallowed finish reason")
        if finish == "length" and row.get("completion_tokens") != cap:
            reasons.append(prefix + " length finish did not reach the configured cap")
    return reasons


def save(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def group(base, model, token, clients, kind, numbers, cap, phase, rep=None):
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [pool.submit(request, base, model, kind, number, cap,
                               api_key=token) for number in numbers]
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - start
    rows = [sanitize(result, clients, phase, rep) for result in results]
    failures = ineligible_reasons(rows, cap, exact=(kind == "exact"),
                                  expected_count=1 if kind == "exact" else 4)
    eligible = not failures
    completion_tokens = [row["completion_tokens"] for row in rows
                         if type(row.get("completion_tokens")) is int]
    return {
        "phase": phase,
        "rep": rep,
        "clients": clients,
        "kind": kind,
        "request_count": len(rows),
        "group_elapsed_seconds": elapsed,
        "speed_eligible": eligible,
        "ineligible_reasons": failures,
        "aggregate_output_tokens_per_second": (
            sum(completion_tokens) / elapsed if eligible and elapsed > 0 else None
        ),
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True,
                        help="User-supplied OpenAI-compatible base URL, including /v1")
    parser.add_argument("--model", required=True, help="Served model name accepted by this endpoint")
    parser.add_argument("--label", required=True, help="Short public comparison-arm label")
    parser.add_argument("--identity", type=Path, required=True,
                        help="Path to the constrained image/runtime/model/settings JSON")
    parser.add_argument("--out", type=Path, required=True,
                        help="New local JSON receipt path; response text and endpoint are never written")
    parser.add_argument("--api-key-env", default="SPARK_SERVE_API_KEY",
                        help="Environment variable holding an optional bearer token")
    parser.add_argument("--timing-condition", choices=("unspecified", "staging", "quiet"),
                        default="unspecified")
    parser.add_argument("--deadline-seconds", type=int, default=7200)
    args = parser.parse_args()
    parsed = urlsplit(args.base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        parser.error("--base must be an http(s) URL with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("--base must not embed credentials, query parameters, or fragments; use the bearer-token environment variable")
    if not args.base.rstrip("/").endswith("/v1"):
        parser.error("--base must include the OpenAI-compatible /v1 path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}", args.label):
        parser.error("--label must be a short alphanumeric slug")
    if not args.model or len(args.model) > 160:
        parser.error("--model must be a short nonempty model name")
    if args.deadline_seconds <= 0:
        parser.error("--deadline-seconds must be positive")
    try:
        identity, identity_sha = load_identity(args.identity)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid identity file: {exc}")

    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write("{}\n")
    except FileExistsError:
        parser.error("output path already exists; preserve previous run evidence")

    token = os.environ.get(args.api_key_env) if args.api_key_env else None
    files = ("benchmark_speed.py", "curate_speed.py", "speed_protocol.py", "transport.py")
    data = {
        "schema_version": 1,
        "status": "in_progress",
        "label": args.label,
        "api_model": args.model,
        "identity": identity,
        "identity_sha256": identity_sha,
        "timing_eligible": args.timing_condition == "quiet",
        "timing_condition": args.timing_condition,
        "auth_bearer_used": bool(token),
        "protocol": speed_protocol.metadata(speed_protocol.VERSION),
        "sources_sha256": {name: sha_bytes((ROOT / name).read_bytes()) for name in files},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "warmup_groups": [],
        "measured_groups": [],
    }
    save(output_path, data)

    def deadline(_signum, _frame):
        data["status"] = "in_progress"
        data["deadline_exceeded"] = True
        save(output_path, data)
        raise SystemExit(124)

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        exact = group(args.base, args.model, token, 1, "exact", [99001], 128,
                      phase="warmup-exact")
        data["warmup_groups"].append(exact)
        save(output_path, data)
        if not exact["speed_eligible"]:
            data["status"] = "failed_warmup"
            data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            data["summary"] = curate(data)
            save(output_path, data)
            return 2

        for clients, kind, numbers in speed_protocol.warmup_cells():
            warmup = group(args.base, args.model, token, clients, kind, numbers, 512,
                           phase="warmup-cell")
            data["warmup_groups"].append(warmup)
            save(output_path, data)
            if not warmup["speed_eligible"]:
                data["status"] = "failed_warmup"
                data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                data["summary"] = curate(data)
                save(output_path, data)
                return 2

        for rep in range(4):
            for clients, kind in speed_protocol.measured_cells(rep):
                numbers = [speed_protocol.measured_number(rep, index) for index in range(4)]
                measured = group(args.base, args.model, token, clients, kind,
                                 numbers, 512, phase="measurement", rep=rep)
                data["measured_groups"].append(measured)
                data["summary"] = curate(data)
                save(output_path, data)
        data["status"] = "complete"
        data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["summary"] = curate(data)
        save(output_path, data)
        print(json.dumps(data["summary"], indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        data["status"] = "in_progress"
        data["interrupted"] = True
        data["summary"] = curate(data)
        save(output_path, data)
        return 130
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
