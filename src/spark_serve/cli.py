from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from spark_serve import __version__
from spark_serve import recipes
from spark_serve.config import ConfigError, load_catalog, render_gateway_config
from spark_serve.gateway import (
    GatewayError,
    is_healthy,
    load_model,
    running_models,
    unload_all,
    unload_model,
    wait_healthy,
)
from spark_serve.runtime import active_state, internal_run
from spark_serve.system import CommandError, GIB, probe_node, profile_checks


COMMANDS = {
    "recipes",
    "list",
    "explain",
    "plan",
    "serve",
    "status",
    "endpoint",
    "release",
    "stop",
    "doctor",
    "validate",
    "render-gateway",
    "bootstrap",
    "gateway",
    "internal-run",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spark-serve",
        description="Load and share resource-safe model services on one or more DGX Sparks.",
    )
    parser.add_argument("--config", type=Path, help="cluster configuration (default: /etc/dgx-spark-serve/cluster.json)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    recipes.add_parser(sub)
    sub.add_parser("list", help="list installed model profiles and readiness")
    explain = sub.add_parser("explain", help="explain a model profile and its tradeoffs")
    explain.add_argument("model")
    plan = sub.add_parser("plan", help="perform a read-only admission and conflict check")
    plan.add_argument("model")
    serve = sub.add_parser("serve", help="load or reuse a model and print its stable endpoint")
    serve.add_argument("model")
    sub.add_parser("status", help="show gateway, model, lease, and node state")
    endpoint = sub.add_parser("endpoint", help="print API and OpenCode settings")
    endpoint.add_argument("model", nargs="?")
    release = sub.add_parser("release", help="unload a model and restore the default")
    release.add_argument("model", nargs="?")
    release.add_argument("--no-default", action="store_true", help="leave the resources idle")
    stop = sub.add_parser("stop", help="unload one model or all models")
    stop.add_argument("model", nargs="?")
    stop.add_argument("--all", action="store_true")
    doctor = sub.add_parser("doctor", help="check installation, nodes, gateway, and profiles")
    doctor.add_argument("--offline", action="store_true", help="do not require a running gateway")
    sub.add_parser("validate", help="validate cluster and every installed profile")
    render = sub.add_parser("render-gateway", help="render the derived llama-swap configuration")
    render.add_argument("--output", type=Path)
    render.add_argument(
        "--include-maintenance-profiles",
        action="store_true",
        help="include explicitly maintenance-only benchmark routes",
    )
    sub.add_parser("bootstrap", help=argparse.SUPPRESS)
    sub.add_parser("gateway", help=argparse.SUPPRESS)
    internal = sub.add_parser("internal-run", help=argparse.SUPPRESS)
    internal.add_argument("model")
    internal.add_argument("--port", type=int, required=True)
    return parser


def _emit(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    if isinstance(value, str):
        print(value)
        return
    print(json.dumps(value, indent=2, sort_keys=True))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
    return "\n".join(lines)


def _endpoint_payload(catalog: Any, profile: dict[str, Any]) -> dict[str, Any]:
    gateway = catalog.cluster["gateway"]
    local_base_url = gateway["base_url"].rstrip("/") + "/v1"
    base_url = gateway.get("client_base_url", gateway["base_url"]).rstrip("/") + "/v1"
    host = gateway.get("ssh_tunnel_host", "spark-head")
    port = int(gateway.get("ssh_tunnel_port", 9292))
    model_id = profile["id"]
    provider_options = {"baseURL": base_url}
    if gateway.get("api_key_file"):
        provider_options["apiKey"] = "{env:SPARK_SERVE_API_KEY}"
    model_config = {
        "name": profile["name"],
        "limit": {"context": profile["model"]["context_tokens"]},
    }
    if profile["serving"].get("client_request_options"):
        model_config["options"] = profile["serving"]["client_request_options"]
    opencode = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "dgx-spark": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DGX Spark",
                "options": provider_options,
                "models": {model_id: model_config},
            }
        },
        "model": f"dgx-spark/{model_id}",
    }
    return {
        "model": model_id,
        "base_url": base_url,
        "local_base_url": local_base_url,
        "health_url": gateway.get("client_base_url", gateway["base_url"]).rstrip("/") + "/health",
        "ssh_tunnel": f"ssh -N -L {port}:127.0.0.1:{port} {host}",
        "curl": f"curl {base_url}/models",
        "opencode": opencode,
    }


def _format_endpoint(payload: dict[str, Any]) -> str:
    local_line = ""
    if payload["local_base_url"] != payload["base_url"]:
        local_line = f"Local-only base URL: {payload['local_base_url']}\n"
    return (
        f"Model: {payload['model']}\n"
        f"OpenAI-compatible base URL: {payload['base_url']}\n"
        + local_line
        + f"Remote tunnel (run once on the client): {payload['ssh_tunnel']}\n"
        + f"Verification: {payload['curl']}\n\n"
        + "OpenCode configuration:\n"
        + json.dumps(payload["opencode"], indent=2)
    )


def _list(catalog: Any, as_json: bool) -> int:
    running = running_models(catalog) if is_healthy(catalog) else []
    values = []
    for profile in catalog.profiles.values():
        measured = profile["operator"].get("measured_output_tps", {})
        values.append(
            {
                "id": profile["id"],
                "aliases": profile["aliases"],
                "topology": profile["topology"]["mode"],
                "quantization": profile["model"]["quantization"],
                "context_tokens": profile["model"]["context_tokens"],
                "readiness": profile["operator"]["readiness"],
                "running": profile["id"] in running,
                "measured_output_tps": measured,
            }
        )
    if as_json:
        _emit(values, True)
    else:
        rows = [
            [
                "*" if item["running"] else "",
                item["id"],
                item["topology"],
                item["quantization"],
                f"{item['context_tokens']:,}",
                item["readiness"],
            ]
            for item in values
        ]
        print(_table(["", "MODEL", "TOPOLOGY", "QUANT", "CONTEXT", "READINESS"], rows))
        print("\n* currently loaded; aliases are accepted by every command")
    return 0


def _explain(catalog: Any, model: str, as_json: bool) -> int:
    profile = catalog.resolve(model)
    if as_json:
        _emit(profile, True)
        return 0
    measured = profile["operator"].get("measured_output_tps", {})
    lines = [
        f"{profile['name']} ({profile['id']})",
        profile["description"],
        "",
        f"Model: {profile['model']['repository']} @ {profile['model']['revision']}",
        f"Runtime: {profile['runtime']['engine']} / {profile['runtime']['image']}",
        f"Format: {profile['model']['quantization']}; context: {profile['model']['context_tokens']:,} tokens",
        f"Topology: {profile['topology']['mode']} on {', '.join(profile['topology']['nodes'])}",
        f"Resource claims: {', '.join(profile['resources']['claims'])}",
        f"Readiness: {profile['operator']['readiness']}",
        f"Best for: {', '.join(profile['operator']['use_cases'])}",
    ]
    if measured:
        lines.append("Measured output throughput: " + ", ".join(f"{key}={value:g} tok/s" for key, value in measured.items()))
    if profile["operator"]["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in profile["operator"]["warnings"])
    print("\n".join(lines))
    return 0


def _plan(catalog: Any, model: str, as_json: bool, *, emit: bool = True) -> tuple[int, dict[str, Any]]:
    profile = catalog.resolve(model)
    running = running_models(catalog) if is_healthy(catalog) else []
    target_claims = set(profile["resources"]["claims"])
    conflicts = [
        current
        for current in running
        if current in catalog.profiles
        and target_claims & set(catalog.profiles[current]["resources"]["claims"])
        and current != profile["id"]
    ]
    checks = profile_checks(catalog, profile, strict_resources=False)
    failures = [check for check in checks if not check["pass"]]
    result = {
        "model": profile["id"],
        "already_running": profile["id"] in running,
        "running": running,
        "will_unload": conflicts,
        "checks": checks,
        "admitted": not failures,
        "note": "Memory and swap are rechecked after conflicting models unload; observed values are advisory here.",
    }
    if as_json and emit:
        _emit(result, True)
    elif emit:
        print(f"Model: {profile['id']}")
        print(f"Action: {'reuse existing backend' if result['already_running'] else 'load on demand'}")
        print(f"Conflicting models to unload: {', '.join(conflicts) if conflicts else 'none'}")
        for check in checks:
            marker = "PASS" if check["pass"] else "FAIL"
            observed = ""
            if check.get("observed_pass") is False and check["pass"]:
                observed = " (will recheck after eviction)"
            print(f"  {marker:4} {check['node']:6} {check['kind']:8} {check['detail']}{observed}")
        print(result["note"])
    return (0 if result["admitted"] else 1), result


def _serve(catalog: Any, model: str, as_json: bool) -> int:
    profile = catalog.resolve(model)
    wait_healthy(catalog, timeout=30)
    previous = running_models(catalog)
    if profile["id"] in previous:
        result = {"action": "reused", **_endpoint_payload(catalog, profile), "running": previous}
        _emit(result, True) if as_json else print("Already loaded; no duplicate process was started.\n" + _format_endpoint(result))
        return 0
    plan_code, plan = _plan(catalog, profile["id"], as_json, emit=not as_json)
    if plan_code:
        raise CommandError("model failed static admission; no service was changed")
    timeout = int(profile["runtime"]["launch"]["timeout_seconds"]) + 30
    try:
        load_model(catalog, profile["id"], timeout)
    except GatewayError as original:
        restored: list[str] = []
        rollback_errors: list[str] = []
        for previous_id in previous:
            if previous_id not in catalog.profiles:
                continue
            try:
                previous_profile = catalog.profiles[previous_id]
                load_model(
                    catalog,
                    previous_id,
                    int(previous_profile["runtime"]["launch"]["timeout_seconds"]) + 30,
                )
                restored.append(previous_id)
            except GatewayError as rollback_error:
                rollback_errors.append(str(rollback_error))
        suffix = f"; restored: {', '.join(restored) or 'none'}"
        if rollback_errors:
            suffix += f"; rollback errors: {' | '.join(rollback_errors)}"
        raise GatewayError(str(original) + suffix) from original
    result = {"action": "loaded", **_endpoint_payload(catalog, profile), "replaced": plan["will_unload"]}
    _emit(result, True) if as_json else print("Model is ready.\n" + _format_endpoint(result))
    return 0


def _status(catalog: Any, as_json: bool) -> int:
    gateway = {"healthy": is_healthy(catalog), "base_url": catalog.cluster["gateway"]["base_url"]}
    running = running_models(catalog) if gateway["healthy"] else []
    nodes: dict[str, Any] = {}
    for node_id, node in catalog.cluster["nodes"].items():
        try:
            nodes[node_id] = probe_node(node)
        except Exception as exc:  # status must report every node, including failures
            nodes[node_id] = {"error": str(exc)}
    result = {"gateway": gateway, "running": running, "leases": active_state(catalog), "nodes": nodes}
    if as_json:
        _emit(result, True)
    else:
        print(f"Gateway: {'healthy' if gateway['healthy'] else 'unavailable'} at {gateway['base_url']}")
        print(f"Running: {', '.join(running) if running else 'none'}")
        for node_id, telemetry in nodes.items():
            if "error" in telemetry:
                print(f"  {node_id}: ERROR {telemetry['error']}")
            else:
                print(
                    f"  {node_id}: memory={telemetry['mem_available_bytes'] / GIB:.1f} GiB available "
                    f"swap={telemetry['swap_used_bytes'] / GIB:.2f} GiB used "
                    f"disk={telemetry['disk_free_bytes'] / GIB:.1f} GiB free "
                    f"gpu={telemetry['gpu'].get('name', 'unavailable')}"
                )
    return 0 if gateway["healthy"] and all("error" not in node for node in nodes.values()) else 1


def _doctor(catalog: Any, offline: bool, as_json: bool) -> int:
    checks: list[dict[str, Any]] = []
    binary = Path(catalog.cluster["gateway"]["binary"])
    checks.append({"check": "llama-swap binary", "pass": binary.is_file() and os.access(binary, os.X_OK), "detail": str(binary)})
    for node_id, node in catalog.cluster["nodes"].items():
        try:
            telemetry = probe_node(node)
            checks.append({"check": f"node {node_id}", "pass": bool(telemetry["gpu"].get("available")), "detail": telemetry})
        except Exception as exc:
            checks.append({"check": f"node {node_id}", "pass": False, "detail": str(exc)})
    gateway_ok = is_healthy(catalog)
    checks.append({"check": "gateway", "pass": gateway_ok or offline, "detail": "healthy" if gateway_ok else "unavailable (allowed offline)" if offline else "unavailable"})
    for profile in catalog.profiles.values():
        artifact_checks = [item for item in profile_checks(catalog, profile, strict_resources=False) if item["kind"] == "artifact"]
        checks.append({"check": f"profile {profile['id']}", "pass": all(item["pass"] for item in artifact_checks), "detail": artifact_checks})
    result = {"pass": all(item["pass"] for item in checks), "checks": checks}
    if as_json:
        _emit(result, True)
    else:
        for check in checks:
            print(f"{'PASS' if check['pass'] else 'FAIL'}  {check['check']}: {check['detail']}")
    return 0 if result["pass"] else 1


def _render(
    catalog: Any,
    output: Path | None,
    as_json: bool,
    *,
    include_maintenance: bool = False,
) -> int:
    rendered = render_gateway_config(
        catalog,
        include_maintenance=include_maintenance,
    )
    text = json.dumps(rendered, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o640)
        temporary.replace(output)
        _emit({"written": str(output), "models": len(rendered["models"])}, as_json)
    else:
        print(text, end="")
    return 0


def _bootstrap(catalog: Any, as_json: bool) -> int:
    wait_healthy(catalog, timeout=60)
    preload = catalog.cluster.get("preload_models", [])
    if preload:
        # Use the existing broker readiness endpoint, not GET / (which returns
        # 404 for vLLM). Disjoint models can load in parallel; resource validation
        # has already rejected conflicting preload sets. No second lifecycle owner.
        failures: dict[str, str] = {}
        ready: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(preload), 8)) as pool:
            futures = {
                pool.submit(load_model, catalog, model_id,
                    int(catalog.profiles[model_id]["runtime"]["launch"]["timeout_seconds"]) + 30): model_id
                for model_id in preload
            }
            for future in as_completed(futures):
                model_id = futures[future]
                try:
                    future.result()
                    ready.append(model_id)
                except Exception as exc:
                    failures[model_id] = str(exc)
        _emit({"action": "preloaded" if not failures else "degraded", "ready": sorted(ready), "failures": failures}, as_json)
        if failures:
            raise GatewayError("fleet startup did not make every configured model ready")
        return 0
    running = running_models(catalog)
    if running:
        _emit({"action": "kept", "running": running}, as_json)
        return 0
    return _serve(catalog, catalog.default_profile["id"], as_json)


def _normalize_argv(argv: list[str]) -> list[str]:
    # Keep the convenient ``spark-serve qwen3.8`` form deliberately simple.
    # Global options may still be used with the explicit ``serve`` command;
    # trying to infer which tokens are option values makes a config path look
    # like a model name.
    if argv and not argv[0].startswith("-") and argv[0] not in COMMANDS:
        return ["serve", *argv]
    return argv


def main(argv: list[str] | None = None) -> None:
    raw = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = _parser()
    args = parser.parse_args(raw)
    try:
        if args.command == "recipes":
            raise SystemExit(recipes.execute(args))
        catalog = load_catalog(args.config)
        if args.command == "list":
            code = _list(catalog, args.json)
        elif args.command == "explain":
            code = _explain(catalog, args.model, args.json)
        elif args.command == "plan":
            code, _ = _plan(catalog, args.model, args.json)
        elif args.command == "serve":
            code = _serve(catalog, args.model, args.json)
        elif args.command == "status":
            code = _status(catalog, args.json)
        elif args.command == "endpoint":
            profile = catalog.resolve(args.model) if args.model else catalog.default_profile
            payload = _endpoint_payload(catalog, profile)
            _emit(payload, True) if args.json else print(_format_endpoint(payload))
            code = 0
        elif args.command == "release":
            running = running_models(catalog)
            target = catalog.resolve(args.model)["id"] if args.model else (running[0] if len(running) == 1 else None)
            if target:
                unload_model(catalog, target)
            if args.no_default:
                _emit({"released": target, "running": running_models(catalog)}, args.json)
                code = 0
            else:
                code = _serve(catalog, catalog.default_profile["id"], args.json)
        elif args.command == "stop":
            if args.all or not args.model:
                unload_all(catalog)
                _emit({"stopped": "all"}, args.json)
            else:
                target = catalog.resolve(args.model)["id"]
                unload_model(catalog, target)
                _emit({"stopped": target}, args.json)
            code = 0
        elif args.command == "doctor":
            code = _doctor(catalog, args.offline, args.json)
        elif args.command == "validate":
            _emit({"valid": True, "cluster": catalog.cluster["cluster_id"], "profiles": sorted(catalog.profiles)}, args.json)
            code = 0
        elif args.command == "render-gateway":
            code = _render(
                catalog,
                args.output,
                args.json,
                include_maintenance=args.include_maintenance_profiles,
            )
        elif args.command == "bootstrap":
            code = _bootstrap(catalog, args.json)
        elif args.command == "gateway":
            binary = catalog.cluster["gateway"]["binary"]
            os.execv(
                binary,
                [
                    binary,
                    "--config",
                    catalog.cluster["gateway"]["config_path"],
                    "--listen",
                    catalog.cluster["gateway"]["listen"],
                ],
            )
            return
        elif args.command == "internal-run":
            profile = catalog.resolve(args.model)
            code = internal_run(catalog, profile, args.port)
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (ConfigError, CommandError, GatewayError, subprocess.TimeoutExpired) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "type": type(exc).__name__}, indent=2), file=sys.stderr)
        else:
            print(f"spark-serve: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
