from __future__ import annotations

import itertools
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG = Path("/etc/dgx-spark-serve/cluster.json")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


class ConfigError(ValueError):
    """Raised when cluster or profile configuration is unsafe or incomplete."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"top-level JSON value must be an object: {path}")
    return value


def _require(obj: dict[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ConfigError(f"{where}: missing required keys: {', '.join(missing)}")


def _only(obj: dict[str, Any], keys: Iterable[str], where: str) -> None:
    extra = sorted(set(obj) - set(keys))
    if extra:
        raise ConfigError(f"{where}: unknown keys: {', '.join(extra)}")


@dataclass(frozen=True)
class Catalog:
    config_path: Path
    cluster: dict[str, Any]
    profiles: dict[str, dict[str, Any]]
    aliases: dict[str, str]

    def resolve(self, requested: str) -> dict[str, Any]:
        model_id = self.aliases.get(requested, requested)
        try:
            return self.profiles[model_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self.profiles))
            raise ConfigError(f"unknown model {requested!r}; available: {choices}") from exc

    @property
    def default_profile(self) -> dict[str, Any]:
        return self.resolve(str(self.cluster["default_model"]))


def _validate_command(command: dict[str, Any], nodes: set[str], where: str) -> None:
    _require(command, ("node", "mode", "argv", "timeout_seconds"), where)
    _only(command, ("node", "mode", "argv", "environment", "timeout_seconds"), where)
    if command["node"] not in nodes:
        raise ConfigError(f"{where}.node references unknown node {command['node']!r}")
    if command["mode"] not in {"foreground", "detached"}:
        raise ConfigError(f"{where}.mode must be foreground or detached")
    if not isinstance(command["argv"], list) or not command["argv"] or not all(
        isinstance(value, str) and value for value in command["argv"]
    ):
        raise ConfigError(f"{where}.argv must be a non-empty string array")
    if not isinstance(command["timeout_seconds"], int) or command["timeout_seconds"] < 1:
        raise ConfigError(f"{where}.timeout_seconds must be a positive integer")
    environment = command.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        raise ConfigError(f"{where}.environment must map strings to strings")


def validate_cluster(cluster: dict[str, Any], path: Path) -> None:
    _require(
        cluster,
        ("schema_version", "cluster_id", "service_user", "gateway", "nodes", "profiles_dir", "default_model"),
        str(path),
    )
    _only(
        cluster,
        (
            "schema_version",
            "cluster_id",
            "service_user",
            "cli_path",
            "gateway",
            "nodes",
            "profiles_dir",
            "default_model",
            "preload_models",
            "worker_pool",
            "model_routes",
            "runtime_dir",
            "state_dir",
        ),
        str(path),
    )
    preload = cluster.get("preload_models", [])
    if not isinstance(preload, list) or not all(isinstance(x, str) and ID_PATTERN.fullmatch(x) for x in preload) or len(set(preload)) != len(preload):
        raise ConfigError(f"{path}: preload_models must contain unique model IDs")
    pool = cluster.get("worker_pool")
    routes = cluster.get("model_routes", [])
    if not isinstance(routes, list):
        raise ConfigError("model_routes must be an array")
    if pool is not None and routes:
        raise ConfigError("replace worker_pool with model_routes; do not configure both")
    route_ids = set()
    for route in routes:
        if not isinstance(route, dict):
            raise ConfigError("model_routes entries must be objects")
        _require(route, ("id", "targets", "max_output_tokens", "client_compat"), "model_routes")
        _only(route, ("id", "targets", "max_output_tokens", "client_compat", "thinking_levels", "name", "spillover", "ready_only", "maintenance_profiles"), "model_routes")
        rid = route["id"]
        if not isinstance(rid, str) or not ID_PATTERN.fullmatch(rid) or rid in route_ids:
            raise ConfigError("model_routes IDs must be valid and unique")
        route_ids.add(rid)
        for key in ("ready_only", "maintenance_profiles"):
            if key in route and type(route[key]) is not bool:
                raise ConfigError(f"model_routes {key} must be a boolean")
        if route.get("maintenance_profiles") and not route.get("ready_only"):
            raise ConfigError("model_routes maintenance_profiles requires ready_only")
        targets = route["targets"]
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets) or len(set(targets)) != len(targets):
            raise ConfigError("model_routes targets must be unique model IDs")
        for key, value in (("spillover", route.get("spillover", 1)), ("max_output_tokens", route["max_output_tokens"])):
            if type(value) is not int or value < 1:
                raise ConfigError(f"model_routes {key} must be a positive integer")
        if not isinstance(route["client_compat"], dict):
            raise ConfigError("model_routes client_compat must be an object")
        levels = route.get("thinking_levels", {})
        if not isinstance(levels, dict) or any(k not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"} or (v is not None and not isinstance(v, str)) for k, v in levels.items()):
            raise ConfigError("model_routes thinking_levels must map native levels to wire values or null")
        if "name" in route and (not isinstance(route["name"], str) or not route["name"]):
            raise ConfigError("model_routes name must be a nonempty string")
    if pool is not None:
        if not isinstance(pool, dict):
            raise ConfigError(f"{path}: worker_pool must be an object")
        _require(pool, ("id", "targets", "spillover"), "worker_pool")
        _only(pool, ("id", "targets", "spillover"), "worker_pool")
        if not isinstance(pool["id"], str) or not ID_PATTERN.fullmatch(pool["id"]):
            raise ConfigError("worker_pool.id must be a model ID")
        targets = pool["targets"]
        if not isinstance(targets, list) or not targets or not all(isinstance(x, str) for x in targets) or len(set(targets)) != len(targets):
            raise ConfigError("worker_pool.targets must contain unique model IDs")
        if type(pool["spillover"]) is not int or pool["spillover"] < 1:
            raise ConfigError("worker_pool.spillover must be a positive integer")
    if cluster["schema_version"] != 1:
        raise ConfigError(f"{path}: unsupported schema_version {cluster['schema_version']!r}")
    if not isinstance(cluster["nodes"], dict) or not cluster["nodes"]:
        raise ConfigError(f"{path}: nodes must be a non-empty object")
    for node_id, node in cluster["nodes"].items():
        if not ID_PATTERN.match(node_id):
            raise ConfigError(f"{path}: invalid node id {node_id!r}")
        if not isinstance(node, dict):
            raise ConfigError(f"{path}: node {node_id} must be an object")
        _require(node, ("host", "user", "model_root", "swap_devices"), f"node {node_id}")
        _only(
            node,
            ("host", "user", "model_root", "swap_devices", "fabric_ip", "tailscale_name"),
            f"node {node_id}",
        )
        if not isinstance(node["swap_devices"], list) or not all(
            isinstance(value, str) and value.startswith("/") for value in node["swap_devices"]
        ):
            raise ConfigError(f"node {node_id}.swap_devices must contain absolute paths")
    gateway = cluster["gateway"]
    if not isinstance(gateway, dict):
        raise ConfigError(f"{path}: gateway must be an object")
    _require(gateway, ("listen", "base_url", "binary", "config_path"), "gateway")
    _only(
        gateway,
        (
            "listen",
            "base_url",
            "client_base_url",
            "binary",
            "config_path",
            "api_key_file",
            "ssh_tunnel_host",
            "ssh_tunnel_port",
        ),
        "gateway",
    )
    for key in ("profiles_dir", "cli_path", "runtime_dir", "state_dir"):
        if key in cluster and not str(cluster[key]).startswith("/"):
            raise ConfigError(f"{path}: {key} must be an absolute path")


def validate_profile(profile: dict[str, Any], nodes: set[str], source: Path) -> None:
    required = (
        "schema_version",
        "id",
        "aliases",
        "name",
        "description",
        "model",
        "runtime",
        "topology",
        "resources",
        "artifacts",
        "serving",
        "operator",
    )
    _require(profile, required, str(source))
    _only(profile, ("$schema", *required), str(source))
    if profile["schema_version"] != 1:
        raise ConfigError(f"{source}: unsupported schema_version {profile['schema_version']!r}")
    model_id = profile["id"]
    if not isinstance(model_id, str) or not ID_PATTERN.match(model_id):
        raise ConfigError(f"{source}: invalid id {model_id!r}")
    aliases = profile["aliases"]
    if not isinstance(aliases, list) or not all(isinstance(value, str) and value for value in aliases):
        raise ConfigError(f"{source}: aliases must be a string array")
    if len(set(aliases)) != len(aliases):
        raise ConfigError(f"{source}: aliases must be unique")

    model = profile["model"]
    _require(model, ("repository", "revision", "served_name", "quantization", "context_tokens"), f"{model_id}.model")
    _only(
        model,
        ("repository", "revision", "served_name", "quantization", "context_tokens", "input_modalities"),
        f"{model_id}.model",
    )
    if not isinstance(model["context_tokens"], int) or model["context_tokens"] < 1:
        raise ConfigError(f"{model_id}.model.context_tokens must be positive")
    modalities = model.get("input_modalities", ["text"])
    if (
        not isinstance(modalities, list)
        or not modalities
        or len(set(modalities)) != len(modalities)
        or any(value not in {"text", "image", "audio"} for value in modalities)
        or "text" not in modalities
    ):
        raise ConfigError(
            f"{model_id}.model.input_modalities must be unique llama-swap modalities including text"
        )

    runtime = profile["runtime"]
    _require(runtime, ("engine", "image", "launch", "backend"), f"{model_id}.runtime")
    _only(runtime, ("engine", "image", "launch", "stop", "backend"), f"{model_id}.runtime")
    _validate_command(runtime["launch"], nodes, f"{model_id}.runtime.launch")
    if "stop" in runtime:
        _validate_command(runtime["stop"], nodes, f"{model_id}.runtime.stop")
    backend = runtime["backend"]
    _require(backend, ("node", "port", "health_path"), f"{model_id}.runtime.backend")
    _only(backend, ("node", "port", "health_path", "models_path"), f"{model_id}.runtime.backend")
    if backend["node"] not in nodes:
        raise ConfigError(f"{model_id}.runtime.backend.node references unknown node")
    if not isinstance(backend["port"], int) or not 1 <= backend["port"] <= 65535:
        raise ConfigError(f"{model_id}.runtime.backend.port must be 1..65535")

    topology = profile["topology"]
    _require(topology, ("mode", "nodes", "fabric_required"), f"{model_id}.topology")
    _only(topology, ("mode", "nodes", "fabric_required"), f"{model_id}.topology")
    if not isinstance(topology["nodes"], list) or not topology["nodes"]:
        raise ConfigError(f"{model_id}.topology.nodes must be non-empty")
    unknown_nodes = set(topology["nodes"]) - nodes
    if unknown_nodes:
        raise ConfigError(f"{model_id}.topology.nodes references unknown nodes: {sorted(unknown_nodes)}")

    resources = profile["resources"]
    _require(resources, ("claims", "nodes"), f"{model_id}.resources")
    _only(resources, ("claims", "nodes"), f"{model_id}.resources")
    claims = resources["claims"]
    if not isinstance(claims, list) or not claims or len(set(claims)) != len(claims):
        raise ConfigError(f"{model_id}.resources.claims must be a non-empty unique array")
    gpu_nodes = {claim[4:] for claim in claims if claim.startswith("gpu:")}
    if gpu_nodes != set(topology["nodes"]):
        raise ConfigError(f"{model_id}: GPU claims must exactly match topology.nodes")
    if backend["node"] not in gpu_nodes:
        raise ConfigError(f"{model_id}: backend must run on a claimed GPU node")
    if len(gpu_nodes) == 1 and runtime["launch"]["node"] not in gpu_nodes:
        raise ConfigError(f"{model_id}: single GPU launch must run on its claimed node")
    if set(resources["nodes"]) != set(topology["nodes"]):
        raise ConfigError(f"{model_id}.resources.nodes must exactly match topology.nodes")
    for node_id, requirement in resources["nodes"].items():
        _require(
            requirement,
            ("min_memory_available_gib", "min_disk_free_gib", "swap_policy"),
            f"{model_id}.resources.nodes.{node_id}",
        )
        _only(
            requirement,
            ("min_memory_available_gib", "min_disk_free_gib", "swap_policy"),
            f"{model_id}.resources.nodes.{node_id}",
        )
        if requirement["swap_policy"] not in {"unchanged", "off_during_run"}:
            raise ConfigError(f"{model_id}: invalid swap_policy on {node_id}")
        for key in ("min_memory_available_gib", "min_disk_free_gib"):
            if not isinstance(requirement[key], (int, float)) or requirement[key] < 0:
                raise ConfigError(f"{model_id}: {key} on {node_id} must be non-negative")

    artifacts = profile["artifacts"]
    if not isinstance(artifacts, list):
        raise ConfigError(f"{model_id}.artifacts must be an array")
    for index, artifact in enumerate(artifacts):
        where = f"{model_id}.artifacts[{index}]"
        _require(artifact, ("node", "kind", "value"), where)
        _only(artifact, ("node", "kind", "value", "revision", "min_size_bytes"), where)
        if artifact["node"] not in nodes:
            raise ConfigError(f"{where}.node references unknown node")
        if artifact["kind"] not in {"file", "directory", "docker_image", "git_checkout"}:
            raise ConfigError(f"{where}.kind is unsupported")
        if artifact["kind"] == "git_checkout" and not artifact.get("revision"):
            raise ConfigError(f"{where}.revision is required for git_checkout")

    serving = profile["serving"]
    _require(serving, ("concurrency_limit", "ttl_seconds", "tools"), f"{model_id}.serving")
    _only(
        serving,
        (
            "concurrency_limit",
            "ttl_seconds",
            "tools",
            "reasoning",
            "client_request_options",
            "request_filters",
            "eviction_cost",
            "scheduler",
        ),
        f"{model_id}.serving",
    )
    if not isinstance(serving["concurrency_limit"], int) or serving["concurrency_limit"] < 1:
        raise ConfigError(f"{model_id}.serving.concurrency_limit must be positive")
    if not isinstance(serving.get("client_request_options", {}), dict):
        raise ConfigError(f"{model_id}.serving.client_request_options must be an object")
    filters = serving.get("request_filters", {})
    if not isinstance(filters, dict):
        raise ConfigError(f"{model_id}.serving.request_filters must be an object")
    _only(filters, ("setParams", "setParamsByID", "stripParams"), "request_filters")
    for key in ("setParams", "setParamsByID"):
        if key in filters and not isinstance(filters[key], dict):
            raise ConfigError(f"request_filters.{key} must be an object")
    if "stripParams" in filters and (not isinstance(filters["stripParams"], list) or not all(isinstance(x, str) for x in filters["stripParams"])):
        raise ConfigError("request_filters.stripParams must be a string array")
    scheduler = serving.get("scheduler")
    if scheduler is not None:
        if not isinstance(scheduler, dict):
            raise ConfigError(f"{model_id}.serving.scheduler must be an object")
        _require(
            scheduler,
            (
                "engine",
                "policy",
                "priority_order",
                "max_running_requests",
                "max_num_batched_tokens",
            ),
            f"{model_id}.serving.scheduler",
        )
        _only(
            scheduler,
            (
                "engine",
                "policy",
                "priority_order",
                "max_running_requests",
                "max_num_batched_tokens",
            ),
            f"{model_id}.serving.scheduler",
        )
        if scheduler["engine"] != "vllm":
            raise ConfigError(f"{model_id}.serving.scheduler.engine must be vllm")
        if scheduler["policy"] not in {"fcfs", "priority"}:
            raise ConfigError(f"{model_id}.serving.scheduler.policy must be fcfs or priority")
        if scheduler["priority_order"] != "lower-first":
            raise ConfigError(f"{model_id}.serving.scheduler.priority_order must be lower-first")
        for key in ("max_running_requests", "max_num_batched_tokens"):
            if not isinstance(scheduler[key], int) or isinstance(scheduler[key], bool) or scheduler[key] < 1:
                raise ConfigError(f"{model_id}.serving.scheduler.{key} must be positive")
        if scheduler["max_running_requests"] > serving["concurrency_limit"]:
            raise ConfigError(
                f"{model_id}.serving.scheduler.max_running_requests cannot exceed concurrency_limit"
            )
        reserved = {
            "MAX_MODEL_LEN",
            "MAX_NUM_SEQS",
            "MAX_NUM_BATCHED_TOKENS",
            "EXTRA_ARGS",
        }
        declared = set(runtime["launch"].get("environment", {}))
        overlap = sorted(reserved & declared)
        if overlap:
            raise ConfigError(
                f"{model_id}.runtime.launch.environment duplicates scheduler authority: {', '.join(overlap)}"
            )
        if runtime["launch"]["argv"] == [
            "/usr/local/libexec/dgx-mia-flash",
            "launch",
            "exl3",
        ] and (
            scheduler["policy"] != "priority"
            or scheduler["max_running_requests"] not in {4, 8, 16, 32}
            or scheduler["max_num_batched_tokens"] not in {2048, 4096, 8192}
        ):
            raise ConfigError(
                f"{model_id}.serving.scheduler is unsupported by the pinned EXL3 launcher"
            )

    operator = profile["operator"]
    _require(operator, ("readiness", "use_cases", "warnings"), f"{model_id}.operator")
    _only(
        operator,
        ("readiness", "exposure", "use_cases", "warnings", "measured_output_tps"),
        f"{model_id}.operator",
    )
    if operator.get("exposure", "production") not in {
        "production",
        "maintenance-only",
    }:
        raise ConfigError(f"{model_id}.operator.exposure is invalid")
    if operator["readiness"] == "disabled":
        raise ConfigError(f"{model_id}: disabled profiles must not be installed in the active catalog")


def load_catalog(config_path: Path | str | None = None) -> Catalog:
    path = Path(config_path or os.environ.get("SPARK_SERVE_CONFIG", DEFAULT_CONFIG)).resolve()
    cluster = _load_json(path)
    validate_cluster(cluster, path)
    profiles_dir = Path(cluster["profiles_dir"])
    if not profiles_dir.is_absolute():
        profiles_dir = (path.parent / profiles_dir).resolve()
    profiles: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for source in sorted(profiles_dir.glob("*.json")):
        profile = _load_json(source)
        validate_profile(profile, set(cluster["nodes"]), source)
        model_id = profile["id"]
        if model_id in profiles:
            raise ConfigError(f"duplicate model id {model_id!r}")
        profiles[model_id] = profile
    if not profiles:
        raise ConfigError(f"no model profiles found in {profiles_dir}")
    for model_id, profile in profiles.items():
        for alias in profile["aliases"]:
            if alias in profiles and alias != model_id:
                raise ConfigError(f"alias {alias!r} collides with model id")
            prior = aliases.setdefault(alias, model_id)
            if prior != model_id:
                raise ConfigError(f"alias {alias!r} is defined by both {prior!r} and {model_id!r}")
    catalog = Catalog(path, cluster, profiles, aliases)
    catalog.default_profile
    targets = list(cluster.get("preload_models", []))
    pool = cluster.get("worker_pool")
    if pool:
        if pool["id"] in profiles or pool["id"] in aliases:
            raise ConfigError("worker_pool.id collides with a model or alias")
        targets += pool["targets"]
    for route in cluster.get("model_routes", []):
        if route["id"] in profiles or route["id"] in aliases:
            raise ConfigError("model_routes ID collides with a model or alias")
        if route.get("maintenance_profiles"):
            reserved = set(profiles) | set(aliases) | {r["id"] for r in cluster["model_routes"]}
            for i in range(len(route["targets"])):
                if f"{route['id']}--only--{i + 1}" in reserved:
                    raise ConfigError("generated maintenance selector collides with a model, alias or route")
        targets += route["targets"]
    for target in targets:
        if target not in profiles:
            raise ConfigError(f"fleet target {target!r} must be a concrete model ID")
        if profiles[target]["operator"].get("exposure", "production") != "production":
            raise ConfigError(f"fleet target {target!r} is not exposed in production")
    for route in cluster.get("model_routes", []):
        claims: set[str] = set()
        for target in route["targets"]:
            current = set(profiles[target]["resources"]["claims"])
            if claims & current:
                raise ConfigError("model_routes targets must coexist without resource conflicts")
            claims.update(current)
        if route["max_output_tokens"] >= min(profiles[t]["model"]["context_tokens"] for t in route["targets"]):
            raise ConfigError("model_routes output reserve must be smaller than every target context")
    claimed: set[str] = set()
    for model_id in cluster.get("preload_models", []):
        claims = set(profiles[model_id]["resources"]["claims"])
        if claimed & claims:
            raise ConfigError("preload_models have conflicting resources")
        claimed.update(claims)
    return catalog


def launch_environment(profile: dict[str, Any]) -> dict[str, str]:
    """Return the exact launcher environment derived from one profile authority."""

    environment = dict(profile["runtime"]["launch"].get("environment", {}))
    scheduler = profile["serving"].get("scheduler")
    if scheduler is None:
        return environment
    environment["MAX_NUM_SEQS"] = str(scheduler["max_running_requests"])
    environment["MAX_NUM_BATCHED_TOKENS"] = str(scheduler["max_num_batched_tokens"])
    environment["MAX_MODEL_LEN"] = str(profile["model"]["context_tokens"])
    environment["EXTRA_ARGS"] = f"--scheduling-policy {scheduler['policy']}"
    return environment


def _read_api_keys(path_value: str | None) -> list[str]:
    if not path_value:
        return []
    path = Path(path_value)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"gateway api_key_file not found: {path}") from exc
    keys = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if len(set(keys)) != len(keys):
        raise ConfigError(f"gateway api_key_file contains duplicate keys: {path}")
    return keys


def compatible_sets(profiles: dict[str, dict[str, Any]]) -> list[tuple[str, ...]]:
    """Return every maximal resource-compatible profile set.

    The catalog is intentionally small on a Spark pair, so exhaustive subsets
    make the generated policy obvious and keep scheduling decisions in the
    profile claims rather than in hand-maintained routing rules.
    """

    ids = sorted(profiles)
    compatible: list[tuple[str, ...]] = []
    for size in range(1, len(ids) + 1):
        for subset in itertools.combinations(ids, size):
            claimed: set[str] = set()
            allowed = True
            for model_id in subset:
                model_claims = set(profiles[model_id]["resources"]["claims"])
                if claimed & model_claims:
                    allowed = False
                    break
                claimed.update(model_claims)
            if allowed:
                compatible.append(subset)
    maximal = [
        subset
        for subset in compatible
        if not any(set(subset) < set(other) for other in compatible)
    ]
    return maximal or [(model_id,) for model_id in ids]


def render_gateway_config(
    catalog: Catalog,
    *,
    include_maintenance: bool = False,
) -> dict[str, Any]:
    cluster = catalog.cluster
    models: dict[str, Any] = {}
    maximum_startup = 15
    maximum_stop = 30
    executable = str(cluster.get("cli_path", "/usr/local/bin/spark-serve"))
    routable_profiles = {
        model_id: profile
        for model_id, profile in catalog.profiles.items()
        if include_maintenance
        or profile["operator"].get("exposure", "production") == "production"
    }
    for model_id, profile in sorted(routable_profiles.items()):
        launch_timeout = int(profile["runtime"]["launch"]["timeout_seconds"])
        stop_timeout = int(profile["runtime"].get("stop", {}).get("timeout_seconds", 30))
        maximum_startup = max(maximum_startup, launch_timeout)
        maximum_stop = max(maximum_stop, stop_timeout)
        backend = profile["runtime"]["backend"]
        command = shlex.join(
            [executable, "--config", str(catalog.config_path), "internal-run", model_id, "--port", "${PORT}"]
        )
        if cluster["nodes"][backend["node"]]["host"] in {"local", "localhost", "127.0.0.1"}:
            proxy = f"http://127.0.0.1:{backend['port']}"
        else:
            proxy = "http://127.0.0.1:${PORT}"
        models[model_id] = {
            "name": profile["name"],
            "description": profile["description"],
            "cmd": command,
            "proxy": proxy,
            "checkEndpoint": backend["health_path"],
            "ttl": int(profile["serving"]["ttl_seconds"]),
            "unloadTimeout": stop_timeout,
            "aliases": profile["aliases"],
            "useModelName": profile["model"]["served_name"],
            "concurrencyLimit": int(profile["serving"]["concurrency_limit"]),
            "sendLoadingState": True,
            "capabilities": {
                "in": profile["model"].get("input_modalities", ["text"]),
                "out": ["text"],
                "tools": bool(profile["serving"]["tools"]),
                "context": int(profile["model"]["context_tokens"]),
            },
            "metadata": {
                "topology": profile["topology"]["mode"],
                "quantization": profile["model"]["quantization"],
                "readiness": profile["operator"]["readiness"],
                "engine": profile["runtime"]["engine"],
            },
        }
        if profile["serving"].get("request_filters"):
            models[model_id]["filters"] = profile["serving"]["request_filters"]
    sets = compatible_sets(routable_profiles)
    matrix_sets = {f"allowed-{index:03d}": " & ".join(values) for index, values in enumerate(sets, 1)}
    config: dict[str, Any] = {
        "healthCheckTimeout": maximum_startup,
        "unloadTimeout": maximum_stop,
        "logLevel": "info",
        "logTimeFormat": "rfc3339",
        "logToStdout": "both",
        "captureBuffer": 0,
        "includeAliasesInList": True,
        "sendLoadingState": True,
        "startPort": 9300,
        "models": models,
        "routing": {
            "router": {
                "use": "matrix",
                "settings": {
                    "matrix": {
                        "sets": matrix_sets,
                        "evict_costs": {
                            model_id: int(profile["serving"].get("eviction_cost", 1))
                            for model_id, profile in sorted(routable_profiles.items())
                        },
                    }
                },
            },
            "scheduler": {"use": "fifo"},
        },
    }
    pool = cluster.get("worker_pool")
    if pool:
        config["selectors"] = {pool["id"]: {
            "strategy": "spillover", "targets": pool["targets"],
            "settings": {"spillover": pool["spillover"]},
            "name": "Local model pool",
            "description": "Centrally configured model targets.",
        }}
    for route in cluster.get("model_routes", []):
        contract = route_contract(catalog, route)
        name = route.get("name", route["id"])
        config.setdefault("selectors", {})[route["id"]] = {
            "strategy": "ready" if route.get("ready_only") else "spillover", "targets": route["targets"],
            "settings": {"spillover": route.get("spillover", 1)},
            "name": name, "description": "Automatic local serving route; placement is managed centrally.",
            "metadata": {"route_contract": contract, "pi": {
                "name": name, "api": "openai-completions",
                "reasoning": contract["supports_reasoning"], "input": contract["input_modalities"],
                "contextWindow": contract["context_window"], "maxTokens": contract["max_output_tokens"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "compat": route["client_compat"],
                **({"thinkingLevelMap": route["thinking_levels"]} if "thinking_levels" in route else {}),
            }},
        }
        if route.get("maintenance_profiles"):
            # Native profile activation rewrites only the logical route and
            # does not reload configuration or stop either physical model.
            # The hidden single-target selector still enforces readiness.
            for i, target in enumerate(route["targets"], 1):
                selector_id = f"{route['id']}--only--{i}"
                config["selectors"][selector_id] = {
                    "strategy": "ready", "targets": [target],
                    "unlisted": True,
                    "name": name,
                    "metadata": config["selectors"][route["id"]]["metadata"],
                }
                config.setdefault("profiles", {})[selector_id] = {
                    "description": f"Route {route['id']} only to ready {target}; other replicas remain available for maintenance.",
                    "pins": {route["id"]: selector_id},
                }
    api_keys = _read_api_keys(cluster["gateway"].get("api_key_file"))
    if api_keys:
        config["apiKeys"] = api_keys
    return config


def route_contract(catalog: Catalog, route: dict[str, Any]) -> dict[str, Any]:
    """Advertise the intersection that every centrally selected target supports."""
    profiles = [catalog.profiles[t] for t in route["targets"]]
    modalities = set.intersection(*(set(p["model"].get("input_modalities", ["text"])) for p in profiles))
    return {
        "schema_version": 1, "api": "chat_completions",
        "context_window": min(p["model"]["context_tokens"] for p in profiles),
        "max_output_tokens": route["max_output_tokens"],
        "supports_tools": all(p["serving"]["tools"] for p in profiles),
        "supports_reasoning": all(p["serving"]["reasoning"] for p in profiles),
        "input_modalities": sorted(modalities, key=lambda x: (x != "text", x)),
        "client_compat": route["client_compat"],
    }
