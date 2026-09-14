from __future__ import annotations

import json
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

from spark_serve.config import Catalog, launch_environment
from spark_serve.system import (
    ClaimLocks,
    CommandError,
    backend_matches,
    popen_command,
    probe_node,
    profile_checks,
    run_command,
    set_swap,
    start_tunnel,
    stop_process,
    wait_backend,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o640)
    temporary.replace(path)


def _service_identity_ok(catalog: Catalog) -> bool:
    current = pwd.getpwuid(os.geteuid()).pw_name
    return current in {"root", catalog.cluster["service_user"]}


def _owner_environment(catalog: Catalog, profile: dict[str, Any]) -> dict[str, str]:
    return {"DGX_SPARK_SERVE_MANAGED": "1",
            "DGX_SPARK_SERVE_DEPLOYMENT_ID": catalog.cluster["cluster_id"],
            "DGX_SPARK_SERVE_PROFILE_ID": profile["id"]}


def _run_stop(catalog: Catalog, profile: dict[str, Any]) -> None:
    command = profile["runtime"].get("stop")
    if not command:
        return
    node = catalog.cluster["nodes"][command["node"]]
    result = run_command(
        node,
        command["argv"],
        environment={**command.get("environment", {}), **_owner_environment(catalog, profile)},
        timeout=int(command["timeout_seconds"]),
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        print(f"spark-serve: stop warning for {profile['id']}: {detail[-800:]}", file=sys.stderr, flush=True)


def _write_state(catalog: Catalog, profile: dict[str, Any], attached: bool) -> Path:
    state_dir = Path(catalog.cluster.get("state_dir", "/var/lib/dgx-spark-serve"))
    path = state_dir / f"active-{profile['id']}.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "cluster_id": catalog.cluster["cluster_id"],
            "model": profile["id"],
            "pid": os.getpid(),
            "attached": attached,
            "started_unix": int(time.time()),
            "claims": profile["resources"]["claims"],
        },
    )
    return path


def _static_artifacts_ready(catalog: Catalog, profile: dict[str, Any]) -> None:
    checks = [
        check
        for check in profile_checks(catalog, profile, strict_resources=False)
        if check["kind"] in {"launcher", "artifact", "disk", "gpu", "fabric"}
    ]
    failures = [check for check in checks if not check["pass"]]
    if failures:
        detail = "; ".join(f"{item['node']} {item['kind']}: {item['detail']}" for item in failures)
        raise CommandError(f"preflight failed: {detail}")


def _adopt_existing_backend(catalog: Catalog, profile: dict[str, Any]) -> bool:
    """Adopt only when the profile has no unverifiable scheduler contract."""

    if not backend_matches(catalog, profile):
        return False
    if profile["serving"].get("scheduler") is None:
        return True

    print(
        "spark-serve: replacing healthy backend because scheduler launch semantics "
        f"cannot be proven for {profile['id']}",
        flush=True,
    )
    _run_stop(catalog, profile)
    stop_timeout = int(
        profile["runtime"].get("stop", {}).get("timeout_seconds", 30)
    )
    deadline = time.monotonic() + stop_timeout
    while time.monotonic() < deadline:
        if not backend_matches(catalog, profile):
            return False
        time.sleep(1)
    raise CommandError(
        f"existing backend for {profile['id']} remained healthy after scheduler-safe stop"
    )


def internal_run(catalog: Catalog, profile: dict[str, Any], gateway_port: int) -> int:
    if not _service_identity_ok(catalog):
        raise CommandError(
            f"internal-run is restricted to root or service user {catalog.cluster['service_user']}"
        )
    runtime_dir = Path(catalog.cluster.get("runtime_dir", "/run/dgx-spark-serve"))
    stop_event = Event()

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"spark-serve: received signal {signum}; stopping {profile['id']}", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGHUP, request_stop)

    child: subprocess.Popen[str] | None = None
    tunnel: subprocess.Popen[str] | None = None
    state_path: Path | None = None
    changed_swap: list[str] = []
    attached = False
    owns_backend = False
    backend = profile["runtime"]["backend"]
    backend_node = catalog.cluster["nodes"][backend["node"]]
    launch = profile["runtime"]["launch"]
    launch_node = catalog.cluster["nodes"][launch["node"]]
    launch_env = {**launch_environment(profile), **_owner_environment(catalog, profile)}

    with ClaimLocks(runtime_dir, profile["resources"]["claims"], profile["id"]):
        try:
            _static_artifacts_ready(catalog, profile)
            attached = _adopt_existing_backend(catalog, profile)
            if attached:
                owns_backend = True
                print(f"spark-serve: adopting healthy existing backend for {profile['id']}", flush=True)
            else:
                for node_id, requirement in profile["resources"]["nodes"].items():
                    if requirement["swap_policy"] != "off_during_run":
                        continue
                    node = catalog.cluster["nodes"][node_id]
                    telemetry = probe_node(node)
                    if telemetry["swap_total_bytes"]:
                        print(f"spark-serve: disabling swap on {node_id} for {profile['id']}", flush=True)
                        set_swap(node, "off", node["swap_devices"])
                        changed_swap.append(node_id)
                checks = profile_checks(catalog, profile, strict_resources=True)
                failures = [check for check in checks if not check["pass"]]
                if failures:
                    detail = "; ".join(
                        f"{item['node']} {item['kind']}: {item['detail']}" for item in failures
                    )
                    raise CommandError(f"resource admission failed: {detail}")
                print(f"spark-serve: launching {profile['id']} ({profile['topology']['mode']})", flush=True)
                owns_backend = True
                if launch["mode"] == "foreground":
                    child = popen_command(launch_node, launch["argv"], launch_env)
                else:
                    run_command(
                        launch_node,
                        launch["argv"],
                        environment=launch_env,
                        timeout=int(launch["timeout_seconds"]),
                    )
                wait_backend(catalog, profile, child)

            if backend_node["host"] not in {"local", "localhost", "127.0.0.1"}:
                tunnel = start_tunnel(backend_node, gateway_port, int(backend["port"]))
            state_path = _write_state(catalog, profile, attached)
            print(
                f"spark-serve: ready model={profile['id']} backend_node={backend['node']} backend_port={backend['port']}",
                flush=True,
            )
            misses = 0
            while not stop_event.wait(5):
                if child is not None and child.poll() is not None:
                    raise CommandError(f"launcher exited unexpectedly with status {child.returncode}")
                if tunnel is not None and tunnel.poll() is not None:
                    raise CommandError(f"SSH tunnel exited unexpectedly with status {tunnel.returncode}")
                if backend_matches(catalog, profile):
                    misses = 0
                else:
                    misses += 1
                    if misses >= 3:
                        raise CommandError("backend failed three consecutive health checks")
            return 0
        finally:
            stop_process(tunnel)
            stop_process(child, timeout=30)
            if owns_backend:
                _run_stop(catalog, profile)
            if state_path is not None:
                try:
                    state_path.unlink()
                except FileNotFoundError:
                    pass
            for node_id in reversed(changed_swap):
                node = catalog.cluster["nodes"][node_id]
                try:
                    print(f"spark-serve: restoring swap on {node_id}", flush=True)
                    set_swap(node, "on", node["swap_devices"])
                except (CommandError, subprocess.TimeoutExpired) as exc:
                    print(f"spark-serve: swap restore warning on {node_id}: {exc}", file=sys.stderr, flush=True)


def stop_known_backend(catalog: Catalog, profile: dict[str, Any]) -> None:
    _run_stop(catalog, profile)


def active_state(catalog: Catalog) -> list[dict[str, Any]]:
    state_dir = Path(catalog.cluster.get("state_dir", "/var/lib/dgx-spark-serve"))
    result: list[dict[str, Any]] = []
    for path in sorted(state_dir.glob("active-*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        state["state_file"] = str(path)
        result.append(state)
    return result
