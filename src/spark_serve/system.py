from __future__ import annotations

import base64
import fcntl
import http.client
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spark_serve.config import Catalog, ConfigError


GIB = 1024**3
RESOURCE_HELPER = "/usr/local/libexec/dgx-spark-serve-resource"

_PROBE_SOURCE = r"""
import json, os, socket, subprocess, sys

model_root = sys.argv[1]
port = int(sys.argv[2])
mem = {}
with open('/proc/meminfo', encoding='utf-8') as handle:
    for line in handle:
        key, value = line.split(':', 1)
        fields = value.split()
        if fields:
            mem[key] = int(fields[0]) * 1024
stat = os.statvfs(model_root)
gpu = {'available': False}
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, timeout=8, check=True,
    )
    first = result.stdout.strip().splitlines()[0].split(', ')
    gpu = {'available': True, 'name': first[0], 'temperature_c': float(first[1]), 'power_w': float(first[2])}
except Exception as exc:
    gpu = {'available': False, 'error': type(exc).__name__}
listening = False
if port:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        listening = sock.connect_ex(('127.0.0.1', port)) == 0
    finally:
        sock.close()
print(json.dumps({
    'mem_total_bytes': mem.get('MemTotal', 0),
    'mem_available_bytes': mem.get('MemAvailable', 0),
    'swap_total_bytes': mem.get('SwapTotal', 0),
    'swap_free_bytes': mem.get('SwapFree', 0),
    'swap_used_bytes': max(0, mem.get('SwapTotal', 0) - mem.get('SwapFree', 0)),
    'disk_free_bytes': stat.f_bavail * stat.f_frsize,
    'load_1m': os.getloadavg()[0],
    'backend_port_listening': listening,
    'gpu': gpu,
}))
"""


class CommandError(RuntimeError):
    pass


def _is_local(node: dict[str, Any]) -> bool:
    return node["host"] in {"local", "localhost", "127.0.0.1"}


def command_argv(node: dict[str, Any], argv: list[str], environment: dict[str, str] | None = None) -> list[str]:
    environment = environment or {}
    command = ["env", *[f"{key}={value}" for key, value in sorted(environment.items())], *argv]
    if _is_local(node):
        return command
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        node["host"],
        shlex.join(command),
    ]


def run_command(
    node: dict[str, Any],
    argv: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: int = 60,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command_argv(node, argv, environment),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise CommandError(f"command failed on {node['host']} ({result.returncode}): {detail[-1200:]}")
    return result


def popen_command(
    node: dict[str, Any], argv: list[str], environment: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command_argv(node, argv, environment),
        text=True,
        start_new_session=True,
    )


def probe_node(node: dict[str, Any], port: int = 0) -> dict[str, Any]:
    encoded = base64.b64encode(_PROBE_SOURCE.encode()).decode()
    program = f"import base64;exec(base64.b64decode({encoded!r}))"
    result = run_command(
        node,
        ["python3", "-c", program, node["model_root"], str(port)],
        timeout=20,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CommandError(f"invalid telemetry response from {node['host']}") from exc


def _remote_url_request(node: dict[str, Any], url: str, timeout: int) -> tuple[int, bytes]:
    if _is_local(node):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            return 0, b""
    result = run_command(
        node,
        ["curl", "--silent", "--show-error", "--max-time", str(timeout), "--write-out", "\n%{http_code}", url],
        timeout=timeout + 5,
        check=False,
    )
    if result.returncode:
        return 0, b""
    body, _, status = result.stdout.rpartition("\n")
    try:
        return int(status), body.encode()
    except ValueError:
        return 0, b""


def backend_matches(catalog: Catalog, profile: dict[str, Any], timeout: int = 5) -> bool:
    backend = profile["runtime"]["backend"]
    node = catalog.cluster["nodes"][backend["node"]]
    base = f"http://127.0.0.1:{backend['port']}"
    status, _ = _remote_url_request(node, base + backend["health_path"], timeout)
    if status != 200:
        return False
    models_path = backend.get("models_path", "/v1/models")
    status, body = _remote_url_request(node, base + models_path, timeout)
    if status != 200:
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    expected = profile["model"]["served_name"]
    ids = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    return expected in ids or profile["id"] in ids


def wait_backend(catalog: Catalog, profile: dict[str, Any], child: subprocess.Popen[str] | None) -> None:
    timeout = int(profile["runtime"]["launch"]["timeout_seconds"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if backend_matches(catalog, profile):
            return
        if child is not None and child.poll() is not None:
            raise CommandError(f"launcher exited before readiness with status {child.returncode}")
        time.sleep(2)
    raise CommandError(f"backend did not become ready within {timeout} seconds")


def check_artifact(catalog: Catalog, artifact: dict[str, Any]) -> tuple[bool, str]:
    node = catalog.cluster["nodes"][artifact["node"]]
    kind = artifact["kind"]
    value = artifact["value"]
    if kind == "docker_image":
        result = run_command(node, ["docker", "image", "inspect", value], timeout=30, check=False)
        return result.returncode == 0, f"docker image {value}"
    if kind == "git_checkout":
        revision = artifact["revision"]
        result = run_command(
            node,
            ["git", "-C", value, "status", "--porcelain", "--untracked-files=no"],
            timeout=15,
            check=False,
        )
        if result.returncode or result.stdout.strip():
            return False, f"clean git checkout {value}"
        result = run_command(node, ["git", "-C", value, "rev-parse", "HEAD"], timeout=15, check=False)
        return result.returncode == 0 and result.stdout.strip() == revision, f"git {value}@{revision[:12]}"
    test_flag = "-f" if kind == "file" else "-d"
    result = run_command(node, ["test", test_flag, value], timeout=10, check=False)
    if result.returncode:
        return False, f"{kind} {value}"
    minimum = int(artifact.get("min_size_bytes", 0))
    if minimum:
        if kind == "file":
            size_result = run_command(node, ["stat", "-c", "%s", value], timeout=10, check=False)
        else:
            size_result = run_command(node, ["du", "-sb", value], timeout=60, check=False)
        try:
            size = int(size_result.stdout.split()[0])
        except (IndexError, ValueError):
            return False, f"size of {value}"
        if size < minimum:
            return False, f"{value} is {size} bytes; require {minimum}"
    return True, f"{kind} {value}"


def profile_checks(catalog: Catalog, profile: dict[str, Any], *, strict_resources: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for command_name in ("launch", "stop"):
        command = profile["runtime"].get(command_name)
        if not command:
            continue
        node = catalog.cluster["nodes"][command["node"]]
        executable = command["argv"][0]
        if executable.startswith("/"):
            result = run_command(node, ["test", "-x", executable], timeout=10, check=False)
        else:
            result = run_command(node, ["sh", "-c", "command -v \"$1\" >/dev/null", "sh", executable], timeout=10, check=False)
        checks.append(
            {
                "kind": "launcher",
                "node": command["node"],
                "pass": result.returncode == 0,
                "detail": f"{command_name} executable {executable}",
            }
        )
    for artifact in profile["artifacts"]:
        try:
            passed, detail = check_artifact(catalog, artifact)
        except (CommandError, subprocess.TimeoutExpired) as exc:
            passed, detail = False, str(exc)
        checks.append({"kind": "artifact", "node": artifact["node"], "pass": passed, "detail": detail})
    for node_id, requirement in profile["resources"]["nodes"].items():
        node = catalog.cluster["nodes"][node_id]
        backend_port = profile["runtime"]["backend"]["port"] if profile["runtime"]["backend"]["node"] == node_id else 0
        try:
            telemetry = probe_node(node, backend_port)
        except (CommandError, subprocess.TimeoutExpired) as exc:
            checks.append({"kind": "node", "node": node_id, "pass": False, "detail": str(exc)})
            continue
        memory_required = float(requirement["min_memory_available_gib"]) * GIB
        disk_required = float(requirement["min_disk_free_gib"]) * GIB
        memory_pass = telemetry["mem_available_bytes"] >= memory_required
        disk_pass = telemetry["disk_free_bytes"] >= disk_required
        swap_pass = requirement["swap_policy"] != "off_during_run" or telemetry["swap_used_bytes"] == 0
        checks.extend(
            [
                {
                    "kind": "memory",
                    "node": node_id,
                    "pass": memory_pass if strict_resources else True,
                    "observed_pass": memory_pass,
                    "detail": f"{telemetry['mem_available_bytes'] / GIB:.1f} GiB available; require {memory_required / GIB:.1f} GiB",
                },
                {
                    "kind": "disk",
                    "node": node_id,
                    "pass": disk_pass,
                    "detail": f"{telemetry['disk_free_bytes'] / GIB:.1f} GiB free; require {disk_required / GIB:.1f} GiB",
                },
                {
                    "kind": "swap",
                    "node": node_id,
                    "pass": swap_pass if strict_resources else True,
                    "observed_pass": swap_pass,
                    "detail": f"{telemetry['swap_used_bytes'] / GIB:.2f} GiB used; policy {requirement['swap_policy']}",
                },
                {
                    "kind": "gpu",
                    "node": node_id,
                    "pass": bool(telemetry["gpu"].get("available")),
                    "detail": telemetry["gpu"].get("name", telemetry["gpu"].get("error", "unavailable")),
                },
            ]
        )
    if profile["topology"]["fabric_required"]:
        head_node = catalog.cluster["nodes"][profile["topology"]["nodes"][0]]
        for node_id in profile["topology"]["nodes"][1:]:
            target = catalog.cluster["nodes"][node_id].get("fabric_ip")
            if not target:
                checks.append({"kind": "fabric", "node": node_id, "pass": False, "detail": "fabric_ip missing"})
                continue
            result = run_command(head_node, ["bash", "-lc", f"timeout 3 bash -c '</dev/tcp/{target}/22'"], timeout=6, check=False)
            checks.append({"kind": "fabric", "node": node_id, "pass": result.returncode == 0, "detail": f"SSH over fabric to {target}"})
    return checks


class ClaimLocks:
    def __init__(self, runtime_dir: Path, claims: list[str], model_id: str):
        self.runtime_dir = runtime_dir
        self.claims = sorted(claims)
        self.model_id = model_id
        self.handles: list[Any] = []

    def __enter__(self) -> "ClaimLocks":
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        try:
            for claim in self.claims:
                safe = re_safe(claim)
                handle = (self.runtime_dir / f"claim-{safe}.lock").open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    handle.seek(0)
                    owner = handle.read().strip() or "another broker process"
                    handle.close()
                    raise CommandError(f"resource claim {claim} is owned by {owner}") from exc
                handle.seek(0)
                handle.truncate()
                handle.write(f"model={self.model_id} pid={os.getpid()}\n")
                handle.flush()
                self.handles.append(handle)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in reversed(self.handles):
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            except OSError:
                pass
        self.handles.clear()


def re_safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def set_swap(node: dict[str, Any], action: str, devices: list[str]) -> None:
    if action == "off":
        argv = ["sudo", "-n", RESOURCE_HELPER, "swap-off"]
    elif action == "on":
        argv = ["sudo", "-n", RESOURCE_HELPER, "swap-on", *devices]
    else:
        raise ValueError(action)
    run_command(node, argv, timeout=300)


def stop_process(child: subprocess.Popen[str] | None, timeout: int = 30) -> None:
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def start_tunnel(node: dict[str, Any], local_port: int, remote_port: int) -> subprocess.Popen[str]:
    if _is_local(node):
        raise CommandError("a tunnel was requested for a local backend")
    process = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            node["host"],
        ],
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CommandError(f"SSH tunnel exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{local_port}/health", timeout=1) as response:
                if response.status == 200:
                    return process
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            time.sleep(0.25)
    stop_process(process)
    raise CommandError("SSH backend tunnel did not become ready")
