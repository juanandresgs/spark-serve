from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from spark_serve.config import Catalog, ConfigError


class GatewayError(RuntimeError):
    pass


def _api_key(catalog: Catalog) -> str | None:
    path_value = catalog.cluster["gateway"].get("api_key_file")
    if not path_value:
        return None
    value: str | None = None
    try:
        for line in Path(path_value).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    except PermissionError:
        value = None
    except FileNotFoundError as exc:
        raise GatewayError(f"gateway API key file is missing: {path_value}") from exc
    if value is None:
        raise GatewayError(
            f"cannot read gateway API key file {path_value}; use SPARK_SERVE_API_KEY"
        )
    return None


def request(
    catalog: Catalog,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> tuple[int, bytes]:
    base = catalog.cluster["gateway"]["base_url"].rstrip("/")
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    key = os.environ.get("SPARK_SERVE_API_KEY") or _api_key(catalog)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise GatewayError(f"gateway unavailable at {base}: {exc}") from exc


def json_request(
    catalog: Catalog,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    status, body = request(catalog, method, path, payload=payload, timeout=timeout)
    if not 200 <= status < 300:
        detail = body.decode(errors="replace")[-1200:]
        raise GatewayError(f"gateway returned HTTP {status} for {path}: {detail}")
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")


def is_healthy(catalog: Catalog) -> bool:
    try:
        status, _ = request(catalog, "GET", "/health", timeout=3)
        return status == 200
    except GatewayError:
        return False


def wait_healthy(catalog: Catalog, timeout: int = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_healthy(catalog):
            return
        time.sleep(0.5)
    raise GatewayError(f"gateway did not become healthy within {timeout} seconds")


def running_models(catalog: Catalog) -> list[str]:
    payload = json_request(catalog, "GET", "/running", timeout=10)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("models", payload.get("running", []))
        if isinstance(records, dict):
            records = list(records.values())
    else:
        records = []
    result: list[str] = []
    for record in records:
        if isinstance(record, str):
            result.append(record)
        elif isinstance(record, dict):
            value = record.get("id") or record.get("model") or record.get("name")
            if isinstance(value, str):
                result.append(value)
    return sorted(set(result))


def load_model(catalog: Catalog, model_id: str, timeout: int) -> None:
    profile = catalog.profiles.get(model_id)
    if profile is None:
        raise GatewayError(f"unknown model {model_id}")
    backend = profile["runtime"]["backend"]
    models_path = backend.get("models_path", "/v1/models")
    encoded = urllib.parse.quote(model_id, safe="")
    status, body = request(catalog, "GET", f"/upstream/{encoded}{models_path}", timeout=timeout)
    if status != 200:
        detail = body.decode(errors="replace")[-1600:]
        raise GatewayError(f"model {model_id} failed to load: HTTP {status}: {detail}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GatewayError(f"model {model_id} readiness response was not JSON") from exc
    if not isinstance(payload, dict):
        raise GatewayError(f"model {model_id} readiness response was not an object")
    expected = profile["model"]["served_name"]
    ids = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    if expected not in ids and model_id not in ids:
        observed = ", ".join(sorted(value for value in ids if isinstance(value, str))) or "none"
        raise GatewayError(
            f"model {model_id} readiness identity mismatch: expected {expected}, observed {observed}"
        )


def unload_model(catalog: Catalog, model_id: str, timeout: int = 900) -> None:
    encoded = urllib.parse.quote(model_id, safe="")
    status, body = request(catalog, "POST", f"/api/models/unload/{encoded}", timeout=timeout)
    if not 200 <= status < 300:
        detail = body.decode(errors="replace")[-1200:]
        raise GatewayError(f"model {model_id} failed to unload: HTTP {status}: {detail}")


def unload_all(catalog: Catalog, timeout: int = 900) -> None:
    status, body = request(catalog, "POST", "/api/models/unload", timeout=timeout)
    if not 200 <= status < 300:
        detail = body.decode(errors="replace")[-1200:]
        raise GatewayError(f"models failed to unload: HTTP {status}: {detail}")
