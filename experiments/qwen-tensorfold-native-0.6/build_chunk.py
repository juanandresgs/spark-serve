#!/usr/bin/env python3
"""Build and CPU-test the separate burst-4 plus 1024-row prefill derivative."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

ROOT = Path(__file__).resolve().parent
EXPECTED_TESTS = 19
PARENT_VARIANT = "experimental-native-prefill-cancellation-decode-burst4"
CHUNK_VARIANT = "experimental-native-prefill-cancellation-decode-burst4-prefill1024"
RUNTIME_REVISION = "c4646171139ee8a3c38103eaa1699dad226ec12b"
BURST_PATCH = "075f814bef774f8dbb33096e91a9335f7faff0be6e6f987da48e6bdaf282f09a"


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(image):
    return json.loads(output(["docker", "image", "inspect", image]))[0]


def valid_cpu_result(result):
    return (result.get("tests_run") == EXPECTED_TESTS and result.get("failures") == 0 and
            result.get("errors") == 0 and result.get("skipped") == 0 and
            result.get("successful") is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-burst-image-id", required=True,
                        help="Exact local image ID of the CPU-tested burst-4 derivative on this host")
    parser.add_argument("--tag", default="spark-serve/qwen-tensorfold-native:0.6.0-cancel-burst4-prefill1024")
    parser.add_argument("--receipt", type=Path, default=Path("artifacts/chunk-build-receipt.json"))
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.parent_burst_image_id):
        raise SystemExit("--parent-burst-image-id must be an exact local sha256 image ID")

    burst_manifest_path = ROOT / "decode-burst" / "patch-manifest.json"
    chunk_manifest_path = ROOT / "scheduling" / "chunk" / "patch-manifest.json"
    chunk_patch_path = ROOT / "scheduling" / "chunk" / "prefill-rows.patch"
    burst = json.loads(burst_manifest_path.read_text())
    chunk = json.loads(chunk_manifest_path.read_text())
    if burst["patch_sha256"] != BURST_PATCH:
        raise SystemExit("burst parent manifest does not match the retained burst-4 baseline")
    if chunk["upstream_commit"] != RUNTIME_REVISION:
        raise SystemExit("chunk manifest runtime revision does not match the pinned runtime")
    if chunk["parent_burst_patch_sha256"] != burst["patch_sha256"]:
        raise SystemExit("chunk derivative does not name the retained burst-4 patch")
    if sha(chunk_patch_path) != chunk["patch_sha256"]:
        raise SystemExit("prefill-rows patch hash does not match its manifest")

    parent = inspect(args.parent_burst_image_id)
    if parent.get("Id") != args.parent_burst_image_id:
        raise SystemExit("Docker did not resolve the exact supplied parent image ID")
    labels = parent.get("Config", {}).get("Labels") or {}
    if labels.get("org.dgx-spark-serve.tensorfold.variant") != PARENT_VARIANT:
        raise SystemExit("parent image is not the retained burst-4 variant")
    if labels.get("org.dgx-spark-serve.tensorfold.decode-burst-patch-sha256") != BURST_PATCH:
        raise SystemExit("parent image decode-burst patch label does not match the baseline")
    if labels.get("org.opencontainers.image.revision") != RUNTIME_REVISION:
        raise SystemExit("parent image runtime revision does not match the pinned source")
    if parent.get("Architecture") != "arm64" or parent.get("Os") != "linux":
        raise SystemExit("parent image must be linux/arm64")
    parent_env = parent.get("Config", {}).get("Env") or []
    if "TF_FLASH_DECODE_BURST=4" not in parent_env:
        raise SystemExit("parent image does not retain the burst-4 baseline")
    if any(entry.startswith("TF_FLASH_PREFILL_ROWS=") for entry in parent_env):
        raise SystemExit("parent image must not override the burst-4 baseline prefill rows")

    temporary_parent = "spark-serve/qwen-native-parent:chunk-" + uuid.uuid4().hex[:12]
    keep_temporary_tag = not parent.get("RepoTags")
    subprocess.run(["docker", "image", "tag", parent["Id"], temporary_parent], check=True)
    try:
        if inspect(temporary_parent).get("Id") != parent["Id"]:
            raise SystemExit("temporary parent tag does not resolve to the supplied image ID")
        subprocess.run([
            "docker", "build", "--pull=false", "--network=none",
            "--file", str(ROOT / "Dockerfile.chunk"),
            "--build-arg", "PARENT_IMAGE=" + temporary_parent,
            "--tag", args.tag, str(ROOT),
        ], check=True)
    finally:
        if not keep_temporary_tag:
            subprocess.run(["docker", "image", "rm", temporary_parent], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    child = inspect(args.tag)
    child_labels = child.get("Config", {}).get("Labels") or {}
    child_env = child.get("Config", {}).get("Env") or []
    if child_labels.get("org.dgx-spark-serve.tensorfold.variant") != CHUNK_VARIANT:
        raise SystemExit("built image does not identify the 1024-row derivative")
    if child_labels.get("org.dgx-spark-serve.tensorfold.prefill-rows-patch-sha256") != chunk["patch_sha256"]:
        raise SystemExit("built image patch label does not match source manifest")
    if "TF_FLASH_DECODE_BURST=4" not in child_env or "TF_FLASH_PREFILL_ROWS=1024" not in child_env:
        raise SystemExit("built image does not retain burst4 and select 1024 prefill rows")
    parent_layers = parent.get("RootFS", {}).get("Layers", [])
    child_layers = child.get("RootFS", {}).get("Layers", [])
    if child_layers[:len(parent_layers)] != parent_layers:
        raise SystemExit("built image does not retain the selected parent rootfs layers")

    test_program = r'''
import importlib.util, io, json, sys, unittest
modules = []
for name, path in (
    ('test_prefill_cancel', '/opt/spark-serve/cancellation/test_prefill_cancel.py'),
    ('test_decode_burst', '/opt/spark-serve/decode-burst/test_decode_burst.py'),
    ('test_prefill_rows', '/opt/spark-serve/scheduling/chunk/test_prefill_rows.py'),
):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    modules.append(module)
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
payload = {'tests_run': result.testsRun, 'failures': len(result.failures),
           'errors': len(result.errors), 'skipped': len(result.skipped),
           'successful': result.wasSuccessful()}
print(json.dumps(payload, sort_keys=True))
if result.testsRun != 19 or not result.wasSuccessful() or result.skipped:
    sys.exit(1)
'''
    tested = subprocess.run([
        "docker", "run", "--rm", "--network=none", "--entrypoint", "python", child["Id"],
        "-c", test_program,
    ], check=False, capture_output=True, text=True)
    if tested.returncode:
        raise SystemExit("19-test cancellation/burst/chunk CPU gate failed")
    try:
        cpu_result = json.loads(tested.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SystemExit("CPU test runner did not emit its structured result") from exc
    if not valid_cpu_result(cpu_result):
        raise SystemExit("structured cancellation/burst/chunk CPU result failed validation")

    receipt = {
        "schema_version": 1,
        "variant": CHUNK_VARIANT,
        "qualification": "Experimental derivative; 19 mocked-GPU cancellation/scheduler CPU checks required; GPU latency, speed, quality and serving qualification pending.",
        "local_parent_burst_image_id": parent["Id"],
        "local_image_id": child["Id"],
        "local_image_id_semantics": "daemon-store-specific; may identify config or manifest",
        "architecture": child.get("Architecture"),
        "os": child.get("Os"),
        "parent_rootfs_diff_ids": parent_layers,
        "rootfs_diff_ids": child_layers,
        "runtime_revision": RUNTIME_REVISION,
        "parent_burst_patch_sha256": burst["patch_sha256"],
        "chunk_patch_sha256": chunk["patch_sha256"],
        "chunk_patch_manifest_sha256": sha(chunk_manifest_path),
        "chunk_test_source_sha256": sha(ROOT / "scheduling" / "chunk" / "test_prefill_rows.py"),
        "cpu_test_result": cpu_result,
        "settings": {"TF_FLASH_DECODE_BURST": 4, "TF_FLASH_PREFILL_ROWS": 1024},
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "local_image_id": child["Id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
