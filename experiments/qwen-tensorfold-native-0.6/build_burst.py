#!/usr/bin/env python3
"""Build and CPU-test the separate cancellation plus EXL3 burst-4 derivative."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

ROOT = Path(__file__).resolve().parent
EXPECTED_TESTS = 14
BASE_VARIANT = "experimental-native-prefill-cancellation"
BURST_VARIANT = "experimental-native-prefill-cancellation-decode-burst4"
RUNTIME_REVISION = "c4646171139ee8a3c38103eaa1699dad226ec12b"


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
    parser.add_argument("--parent-cancellation-image-id", required=True,
                        help="Exact local image ID of the CPU-tested cancellation derivative on this host")
    parser.add_argument("--tag", default="spark-serve/qwen-tensorfold-native:0.6.0-cancel-burst4")
    parser.add_argument("--receipt", type=Path, default=Path("artifacts/burst4-build-receipt.json"))
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.parent_cancellation_image_id):
        raise SystemExit("--parent-cancellation-image-id must be an exact local sha256 image ID")

    cancellation_path = ROOT / "cancellation" / "patch-manifest.json"
    burst_manifest_path = ROOT / "decode-burst" / "patch-manifest.json"
    burst_patch_path = ROOT / "decode-burst" / "decode-burst.patch"
    cancellation = json.loads(cancellation_path.read_text())
    burst = json.loads(burst_manifest_path.read_text())
    if burst["upstream_commit"] != RUNTIME_REVISION:
        raise SystemExit("burst manifest runtime revision does not match the pinned runtime")
    if burst["parent_cancellation_patch_sha256"] != cancellation["patch_sha256"]:
        raise SystemExit("burst variant does not name the retained cancellation patch")
    if sha(burst_patch_path) != burst["patch_sha256"]:
        raise SystemExit("decode-burst patch hash does not match its manifest")

    parent = inspect(args.parent_cancellation_image_id)
    if parent.get("Id") != args.parent_cancellation_image_id:
        raise SystemExit("Docker did not resolve the exact supplied parent image ID")
    labels = parent.get("Config", {}).get("Labels") or {}
    if labels.get("org.dgx-spark-serve.tensorfold.variant") != BASE_VARIANT:
        raise SystemExit("parent image is not labeled as the cancellation-only variant")
    if labels.get("org.opencontainers.image.revision") != RUNTIME_REVISION:
        raise SystemExit("parent image runtime revision does not match the pinned source")
    if parent.get("Architecture") != "arm64" or parent.get("Os") != "linux":
        raise SystemExit("parent image must be linux/arm64")

    temporary_parent = "spark-serve/qwen-native-parent:burst-" + uuid.uuid4().hex[:12]
    keep_temporary_tag = not parent.get("RepoTags")
    subprocess.run(["docker", "image", "tag", parent["Id"], temporary_parent], check=True)
    try:
        if inspect(temporary_parent).get("Id") != parent["Id"]:
            raise SystemExit("temporary parent tag does not resolve to the supplied image ID")
        subprocess.run([
            "docker", "build", "--pull=false", "--network=none",
            "--file", str(ROOT / "Dockerfile.burst"),
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
    if child_labels.get("org.dgx-spark-serve.tensorfold.variant") != BURST_VARIANT:
        raise SystemExit("built image does not identify the burst-4 variant")
    if child_labels.get("org.dgx-spark-serve.tensorfold.decode-burst-patch-sha256") != burst["patch_sha256"]:
        raise SystemExit("built image patch label does not match source manifest")
    if "TF_FLASH_DECODE_BURST=4" not in child_env:
        raise SystemExit("built image does not select the documented burst-4 setting")
    parent_layers = parent.get("RootFS", {}).get("Layers", [])
    child_layers = child.get("RootFS", {}).get("Layers", [])
    if child_layers[:len(parent_layers)] != parent_layers:
        raise SystemExit("built image does not retain the selected parent rootfs layers")

    test_program = r'''
import importlib.util, io, json, pathlib, sys, unittest
modules = []
for name, path in (
    ('test_prefill_cancel', '/opt/spark-serve/cancellation/test_prefill_cancel.py'),
    ('test_decode_burst', '/opt/spark-serve/decode-burst/test_decode_burst.py'),
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
if result.testsRun != 14 or not result.wasSuccessful() or result.skipped:
    sys.exit(1)
'''
    tested = subprocess.run([
        "docker", "run", "--rm", "--network=none", "--entrypoint", "python", child["Id"],
        "-c", test_program,
    ], check=False, capture_output=True, text=True)
    if tested.returncode:
        raise SystemExit("14-test cancellation/burst CPU gate failed")
    try:
        cpu_result = json.loads(tested.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SystemExit("CPU test runner did not emit its structured result") from exc
    if not valid_cpu_result(cpu_result):
        raise SystemExit("structured cancellation/burst CPU result failed validation")

    receipt = {
        "schema_version": 1,
        "variant": BURST_VARIANT,
        "qualification": "Publicly rebuilt derivative; 14 mocked-GPU cancellation/scheduler CPU checks passed; GPU cancellation latency, speed, quality and serving qualification pending.",
        "local_parent_cancellation_image_id": parent["Id"],
        "local_image_id": child["Id"],
        "local_image_id_semantics": "daemon-store-specific; may identify config or manifest",
        "architecture": child.get("Architecture"),
        "os": child.get("Os"),
        "parent_rootfs_diff_ids": parent_layers,
        "rootfs_diff_ids": child_layers,
        "runtime_revision": RUNTIME_REVISION,
        "parent_cancellation_patch_sha256": cancellation["patch_sha256"],
        "burst_patch_sha256": burst["patch_sha256"],
        "burst_patch_manifest_sha256": sha(burst_manifest_path),
        "burst_test_source_sha256": sha(ROOT / "decode-burst" / "test_decode_burst.py"),
        "cancellation_test_source_sha256": sha(ROOT / "cancellation" / "test_prefill_cancel.py"),
        "cpu_test_result": cpu_result,
        "settings": {"TF_FLASH_DECODE_BURST": 4},
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "local_image_id": child["Id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
