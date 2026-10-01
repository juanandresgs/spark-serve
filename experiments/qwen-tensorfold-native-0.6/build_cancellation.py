#!/usr/bin/env python3
"""Build and CPU-test a separate cancellation derivative from a local parent image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parent


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(image):
    return json.loads(output(["docker", "image", "inspect", image]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-image-id", required=True,
                        help="Exact local image ID emitted by build.py on this same host")
    parser.add_argument("--tag", default="spark-serve/qwen-tensorfold-native:0.6.0-cancel")
    parser.add_argument("--receipt", type=Path, default=Path("cancellation-build-receipt.json"))
    args = parser.parse_args()
    manifest_path = ROOT / "cancellation" / "patch-manifest.json"
    patch_path = ROOT / "cancellation" / "prefill-cancellation.patch"
    test_path = ROOT / "cancellation" / "test_prefill_cancel.py"
    manifest = json.loads(manifest_path.read_text())
    if not args.parent_image_id.startswith("sha256:"):
        raise SystemExit("parent image must be an exact local sha256 image ID")
    parent = inspect(args.parent_image_id)
    if parent["Id"] != args.parent_image_id:
        raise SystemExit("Docker resolved parent ID to a different local image")
    if sha(patch_path) != manifest["patch_sha256"]:
        raise SystemExit("cancellation patch hash does not match manifest")

    parent_ref = "spark-serve/qwen-native-parent:" + uuid.uuid4().hex
    subprocess.run(["docker", "image", "tag", args.parent_image_id, parent_ref], check=True)
    try:
        if inspect(parent_ref)["Id"] != args.parent_image_id:
            raise SystemExit("temporary parent reference does not identify requested image")
        subprocess.run([
            "docker", "build", "--pull=false", "--network=none", "--file",
            str(ROOT / "Dockerfile.cancellation"), "--build-arg", "PARENT_IMAGE=" + parent_ref,
            "--tag", args.tag, str(ROOT)
        ], check=True)
    finally:
        subprocess.run(["docker", "image", "rm", parent_ref], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    child = inspect(args.tag)
    test_output = output(["docker", "run", "--rm", "--network=none", "--entrypoint",
                          "python", child["Id"],
                          "/opt/spark-serve/cancellation/test_prefill_cancel.py"])
    rootfs = child.get("RootFS", {}).get("Layers", [])
    receipt = {
        "schema_version": 1,
        "variant": "experimental-native-prefill-cancellation",
        "qualification": "CPU App.run/scheduler regression passed; GPU cancellation latency and serving qualification pending",
        "local_parent_image_id": parent["Id"],
        "local_image_id": child["Id"],
        "local_image_id_semantics": "daemon-store-specific; may identify config or manifest",
        "architecture": child.get("Architecture"),
        "os": child.get("Os"),
        "parent_rootfs_diff_ids": parent.get("RootFS", {}).get("Layers", []),
        "rootfs_diff_ids": rootfs,
        "upstream_revision": manifest["upstream_revision"],
        "patch_sha256": manifest["patch_sha256"],
        "patch_manifest_sha256": sha(manifest_path),
        "test_source_sha256": sha(test_path),
        "cpu_test_output": test_output,
        "cpu_tests": "5 passed; real App.run callback wrapper, scheduler, socket-pair cancellation; GPU work mocked",
    }
    out = args.receipt.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(out), "local_image_id": child["Id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
