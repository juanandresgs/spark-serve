#!/usr/bin/env python3
"""Build the pinned native runtime and write a daemon-specific image receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def run(args, *, text=True):
    return subprocess.check_output(args, text=text)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="spark-serve/qwen-tensorfold-native:0.6.0")
    parser.add_argument("--receipt", type=Path, default=Path("build-receipt.json"))
    args = parser.parse_args()
    pins = json.loads((ROOT / "pins.json").read_text())
    subprocess.run(["docker", "build", "--pull", "--tag", args.tag,
                    "--build-arg", "BASE=" + pins["base_image"],
                    "--build-arg", "TENSORFOLD_REV=" + pins["runtime_revision"],
                    str(ROOT)], check=True)
    inspected = json.loads(run(["docker", "image", "inspect", args.tag]))[0]
    image_id = inspected["Id"]
    version = run(["docker", "run", "--rm", "--entrypoint", "tensorfold", image_id, "--version"]).strip()
    if version != "tensorfold " + pins["runtime_version"]:
        raise SystemExit(f"unexpected runtime version: {version}")
    inventory_receipt = json.loads(run([
        "docker", "run", "--rm", "--entrypoint", "cat", image_id,
        "/opt/spark-serve/build-inventory-receipt.json"]))
    rootfs = inspected.get("RootFS", {}).get("Layers", [])
    receipt = {
        "schema_version": 1,
        "qualification": "source-build-only; GPU and serving behavior not qualified",
        "local_image_id": image_id,
        "local_image_id_semantics": "daemon-store-specific; may identify config or manifest",
        "repository_digests": inspected.get("RepoDigests", []),
        "architecture": inspected.get("Architecture"),
        "os": inspected.get("Os"),
        "rootfs_diff_ids": rootfs,
        "rootfs_layers_sha256": hashlib.sha256("\n".join(rootfs).encode()).hexdigest(),
        "base_image": pins["base_image"],
        "runtime_repository": pins["runtime_repository"],
        "runtime_revision": pins["runtime_revision"],
        "runtime_version": version,
        "model_repository": pins["model_repository"],
        "model_revision": pins["model_revision"],
        "model_branch": pins["model_branch"],
        "inventory_check": inventory_receipt,
        "source_sha256": {name: sha(ROOT / name) for name in
                          ("Dockerfile", "build.py", "inventory.py", "verify_runtime.py",
                           "pins.json", "runtime-constraints.txt", "expected-base-inventory.json",
                           "expected-primary-inventory.json")},
        "identity_note": "Local image IDs are daemon-store-specific. Compare repository digests and rootfs diff IDs across stores; do not infer a config digest from `.Id`.",
    }
    output = args.receipt.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(output), "local_image_id": image_id,
                      "runtime_version": version}, sort_keys=True))


if __name__ == "__main__":
    main()
