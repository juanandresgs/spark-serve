#!/usr/bin/env python3
"""Build the optional grammar image from a caller-selected local parent image ID."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

ROOT = Path(__file__).resolve().parent


def output(args):
    return subprocess.check_output(args, text=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(image):
    return json.loads(output(["docker", "image", "inspect", image]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-image-id", required=True,
                        help="exact local image ID from the separately built native runtime")
    parser.add_argument("--tag", default="spark-serve/qwen-tensorfold-native:0.6.0-grammar")
    parser.add_argument("--receipt", type=Path, default=Path("grammar-build-receipt.json"))
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.parent_image_id):
        raise SystemExit("--parent-image-id must be an exact local sha256 image ID")
    parent = inspect(args.parent_image_id)
    if parent.get("Id") != args.parent_image_id:
        raise SystemExit("Docker did not resolve the requested parent image ID exactly")
    keep_temporary_tag = not parent.get("RepoTags")

    # Dockerfile FROM requires a named reference. Bind a unique throwaway tag to
    # the inspected immutable local ID and verify the binding immediately before build.
    temp_tag = "spark-serve/qwen-native-parent:build-" + uuid.uuid4().hex[:12]
    subprocess.run(["docker", "image", "tag", args.parent_image_id, temp_tag], check=True)
    try:
        if inspect(temp_tag).get("Id") != args.parent_image_id:
            raise SystemExit("temporary parent tag does not resolve to the requested image ID")
        subprocess.run([
            "docker", "build", "--pull=false", "--file", str(ROOT / "Dockerfile.grammar"),
            "--tag", args.tag, "--build-arg", "PARENT_IMAGE=" + temp_tag,
            "--build-arg", "PARENT_LOCAL_IMAGE_ID=" + args.parent_image_id, str(ROOT),
        ], check=True)
    finally:
        if not keep_temporary_tag:
            subprocess.run(["docker", "image", "rm", temp_tag], check=False, stdout=subprocess.DEVNULL)

    child = inspect(args.tag)
    labels = child.get("Config", {}).get("Labels") or {}
    if labels.get("org.spark-serve.parent-local-image-id") != args.parent_image_id:
        raise SystemExit("grammar image parent label does not match requested parent image ID")
    inventory_receipt = json.loads(output([
        "docker", "run", "--rm", "--entrypoint", "cat", child["Id"],
        "/opt/spark-serve/grammar-inventory-receipt.json"]))
    layers = child.get("RootFS", {}).get("Layers", [])
    receipt = {
        "schema_version": 1,
        "qualification": "optional grammar package build only; API/tool grammar behavior not qualified",
        "parent_local_image_id": args.parent_image_id,
        "parent_repository_digests": parent.get("RepoDigests", []),
        "parent_rootfs_diff_ids": parent.get("RootFS", {}).get("Layers", []),
        "child_local_image_id": child["Id"],
        "child_repository_digests": child.get("RepoDigests", []),
        "child_rootfs_diff_ids": layers,
        "child_rootfs_layers_sha256": hashlib.sha256("\n".join(layers).encode()).hexdigest(),
        "runtime_revision": "c4646171139ee8a3c38103eaa1699dad226ec12b",
        "grammar_constraints_sha256": sha(ROOT / "grammar-constraints.txt"),
        "expected_inventory_sha256": sha(ROOT / "expected-grammar-inventory.json"),
        "inventory_check": inventory_receipt,
        "source_sha256": {name: sha(ROOT / name) for name in
                          ("Dockerfile.grammar", "build_grammar.py", "inventory.py", "verify_runtime.py",
                           "grammar-requirements.txt", "grammar-constraints.txt",
                           "expected-primary-inventory.json", "expected-grammar-inventory.json")},
        "identity_note": "Local image IDs are daemon-store-specific; compare repository and rootfs provenance across stores.",
    }
    output_path = args.receipt.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(output_path), "child_local_image_id": child["Id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
