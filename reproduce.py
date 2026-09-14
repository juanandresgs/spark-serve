#!/usr/bin/env python3
"""Fetch exact recipe sources or prepare E3 build inputs; never deploy a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def fetch(root, name, output):
    source = json.loads((root / "sources.json").read_text())["sources"][name]
    url, revision = source["url"], source["revision"]
    if not re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url):
        raise ValueError("Source must be a public HTTPS GitHub repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Source must name a full immutable revision")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    # No source hooks or repository scripts are executed; disable checkout filters.
    run(["git", "init", "--quiet", str(output)])
    run(["git", "-C", str(output), "remote", "add", "origin", url])
    run(["git", "-C", str(output), "fetch", "--quiet", "--depth=1", "origin", revision])
    run(["git", "-c", "core.hooksPath=/dev/null", "-c", "filter.lfs.required=false",
         "-c", "filter.lfs.smudge=", "-C", str(output), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    actual = run(["git", "-C", str(output), "rev-parse", "HEAD"])
    if actual != revision:
        raise ValueError("Fetched source revision does not match the pin")
    hashes = {}
    for rel, expected in source.get("files", {}).items():
        path = output / rel
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Unsafe source file")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected:
            raise ValueError(f"Source digest mismatch: {rel}")
        hashes[rel] = actual_hash
    return {"source": name, "revision": actual, "verified_files": hashes,
            "build_executed": False, "hardware_qualified": False}


def prepare_e3(root, source, integration, selfcheck, output):
    result = run([sys.executable, str(root / "cluster/prepare_orchestrator_e3_inputs.py"),
                  "--source", str(source.absolute()), "--e2-integration", str(integration.absolute()),
                  "--e2-selfcheck", str(selfcheck.absolute()), "--output", str(output.absolute())])
    return {"files": json.loads(result), "build_executed": False,
            "next": "Compile on an idle compatible Spark with the pinned E2 image; see REPRODUCTION.md"}


def prepare_e2(root, source, output):
    pins = json.loads((root / "sources.json").read_text())
    expected = pins["sources"]["glm-e2"]["revision"]
    source = source.resolve()
    if run(["git", "-C", str(source), "rev-parse", "HEAD"]) != expected:
        raise ValueError("E2 checkout revision differs from the pin")
    if run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("E2 checkout has tracked modifications")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    names = run(["git", "-C", str(source), "ls-files", "-z"]).split("\0")
    for name in filter(None, names):
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Unsafe source file: {name}")
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        dest.chmod(path.stat().st_mode & 0o777)
    patch = root / "cluster/mia-flash/exl3-e2-pinned-integration.patch"
    # Apply only the checked-in project patch; no upstream build script is executed.
    # git apply inside a nested non-repository can silently skip every patch.
    # patch operates explicitly on this directory, independent of any parent Git tree.
    run(["patch", "--batch", "--fuzz=0", "--dry-run", "-p1", "-i", str(patch.absolute())], cwd=output)
    run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch.absolute())], cwd=output)
    for rel, expected_hash in pins["sources"]["glm-e2"].get("patched_files", {}).items():
        if hashlib.sha256((output / rel).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Patched E2 output differs from the recorded source: {rel}")
    files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob("*")) if p.is_file()}
    # Original site stamp incorporated absolute paths. This portable stamp is relative.
    stamp = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    image = pins["image_reproduction"]
    argv = ["docker", "build", "--pull=false", "--build-arg", "BASE=" + image["E2_base"],
            "--build-arg", "EXLLAMAV3_COMMIT=" + image["exllamav3_revision"],
            "--build-arg", "GLM53_RECIPE_STAMP=" + stamp,
            "--tag", "dgx-local/glm53-exl3-e2:eb0469f-905c0293-c5d9c657", str(output)]
    receipt = {"source_revision": expected, "files": files, "portable_stamp": stamp,
               "build_argv": argv, "build_executed": False,
               "note": "Relative source stamp changes image metadata; this is not the historical image identity. Requalify the rebuilt image."}
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fetch", help="download exact upstream source, without executing it")
    f.add_argument("source")
    f.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("prepare-e2", help="assemble a pinned E2 Docker context without compiling")
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("prepare-e3", help="verify input hashes and assemble an offline build context")
    e.add_argument("--source", type=Path, required=True)
    e.add_argument("--e2-integration", type=Path, required=True)
    e.add_argument("--e2-selfcheck", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "fetch":
        result = fetch(a.root, a.source, a.output)
    elif a.action == "prepare-e2":
        result = prepare_e2(a.root, a.source, a.output)
    else:
        result = prepare_e3(a.root, a.source, a.e2_integration, a.e2_selfcheck, a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
