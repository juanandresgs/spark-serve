#!/usr/bin/env python3
"""Download and verify the pinned Qwen EXL3 snapshot into a local directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REPOSITORY = "turboderp/Qwen3.8-Flash-Next-exl3"
REVISION = "69e33439ae950f17bcbe95c98f117d80f759ab6d"


def check(directory: Path, siblings: list[dict]) -> tuple[int, int]:
    count = total = 0
    for item in siblings:
        relative = Path(item["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("The model repository returned an unsafe path")
        path = directory / relative
        if not path.resolve(strict=True).is_relative_to(directory.resolve()):
            raise ValueError("The model repository returned a path outside the snapshot")
        before = path.stat()
        if before.st_size != item["size"]:
            raise ValueError(f"File size mismatch: {relative.as_posix()}")
        lfs = item.get("lfs") or {}
        expected = lfs.get("sha256")
        digest = hashlib.sha256() if expected else hashlib.sha1()
        if not expected:
            expected = item.get("blobId")
            if not expected:
                raise ValueError(f"The repository did not provide a verifiable digest: {relative.as_posix()}")
            digest.update(f"blob {before.st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected or path.stat().st_mtime_ns != before.st_mtime_ns:
            raise ValueError(f"Digest mismatch or file changed during verification: {relative.as_posix()}")
        count += 1
        total += before.st_size
    return count, total


def main() -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True,
                        help="absolute local directory for the model snapshot")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an already-downloaded snapshot without downloading files")
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        manifest = json.load(response)
    if manifest.get("sha") != REVISION:
        raise ValueError("Hugging Face resolved a different model revision")
    if not args.verify_only:
        snapshot_download(REPOSITORY, revision=REVISION, local_dir=directory, max_workers=4)
    siblings = manifest.get("siblings", [])
    count, total = check(directory, siblings)
    print(json.dumps({"repository": REPOSITORY, "revision": REVISION,
                      "verified_files": count, "verified_bytes": total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
