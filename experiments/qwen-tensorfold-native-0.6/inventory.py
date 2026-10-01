#!/usr/bin/env python3
"""Emit installed Python distribution names and versions as deterministic JSON."""
import json
import re
import subprocess


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


installed = json.loads(subprocess.check_output(
    ["python", "-m", "pip", "list", "--format=json"], text=True))
items = sorted(({"name": canonical(item["name"]), "version": item["version"]}
                for item in installed), key=lambda item: item["name"])
names = [item["name"] for item in items]
if len(names) != len(set(names)):
    raise SystemExit("duplicate normalized distribution names")
print(json.dumps(items, sort_keys=True, separators=(",", ":")))
