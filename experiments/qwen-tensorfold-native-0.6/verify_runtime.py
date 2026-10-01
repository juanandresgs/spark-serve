#!/usr/bin/env python3
"""Check expected runtime packages, declared dependencies, and source identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import importlib.metadata


def read_inventory(path):
    rows = json.loads(Path(path).read_text())
    return {row["name"]: row["version"] for row in rows}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_constraints(path):
    pins = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!-]*)", line)
        if not match:
            raise ValueError(f"constraint is not a portable exact pin: {line!r}")
        name, version = canonical(match.group(1)), match.group(2)
        if name in pins:
            raise ValueError(f"duplicate constraint for {name}")
        pins[name] = version
    return pins


def verify_declared_dependencies(runtime):
    """Ensure every active TensorFold requirement is present at a satisfying version."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    distribution = importlib.metadata.distribution("tensorfold")
    installed = runtime
    environment = default_environment()
    missing = []
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonical(requirement.name)
        version = installed.get(name)
        if version is None or (requirement.specifier and not requirement.specifier.contains(version, prereleases=True)):
            missing.append({"requirement": raw, "installed": version})
    if missing:
        raise SystemExit("unsatisfied TensorFold runtime dependencies: " + json.dumps(missing, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--expected-base", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-dependencies", action="store_true")
    args = parser.parse_args()
    base, runtime = read_inventory(args.base), read_inventory(args.runtime)
    pins = parse_constraints(args.constraints)
    expected = read_inventory(args.expected)
    expected_base = read_inventory(args.expected_base)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("runtime revision must be a full git commit")
    if base != expected_base:
        raise SystemExit("base package inventory differs from pinned expected base")
    changed = {name for name in set(base) | set(runtime) if base.get(name) != runtime.get(name)}
    mismatches = {name: {"expected": version, "actual": runtime.get(name)}
                  for name, version in expected.items() if runtime.get(name) != version}
    mismatches.update({name: {"expected": "absent", "actual": version}
                       for name, version in runtime.items() if name not in expected})
    if mismatches:
        raise SystemExit("built package inventory differs from tested runtime: " + json.dumps(mismatches, sort_keys=True))
    expected_constraints = {name: version for name, version in expected.items() if name != "tensorfold"}
    if pins != expected_constraints:
        raise SystemExit("constraints must exactly match tested inventory excluding TensorFold")
    if runtime.get("tensorfold") != "0.6.0":
        raise SystemExit("TensorFold 0.6.0 distribution is not installed")
    if args.check_dependencies:
        verify_declared_dependencies(runtime)
    receipt = {
        "schema_version": 1,
        "result": "passed",
        "source_revision": args.revision,
        "tensorfold_version": runtime["tensorfold"],
        "base_distribution_count": len(base),
        "expected_base_inventory_sha256": hashlib.sha256(args.expected_base.read_bytes()).hexdigest(),
        "runtime_distribution_count": len(runtime),
        "base_to_tested_runtime_delta": {name: {"base": base.get(name), "tested_runtime": expected.get(name)}
                                         for name in sorted(changed)},
        "runtime_matches_tested_inventory": True,
        "declared_dependencies_satisfied": args.check_dependencies,
        "constraint_count": len(pins),
        "constraints_sha256": hashlib.sha256(args.constraints.read_bytes()).hexdigest(),
        "base_inventory_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "runtime_inventory_sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
    }
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
