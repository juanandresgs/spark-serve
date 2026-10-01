#!/usr/bin/env python3
"""Curate only numeric summaries from a sanitized speed-replay receipt."""
import argparse
import json
import math
from pathlib import Path
import statistics


def stats(values):
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95_nearest_rank": sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)],
    }


def curate(data):
    cells = []
    measured = data.get("measured_groups", [])
    protocol = data.get("protocol", {})
    reps_expected = protocol.get("measured_repetitions_primary")
    for clients, kind in ((1, "code"), (1, "prose"), (4, "code"), (4, "prose")):
        selected = sorted(
            (group for group in measured
             if group.get("clients") == clients and group.get("kind") == kind),
            key=lambda group: group.get("rep", -1),
        )
        observed_reps = [group.get("rep") for group in selected]
        complete = len(selected) == reps_expected and all(
            group.get("speed_eligible") is True and len(group.get("requests", [])) == 4 and
            isinstance(group.get("aggregate_output_tokens_per_second"), (int, float))
            for group in selected
        ) and observed_reps == list(range(reps_expected or 0))
        reasons = sorted({
            reason for group in selected for reason in group.get("ineligible_reasons", [])
        })
        if len(selected) != reps_expected:
            reasons.append(f"expected {reps_expected} measured groups, found {len(selected)}")
        if observed_reps != list(range(reps_expected or 0)):
            reasons.append("measured repetition indices are missing, duplicated, or outside the expected range")
        cells.append({
            "clients": clients,
            "workload": kind,
            "repetitions_expected": reps_expected,
            "repetitions_observed": len(selected),
            "eligible": complete,
            "ineligible_reasons": [] if complete else sorted(set(reasons)),
            "group_aggregate_output_tokens_per_second": stats(
                [group.get("aggregate_output_tokens_per_second") for group in selected]
            ) if complete else None,
            "per_request_decode_proxy_tokens_per_second": stats(
                [request.get("decode_tokens_per_second_proxy")
                 for group in selected for request in group.get("requests", [])]
            ) if complete else None,
            "per_request_first_output_event_seconds": stats(
                [request.get("first_output_event_seconds")
                 for group in selected for request in group.get("requests", [])]
            ) if complete else None,
        })
    return {
        "summary_version": "portable-speed-summary-v1",
        "warmup_groups_excluded": True,
        "measured_request_count": sum(
            len(group.get("requests", [])) for group in measured
        ),
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.receipt.read_text())
    result = curate(data)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(output), "complete_cells": sum(c["eligible"] for c in result["cells"])}))


if __name__ == "__main__":
    main()
