#!/usr/bin/env python
"""Export target/generated pairs from an ELF metrics JSON to a compact CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    targets = payload["targets"]
    generated = payload["generated"]
    if len(targets) != len(generated):
        raise ValueError(
            f"Target/generated row mismatch: {len(targets)} != {len(generated)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "generated"])
        writer.writerows(zip(targets, generated))
    print(json.dumps({"output": str(output), "rows": len(targets)}, indent=2))


if __name__ == "__main__":
    main()
