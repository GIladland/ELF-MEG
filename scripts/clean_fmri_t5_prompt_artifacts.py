#!/usr/bin/env python
"""Remove leaked lexical-prompt labels from deterministic T5 generations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from train_fmri_minilm_t5_prefix import compact


LABEL_RE = re.compile(
    r"(?i)(?:\bkey\s+phrases?\b|\bkey\s+words?\b|\bkeywords?\b|\bwords?\b)\s*:\s*"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def clean(text: str) -> str:
    return " ".join(LABEL_RE.sub("", text).split())


def main() -> None:
    args = parse_args()
    source = Path(args.metrics_json)
    payload = json.loads(source.read_text())
    targets = [str(value) for value in payload["targets"]]
    original = [str(value) for value in payload["generated"]]
    generated = [clean(value) for value in original]
    changed = sum(left != right for left, right in zip(original, generated))
    metrics = compact(generated, targets)
    output = {
        **payload,
        **metrics,
        "source_metrics_json": str(source),
        "postprocess": {
            "name": "strip_lexical_prompt_labels_v1",
            "target_free": True,
            "changed_rows": changed,
            "pattern": LABEL_RE.pattern,
        },
        "generated": generated,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"changed_rows": changed, **metrics}, indent=2))


if __name__ == "__main__":
    main()
