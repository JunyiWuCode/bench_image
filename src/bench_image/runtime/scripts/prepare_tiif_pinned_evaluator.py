#!/usr/bin/env python3
"""Make the pinned TIIF evaluator's sample-index filter partial-safe."""

from __future__ import annotations

import argparse
from pathlib import Path


UPSTREAM_LOOKUP = "line_indices = sample_idx[attr_type]"
PARTIAL_SAFE_LOOKUP = "line_indices = sample_idx.get(attr_type, [])"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deterministic-fixture", action="store_true",
        help="Officially supported temperature=0 and stable per-question template choice for diagnostic replay only")
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    if source.count(UPSTREAM_LOOKUP) != 1:
        raise RuntimeError("Expected exactly one TIIF sample-index lookup")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source=source.replace(UPSTREAM_LOOKUP, PARTIAL_SAFE_LOOKUP)
    if args.deterministic_fixture:
        replacements={
            'temperature=1.0 # You may set it to 0 if you require stricter reproducibility.':
                'temperature=0.0, seed=0 # Deterministic diagnostic replay; full protocol keeps upstream temperature=1.',
            'random.choice([raw_prompt, raw_prompt_1, raw_prompt_2])':
                'random.Random("fixture-v1:" + raw_prompt + repr(questions)).choice([raw_prompt, raw_prompt_1, raw_prompt_2])',
        }
        for old,new in replacements.items():
            if source.count(old)!=1:raise RuntimeError('Unexpected TIIF source: '+old)
            source=source.replace(old,new)
    args.output.write_text(source, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
