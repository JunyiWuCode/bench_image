#!/usr/bin/env python3
"""Build only the input needed by one scorer, inside that scorer's job."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from bench_image._legacy.official_layouts import materialize_official_layouts, read_records


LAYOUT_BENCHMARKS = {
    "geneval",
    "dpgbench",
    "tiif_short",
    "oneig_en",
    "t2i_corebench",
    "wise_verified",
}


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--layout-workers", type=int, default=8)
    args = parser.parse_args()

    requested = {args.benchmark}
    if args.benchmark == "tiif_short":
        requested.add("tiif_long")

    expected = [row for row in read_records(args.records) if row["benchmark"] in requested]
    generated = [
        row
        for row in read_records(args.generation_manifest)
        if row["benchmark"] in requested
    ]
    expected_ids = Counter(str(row["artifact_id"]) for row in expected)
    generated_ids = Counter(str(row["artifact_id"]) for row in generated)
    if expected_ids != generated_ids or any(count != 1 for count in generated_ids.values()):
        missing = sorted((expected_ids - generated_ids).elements())[:20]
        unexpected = sorted((generated_ids - expected_ids).elements())[:20]
        raise RuntimeError(
            f"{args.benchmark} generation mismatch: missing={missing}, unexpected={unexpected}"
        )

    input_rows = [row for row in generated if row["benchmark"] == args.benchmark]
    if not input_rows:
        raise RuntimeError(f"No generated records for {args.benchmark}")
    write_jsonl_atomic(args.input, input_rows)

    if args.benchmark in LAYOUT_BENCHMARKS:
        materialize_official_layouts(
            generated,
            args.image_root,
            args.layout_root,
            model_name=args.model_name,
            grid_workers=args.layout_workers,
        )

    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "input_rows": len(input_rows),
                "layout_rows": len(generated) if args.benchmark in LAYOUT_BENCHMARKS else 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
