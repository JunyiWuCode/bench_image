#!/usr/bin/env python3
"""Run the official TIIF evaluator through parallel, disjoint API clients."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def split_indices(sample_indices: dict[str, list[int]], workers: int) -> list[dict[str, list[int]]]:
    """Partition every (attribute, sample index) exactly once across workers."""
    shards: list[dict[str, list[int]]] = [{} for _ in range(workers)]
    items = [
        (attribute, int(index))
        for attribute, indices in sorted(sample_indices.items())
        for index in indices
    ]
    for position, (attribute, index) in enumerate(items):
        shard = shards[position % workers]
        shard.setdefault(attribute, []).append(index)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--eval-model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local-open-model")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("TIIF scoring requires at least one client")

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    all_sample_indices = json.loads(args.sample_indices.read_text(encoding="utf-8"))
    all_items = [
        (attribute, int(index))
        for attribute, indices in sorted(all_sample_indices.items())
        for index in indices
    ]
    selected_items = all_items[args.shard_index :: args.num_shards]
    sample_indices: dict[str, list[int]] = {}
    for attribute, index in selected_items:
        sample_indices.setdefault(attribute, []).append(index)
    shards = split_indices(sample_indices, args.workers)
    observed = {
        (attribute, index)
        for shard in shards
        for attribute, indices in shard.items()
        for index in indices
    }
    expected = {
        (attribute, int(index))
        for attribute, indices in sample_indices.items()
        for index in indices
    }
    if observed != expected or sum(len(v) for shard in shards for v in shard.values()) != len(expected):
        raise RuntimeError("TIIF client shards are incomplete or overlapping")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    for rank, shard in enumerate(shards):
        shard_path = args.work_dir / f"sample_indices_rank_{rank:02d}.json"
        shard_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_handle = (args.work_dir / f"client_{rank:02d}.log").open("wb")
        command = [
            sys.executable,
            str(args.evaluator),
            "--jsonl_dir", "data/test_eval_prompts",
            "--image_dir", str(args.image_dir),
            "--eval_model", args.eval_model,
            "--output_dir", str(args.output_dir),
            "--api_key", args.api_key,
            "--base_url", args.base_url,
            "--model", args.model,
            "--sample_idx_file", str(shard_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=args.source_root,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((rank, process, log_handle))

    failures = []
    try:
        for rank, process, log_handle in processes:
            status = process.wait()
            log_handle.close()
            if status != 0:
                failures.append((rank, status))
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    if failures:
        raise RuntimeError(f"TIIF parallel clients failed: {failures}")
    print(
        f"TIIF parallel clients complete: workers={args.workers}, samples={len(expected)}, "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
