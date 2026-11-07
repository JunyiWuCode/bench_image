#!/usr/bin/env python3
"""Validate and collect the five official OneIG-EN metrics."""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


MODEL_NAME = "Z-Image-Base-cfg4-nfe50"
ALIGNMENT_COUNTS = {"anime": 245, "human": 244, "object": 206}


def _latest(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} under {root}")
    return matches[-1]


def _single_metric(path: Path, column: str, model_name: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((row for row in rows if model_name in row.values()), None)
    if row is None:
        raise RuntimeError(f"No row for {model_name} in {path}")
    return float(row[column])


def collect_alignment(results_root: Path, model_name: str, expected_shards: int, image_root=None) -> tuple[float, Path]:
    # Resumed evaluations can retain complete shards from an earlier world
    # size. Select only the shard family requested by this invocation so
    # unrelated historical CSVs do not invalidate an otherwise exact run.
    shard_paths = sorted(
        (results_root / "far_alignment_shards").glob(
            f"global-shard-*-of-{expected_shards:02d}.csv"
        )
    )
    if len(shard_paths) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} alignment shard files, found {len(shard_paths)}"
        )
    frames = [pd.read_csv(path, index_col=0) for path in shard_paths]
    merged = pd.concat(frames).sort_index()
    if merged.index.duplicated().any():
        raise RuntimeError("Duplicate rows found across alignment shards")
    if list(merged.columns) != [model_name]:
        raise RuntimeError(f"Unexpected alignment columns: {list(merged.columns)}")
    expected_rows = {
        f"{class_item}_{index:03d}"
        for class_item, count in ALIGNMENT_COUNTS.items()
        for index in range(count)
    }
    if image_root is not None:
        expected_rows={f'{kind}_{path.stem}' for kind in ALIGNMENT_COUNTS
                       for path in (image_root/kind/model_name).glob('*.webp')}
        if not expected_rows:raise ValueError('No selected OneIG alignment inputs')
    actual_rows = set(merged.index)
    if actual_rows != expected_rows:
        missing = sorted(expected_rows - actual_rows)
        extra = sorted(actual_rows - expected_rows)
        raise RuntimeError(f"Alignment row mismatch missing={missing[:5]} extra={extra[:5]}")
    if merged[model_name].isna().any():
        raise RuntimeError("Alignment shards contain null scores")

    prompt_output = results_root / "alignment_prompt_score_EN_FAR_merged.csv"
    merged.to_csv(prompt_output)
    score = float(merged[model_name].mean())
    pd.DataFrame({"alignment": [score]}, index=[model_name]).to_csv(
        results_root / "alignment_score_EN_FAR_merged.csv"
    )
    return score, prompt_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--alignment-shards", type=int, default=6)
    parser.add_argument('--subset-image-root',type=Path)
    args = parser.parse_args()

    alignment, alignment_file = collect_alignment(
        args.results_root, args.model_name, args.alignment_shards,args.subset_image_root
    )
    metric_files = {
        "text": _latest(args.results_root, "text_score_EN_*.csv"),
        "reasoning": _latest(args.results_root, "reasoning_score_EN_*.csv"),
        "style": _latest(args.results_root, "style_score_EN_*.csv"),
        "diversity": _latest(args.results_root, "diversity_score_EN_*.csv"),
    }
    scores = {
        "alignment": alignment,
        "text": _single_metric(metric_files["text"], "text score", args.model_name),
        "reasoning": _single_metric(metric_files["reasoning"], "reasoning", args.model_name),
        "style": _single_metric(metric_files["style"], "style", args.model_name),
        "diversity": _single_metric(metric_files["diversity"], "total average", args.model_name),
    }
    scores["overall"] = sum(scores.values()) / len(scores)
    result = {
        "benchmark": "OneIG-Bench-EN",
        "model": args.model_name,
        "scores": scores,
        "artifacts": {
            "alignment": str(alignment_file),
            **{name: str(path) for name, path in metric_files.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
