#!/usr/bin/env python3
"""Run a resumable shard of the official OneIG alignment metric."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


EXPECTED_CLASS_COUNTS = {
    "anime": 245,
    "human": 244,
    "object": 206,
}


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.sort_index().to_csv(temporary)
    temporary.replace(path)


def _load_rows(oneig_root: Path, class_item: str | None) -> list[tuple[str, str, dict]]:
    rows = []
    class_items = [class_item] if class_item else list(EXPECTED_CLASS_COUNTS)
    for item_class in class_items:
        question_path = oneig_root / "scripts" / "alignment" / "Q_D" / f"{item_class}.json"
        dependency = json.loads(question_path.read_text(encoding="utf-8"))
        keys = sorted(dependency, key=lambda value: int(value))
        expected_count = EXPECTED_CLASS_COUNTS[item_class]
        if len(keys) != expected_count:
            raise RuntimeError(
                f"Unexpected {item_class} prompt count: {len(keys)} != {expected_count}"
            )
        rows.extend((item_class, key, dependency[key]) for key in keys)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oneig-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-name", default="Z-Image-Base-cfg4-nfe50")
    parser.add_argument("--class-item", choices=sorted(EXPECTED_CLASS_COUNTS))
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.all_classes == (args.class_item is not None):
        parser.error("set exactly one of --class-item or --all-classes")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")

    oneig_root = args.oneig_root.resolve()
    sys.path.insert(0, str(oneig_root))
    os.chdir(oneig_root)

    # Subset fixtures retain original IDs. Filter by present official grids before
    # sharding; the full suite still visits the exact same 695 alignment prompts.
    available=[row for row in _load_rows(oneig_root,args.class_item)
               if list((args.image_root/row[0]/args.model_name).glob(str(row[1])+'*'))]
    shard_rows = available[args.shard_index :: args.num_shards]

    if args.output.exists():
        scores = pd.read_csv(args.output, index_col=0)
        if list(scores.columns) != [args.model_name]:
            raise RuntimeError(f"Unexpected columns in resumable output {args.output}")
    else:
        scores = pd.DataFrame(columns=[args.model_name], dtype=float)

    completed = set(scores[args.model_name].dropna().index)
    expected_rows = {f"{item_class}_{key}" for item_class, key, _ in shard_rows}
    unexpected_rows = completed - expected_rows
    if unexpected_rows:
        raise RuntimeError(f"Unexpected rows in {args.output}: {sorted(unexpected_rows)[:5]}")
    if completed == expected_rows:
        print(f"Alignment shard already complete: {args.output}")
        return

    # Importing this module constructs the official Qwen2.5-VL inferencer.
    from scripts.alignment.alignment_score import alignment_score
    import megfile

    scratch_parent = os.environ.get("SLURM_TMPDIR")
    cache_dir = tempfile.mkdtemp(
        prefix=f"oneig-align-{args.class_item}-{args.shard_index}-",
        dir=scratch_parent,
    )
    try:
        for item_class, key, item in shard_rows:
            row_key = f"{item_class}_{key}"
            if row_key in completed:
                continue
            questions = item["question"]
            dependencies = item["dependency"]
            if isinstance(questions, str):
                questions = {int(k): v for k, v in json.loads(questions).items()}
            if isinstance(dependencies, str):
                dependencies = {int(k): v for k, v in json.loads(dependencies).items()}

            image_paths = megfile.smart_glob(
                str(args.image_root / item_class / args.model_name / f"{key}*")
            )
            if len(image_paths) != 1:
                raise RuntimeError(f"Expected one grid for {row_key}, found {len(image_paths)}")
            score = alignment_score(image_paths, questions, dependencies, (2, 2), cache_dir)
            if score is None:
                raise RuntimeError(f"Official alignment scorer returned None for {row_key}")
            scores.loc[row_key, args.model_name] = float(score)
            _write_csv_atomic(scores, args.output)
            print(f"alignment row={row_key} score={float(score):.8f}", flush=True)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    completed = set(scores[args.model_name].dropna().index)
    if completed != expected_rows:
        raise RuntimeError(
            f"Incomplete alignment shard: {len(completed)} != {len(expected_rows)} rows"
        )
    print(f"Alignment shard complete rows={len(completed)} output={args.output}")


if __name__ == "__main__":
    main()
