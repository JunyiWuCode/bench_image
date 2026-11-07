#!/usr/bin/env python3
"""Inject the locked base-model snapshot into the pinned HPSv3 config."""

from __future__ import annotations

import argparse
from pathlib import Path


UPSTREAM_VALUE = 'model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    if source.count(UPSTREAM_VALUE) != 1:
        raise RuntimeError("Unexpected HPSv3 base-model config")
    model_path = args.model_path.resolve()
    if not model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        source.replace(UPSTREAM_VALUE, f'model_name_or_path: "{model_path}"'),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
