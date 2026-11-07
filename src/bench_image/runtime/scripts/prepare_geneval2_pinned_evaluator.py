#!/usr/bin/env python3
"""Inject the exact locked Qwen snapshot into the pinned GenEval2 source."""

from __future__ import annotations

import argparse
from pathlib import Path


UPSTREAM_MODEL_LITERAL = '"Qwen/Qwen3-VL-8B-Instruct"'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    if source.count(UPSTREAM_MODEL_LITERAL) != 2:
        raise RuntimeError("Expected exactly two GenEval2 Qwen model literals")
    model_path = args.model_path.resolve()
    if not model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        source.replace(UPSTREAM_MODEL_LITERAL, repr(str(model_path))), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
