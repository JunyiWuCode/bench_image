#!/usr/bin/env python3
"""Validate and summarize official WISE_Verified binary judge outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


CATEGORIES = ("CULTURE", "TIME", "SPACE", "BIOLOGY", "PHYSICS", "CHEMISTRY")
WEIGHTS = {
    "CULTURE": 0.40,
    "TIME": 0.12,
    "SPACE": 0.12,
    "BIOLOGY": 0.12,
    "PHYSICS": 0.12,
    "CHEMISTRY": 0.12,
}


def category_for_id(prompt_id: int) -> str:
    if 1 <= prompt_id <= 400:
        return "CULTURE"
    if 401 <= prompt_id <= 520:
        return "TIME"
    if 521 <= prompt_id <= 640:
        return "SPACE"
    if 641 <= prompt_id <= 760:
        return "BIOLOGY"
    if 761 <= prompt_id <= 880:
        return "PHYSICS"
    if 881 <= prompt_id <= 1000:
        return "CHEMISTRY"
    raise ValueError(f"WISE_Verified prompt_id is outside 1..1000: {prompt_id}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    expected_rows = read_jsonl(args.input)
    expected_ids = [int(row["metadata"]["prompt_id"]) for row in expected_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Duplicate WISE_Verified prompt IDs in generation input")
    if not args.allow_partial and set(expected_ids) != set(range(1, 1001)):
        raise RuntimeError(
            f"A reportable WISE_Verified run requires IDs 1..1000; got {len(expected_ids)}"
        )

    scores: dict[int, float] = {}
    for path in args.score_file:
        for row in read_jsonl(path):
            prompt_id = int(row["prompt_id"])
            score = float(row["score"])
            if prompt_id in scores:
                raise RuntimeError(f"Duplicate WISE_Verified score for prompt_id={prompt_id}")
            if not math.isfinite(score) or score not in (0.0, 1.0):
                raise RuntimeError(f"Invalid binary score for prompt_id={prompt_id}: {score}")
            scores[prompt_id] = score

    if set(scores) != set(expected_ids):
        raise RuntimeError(
            "WISE_Verified score coverage mismatch: "
            f"missing={sorted(set(expected_ids) - set(scores))[:20]}, "
            f"unexpected={sorted(set(scores) - set(expected_ids))[:20]}"
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for prompt_id, score in scores.items():
        grouped[category_for_id(prompt_id)].append(score)
    category_scores = {
        category: sum(grouped[category]) / len(grouped[category])
        for category in CATEGORIES
        if grouped[category]
    }
    covered_weight = sum(WEIGHTS[category] for category in category_scores)
    weighted_sum = sum(
        WEIGHTS[category] * score for category, score in category_scores.items()
    )
    overall = weighted_sum if covered_weight == 1.0 else weighted_sum / covered_weight
    payload = {
        "benchmark": "wise_verified",
        "protocol": "WISE_Verified",
        "complete": set(scores) == set(range(1, 1001)),
        "reportable": not args.allow_partial and set(scores) == set(range(1, 1001)),
        "num_images": len(scores),
        "wiscore": overall,
        "wiscore_percent": 100.0 * overall,
        "category_scores": category_scores,
        "category_scores_percent": {
            category: 100.0 * score for category, score in category_scores.items()
        },
        "category_counts": {
            category: len(grouped[category]) for category in CATEGORIES if grouped[category]
        },
        "official_weights": WEIGHTS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
