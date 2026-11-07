"""Unchanged, previously validated DPG/TIIF aggregation adapters."""
import json, math, re
from pathlib import Path

def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite metric {label}: {value}")
    return number

def collect_dpg(run_root: Path, expected_images: int) -> dict:
    output = run_root / "scoring" / "dpgbench"
    text = (output / "results.txt").read_text(encoding="utf-8")
    matches = re.findall(r"DPG-Bench score:\s*([-+0-9.eE]+)", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one DPG score, found {matches}")
    detail = output / "results_detail.txt"
    if not detail.is_file():
        raise FileNotFoundError(detail)
    image_paths = {
        line.split(", ", 1)[0]
        for line in detail.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected_grids = expected_images // 4
    if len(image_paths) != expected_grids:
        raise RuntimeError(f"DPG scored grids mismatch: {len(image_paths)} != {expected_grids}")
    return {"score": finite(matches[0], "dpgbench.score"), "grid_count": len(image_paths)}

def collect_tiif(
    run_root: Path,
    expected_short: int,
    expected_long: int,
    evaluator_model_name: str = "Z-Image-Base-cfg4-nfe50",
) -> dict:
    root = run_root / "scoring" / "tiif_short" / "raw" / evaluator_model_name
    buckets: dict[str, list[float]] = {"short": [], "long": []}
    for path in root.glob("*/*/*.json"):
        data = load_json(path)
        gt, pred = data.get("gt_answers", []), data.get("model_pred", [])
        if len(gt) != len(pred) or not gt:
            raise RuntimeError(f"Invalid TIIF raw output: {path}")
        key = "short" if path.parent.name == "short" else "long"
        buckets[key].extend(float(str(a).lower() == str(b).lower()) for a, b in zip(gt, pred))
    raw_counts = {
        key: len(list(root.glob(f"*/{key}/*.json"))) for key in ("short", "long")
    }
    if raw_counts != {"short": expected_short, "long": expected_long}:
        raise RuntimeError(f"TIIF score coverage mismatch: {raw_counts}")
    pooled = {
        key: finite(sum(values) / len(values), f"tiif.pooled_question_accuracy.{key}")
        for key, values in buckets.items()
    }
    summary_path = run_root / "scoring" / "tiif_short" / "summary.txt"
    summary = summary_path.read_text(encoding="utf-8")
    section_marker = "==== Overall average score of 9 sub-attributes ===="
    section = summary.rsplit(section_marker, 1)[-1] if section_marker in summary else ""
    rows = []
    for line in section.splitlines():
        match = re.match(
            r"^\s*(\S.*?)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$", line
        )
        if match and match.group(1) not in {"Model", "overall-short"}:
            rows.append(match)
    if len(rows) != 1:
        raise RuntimeError(f"Missing official TIIF overall scores in {summary_path}")
    match = rows[0]
    short = finite(match.group(2), "tiif.short")
    long = finite(match.group(3), "tiif.long")
    return {
        "short": short,
        "long": long,
        "long_minus_short": long - short,
        "pooled_question_accuracy": pooled,
        "raw_counts": raw_counts,
    }
