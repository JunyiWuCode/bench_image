#!/usr/bin/env python3
"""Create an audited OneIG script tree with every indirect model pinned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MCGILL_CODE_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected OneIG source occurrence count for {old!r}")
    return text.replace(old, new)


def _require_snapshot(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.joinpath("config.json").is_file():
        raise FileNotFoundError(resolved / "config.json")
    return str(resolved)


def _link_children(source: Path, output: Path, excluded: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        if child.name not in excluded:
            output.joinpath(child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())


def _isolated_component_cache(text: str, component: str) -> str:
    text = _replace_once(text, "import datetime", "import datetime\nimport tempfile")
    return _replace_once(
        text,
        '    cache_dir = f"tmp_{formatted_time}"',
        '    cache_dir = tempfile.mkdtemp(\n'
        f'        prefix="oneig-{component}-", dir=os.environ.get("SLURM_TMPDIR")\n'
        '    )',
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--llm2clip-model", type=Path, required=True)
    parser.add_argument("--llm2clip-llm", type=Path, required=True)
    parser.add_argument("--llm2vec-code", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--style-model", type=Path, required=True)
    parser.add_argument("--csd-model", type=Path, required=True)
    parser.add_argument("--style-clip-model", type=Path, required=True)
    args = parser.parse_args()

    source_scripts = args.source.resolve() / "scripts"
    inference = _source_text(source_scripts / "utils" / "inference.py")
    style = _source_text(source_scripts / "style" / "style_score.py")
    llm2vec_modeling = _source_text(args.llm2vec_code)
    marker = args.output / "pin_manifest.json"
    requested = {
        "source": str(source_scripts),
        "qwen_model": _require_snapshot(args.qwen_model),
        "llm2clip_model": _require_snapshot(args.llm2clip_model),
        "llm2clip_llm": _require_snapshot(args.llm2clip_llm),
        "llm2vec_code": str(args.llm2vec_code.resolve()),
        "clip_model": _require_snapshot(args.clip_model),
        "style_model": _require_snapshot(args.style_model),
        "csd_model": str(args.csd_model.resolve(strict=True)),
        "style_clip_model": str(args.style_clip_model.resolve(strict=True)),
        "mcgill_code_revision": MCGILL_CODE_REVISION,
        "component_cache_layout": "isolated_node_local_v1",
    }
    if args.output.exists():
        if marker.is_file():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if all(existing.get(key) == value for key, value in requested.items()):
                print(args.output)
                return 0
        raise RuntimeError(f"Refusing to replace existing OneIG worktree: {args.output}")

    component_scripts = {
        "text": "text_score.py",
        "reasoning": "reasoning_score.py",
        "style": "style_score.py",
        "diversity": "diversity_score.py",
    }
    _link_children(source_scripts, args.output, {"utils", *component_scripts})
    _link_children(source_scripts / "utils", args.output / "utils", {"inference.py", "CSD_config.py"})
    csd_config=_replace_once(_source_text(source_scripts / "utils" / "CSD_config.py"),
        'clip.load("scripts/style/models/ViT-L-14.pt")',
        f'clip.load({requested["style_clip_model"]!r})')
    (args.output / "utils" / "CSD_config.py").write_text(csd_config,encoding="utf-8")
    patched_components = {}
    for component, script_name in component_scripts.items():
        _link_children(
            source_scripts / component,
            args.output / component,
            {script_name},
        )
        component_text = (
            style
            if component == "style"
            else (source_scripts / component / script_name).read_text(encoding="utf-8")
        )
        patched_components[component] = _isolated_component_cache(component_text, component)

    replacements = {
        'model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"': f'model_path: str = {requested["qwen_model"]!r}',
        'processor_model="openai/clip-vit-large-patch14-336"': f'processor_model={requested["clip_model"]!r}',
        'model_name="microsoft/LLM2CLIP-Openai-L-14-336"': f'model_name={requested["llm2clip_model"]!r}',
        'llm_model_name="microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"': f'llm_model_name={requested["llm2clip_llm"]!r}',
        'pretrained_path: str = "xingpng/OneIG-StyleEncoder"': f'pretrained_path: str = {requested["style_model"]!r}',
        'from qwen_vl_utils import process_vision_info': (
            'from qwen_vl_utils import process_vision_info\n'
            'from .modeling_llama_encoder import LlamaEncoderModel'
        ),
        'self.llm_model_name, trust_remote_code=True\n        )': (
            'self.llm_model_name, attn_implementation="flash_attention_2"\n        )'
        ),
        'self.llm_model = AutoModel.from_pretrained(\n'
        '            self.llm_model_name, torch_dtype=torch.bfloat16, config=self.config, trust_remote_code=True\n'
        '        )': (
            'self.llm_model = LlamaEncoderModel.from_pretrained(\n'
            '            self.llm_model_name, torch_dtype=torch.bfloat16, config=self.config\n'
            '        )'
        ),
    }
    for old, new in replacements.items():
        inference = _replace_once(inference, old, new)
    (args.output / "utils" / "inference.py").write_text(inference, encoding="utf-8")

    llm2vec_modeling = _replace_once(
        llm2vec_modeling,
        "from transformers.modeling_layers import GradientCheckpointingLayer",
        "GradientCheckpointingLayer = nn.Module  # Transformers 4.50 compatibility",
    )
    llm2vec_modeling = _replace_once(
        llm2vec_modeling,
        '        if not is_transformers_attn_greater_or_equal_4_56_2():\n'
        '            raise ValueError(\n'
        '                "The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.56.2"\n'
        '            )\n',
        "",
    )
    (args.output / "utils" / "modeling_llama_encoder.py").write_text(
        llm2vec_modeling, encoding="utf-8"
    )

    style = _replace_once(
        patched_components["style"],
        'SEStyleEmbedding(pretrained_path="xingpng/OneIG-StyleEncoder")',
        f'SEStyleEmbedding(pretrained_path={requested["style_model"]!r})',
    )
    style = _replace_once(style,
        'CSDStyleEmbedding(model_path="scripts/style/models/checkpoint.pth")',
        f'CSDStyleEmbedding(model_path={requested["csd_model"]!r})')
    (args.output / "style" / "style_score.py").write_text(style, encoding="utf-8")
    for component in ("text", "reasoning", "diversity"):
        script_name = component_scripts[component]
        (args.output / component / script_name).write_text(
            patched_components[component], encoding="utf-8"
        )
    marker.write_text(json.dumps(requested, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
