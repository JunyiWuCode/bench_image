"""Metadata-only probe, executable even before bench-image is installed."""
import importlib.metadata as metadata
import json
import platform
import sys

PACKAGES=('torch','torchvision','mmcv-full','mmdet','transformers','numpy','pandas',
          'modelscope','paddleocr','paddlex','paddlepaddle-gpu','vllm','ms-swift','flash-attn',
          'accelerate','diffusers','peft','trl','dreamsim','qwen-vl-utils','openai-clip','setuptools',
          'open-clip-torch','clip-benchmark')


def probe():
    result={}
    for name in PACKAGES:
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:pass
    return dict(python=platform.python_version(),executable=sys.executable,packages=result)


if __name__=='__main__':print(json.dumps(probe()))
