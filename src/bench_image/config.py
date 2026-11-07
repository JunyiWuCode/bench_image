"""Portable explicit runtime paths; never probe another project's installation."""
import json
import os
from pathlib import Path
import tempfile
import threading

PACKAGE=Path(__file__).resolve().parent
TOOLS=PACKAGE/'runtime/scripts'
WORKERS=PACKAGE/'_legacy/workers'
_IMPORT_CONTEXT=None
_IMPORT_LOCK=threading.Lock()


def python_namespace_root():
    """Expose only our pure Python namespace, never controller site-packages.

    Scorers use different Python/CUDA ABIs. Prepending the controller's complete
    site-packages would shadow their own NumPy/Pillow/Transformers installations.
    This node-local link lives until all synchronous children have finished.
    """
    global _IMPORT_CONTEXT
    with _IMPORT_LOCK:
        if _IMPORT_CONTEXT is None:
            _IMPORT_CONTEXT=tempfile.TemporaryDirectory(prefix='bench-image-import-')
            (Path(_IMPORT_CONTEXT.name)/'bench_image').symlink_to(PACKAGE,target_is_directory=True)
        return Path(_IMPORT_CONTEXT.name)


def cache_root():
    return Path(os.environ.get('BENCH_IMAGE_CACHE',str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'bench-image'))).expanduser().absolute()


def load_runtime(path=None):
    path=Path(path or cache_root()/'runtime.json').expanduser().absolute()
    if not path.is_file():raise FileNotFoundError(f'{path}: run bench-image setup first, or pass --runtime')
    data=json.loads(path.read_text())
    root=Path(data.get('cache_root',path.parent)).expanduser().absolute()
    def expand(value):
        if isinstance(value,str):return os.path.expanduser(value.replace('${CACHE}',str(root)))
        if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        return value
    data=expand(data);data['cache_root']=str(root);data['_file']=str(path)
    return data


def worker_environment(runtime, *,world_size,model_name):
    env=dict(os.environ)
    for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','GH_TOKEN','GITHUB_TOKEN'):env.pop(key,None)
    env.update({k:str(v) for k,v in runtime.get('environment',{}).items()})
    env.update({k:str(v) for k,v in runtime['assets'].items()})
    env['LONGTEXT_QWEN_MODEL']=runtime['assets'].get('LONGTEXT_QWEN_MODEL',runtime['assets']['ONEIG_QWEN_MODEL'])
    env.update({k:str(v) for k,v in runtime['python'].items()})
    env.update(BENCH_REPO_ROOT=str(PACKAGE/'runtime'),BENCH_WORKER_ROOT=str(WORKERS),
               BENCH_SETUP_PYTHON=runtime.get('setup_python',runtime['python']['GENERATION_PYTHON']),
               BENCH_SCORER_WORLD_SIZE=str(world_size),BENCH_EVAL_MODEL_NAME=model_name)
    env['PYTHONPATH']=str(python_namespace_root())+os.pathsep+env.get('PYTHONPATH','')
    return env
