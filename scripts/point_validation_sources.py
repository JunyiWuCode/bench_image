"""Validate new public source checkouts while reusing existing isolated weights/envs."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--setup-python',type=Path)
a=p.parse_args();value=json.loads(a.runtime.read_text());source=a.source_root.resolve()
assert all((source/name).is_dir() for name in ('geneval','geneval2','dpgbench','tiif','oneig_en','hpsv3_scorer','wise_verified','qwen_image_bench','mmdetection'))
value['source_root']=str(source)
value['assets']['QWEN_IMAGE_BENCH_ROOT']=str(source/'qwen_image_bench')
value['assets']['GENEVAL_MODEL_CONFIG']=str(source/'mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')
value['assets']['LONGTEXT_QWEN_MODEL']=value['assets']['ONEIG_QWEN_MODEL']
if a.setup_python:value['setup_python']=str(a.setup_python.absolute())  # Preserve the venv, never resolve its interpreter symlink.
a.output.write_text(json.dumps(value,indent=2)+'\n');print('Fresh-source validation runtime:',a.output)
