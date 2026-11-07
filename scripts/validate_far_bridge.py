"""CPU validation of a clean FAR checkout's pinned submodule and installer bridge."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--checkout',type=Path,required=True)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.checkout.absolute();cache=a.cache.absolute()
env=dict(os.environ,BENCH_IMAGE_CACHE=str(cache))
submodule=root/'third_party/bench_image'
actual=subprocess.check_output(['git','-C',str(submodule),'rev-parse','HEAD'],text=True).strip()
index=subprocess.check_output(['git','-C',str(root),'ls-files','--stage','third_party/bench_image'],text=True).split()
if index[:2]!=['160000',actual]:raise ValueError('Benchmark submodule is not at the committed FAR pin')
for arguments in (['protocols'],['setup','--dry-run']):
    subprocess.run(['bash','scripts/image_benchmark.sh',*arguments],cwd=root,env=env,check=True)
python=cache/'controller/bin/python'
code='''import json
from pathlib import Path
import bench_image
from bench_image.config import WORKERS
from bench_image.protocols import FIELDS
worker=(WORKERS/'score_cvtg.py').read_text()
assert 'recognized_text(raw, separator=" ")' in worker
assert len(FIELDS)==14
assert 'en_PP-OCRv5_mobile_rec' in (WORKERS/'score_ocr.py').read_text()
print(json.dumps(dict(package=bench_image.__file__,workers=str(WORKERS),columns=len(FIELDS))))
'''
observed=json.loads(subprocess.check_output([str(python),'-c',code],cwd=root,env=env,text=True))
if 'site-packages' not in observed['package']:raise ValueError('Bridge did not use its installed controller package')
result=dict(complete=True,benchmark_revision=actual,observed=observed,
    far_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    scope='Clean pinned checkout, isolated package installation, packaged worker behavior, protocol CLI and setup dry-run; not a fresh ML-environment rebuild')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
