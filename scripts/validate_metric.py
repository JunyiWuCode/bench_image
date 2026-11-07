"""Two independent scorer passes over the same frozen reference pixels."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bench_image.config import load_runtime
from bench_image.scoring import evaluate

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--benchmark',required=True);p.add_argument('--devices',default='0')
a=p.parse_args()
for directory in ('run','repeat'):
    result=evaluate(a.root/directory,load_runtime(a.runtime),benchmarks=(a.benchmark,),
                    devices=a.devices,model_name='FLUX1-Raw2Q-3000-reference-v1')
    print(directory,a.benchmark,'completed',result['headline_metrics'],flush=True)
