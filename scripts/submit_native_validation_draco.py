"""Optional single-node, single-GPU native-generation validation launcher."""
import argparse
import json
from pathlib import Path
import shlex
from submit_validation_draco import submit,ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt-map',type=Path)
    p.add_argument('--extra-pythonpath',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    receipt=a.output/'job.json'
    if receipt.exists():raise ValueError('Job already submitted; inspect its live/accounting state before retrying')
    runtime=json.loads(a.runtime.read_text())
    command=[runtime['python']['GENERATION_PYTHON'],'scripts/validate_native_generation.py',
        '--model',a.model,'--reference-run',str(a.reference_run.absolute()),'--output',str(a.output.absolute())]
    if a.prompt_map:command+=['--prompt-map',str(a.prompt_map.absolute())]
    prefix=''
    if a.extra_pythonpath:prefix='export PYTHONPATH='+shlex.quote(str(a.extra_pythonpath.absolute()))+':${PYTHONPATH:-}; '
    job=submit(prefix+shlex.join(command),'bench-image-native',a.output.absolute()/'job_%j.log',
        gpus=1,cpus=4,memory='80G',time='00:30:00')
    data=dict(job_id=job,command=command,single_node=True,gpus=1)
    receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)


if __name__=='__main__':main()
