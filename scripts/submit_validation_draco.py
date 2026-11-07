"""Explicit optional Draco validation recipe; library itself has no Slurm dependency."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
GPU='batch_block3,batch_block4,batch_block1,batch_singlenode,backfill_block3,backfill_block4,backfill_block1,backfill_singlenode'
CPU='cpu,cpu_short,cpu_dataprocessing,cpu_interactive'


def submit(command,name,log,*,gpus=0,cpus=4,memory='32G',dependency=None,time='00:45:00'):
    args=['sbatch','--parsable','-A',os.environ.get('BENCH_SLURM_ACCOUNT','nvr_elm_llm'),
          '-p',GPU if gpus else CPU,'--nodes=1','--ntasks=1','--cpus-per-task='+str(cpus),
          '--mem='+memory,'--time='+time,'--job-name='+name,'--chdir='+str(ROOT),'--output='+str(log)]
    if gpus:args.extend(['--gres=gpu:'+str(gpus),'--exclude=batch-block1-1069,batch-block1-1019'])
    if dependency:
        active=[]
        for job in dependency:
            state=subprocess.check_output(['sacct','-X','-j',str(job),'-n','-o','State'],text=True).strip().split()
            if state and state[0]=='COMPLETED':continue
            if state and state[0] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'):raise RuntimeError(f'Dependency {job} did not succeed')
            active.append(job)
        if active:args.append('--dependency=afterok:'+':'.join(map(str,active)))
    args.extend(['--wrap','set -euo pipefail; export TMPDIR=/tmp; export PYTHONPATH='+shlex.quote(str(ROOT/'src'))+':${PYTHONPATH:-}; '+command])
    env={k:v for k,v in os.environ.items() if k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','GH_TOKEN','GITHUB_TOKEN')}
    return int(subprocess.check_output(args,text=True,env=env).strip().split(';')[0])


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--stage',choices=['build','sources','score'],required=True)
    p.add_argument('--retry-benchmark',help='Explicitly rerun a completed scorer only after both prior scoring directories were archived')
    a=p.parse_args();work=ROOT/'validation_runs/reference-v1';logs=work/'logs';logs.mkdir(parents=True,exist_ok=True)
    runtime_path=a.runtime if a.runtime.exists() else ROOT/'validation_runs/runtime.local.json'
    runtime=json.loads(runtime_path.read_text());python=runtime['python']['GENERATION_PYTHON']
    receipt=work/'jobs.json';jobs=json.loads(receipt.read_text()) if receipt.exists() else {}
    if a.stage=='build':
        if 'build' not in jobs:
            jobs['build']=submit(shlex.join([python,'scripts/build_reference_fixture.py','--reference-run',str(a.reference_run.resolve()),
                '--output',str(work),'--workers','16']),'bench-image-fixture',logs/'build_%j.log',cpus=16,memory='64G',time='00:20:00')
    elif a.stage=='sources':
        if 'sources' not in jobs:
            fresh=work/'fresh_cache'
            command=shlex.join([python,'-m','bench_image','download','--cache-dir',str(fresh),'--only','sources,dpgbench_questions,hpsv3_official'])+' && '+shlex.join([
                python,'scripts/point_validation_sources.py','--runtime',str(a.runtime.resolve()),'--source-root',str(fresh/'sources'),
                '--output',str(ROOT/'validation_runs/runtime.fresh.local.json')])
            jobs['sources']=submit(command,'bench-image-sources',logs/'sources_%j.log',cpus=4,memory='32G',time='00:30:00')
    else:
        jobs.setdefault('scorers',{})
        changed=False
        for name in ('geneval','geneval2','dpgbench','tiif_short','cvtg','longtext_en','oneig_en','qwen_image_bench_en','hpsv3_official','wise_verified'):
            if name in jobs['scorers']:
                old=jobs['scorers'][name]
                state=subprocess.check_output(['sacct','-X','-j',str(old),'-n','-o','State'],text=True).strip().split()
                if state and state[0] in ('RUNNING','COMPLETING','CONFIGURING'):continue
                if state and state[0]=='COMPLETED':
                    if name!=a.retry_benchmark:continue
                    if any((work/directory/'scoring'/name).exists() for directory in ('run','repeat')):
                        raise ValueError('Archive completed scoring evidence before explicit rerun: '+name)
                if state and state[0]=='PENDING':
                    reason=subprocess.run(['squeue','-h','-j',str(old),'-o','%r'],text=True,capture_output=True).stdout.strip()
                    if reason!='DependencyNeverSatisfied':continue
                    subprocess.run(['scancel','--state=PENDING',str(old)],check=True)
                jobs.setdefault('replaced_scorers',[]).append(dict(benchmark=name,job_id=old,state=state))
            gpus=8 if name in ('tiif_short','wise_verified') else 1
            command=shlex.join([python,'scripts/validate_metric.py','--root',str(work),'--runtime',str(a.runtime.resolve()),
                               '--benchmark',name,'--devices',','.join(map(str,range(gpus)))])
            jobs['scorers'][name]=submit(command,'bench-image-'+name,logs/(name+'_%j.log'),gpus=gpus,cpus=4*gpus,
                memory=str(80*gpus)+'G',dependency=[jobs['build'],*([jobs['sources']] if jobs.get('sources') else [])],time='01:00:00' if name=='oneig_en' else '00:45:00')
            changed=True
            receipt.write_text(json.dumps(jobs,indent=2)+'\n')
        aggregate_state=[]
        if jobs.get('aggregate'):
            aggregate_state=subprocess.check_output(['sacct','-X','-j',str(jobs['aggregate']),'-n','-o','State'],text=True).strip().split()
        if jobs.get('aggregate') and (changed or (aggregate_state and aggregate_state[0] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'))):
            old=jobs.pop('aggregate')
            live=subprocess.run(['squeue','-h','-j',str(old),'-o','%T'],capture_output=True,text=True)
            if live.stdout.strip()=='PENDING':subprocess.run(['scancel','--state=PENDING',str(old)],check=True)
            jobs.setdefault('replaced_aggregates',[]).append(old)
            receipt.write_text(json.dumps(jobs,indent=2)+'\n')
        if 'aggregate' not in jobs:
            jobs['aggregate']=submit(shlex.join([python,'scripts/finalize_reference_fixture.py','--root',str(work)]),
                'bench-image-reference-merge',logs/'aggregate_%j.log',dependency=list(jobs['scorers'].values()))
    receipt.write_text(json.dumps(jobs,indent=2)+'\n');print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
