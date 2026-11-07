import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _print(value):print(json.dumps(value,ensure_ascii=False,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description='Portable open image benchmarks: generate, then evaluate.')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('protocols')
    for name in ('setup','download','doctor'):
        child=sub.add_parser(name);child.add_argument('--cache-dir',type=Path)
        child.add_argument('--runtime',type=Path)
        child.add_argument('--dry-run',action='store_true')
        child.add_argument('--gpu',action='store_true')
        child.add_argument('--only',help='Comma-separated environment or resource names')
    child=sub.add_parser('samples');child.add_argument('--runtime',type=Path);child.add_argument('--source-root',type=Path)
    child.add_argument('--output',type=Path,required=True);child.add_argument('--resolution',type=int,default=1024)
    child.add_argument('--seed',type=int,default=0);child.add_argument('--benchmarks');child.add_argument('--selection',type=Path)
    child=sub.add_parser('generate');child.add_argument('--records',type=Path,required=True)
    child.add_argument('--output',type=Path,required=True);child.add_argument('--model',required=True)
    child.add_argument('--revision');child.add_argument('--steps',type=int,default=4);child.add_argument('--guidance',type=float)
    child.add_argument('--prompt-map',type=Path);child.add_argument('--allow-raw-flux3',action='store_true')
    child.add_argument('--trust-remote-code',action='store_true');child.add_argument('--shard-id',type=int,default=0)
    child.add_argument('--num-shards',type=int,default=1)
    child=sub.add_parser('pup');child.add_argument('--records',type=Path,required=True);child.add_argument('--model',required=True)
    child.add_argument('--output',type=Path,required=True);child.add_argument('--revision');child.add_argument('--batch-size',type=int,default=4)
    child=sub.add_parser('evaluate');child.add_argument('--run',type=Path,required=True);child.add_argument('--runtime',type=Path)
    child.add_argument('--benchmarks');child.add_argument('--devices',default='0');child.add_argument('--model-name',default='model')
    child=sub.add_parser('collect');child.add_argument('--run',type=Path,required=True)
    child=sub.add_parser('example');child.add_argument('--archive',type=Path);child.add_argument('--output',type=Path,required=True)
    child=sub.add_parser('verify');child.add_argument('--expected',type=Path,required=True);child.add_argument('--actual',type=Path,required=True)
    child.add_argument('--atol',type=float)
    child=sub.add_parser('run');child.add_argument('--config',type=Path,required=True);child.add_argument('--runtime',type=Path)
    child.add_argument('--output',type=Path,required=True);child.add_argument('--devices',default='0');child.add_argument('--generate-only',action='store_true')
    args=parser.parse_args(argv)
    from .protocols import BENCHMARKS,FIELDS,generation_samples,write_jsonl
    from .config import load_runtime
    if args.command=='protocols':
        from ._legacy.official_suite import OFFICIAL_PROTOCOLS
        _print(dict(benchmarks={n:dict(prompts=OFFICIAL_PROTOCOLS[n].prompts,
            samples_per_prompt=OFFICIAL_PROTOCOLS[n].samples_per_prompt,images=OFFICIAL_PROTOCOLS[n].images) for n in BENCHMARKS},
            headline_columns=list(FIELDS),full_images=33468));return
    if args.command in ('setup','download','doctor'):
        from .setup import setup,download,doctor
        if args.command=='doctor':value=doctor(args.runtime,gpu=args.gpu)
        elif args.command=='download':value=download(args.cache_dir,dry_run=args.dry_run,only=args.only)
        else:value=setup(args.cache_dir,dry_run=args.dry_run,only=args.only)
        _print(value);return
    if args.command=='samples':
        source=args.source_root or Path(load_runtime(args.runtime)['source_root'])
        rows=generation_samples(source_root=source,resolution=args.resolution,seed=args.seed,
            benchmarks=args.benchmarks.split(',') if args.benchmarks else BENCHMARKS,
            selection=json.loads(args.selection.read_text()) if args.selection else None)
        write_jsonl(args.output,rows);_print(dict(records=len(rows),output=str(args.output)));return
    if args.command=='generate':
        from .generation import generate
        _print(generate(args.records,args.output,args.model,revision=args.revision,steps=args.steps,guidance=args.guidance,
            prompt_map=args.prompt_map,allow_raw_flux3=args.allow_raw_flux3,trust_remote_code=args.trust_remote_code,
            shard_id=args.shard_id,num_shards=args.num_shards));return
    if args.command=='pup':
        from .pup import generate_pup
        _print(generate_pup(args.records,args.model,args.output,revision=args.revision,batch_size=args.batch_size));return
    if args.command=='evaluate':
        from .scoring import evaluate
        _print(evaluate(args.run,load_runtime(args.runtime),devices=args.devices,
            benchmarks=args.benchmarks.split(',') if args.benchmarks else None,model_name=args.model_name));return
    if args.command=='collect':
        from .scoring import write_summary
        _print(write_summary(args.run));return
    if args.command=='example':
        from .fixtures import extract_example
        _print(extract_example(args.output,args.archive));return
    if args.command=='verify':
        from .fixtures import verify_reference
        result=verify_reference(args.expected,args.actual,atol=args.atol);_print(result)
        if not result['passed']:raise SystemExit(1)
        return
    if args.command=='run':
        # Generation and scoring are distinct processes/environments: no training
        # or generation weights remain resident while isolated scorers execute.
        config=json.loads(args.config.read_text());runtime=load_runtime(args.runtime)
        if not config.get('trust_remote_code',False):raise ValueError('Review the native model code and set trust_remote_code=true in the config')
        root=args.output.absolute();root.mkdir(parents=True,exist_ok=True)
        records=root/'records.jsonl'
        rows=generation_samples(source_root=runtime['source_root'],resolution=config.get('resolution',1024),
            seed=config.get('seed',0),benchmarks=config.get('benchmarks',BENCHMARKS),selection=config.get('selection'))
        if records.exists():
            from .protocols import read_jsonl
            if read_jsonl(records)!=rows:raise ValueError('Run uses different records')
        else:write_jsonl(records,rows)
        command=[runtime['python']['GENERATION_PYTHON'],'-m','bench_image','generate','--records',str(records),
                 '--output',str(root),'--model',config['model'],'--steps',str(config.get('steps',4)),'--trust-remote-code']
        for key,flag in [('revision','--revision'),('guidance','--guidance'),('prompt_map','--prompt-map')]:
            if config.get(key) is not None:command.extend([flag,str(config[key])])
        # The controller and generation interpreter deliberately live in different
        # environments. Make this installed package visible without reinstalling
        # scorer dependencies into the generation environment.
        from .config import python_namespace_root
        env=dict(os.environ)
        env['PYTHONPATH']=str(python_namespace_root())+os.pathsep+env.get('PYTHONPATH','')
        subprocess.run(command,check=True,env=env)
        if not args.generate_only:
            from .scoring import evaluate,write_summary
            evaluate(root,runtime,devices=args.devices,model_name=config.get('name','model'))
            _print(write_summary(root))
