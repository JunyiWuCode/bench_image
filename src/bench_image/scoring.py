"""Isolate incompatible scorers while preserving their native calculations."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import socket
import shlex
import subprocess
import tempfile
import time
import uuid

from .config import TOOLS, worker_environment
from .protocols import BENCHMARKS,FIELDS,read_jsonl,write_jsonl,validate_records,output_path_for_record
from ._collect_native import collect_dpg,collect_tiif
from .generation import collect_generation,_lock


def _read(path):return json.loads(Path(path).read_text())


def _run(command,env,log,*,cwd=None):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as handle:
        handle.write('\nCOMMAND '+json.dumps(list(map(str,command)))+'\n');handle.flush()
        subprocess.run(list(map(str,command)),cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)


def _atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def _compatible_configuration(first,second):
    """Allow catalog additions, never changed recorded paths or model settings.

    Installing a previously missing independent scorer may add its asset path.
    It must not invalidate unrelated completed scorers, nor conceal a changed
    asset already recorded by a scorer.
    """
    a={k:v for k,v in first.items() if k!='assets'}
    b={k:v for k,v in second.items() if k!='assets'}
    return a==b and all(first['assets'][key]==second['assets'][key]
        for key in set(first['assets']) & set(second['assets']))


def _snapshot_runner(output):
    """Freeze this attempt's executable and keep earlier failure logs intact."""
    directory=output/'logs'/('attempt_'+uuid.uuid4().hex)
    directory.mkdir(parents=True,exist_ok=False)
    text=(TOOLS/'run_zimage_official_benchmark_scorer.sh').read_text()
    text=text.replace('$OUTPUT/logs','$LOGS')
    marker='OUTPUT="$RUN_ROOT/scoring/$BENCHMARK"\n'
    if text.count(marker)!=1:raise ValueError('Unexpected scorer runner layout')
    text=text.replace(marker,marker+'LOGS='+shlex.quote(str(directory))+'\n')
    snapshot=directory/'runner.sh';snapshot.write_text(text)
    return snapshot


def _port():
    with socket.socket() as server:
        server.bind(('127.0.0.1',0));return server.getsockname()[1]


def _device_tokens(devices):
    logical=[int(value) for value in devices.split(',')] if isinstance(devices,str) else list(devices)
    if len(set(logical))!=len(logical) or not logical:raise ValueError('Duplicate or empty GPU list')
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if allocated==['']:allocated=[]
    if allocated:
        if min(logical)<0 or max(logical)>=len(allocated):raise ValueError('Requested GPUs are outside the allocated CUDA mask')
        return [allocated[i] for i in logical]
    if min(logical)<0:raise ValueError('Negative GPU ID')
    return list(map(str,logical))


def _oneig(root,source,env,tokens,*,partial):
    """Same official five components; support exact original-ID subsets as well."""
    output=root/'scoring/oneig_en';work=output/'workdir_pinned_v2'
    images=root/'scorer_layouts/oneig_en/oneig_en'
    setup=env['BENCH_SETUP_PYTHON'];python=env['ONEIG_PYTHON'];name=env['BENCH_EVAL_MODEL_NAME']
    _run([setup,TOOLS/'materialize_zimage_official_scorer.py','--records',root/'records.jsonl',
          '--generation-manifest',root/'generation_manifest.jsonl','--image-root',root,
          '--benchmark','oneig_en','--input',root/'scoring_inputs/oneig_en.jsonl',
          '--layout-root',root/'scorer_layouts/oneig_en','--model-name',name,
          '--layout-workers',str(min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','4'))))],
         env,output/'logs/layout.log')
    work.mkdir(parents=True,exist_ok=True)
    if not (work/'models').exists():(work/'models').symlink_to(Path(env['ONEIG_DREAMSIM_CACHE']).resolve(),target_is_directory=True)
    command=[python,TOOLS/'prepare_oneig_pinned_source.py','--source',source/'oneig_en','--output',work/'scripts']
    for flag,key in [('qwen-model','ONEIG_QWEN_MODEL'),('llm2clip-model','ONEIG_LLM2CLIP_MODEL'),
                     ('llm2clip-llm','ONEIG_LLM2CLIP_LLM'),('llm2vec-code','ONEIG_LLM2VEC_CODE'),
                     ('clip-model','ONEIG_CLIP_MODEL'),('style-model','ONEIG_STYLE_MODEL'),
                     ('csd-model','ONEIG_CSD_MODEL'),('style-clip-model','BENCH_CLIP_MODEL_PATH')]:command.extend(['--'+flag,env[key]])
    _run(command,env,output/'logs/pinned_source.log')
    shards=len(tokens)
    def alignment(rank):
        worker=dict(env,CUDA_VISIBLE_DEVICES=tokens[rank])
        path=work/'results/far_alignment_shards'/f'global-shard-{rank:02d}-of-{shards:02d}.csv'
        _run([python,TOOLS/'run_oneig_alignment_shard.py','--oneig-root',work,'--image-root',images,
              '--model-name',name,'--all-classes','--shard-index',rank,'--num-shards',shards,'--output',path],
             worker,output/f'logs/alignment_{rank}.log')
    with ThreadPoolExecutor(max_workers=shards) as pool:list(pool.map(alignment,range(shards)))
    def component(item):
        index,component=item
        if list((work/'results').glob(component+'_score_EN_*.csv')):return
        directory=images if component=='diversity' else images/{'text':'text','reasoning':'reasoning','style':'anime'}[component]
        command=[python,'-m',f'scripts.{component}.{component}_score','--mode','EN','--image_dirname',directory,
                 '--model_names',name,'--image_grid','2,2']
        if component=='diversity':command.extend(['--class_items','anime','human','object','text','reasoning'])
        worker=dict(env,CUDA_VISIBLE_DEVICES=tokens[index%shards])
        _run(command,worker,output/f'logs/{component}.log',cwd=work)
    with ThreadPoolExecutor(max_workers=min(4,shards)) as pool:
        list(pool.map(component,enumerate(('text','reasoning','style','diversity'))))
    command=[python,TOOLS/'collect_oneig_official_scores.py','--results-root',work/'results',
             '--alignment-shards',shards,'--model-name',name,'--output',output/'summary.json']
    if partial:command.extend(['--subset-image-root',images])
    _run(command,env,output/'logs/summary.log')


def collect_metrics(root,records,selected):
    root=Path(root);counts=Counter(row['benchmark'] for row in records);metrics={}
    def summary(name):
        value=_read(root/'scoring'/name/'summary.json')
        count=value.get('num_images',value.get('count'))
        if count is not None and int(count)!=counts[name]:raise ValueError(f'{name}: score count mismatch')
        for key in ('parse_failures','image_failures','failure_count'):
            if value.get(key,0)!=0:raise ValueError(f'{name}: {key} is nonzero; retain the failure and retry only failed judgments')
        return value
    for name in selected:
        if name in ('tiif_short','tiif_long'):continue
        if name=='dpgbench':metrics[name]=collect_dpg(root,counts[name])
        elif name=='oneig_en':metrics[name]=summary(name)['scores']
        else:metrics[name]=summary(name)
    if 'tiif_short' in selected or 'tiif_long' in selected:
        model_name=_read(root/'evaluation_config.json')['model_name']
        value=collect_tiif(root,counts['tiif_short'],counts['tiif_long'],model_name)
        metrics['tiif_short']={'score':value['short']};metrics['tiif_long']={'score':value['long']}
    headline={}
    for title,(benchmark,key,scale) in FIELDS.items():
        if benchmark not in metrics:continue
        value=float(metrics[benchmark][key])*scale
        if not math.isfinite(value):raise ValueError(f'Nonfinite metric: {title}')
        headline[title]=value
    return metrics,headline


def evaluate(root,runtime,*,benchmarks=None,devices='0',model_name='model',force=False):
    """Score immutable existing PNGs. Subsets are diagnostic, never full results."""
    root=Path(root).expanduser().absolute();records=read_jsonl(root/'records.jsonl')
    validate_records(records)
    requested=tuple(benchmarks or [name for name in BENCHMARKS if any(r['benchmark']==name for r in records)])
    if set(requested)-set(BENCHMARKS) or len(requested)!=len(set(requested)):raise ValueError('Invalid scorer list')
    if 'tiif_long' in requested or 'tiif_short' in requested:
        requested=tuple(dict.fromkeys([*requested,'tiif_short','tiif_long']))
    if not set(requested)<={r['benchmark'] for r in records}:raise ValueError('Requested benchmark has no input records')
    partial=any(not row.get('reportable',False) for row in records)
    if not partial:validate_records(records,require_full=True)
    if not (root/'generation_manifest.jsonl').exists():collect_generation(root)
    manifest=read_jsonl(root/'generation_manifest.jsonl')
    if len(manifest)!=len(records) or {r['artifact_id'] for r in manifest}!={r['artifact_id'] for r in records}:
        raise ValueError('Incomplete generation manifest')
    # This is a scope/config lock, not an image checksum or another preparation pass.
    _lock(root/'evaluation_config.json',dict(model_name=model_name,diagnostic_subset=partial,
            image_count=len(records),source_root=runtime['source_root'],runtime_file=runtime['_file']))
    tokens=_device_tokens(devices)
    if len(tokens) not in (1,8):raise ValueError('Use one GPU or one eight-GPU allocation')
    env=worker_environment(runtime,world_size=len(tokens),model_name=model_name)
    env['CUDA_VISIBLE_DEVICES']=','.join(tokens)
    env['TIIF_PORT']=str(_port());env['WISE_PORT']=str(_port())
    source=Path(runtime['source_root'])
    config=dict(runtime_file=runtime['_file'],assets=runtime['assets'],python=runtime['python'],
                world_size=len(tokens),partial=partial,model_name=model_name)
    for name in requested:
        if name=='tiif_long':continue
        # Fixed diagnostic replay uses one device per small scorer. This keeps
        # request/batch order identical whether the caller supplies one GPU or
        # an eight-GPU allocation; the two large TP judges retain all devices.
        chosen=tokens[:1] if partial and name not in ('tiif_short','wise_verified') else tokens
        current_config=dict(config,world_size=len(chosen))
        output=root/'scoring'/name;output.mkdir(parents=True,exist_ok=True)
        marker=output/'complete.json'
        if marker.exists():
            old=_read(marker)
            if not _compatible_configuration(old['configuration'],current_config):raise ValueError(f'Cached {name} used a different runtime; use a new output directory')
            if not force:continue
            raise ValueError('--force cannot overwrite scoring evidence; use a fresh run directory')
        local=dict(env,BENCH_SCORER_LOCAL_CACHE_ROOT=tempfile.mkdtemp(prefix='bench-image-'+name+'-'),
                   BENCH_SCORER_WORLD_SIZE=str(len(chosen)),CUDA_VISIBLE_DEVICES=','.join(chosen))
        if partial:local.update(TIIF_CLIENTS='1',WISE_MAX_WORKERS='1')
        started=time.perf_counter()
        if name=='oneig_en':_oneig(root,source,local,chosen,partial=partial)
        else:
            _run(['bash',_snapshot_runner(output),'smoke' if partial else 'full',
                  name,root,source],local,output/'logs/runner.log')
        this_metrics,_=collect_metrics(root,records,('tiif_short','tiif_long') if name=='tiif_short' else (name,))
        _atomic_json(marker,dict(complete=True,configuration=current_config,metrics=this_metrics,
                         elapsed_seconds=time.perf_counter()-started))
    available=[]
    for name in BENCHMARKS:
        parent='tiif_short' if name=='tiif_long' else name
        if (root/'scoring'/parent/'complete.json').exists():available.append(name)
    metrics,headline=collect_metrics(root,records,available)
    complete=set(available)=={r['benchmark'] for r in records}
    report=dict(evaluation_mode='fixture' if partial else 'full',benchmark_complete=complete and not partial,
                selection_complete=complete,model=model_name,image_count=len(records),headline_metrics=headline,
                metrics=metrics,missing_benchmarks=sorted({r['benchmark'] for r in records}-set(available)))
    # Separate per-invocation reports avoid two independent benchmark jobs racing
    # on a shared summary. A final `collect` command is the sole report writer.
    (root/'reports').mkdir(exist_ok=True)
    key='_'.join(sorted(requested))
    (root/'reports'/f'{key}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def write_summary(root):
    root=Path(root);records=read_jsonl(root/'records.jsonl')
    names=tuple(name for name in BENCHMARKS if any(row['benchmark']==name for row in records))
    configurations=[]
    for name in names:
        if not (root/'scoring'/('tiif_short' if name=='tiif_long' else name)/'complete.json').exists():
            raise ValueError('Missing completed scorer: '+name)
        marker=_read(root/'scoring'/('tiif_short' if name=='tiif_long' else name)/'complete.json')
        configurations.append({k:v for k,v in marker['configuration'].items() if k!='world_size'})
    if any(not _compatible_configuration(a,b) for i,a in enumerate(configurations) for b in configurations[i+1:]):
        raise ValueError('Scorers used inconsistent runtime/model configurations')
    metrics,headline=collect_metrics(root,records,names)
    partial=any(not r.get('reportable',False) for r in records)
    if not partial:validate_records(records,require_full=True)
    config=_read(root/'evaluation_config.json')
    result=dict(model=config['model_name'],evaluation_mode='fixture' if partial else 'full',
        benchmark_complete=not partial,selection_complete=True,image_count=len(records),
        headline_metrics=headline,metrics=metrics)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'metrics_summary.csv').open('w',newline='') as handle:
        row=dict(model=config['model_name'],evaluation_mode=result['evaluation_mode'],images=len(records),
                 **{key:round(value,6) for key,value in headline.items()})
        writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    return result
