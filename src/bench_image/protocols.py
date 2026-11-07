"""The 14 headline columns, represented by eleven generation protocols."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from ._legacy.official_suite import OFFICIAL_PROTOCOLS, build_records, output_path_for_record

BENCHMARKS=('geneval','geneval2','dpgbench','tiif_short','tiif_long','cvtg',
            'longtext_en','oneig_en','qwen_image_bench_en','hpsv3_official','wise_verified')
FIELDS={
 'GenEval':('geneval','overall',100),
 'GenEval2 AM':('geneval2','soft_tifa_am',1),
 'GenEval2 GM':('geneval2','soft_tifa_gm',1),
 'DPG-Bench':('dpgbench','score',1),
 'TIIF Short':('tiif_short','score',100),
 'TIIF Long':('tiif_long','score',100),
 'CVTG WA':('cvtg','word_accuracy',100),
 'CVTG NED':('cvtg','ned',100),
 'CVTG CLIP':('cvtg','clip_score',100),
 'LongText-EN':('longtext_en','text_score',100),
 'OneIG-EN Overall':('oneig_en','overall',100),
 'Qwen-Image-Bench-EN Overall':('qwen_image_bench_en','overall',1),
 'HPSv3':('hpsv3_official','overall',1),
 'WISE Overall':('wise_verified','wiscore',100),
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))


def validate_records(rows, *, require_full=False):
    if not rows:raise ValueError('Empty generation selection')
    ids=[r['artifact_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate artifact IDs')
    groups=defaultdict(list)
    for r in rows:
        if r['benchmark'] not in BENCHMARKS:raise ValueError('Unsupported benchmark: '+r['benchmark'])
        if any(not isinstance(r[key],int) or isinstance(r[key],bool) or r[key]<64 or r[key]%16 for key in ('width','height')):
            raise ValueError('Image dimensions must be integers >=64 and multiples of 16')
        if any(not isinstance(r[key],int) or isinstance(r[key],bool) or r[key]<0 for key in ('prompt_index','sample_index')):
            raise ValueError('Prompt/sample indices must be nonnegative integers')
        if r['prompt_index']>=OFFICIAL_PROTOCOLS[r['benchmark']].prompts:raise ValueError('Nonofficial prompt index')
        if not isinstance(r['prompt'],str) or not r['prompt'].strip():raise ValueError('Missing original prompt')
        groups[(r['benchmark'],r['prompt_index'])].append(r)
    for (name,index),group in groups.items():
        expected=set(range(OFFICIAL_PROTOCOLS[name].samples_per_prompt))
        if {r['sample_index'] for r in group}!=expected or len(group)!=len(expected):
            raise ValueError(f'Incomplete official repetition group: {name}/{index}')
        if len({r['prompt'] for r in group})!=1:raise ValueError('Prompt differs within a repetition group')
    if require_full:
        counts=Counter(r['benchmark'] for r in rows)
        for name,count in counts.items():
            if count!=OFFICIAL_PROTOCOLS[name].images:raise ValueError(f'Partial {name} input is not a full benchmark')
            indices={r['prompt_index'] for r in rows if r['benchmark']==name}
            if indices!=set(range(OFFICIAL_PROTOCOLS[name].prompts)):
                raise ValueError(f'Noncanonical full prompt IDs: {name}')
        if any(not r.get('reportable',False) for r in rows):raise ValueError('Smoke selection cannot be reported as a full benchmark')


def generation_samples(*,source_root,benchmarks=BENCHMARKS,resolution=1024,seed=0,selection=None):
    """Return official prompts, stable IDs/seeds and raw image paths (no model import).

    ``selection`` maps benchmark names to original prompt IDs. All official
    repetitions are retained. Subsets are explicitly non-reportable.
    """
    names=tuple(benchmarks)
    if len(names)!=len(set(names)) or set(names)-set(BENCHMARKS):raise ValueError('Invalid benchmark list')
    source_root=Path(source_root).expanduser().resolve()
    source_names=('geneval','geneval2','dpgbench','dpgbench_questions','tiif','longtext_en',
                  'oneig_en','hpsv3_official','wise_verified')
    sources={name:source_root/name for name in source_names}
    rows=build_records(sources,names,profile='official_report',fallback_resolution=resolution,hps_resolution=resolution)
    if selection is not None:
        if set(selection)!=set(names):raise ValueError('Selection must explicitly identify every requested benchmark')
        rows=[r for r in rows if r['prompt_index'] in selection[r['benchmark']]]
        for name in names:
            if {r['prompt_index'] for r in rows if r['benchmark']==name}!=set(selection[name]):
                raise ValueError('Unknown selected prompt ID for '+name)
    for i,row in enumerate(rows):
        row['index']=i
        row['seed']=int(seed)+1000*int(row['prompt_index'])+int(row['sample_index'])
        row['image_path']=str(output_path_for_record('.',row))
        if selection is not None:row.update(reportable=False,smoke_selection='explicit_prompt_ids')
    validate_records(rows,require_full=selection is None)
    return rows
