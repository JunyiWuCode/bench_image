"""Verify every official prompt/ID/repetition/size against a completed suite."""
import argparse
import json
from pathlib import Path
from bench_image.protocols import generation_samples,read_jsonl
from bench_image.config import load_runtime

p=argparse.ArgumentParser()
p.add_argument('--reference-run',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
actual=generation_samples(source_root=load_runtime(a.runtime)['source_root'])
reference=read_jsonl(a.reference_run/'run/records.jsonl')
keys=('artifact_id','benchmark','prompt','prompt_index','sample_index','width','height','metadata')
observed={row['artifact_id']:row for row in actual}
expected={row['artifact_id']:row for row in reference}
if observed.keys()!=expected.keys():raise ValueError('Different official image IDs')
errors=[]
for key,row in expected.items():
    for field in keys:
        if row[field]!=observed[key][field]:errors.append(dict(artifact_id=key,field=field))
    if observed[key]['seed']!=1000*row['prompt_index']+row['sample_index']:
        errors.append(dict(artifact_id=key,field='seed'))
result=dict(complete=not errors,image_count=len(actual),compared_fields=list(keys),
    per_image_seed_rule='1000*prompt_index+sample_index',differences=errors,
    scope='All official prompt metadata and repetition/size/seed contracts, not generated model quality')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
if errors:raise SystemExit(1)
