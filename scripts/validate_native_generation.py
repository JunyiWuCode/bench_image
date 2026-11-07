"""GPU-only native export smoke; compare new adapter pixels with a recorded run."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from bench_image.generation import generate,collect_generation
from bench_image.protocols import read_jsonl,write_jsonl,output_path_for_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt-map',type=Path)
    a=p.parse_args()
    rows=read_jsonl(a.reference_run/'run/records.jsonl')
    selected=[]
    for name in ('geneval','cvtg','longtext_en'):
        first=next(row['prompt_index'] for row in rows if row['benchmark']==name)
        selected.extend(dict(row,reportable=False,smoke_selection='native_adapter_validation',
            seed=1000*row['prompt_index']+row['sample_index']) for row in rows
            if row['benchmark']==name and row['prompt_index']==first)
    a.output.mkdir(parents=True,exist_ok=True)
    records=a.output/'records.jsonl'
    if records.exists() and read_jsonl(records)!=selected:raise ValueError('Different existing smoke selection')
    if not records.exists():write_jsonl(records,selected)
    receipt=generate(records,a.output,a.model,trust_remote_code=True,prompt_map=a.prompt_map)
    generated=collect_generation(a.output)
    checks=[]
    for row in generated:
        with Image.open(row['image_path']) as actual,Image.open(output_path_for_record(a.reference_run/'run',row)) as reference:
            actual=np.asarray(actual.convert('RGB'),dtype=np.int16)
            reference=np.asarray(reference.convert('RGB'),dtype=np.int16)
            delta=np.abs(actual-reference)
            checks.append(dict(artifact_id=row['artifact_id'],actual_nfe=row['actual_nfe'],
                pixel_identical=bool(np.array_equal(actual,reference)),mean_absolute_pixel_error=float(delta.mean()),
                image_std=float(actual.std())))
            if row['actual_nfe']!=4 or actual.std()<1:raise ValueError('Invalid native generation')
    report=dict(complete=True,images=len(generated),native_receipt=receipt,comparisons=checks,
        note='Pixel equality is diagnostic, not guaranteed across hardware/library versions.')
    (a.output/'native_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
