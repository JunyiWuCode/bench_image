"""Select original official IDs and package existing measured PNGs losslessly.

Use a complete native run as --reference-run. Building images is CPU work.
The final ZIP is written only after a fresh scoring run has a verified summary.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image,ImageChops
from bench_image.api import register_images
from bench_image.protocols import BENCHMARKS,read_jsonl,write_jsonl,output_path_for_record


def choose(rows):
    first={name:[r for r in rows if r['benchmark']==name and r['sample_index']==0] for name in BENCHMARKS}
    selection={}
    def one_per(name,key):
        found={}
        for row in first[name]:found.setdefault(key(row),row['prompt_index'])
        return list(found.values())
    selection['geneval']=one_per('geneval',lambda r:r['metadata']['tag'])
    selection['geneval2']=[r['prompt_index'] for r in first['geneval2'][:2]]
    selection['dpgbench']=[r['prompt_index'] for r in first['dpgbench'][:2]]
    dimensions=('shape+color','2d_spatial_relation','numeracy','action+color','numeracy+color',
                'numeracy+2d','text','style','real_world')
    selection['tiif_short']=[next(r['prompt_index'] for r in first['tiif_short'] if r['metadata']['dimension']==d) for d in dimensions]
    selection['tiif_long']=list(selection['tiif_short'])
    selection['cvtg']=[r['prompt_index'] for r in first['cvtg'][:2]]
    selection['longtext_en']=[first['longtext_en'][0]['prompt_index'],
        next(r['prompt_index'] for r in first['longtext_en'] if r['metadata']['category']!=first['longtext_en'][0]['metadata']['category'])]
    selected=one_per('oneig_en',lambda r:r['metadata']['category'])
    style=next(r['prompt_index'] for r in first['oneig_en'] if r['metadata'].get('style_class'))
    selection['oneig_en']=sorted(set([*selected,style]))
    selection['qwen_image_bench_en']=[0,100]
    selection['hpsv3_official']=one_per('hpsv3_official',lambda r:r['metadata']['category'])
    selected_ids={1,401,521,641,761,881}
    selection['wise_verified']=[r['prompt_index'] for r in first['wise_verified'] if r['metadata']['prompt_id'] in selected_ids]
    return selection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--pack',type=Path,help='Final ZIP path; requires output/expected_metrics.json and verification.json')
    args=parser.parse_args();source=args.reference_run.resolve();root=args.output.resolve()
    if args.pack:
        expected=json.loads((root/'expected_metrics.json').read_text())
        verified=json.loads((root/'verification.json').read_text())
        assert verified['passed'] and len(expected['headline_metrics'])==14
        args.pack.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.pack,'w',compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted((root/'payload').rglob('*')):
                if path.is_file():bundle.write(path,str(path.relative_to(root/'payload')))
            bundle.write(root/'expected_metrics.json','expected_metrics.json')
            bundle.write(root/'verification.json','verification.json')
        if args.pack.stat().st_size>=99*1024**2:raise ValueError('Fixture exceeds the ordinary GitHub-file limit; reduce selection, never rescale reference pixels')
        print(json.dumps(dict(zip=str(args.pack),bytes=args.pack.stat().st_size)));return
    rows=read_jsonl(source/'run/records.jsonl');selection=choose(rows)
    selected=[dict(r) for r in rows if r['prompt_index'] in selection[r['benchmark']]]
    for i,row in enumerate(selected):
        row.update(index=i,reportable=False,smoke_selection='reference_fixture_v1',
                   seed=1000*row['prompt_index']+row['sample_index'],model_input_prompt=row['prompt'],actual_nfe=4)
        row['fixture_image']=str(output_path_for_record('images',row).with_suffix('.webp'))
    root.mkdir(parents=True,exist_ok=True)
    register_images(selected,source/'run',root/'run')
    register_images(selected,source/'run',root/'repeat')
    payload=root/'payload';payload.mkdir(exist_ok=True)
    write_jsonl(payload/'records.jsonl',selected)
    (payload/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    model=json.loads((source/'model_download.json').read_text())
    config=dict(model=model['repo'],revision=model['revision'],pipeline_class='Flux1FourStepPipeline',steps=4,
        actual_nfe=4,resolution=[1024,1024],guidance_embedding=3.5,external_cfg=False,prompt_mode='native_raw',
        noise_generator_device='cpu',noise_layout='BCHW',noise_dtype='bfloat16',vae_dtype='float32',
        shift=3.158192909689767,final_step_size_scale=.3)
    (payload/'generation_config.json').write_text(json.dumps(config,indent=2)+'\n')
    reference=dict(model=model['repo'],model_revision=model['revision'],sampling_steps=4,actual_nfe=4,
        image_count=len(selected),counts=dict(Counter(r['benchmark'] for r in selected)),
        example_mode='diagnostic subset, not full-benchmark quality',
        pixels='Lossless RGB WebP storage; extraction restores identical RGB PNG pixels',
        prompt_field='original prompt, no PUP',generation_hardware='NVIDIA A100-SXM4-80GB',
        upstream_terms='Official benchmark prompt/dataset/model terms continue to apply; see source lock and README.',
        purpose='Reuse these same images to validate scorer consistency, or regenerate native samples separately.')
    (payload/'reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    def image(row):
        original=output_path_for_record(source/'run',row)
        target=payload/row['fixture_image'];target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(original) as raw:
            raw=raw.convert('RGB');raw.save(target,format='WEBP',lossless=True,quality=100,method=6,exact=True)
            with Image.open(target) as encoded:
                if ImageChops.difference(raw,encoded.convert('RGB')).getbbox() is not None:
                    raise ValueError('Fixture encoding changed pixels')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(image,selected))
    print(json.dumps(reference),flush=True)


if __name__=='__main__':main()
