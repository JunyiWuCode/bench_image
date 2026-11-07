"""Use an export's native Diffusers pipeline; never emulate interval sampling."""
import json
import os
from pathlib import Path
import time
import uuid
from .protocols import read_jsonl, validate_records, write_jsonl, output_path_for_record


def _lock(path,value):
    """Publish one immutable config atomically, including concurrent array jobs."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    try:
        try:os.link(temporary,path)
        except FileExistsError:
            if json.loads(path.read_text())!=value:
                raise ValueError(f'Refusing incompatible resume: {path}. Use a new run directory.')
    finally:temporary.unlink()


def generate(records,output_dir,model,*,revision=None,steps=4,guidance=None,
             prompt_map=None,allow_raw_flux3=False,trust_remote_code=False,
             shard_id=0,num_shards=1):
    """Generate one independent single-GPU shard with stable per-image CPU seeds.

    Native Raw2Q exports accept one CPU Generator, not a generator list. We use
    one prompt per native call to preserve the published BCHW/packed noise law;
    independent shards provide concurrency without changing sampling semantics.
    """
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import hf_hub_download, model_info
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('Generation requires an allocated CUDA GPU')
    if not 0<=shard_id<num_shards:raise ValueError('Invalid shard ID/count')
    rows=read_jsonl(records) if isinstance(records,(str,Path)) else list(records)
    validate_records(rows)
    root=Path(output_dir).expanduser().absolute();root.mkdir(parents=True,exist_ok=True)
    local=Path(model).expanduser()
    if local.is_dir():
        model=str(local.resolve());index=local/'model_index.json'
    else:
        revision=model_info(model,revision=revision).sha
        index=Path(hf_hub_download(model,'model_index.json',revision=revision))
    class_name=json.loads(index.read_text())['_class_name']
    backend={'Flux1FourStepPipeline':'flux1-raw2q','Flux3FourStepPipeline':'flux3-raw2q'}.get(class_name)
    if backend is None:
        raise ValueError(f'Unsupported native export {class_name}; do not substitute an ordinary pipeline for an interval model')
    if not trust_remote_code:raise ValueError('Review the model code and explicitly enable --trust-remote-code')
    native_guidance=3.5 if backend=='flux1-raw2q' else 1.0
    if steps!=4 or (guidance is not None and guidance!=native_guidance):
        raise ValueError(f'{backend} requires four steps and native guidance {native_guidance}')
    captions={}
    if prompt_map:
        for row in read_jsonl(prompt_map):
            if row['prompt'] in captions and captions[row['prompt']]!=row['caption']:
                raise ValueError('Conflicting cached PUP captions')
            captions[row['prompt']]=row['caption']
    if backend=='flux1-raw2q' and captions:
        raise ValueError('FLUX.1 Raw2Q uses native raw prompts, not FLUX.3 PUP')
    if backend=='flux3-raw2q' and not captions and not allow_raw_flux3:
        raise ValueError('FLUX.3 comparison requires --prompt-map (or explicit --allow-raw-flux3 ablation)')
    if captions and any(row['prompt'] not in captions for row in rows):raise ValueError('Incomplete PUP cache')
    config=dict(model=model,model_revision=revision,pipeline_class=class_name,backend=backend,
                sampling_steps=steps,guidance_scale=native_guidance,
                prompt_mode='cached_official_pup' if captions else 'native_raw',
                seed_rule='record.seed, otherwise 1000*prompt_index+sample_index',
                num_shards=num_shards,model_dtype='bfloat16',vae_dtype='float32')
    _lock(root/'generation_config.json',config)
    # The CLI normally creates this before submitting an array. Exclusive publication
    # also makes direct API calls safe when many shards start simultaneously.
    record_path=root/'records.jsonl'
    if not record_path.exists():
        temp=root/f'records.{uuid.uuid4().hex}.tmp';write_jsonl(temp,rows)
        try:
            try:os.link(temp,record_path)
            except FileExistsError:pass
        finally:temp.unlink()
    if not isinstance(records,(str,Path)) or Path(records).absolute()!=record_path:
        if read_jsonl(record_path)!=rows:raise ValueError('Run contains different benchmark records')
    selected=[row for i,row in enumerate(rows) if i%num_shards==shard_id]
    manifest=root/'manifests'/f'shard_{shard_id:05d}_of_{num_shards:05d}'/'results.jsonl'
    done={r['artifact_id']:r for r in read_jsonl(manifest)} if manifest.exists() else {}
    expected={row['artifact_id'] for row in selected}
    if set(done)-expected:raise ValueError('Unexpected artifacts in resume manifest')
    pending=[]
    for row in selected:
        path=output_path_for_record(root,row)
        actual_prompt=captions.get(row['prompt'],row['prompt'])
        if row['artifact_id'] in done:
            saved=done[row['artifact_id']]
            if saved['model_input_prompt']!=actual_prompt:raise ValueError('Cached image used a different model input prompt')
            with Image.open(path) as image:
                if image.format!='PNG' or image.size!=(row['width'],row['height']):raise ValueError('Invalid cached PNG')
        else:
            if path.exists():raise ValueError(f'Orphan image without generation receipt: {path}')
            pending.append(row)
    if not pending:return dict(complete=True,shard_id=shard_id,images=len(selected),reused=len(done))
    pipe=DiffusionPipeline.from_pretrained(model,revision=revision,custom_pipeline=model,
        custom_revision=revision,trust_remote_code=True,torch_dtype=torch.bfloat16)
    pipe.to('cuda')  # Never cast the complete pipeline: native VAE loader retains FP32.
    for name in ('transformer','text_encoder','text_encoder_2'):
        module=getattr(pipe,name,None)
        if module is not None and module.dtype!=torch.bfloat16:raise ValueError(f'{name} is not BF16')
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in [*pipe.vae.parameters(),*pipe.vae.buffers()]):
        raise ValueError('Native VAE must remain entirely FP32')
    scale=float(pipe.scheduler.config.final_step_size_scale)
    if scale!=.3:raise ValueError('Unexpected Raw2Q final interval scale')
    shift=float(pipe.scheduler.config.shift)
    expected_shift=3.158192909689767 if backend=='flux1-raw2q' else 6.93
    if shift!=expected_shift:raise ValueError(f'Unexpected native scheduler shift: {shift}')
    raw_times=pipe.scheduler.get_raw_timesteps(4,torch.device('cuda'))
    times=pipe.scheduler.apply_shift(raw_times)
    trace=dict(final_step_size_scale=scale,scheduler_shift=shift,
        raw_times=raw_times.cpu().tolist(),shifted_times=times.cpu().tolist(),
        model_dtype='bfloat16',vae_dtype='float32',integration_dtype='float32',
        noise_generator_device='cpu',noise_layout='BCHW' if backend=='flux1-raw2q' else 'packed_token_major',
        noise_dtype='bfloat16' if backend=='flux1-raw2q' else 'float32_rounded_to_bfloat16')
    calls=[0]
    hook=pipe.transformer.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
    started=time.perf_counter();manifest.parent.mkdir(parents=True,exist_ok=True)
    try:
        for row in pending:
            calls[0]=0
            seed=int(row.get('seed',1000*row['prompt_index']+row['sample_index']))
            actual_prompt=captions.get(row['prompt'],row['prompt'])
            image=pipe(prompt=actual_prompt,height=row['height'],width=row['width'],
                       num_inference_steps=4,guidance_scale=native_guidance,
                       generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
            if calls[0]!=4:raise RuntimeError(f'Unexpected actual transformer NFE: {calls[0]}')
            path=output_path_for_record(root,row);path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.tmp.png');image.save(temp,format='PNG');os.replace(temp,path)
            result=dict(row,seed=seed,image_path=str(path),model_input_prompt=actual_prompt,
                        model=model,model_revision=revision,sampling_steps=4,actual_nfe=calls[0],
                        guidance_scale=native_guidance,schedule=trace)
            with manifest.open('a') as handle:handle.write(json.dumps(result,ensure_ascii=False)+'\n');handle.flush()
    finally:hook.remove()
    receipt=dict(complete=True,shard_id=shard_id,images=len(selected),generated=len(pending),
                 elapsed_seconds=time.perf_counter()-started,includes_model_loading=False,
                 controlled_latency_benchmark=False,schedule=trace,config=config)
    (manifest.parent/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    del pipe
    import gc
    gc.collect();torch.cuda.empty_cache()
    return receipt


def collect_generation(root):
    """Join receipts once at the generation/scoring boundary; require exact coverage."""
    from PIL import Image
    root=Path(root).absolute();rows=read_jsonl(root/'records.jsonl');validate_records(rows)
    generated=[]
    for path in sorted((root/'manifests').glob('shard_*/results.jsonl')):generated.extend(read_jsonl(path))
    if len(generated)!=len(rows) or {r['artifact_id'] for r in generated}!={r['artifact_id'] for r in rows}:
        raise ValueError('Generation incomplete, duplicated or mismatched; no partial scoring')
    by_id={r['artifact_id']:r for r in generated}
    for row in rows:
        result=by_id[row['artifact_id']]
        if result['prompt']!=row['prompt']:raise ValueError('Generation prompt mismatch')
        path=output_path_for_record(root,row)
        with Image.open(path) as image:
            if image.size!=(row['width'],row['height']) or image.format!='PNG':raise ValueError(f'Invalid image {path}')
        result['image_path']=str(path)
    ordered=[by_id[r['artifact_id']] for r in rows]
    write_jsonl(root/'generation_manifest.jsonl',ordered)
    (root/'generation_summary.json').write_text(json.dumps(dict(complete=True,record_count=len(rows)),indent=2)+'\n')
    return ordered
