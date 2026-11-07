"""Official FLUX.3 T2I PUP prompts/constants using the validated HF backend.

Greedy decoding matches the official temperature=0 setting. A frozen cache is
the reproducibility boundary; different inference engines/hardware can still
produce different expansions. Scoring always uses the original benchmark text.
"""
import json
from pathlib import Path
from .protocols import read_jsonl
from .generation import _lock


def generate_pup(records,model,output,*,revision=None,batch_size=4):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,LogitsProcessorList
    if not torch.cuda.is_available():raise RuntimeError('PUP requires an allocated CUDA GPU')
    if not 1<=batch_size<=16:raise ValueError('Official PUP max_num_seqs is 16')
    path=Path(model)
    if not path.is_dir():
        path=Path(snapshot_download(model,revision=revision));revision=path.name
    prompts=list(dict.fromkeys(row['prompt'] for row in read_jsonl(records)))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    config=dict(model=model,revision=revision,backend='transformers_greedy',max_new_tokens=3584,
                presence_penalty=1.5,penalty_scope='generated_tokens_only',enable_thinking=False,
                system_prompt_key='t2i',batch_size=batch_size)
    _lock(output.with_suffix('.protocol.json'),config)
    done={row['prompt']:row for row in read_jsonl(output)} if output.exists() else {}
    if set(done)-set(prompts):raise ValueError('PUP cache contains prompts from another selection')
    pending=[prompt for prompt in prompts if prompt not in done]
    if not pending:return dict(complete=True,prompts=len(prompts),reused=len(done))
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    processor.tokenizer.padding_side='left'
    generator=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    system=json.loads((path/'system_prompts.json').read_text())['t2i']
    stop=[processor.tokenizer.convert_tokens_to_ids(token) for token in ('<|im_end|>','<|endoftext|>')]
    stop=[token for token in stop if token is not None and token>=0]
    class PresencePenalty:
        def __init__(self,input_length):self.input_length=input_length
        def __call__(self,input_ids,scores):
            generated=input_ids[:,self.input_length:]
            if generated.shape[1]:
                seen=torch.zeros_like(scores,dtype=torch.bool);seen.scatter_(1,generated,True)
                scores=scores-seen.to(scores.dtype)*1.5
            return scores
    for start in range(0,len(pending),batch_size):
        batch=pending[start:start+batch_size]
        chats=[[{'role':'system','content':system},{'role':'user','content':prompt}] for prompt in batch]
        inputs=processor.apply_chat_template(chats,add_generation_prompt=True,tokenize=True,
            return_dict=True,return_tensors='pt',padding=True,enable_thinking=False).to('cuda')
        length=inputs['input_ids'].shape[1]
        with torch.inference_mode():
            generated=generator.generate(**inputs,max_new_tokens=3584,do_sample=False,eos_token_id=stop,
                pad_token_id=processor.tokenizer.pad_token_id,logits_processor=LogitsProcessorList([PresencePenalty(length)]))
        captions=processor.batch_decode(generated[:,length:],skip_special_tokens=True)
        with output.open('a') as handle:
            for prompt,caption in zip(batch,captions):
                if not caption.strip():raise ValueError('Empty PUP caption')
                handle.write(json.dumps(dict(prompt=prompt,caption=caption.strip()),ensure_ascii=False)+'\n')
            handle.flush()
    return dict(complete=True,prompts=len(prompts),generated=len(pending),configuration=config)
