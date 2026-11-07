"""Diagnostic-only offline client; full TIIF keeps the upstream online path.

vLLM documents deterministic scheduling with in-process offline execution.
Keep the original evaluator's messages/images and its parsing/aggregation.
"""
import os
from types import SimpleNamespace


class OfflineClient:
    def __init__(self,**_):
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        from vllm import LLM
        self.engine=LLM(model=os.environ['TIIF_MODEL'],
            tensor_parallel_size=int(os.environ['BENCH_SCORER_WORLD_SIZE']),
            dtype='bfloat16',seed=0,max_num_seqs=1,enforce_eager=True,
            enable_prefix_caching=False,enable_chunked_prefill=False)
        self.chat=SimpleNamespace(completions=self)

    def create(self,*,model,messages,temperature,seed):
        if temperature!=0 or seed!=0:raise ValueError('Offline fixture requires temperature=0 and seed=0')
        params=self.engine.get_default_sampling_params()
        params.temperature=0.0;params.seed=0;params.max_tokens=4096
        result=self.engine.chat(messages,sampling_params=params,use_tqdm=False)[0].outputs[0]
        if result.finish_reason=='length':raise RuntimeError('TIIF diagnostic response reached its token cap')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result.text))])
