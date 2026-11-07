"""Create a local, explicitly supplied runtime mapping for an existing cluster.

No installations are modified. This is an optional migration utility, not a
fresh-machine prerequisite. Its output contains local paths: do not commit it.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ('env-root','far-rl-root','source-root','hub-root','hf-hub-root','dreamsim-root',
                 'mplug-root','wise-model','generation-python','paddle-models','clip-model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cuda-compat',type=Path)
    p.add_argument('--oneig-csd-model',type=Path,required=True)
    a=p.parse_args()
    envnames=dict(GENEVAL_PYTHON='flow_opd_geneval_reward',GENEVAL2_PYTHON='geneval2_official',
        DPGBENCH_PYTHON='flow_opd_dpg_reward',CVTG_OCR_PYTHON='paddleocr_gpu_official',CVTG_CLIP_PYTHON='cvtg_eval',
        LONGTEXT_PYTHON='longtext_ocr',ONEIG_PYTHON='oneig_official',HPSV3_PYTHON='hpsv3_official_torch26',
        QJUDGER_PYTHON='q_judger',TIIF_VLLM_PYTHON='vllm',WISE_PYTHON='qwen35_batch')
    python={key:str(a.env_root/name/'bin/python') for key,name in envnames.items()}
    python['GENERATION_PYTHON']=str(a.generation_python)
    def snapshot(repo,revision):return a.hub_root/('models--'+repo.replace('/','--'))/'snapshots'/revision
    assets=dict(
      GENEVAL_MODEL_CONFIG=str(a.far_rl_root/'third_party/reward-server/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
      GENEVAL_MODEL_PATH=str(a.far_rl_root/'third_party/reward-server/model/mask2former2'),
      DPGBENCH_MPLUG_MODEL=str(a.mplug_root),
      GENEVAL2_MODEL=str(snapshot('Qwen/Qwen3-VL-8B-Instruct','0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')),
      TIIF_MODEL=str(a.hf_hub_root/'models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5'),
      ONEIG_QWEN_MODEL=str(snapshot('Qwen/Qwen2.5-VL-7B-Instruct','cc594898137f460bfe9f0759e9844b3ce807cfb5')),
      ONEIG_LLM2CLIP_MODEL=str(snapshot('microsoft/LLM2CLIP-Openai-L-14-336','92512331f393a003c3d98404677f991c188162c9')),
      ONEIG_LLM2CLIP_LLM=str(snapshot('microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned','97f7d164a1eb89a79e344b21c7a1be7eee265d2e')),
      ONEIG_LLM2VEC_CODE=str(snapshot('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','31474e395ada192e8ed1586db6be79fb3b70c9c0')/'modeling_llama_encoder.py'),
      ONEIG_CLIP_MODEL=str(snapshot('openai/clip-vit-large-patch14-336','ce19dc912ca5cd21c8a653c79e251e808ccabcd1')),
      ONEIG_STYLE_MODEL=str(snapshot('xingpng/OneIG-StyleEncoder','9e8104621cbc8c8f04ce663639ffbec63e0a9693')),
      ONEIG_CSD_MODEL=str(a.oneig_csd_model),
      ONEIG_DREAMSIM_CACHE=str(a.dreamsim_root),
      QWEN_IMAGE_BENCH_ROOT=str(a.far_rl_root/'third_party/reference_repos/Qwen-Image-Bench'),
      QJUDGER_MODEL=str(a.far_rl_root/'third_party/reference_models/Qwen-Image-Bench'),
      HPSV3_BASE_MODEL=str(snapshot('Qwen/Qwen2-VL-7B-Instruct','eed13092ef92e448dd6875b2a00151bd3f7db0ac')),
      HPSV3_CHECKPOINT=str(snapshot('MizzenAI/HPSv3','4f81e3e09edd82fe3c5f636444c721b592a735ca')/'HPSv3.safetensors'),
      WISE_MODEL=str(a.wise_model),BENCH_CLIP_MODEL_PATH=str(a.clip_model),
      PADDLEOCR_DET_MODEL_DIR=str(a.paddle_models/'PP-OCRv5_server_det'),
      PADDLEOCR_REC_MODEL_DIR=str(a.paddle_models/'en_PP-OCRv5_mobile_rec'))
    if a.cuda_compat:assets['COREBENCH_CUDA_COMPAT_LIB']=str(a.cuda_compat)
    for path in [*python.values(),*assets.values(),str(a.source_root)]:
        if not Path(path).exists():raise FileNotFoundError(path)
    result=dict(cache_root=str(a.output.parent),source_root=str(a.source_root),python=python,assets=assets,
        setup_python=str(a.generation_python),environment={'TOKENIZERS_PARALLELISM':'false'},
        provenance='Explicit adoption of existing environments; no installation/weights mutated')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Local runtime mapping written:',a.output)


if __name__=='__main__':main()
