"""One entry point for isolated environments, official sources and public assets.

Run installation/build work on an allocated compute node, never a login node.
Existing environments are never silently upgraded: a completed environment has
an immutable recipe receipt; incomplete installations are resumed explicitly.
"""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from urllib.request import urlopen
from .config import PACKAGE,cache_root,load_runtime


def _data(name):return json.loads((PACKAGE/'data'/name).read_text())


def _command(command,*,env=None):
    print('RUN',json.dumps(list(map(str,command))),flush=True)
    subprocess.run(list(map(str,command)),env=env,check=True)


def environment_plan(root,only=None):
    specs=_data('environments.json');selected=set(only.split(',')) if only else set(specs)
    unknown=selected-set(specs)
    if unknown:raise ValueError('Unknown environments: '+str(sorted(unknown)))
    conda=os.environ.get('CONDA_EXE') or shutil.which('conda') or 'conda'
    plan=[]
    for name,spec in specs.items():
        if name not in selected:continue
        prefix=root/'envs'/name;python=prefix/'bin/python'
        commands=[[conda,'create','-y','-p',str(prefix),'python='+spec['python'],'pip'],
                  [str(python),'-m','pip','install','--upgrade','pip','wheel','setuptools']]
        if 'torch' in spec:
            torch,vision,cuda=spec['torch']
            commands.append([str(python),'-m','pip','install','--index-url','https://download.pytorch.org/whl/'+cuda,
                             'torch=='+torch,'torchvision=='+vision])
        torch_pins=['torch=='+spec['torch'][0],'torchvision=='+spec['torch'][1]] if 'torch' in spec else []
        commands.append([str(python),'-m','pip','install',*torch_pins,*spec['packages']])
        if spec.get('paddle'):
            commands.append([str(python),'-m','pip','install','paddlepaddle-gpu=='+spec['paddle'],
                             '--index-url','https://www.paddlepaddle.org.cn/packages/stable/cu126/'])
        if spec.get('compile_mmcv'):
            commands.append([str(python),'-m','pip','install','--no-build-isolation','--no-binary=mmcv-full',
                             'mmcv-full=='+spec['compile_mmcv']])
        if spec.get('flash_attention_wheel'):
            commands.append([str(python),'-m','pip','install',spec['flash_attention_wheel']])
        if spec.get('source_package'):
            commands.append([str(python),'-m','pip','install','--no-deps',str(root/'sources'/spec['source_package'])])
        plan.append(dict(name=name,prefix=str(prefix),recipe=spec,commands=commands))
    return plan


def _git(root,name,url,revision):
    target=root/'sources'/name
    if (target/'.git').exists():
        observed=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        expected=subprocess.check_output(['git','-C',str(target),'rev-parse',revision+'^{commit}'],text=True).strip()
        if observed!=expected:raise ValueError('Existing source revision mismatch: '+str(target))
        if subprocess.check_output(['git','-C',str(target),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise ValueError('Refusing to overwrite modified upstream source: '+str(target))
        return
    if target.exists() and any(target.iterdir()):raise ValueError('Incomplete existing source: '+str(target))
    target.mkdir(parents=True,exist_ok=True)
    _command(['git','init',str(target)])
    _command(['git','-C',str(target),'remote','add','origin',url])
    _command(['git','-C',str(target),'fetch','--depth=1','origin',revision])
    _command(['git','-C',str(target),'checkout','--detach','FETCH_HEAD'])


def _http(url,target):
    target=Path(target)
    if target.is_file() and target.stat().st_size:return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.download')
    with urlopen(url,timeout=120) as remote,temporary.open('wb') as output:
        while chunk:=remote.read(4*1024*1024):output.write(chunk)
    temporary.replace(target)


def _dpg_csv(root):
    import pandas as pd
    folder=root/'sources/dpgbench_questions';target=folder/'dpgbench.csv'
    frame=pd.read_parquet(folder/'data/test-00000-of-00001.parquet')
    prompts=json.loads((root/'sources/dpgbench/src/eval/dpgbench/eval_prompts/dpgbench_prompts.json').read_text())
    if len(frame)!=1065 or set(frame['item_id'].astype(str))!={Path(key).stem for key in prompts}:
        raise ValueError('Official DPG questions and generation prompt IDs do not match')
    rows=[dict(item_id='__gear_skip_row__',text='',keywords='',proposition_id=0,dependency='0',
               category_broad='sentinel',category_detailed='sentinel',tuple='sentinel',question_natural_language='sentinel')]
    for item in frame.itertuples(index=False):
        for q in item.questions:
            broad,detailed=str(q['category_broad']),str(q['category_detailed'])
            rows.append(dict(item_id=str(item.item_id),text=str(item.prompt),keywords='',
                proposition_id=int(q['proposition_id']),dependency=str(q['dependency']),category_broad=broad,
                category_detailed=detailed,tuple=f'{broad}({detailed})' if detailed else broad,
                question_natural_language=str(q['question'])))
    pd.DataFrame(rows).to_csv(target,index=False)


def download(root=None,*,dry_run=False,only=None):
    root=Path(root or cache_root()).expanduser().absolute();catalog=_data('resources.json')
    if dry_run:return dict(cache_root=str(root),resources=catalog,note='Optional PUP is fetched only when named explicitly')
    selected=set(only.split(',')) if only else None
    known={*catalog['git'],*catalog['huggingface'],*catalog['optional_huggingface'],*catalog['datasets'],*catalog['http'],*catalog['modelscope'],*catalog['google_drive'],'sources','dreamsim'}
    if selected and selected-known:raise ValueError('Unknown resources: '+str(sorted(selected-known)))
    assets={};receipt=root/'download_receipt.json'
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    assets.update(previous.get('assets',{}))
    def want(name):return selected is None or name in selected
    for name,(url,revision) in catalog['git'].items():
        if want(name) or (selected and 'sources' in selected):_git(root,name,url,revision)
    from huggingface_hub import snapshot_download,hf_hub_download
    for section in ('huggingface','optional_huggingface'):
        for key,spec in catalog[section].items():
            if section=='optional_huggingface' and (selected is None or key not in selected):continue
            if not want(key):continue
            destination=root/'weights'/key.lower()
            if 'file' in spec:
                value=hf_hub_download(spec['repo'],spec['file'],revision=spec['revision'],local_dir=str(destination))
            else:
                value=snapshot_download(spec['repo'],revision=spec['revision'],local_dir=str(destination),max_workers=4,
                    ignore_patterns=['*.gguf','*.png','*.jpg','*.jpeg','*.mp4','*.gif'])
            assets[key]=str(value)
    for name,spec in catalog['datasets'].items():
        if not want(name):continue
        destination=root/'sources'/name
        if 'file' in spec:hf_hub_download(spec['repo'],spec['file'],repo_type='dataset',revision=spec['revision'],local_dir=str(destination))
        else:snapshot_download(spec['repo'],repo_type='dataset',revision=spec['revision'],allow_patterns=spec['allow_patterns'],local_dir=str(destination))
    if (root/'sources/dpgbench_questions/data/test-00000-of-00001.parquet').exists() and (root/'sources/dpgbench').exists():_dpg_csv(root)
    for key,spec in catalog['http'].items():
        if not want(key):continue
        path=root/'weights'/spec['file'];_http(spec['url'],path)
        assets[key]=str(path.parent if spec.get('return_directory') else path)
    for key,spec in catalog['modelscope'].items():
        if not want(key):continue
        python=root/'envs/dpg/bin/python'
        if not python.exists():raise ValueError('Install dpg environment first (bench-image setup)')
        destination=root/'weights/mplug';result_file=root/'mplug_download.json'
        code=('from modelscope.hub.snapshot_download import snapshot_download;import json;from pathlib import Path;'
              f'p=snapshot_download({spec["repo"]!r},revision={spec["revision"]!r},cache_dir={str(destination)!r});'
              f'Path({str(result_file)!r}).write_text(json.dumps(dict(path=p,upstream_revision="master")))')
        _command([python,'-c',code]);assets[key]=json.loads(result_file.read_text())['path']
    for key,spec in catalog['google_drive'].items():
        if not want(key):continue
        destination=root/'weights'/spec['file']
        if not destination.is_file():
            python=root/'envs/oneig/bin/python'
            if not python.exists():raise ValueError('Install oneig environment first (bench-image setup)')
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.download')
            _command([python,'-m','gdown','https://drive.google.com/uc?id='+spec['id'],'-O',temporary])
            temporary.replace(destination)
        assets[key]=str(destination)
    if want('dreamsim'):
        python=root/'envs/oneig/bin/python';destination=root/'weights/dreamsim'
        if not python.exists():raise ValueError('Install oneig environment first (bench-image setup)')
        _command([python,'-c',f'from dreamsim import dreamsim; model,_=dreamsim(pretrained=True,device="cpu",cache_dir={str(destination)!r});del model'])
        assets['ONEIG_DREAMSIM_CACHE']=str(destination)
    if 'ONEIG_QWEN_MODEL' in assets:assets['LONGTEXT_QWEN_MODEL']=assets['ONEIG_QWEN_MODEL']
    assets['GENEVAL_MODEL_CONFIG']=str(root/'sources/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')
    assets['QWEN_IMAGE_BENCH_ROOT']=str(root/'sources/qwen_image_bench')
    result=dict(cache_root=str(root),assets=assets,catalog=catalog,
                limitations=['Legacy official mPLUG ModelScope endpoint exposes master, not an immutable revision; see README.'])
    root.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2)+'\n')
    return result


def setup(root=None,*,dry_run=False,only=None):
    root=Path(root or cache_root()).expanduser().absolute();plan=environment_plan(root,only)
    if dry_run:return dict(cache_root=str(root),environments=plan,resources=_data('resources.json'),
        prerequisites=['Conda','CUDA-capable allocated compute node','matching nvcc for MMCV source build','upstream licenses/access'],
        note='Never installs heavy scorers into the training environment; no Slurm dependency')
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Bundled CUDA wheel recipes target Linux x86_64; provide an explicit runtime for other platforms')
    if not (os.environ.get('CONDA_EXE') or shutil.which('conda')):raise RuntimeError('Conda is required for isolated scorer environments')
    root.mkdir(parents=True,exist_ok=True)
    download(root,only='sources')
    for entry in plan:
        prefix=Path(entry['prefix']);marker=prefix/'bench_image_recipe.json'
        if marker.exists():
            if json.loads(marker.read_text())!=entry['recipe']:raise ValueError('Environment recipe changed; choose a new cache root')
            continue
        commands=entry['commands']
        if (prefix/'bin/python').exists():commands=commands[1:]
        env=dict(os.environ)
        if entry['recipe'].get('compile_mmcv'):
            if not shutil.which('nvcc'):raise RuntimeError('GenEval MMCV build requires nvcc on an allocated GPU node')
            env['FORCE_CUDA']='1'
            env['MAX_JOBS']=os.environ.get('MAX_JOBS',os.environ.get('SLURM_CPUS_PER_TASK','4'))
        for command in commands:_command(command,env=env)
        marker.write_text(json.dumps(entry['recipe'],indent=2)+'\n')
    if only:return dict(environments_installed=[entry['name'] for entry in plan],complete=False,
                        next='Run setup without --only to install all scorers and weights')
    downloaded=download(root)
    python={key:str(root/'envs'/entry['name']/'bin/python') for entry in plan for key in entry['recipe']['exports']}
    runtime=dict(cache_root=str(root),source_root=str(root/'sources'),python=python,assets=downloaded['assets'],
        setup_python=sys.executable,environment={'TOKENIZERS_PARALLELISM':'false'})
    (root/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    return doctor(root/'runtime.json')


def doctor(runtime=None,*,gpu=False):
    value=load_runtime(runtime);result={};failures=[]
    catalog=_data('resources.json')
    required_python={key for spec in _data('environments.json').values() for key in spec['exports']}
    required_assets={*catalog['huggingface'],*catalog['http'],*catalog['modelscope'],*catalog['google_drive'],
        'ONEIG_DREAMSIM_CACHE','GENEVAL_MODEL_CONFIG','QWEN_IMAGE_BENCH_ROOT'}
    failures.extend('Missing interpreter key: '+key for key in sorted(required_python-set(value['python'])))
    failures.extend('Missing asset key: '+key for key in sorted(required_assets-set(value['assets'])))
    for key,path in value['python'].items():
        if not Path(path).is_file():failures.append('Missing interpreter: '+key);continue
        process=subprocess.run([path,str(PACKAGE/'probe.py')],capture_output=True,text=True)
        if process.returncode:failures.append('Metadata probe failed: '+key);continue
        result[key]=json.loads(process.stdout)
    for key,path in value['assets'].items():
        if not Path(path).exists():failures.append('Missing asset: '+key)
    sources={}
    for key,(_,revision) in catalog['git'].items():
        source=Path(value['source_root'])/key
        if not source.is_dir():failures.append('Missing official source: '+key);continue
        process=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
        observed=process.stdout.strip();sources[key]=observed
        if process.returncode or observed!=revision:failures.append('Wrong official source revision: '+key)
    if gpu:
        for key,path in value['python'].items():
            code=('import paddle;assert paddle.is_compiled_with_cuda();assert paddle.device.cuda.device_count()>0'
                  if key=='CVTG_OCR_PYTHON' else 'import torch;assert torch.cuda.is_available()')
            process=subprocess.run([path,'-c',code],capture_output=True,text=True)
            if process.returncode:failures.append('CUDA probe failed: '+key)
    report=dict(complete=not failures,failures=failures,interpreters=result,sources=sources,runtime=value['_file'],
        scope='Path/source/metadata checks plus CUDA checks when requested; not a full benchmark result')
    if failures:raise RuntimeError(json.dumps(report,indent=2))
    return report
