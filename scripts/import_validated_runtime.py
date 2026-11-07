"""Import the previously validated Apache-2.0 adapter code, without model weights.

This maintenance command is not required for installation or execution.
The source argument is an existing FAR image-benchmark checkout. It performs
mechanical namespace/path changes only; subsequent functional changes are
tracked normally in this repository.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    source=args.source.resolve()
    root=Path(__file__).resolve().parents[1]
    package=root/'src/bench_image'
    legacy=package/'_legacy';legacy.mkdir(parents=True,exist_ok=True)
    if (legacy/'official_suite.py').exists():
        raise RuntimeError('Import is a one-time bootstrap, not an update command: refusing to overwrite maintained adapters')
    (legacy/'__init__.py').touch()
    upstream=source/'third_party/benchmark_image/benchmark_image'
    def rewrite(text):
        return (text.replace('from benchmark_image.', 'from bench_image._legacy.')
                    .replace('files("benchmark_image")', 'files("bench_image._legacy")'))
    for name in ('official_suite.py','official_layouts.py','q_judger_backends.py'):
        (legacy/name).write_text(rewrite((upstream/name).read_text()))
    workers=legacy/'workers';workers.mkdir(exist_ok=True)
    for name in ('score_geneval.py','summarize_geneval.py','score_geneval2.py','summarize_geneval2.py',
                 'score_cvtg.py','score_ocr.py','score_longtext.py','score_hpsv3_official.py','score_qwen_image_bench.py'):
        (workers/name).write_text(rewrite((upstream/'workers'/name).read_text()))
    for name in ('cvtg','qwen_image_bench'):
        shutil.copytree(upstream/'assets'/name,legacy/'assets'/name,dirs_exist_ok=True)
    lock=json.loads((upstream/'assets/source_locks/zimage_base_english_official_v1.json').read_text())
    selected={'geneval','geneval2','dpgbench','tiif_short','tiif_long','cvtg','longtext_en','oneig_en',
              'qwen_image_bench_en','hpsv3_official','wise_verified'}
    selected_sources=selected-{'tiif_short','tiif_long'}|{'tiif','dpgbench_questions'}
    lock['benchmarks']=[n for n in lock['benchmarks'] if n in selected]
    for key,keep in [('sources',selected_sources),('protocols',selected),('scorers',selected_sources)]:
        lock[key]={k:v for k,v in lock[key].items() if k in keep}
    target=legacy/'assets/source_locks/zimage_base_english_official_v1.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(lock,indent=2)+'\n')
    scripts=package/'runtime/scripts';scripts.mkdir(parents=True,exist_ok=True)
    names=('materialize_zimage_official_scorer.py','prepare_geneval2_pinned_evaluator.py',
           'prepare_tiif_pinned_evaluator.py','run_tiif_parallel_clients.py','prepare_oneig_pinned_source.py',
           'run_oneig_alignment_shard.py','collect_oneig_official_scores.py','collect_wise_verified_scores.py',
           'prepare_hpsv3_pinned_config.py','run_zimage_official_benchmark_scorer.sh')
    for name in names:
        text=rewrite((source/'scripts'/name).read_text())
        text=text.replace('WORKER_ROOT="$REPO_ROOT/third_party/benchmark_image/benchmark_image/workers"',
                          'WORKER_ROOT=${BENCH_WORKER_ROOT:?set BENCH_WORKER_ROOT}')
        text=text.replace('${BENCH_SETUP_PYTHON:-/home/hcai/workspace/anaconda3/envs/far/bin/python}',
                          '${BENCH_SETUP_PYTHON:-python}')
        (scripts/name).write_text(text)
    collector=(source/'scripts/collect_zimage_official_benchmark_scores.py').read_text()
    functions=[ast.get_source_segment(collector,node) for node in ast.parse(collector).body
               if isinstance(node,ast.FunctionDef) and node.name in ('finite','load_json','collect_dpg','collect_tiif')]
    (package/'_collect_native.py').write_text('"""Unchanged, previously validated DPG/TIIF aggregation adapters."""\nimport json, math, re\nfrom pathlib import Path\n\n'+'\n\n'.join(functions)+'\n')
    shutil.copy2(source/'third_party/benchmark_image/LICENSE',root/'LICENSE')
    (package/'data').mkdir(exist_ok=True)
    (package/'data/provenance.json').write_text(json.dumps(dict(
        origin='JunyiWuCode/benchmark_image (Apache-2.0) and FAR-Dev benchmark integration',
        benchmark_revision='b8159ebc84ca2197fd5072fe957c9621ad6a5efe',
        project_revision='d1453ab977b3678297ecba9ee2a8aa8c6078431f',
        source='Previously validated 33468-image English open benchmark suite; local adapter corrections retained',
        note='Upstream dataset and model licenses remain separate. No weights are included.'),indent=2)+'\n')
    print('Imported validated runtime into',package)


if __name__=='__main__':main()
