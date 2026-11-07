"""GPU smoke for official PUP constants and the portable HF backend."""
import argparse
import json
from pathlib import Path
from bench_image.pup import generate_pup
from bench_image.protocols import read_jsonl

p=argparse.ArgumentParser()
p.add_argument('--records',type=Path,required=True)
p.add_argument('--model',required=True)
p.add_argument('--reference-cache',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
result=generate_pup(a.records,a.model,a.output/'prompts.jsonl',revision='3f3edb52b2c55de0d49feab28598b68f7f8e4830',batch_size=1)
actual=read_jsonl(a.output/'prompts.jsonl')
reference={row['prompt']:row['caption'] for row in read_jsonl(a.reference_cache)}
assert len(actual)==len({row['prompt'] for row in read_jsonl(a.records)})
comparison=[dict(prompt=row['prompt'],caption=row['caption'],
    identical_to_recorded_cache=reference[row['prompt']]==row['caption']) for row in actual]
report=dict(complete=True,result=result,comparison=comparison,
    note='Cache equality is diagnostic. Frozen shared captions, not inference-engine equality, define the comparison.')
(a.output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
