"""Decode the real release ZIP and compare every image with its source PNG."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageChops
from bench_image.fixtures import extract_example
from bench_image.protocols import read_jsonl,output_path_for_record

p=argparse.ArgumentParser()
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--reference-run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
report=extract_example(a.output,a.archive)
rows=read_jsonl(a.output/'records.jsonl')
for row in rows:
    with Image.open(output_path_for_record(a.output,row)) as restored,Image.open(output_path_for_record(a.reference_run/'run',row)) as original:
        if ImageChops.difference(restored.convert('RGB'),original.convert('RGB')).getbbox() is not None:
            raise ValueError('Release archive changed source pixels: '+row['artifact_id'])
expected=json.loads((a.output/'expected_metrics.json').read_text())
assert len(rows)==106 and len(expected['headline_metrics'])==14
report.update(all_pixels_identical=True,headline_columns=14)
(a.output/'archive_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
