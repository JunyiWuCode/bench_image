"""Freeze measured reference values only after independent repeat validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bench_image.scoring import write_summary
from bench_image.fixtures import verify_reference

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root
first=write_summary(root/'run');second=write_summary(root/'repeat')
assert len(first['headline_metrics'])==14 and len(second['headline_metrics'])==14
expected=dict(image_count=first['image_count'],headline_metrics=first['headline_metrics'],
    evaluation_mode='fixture',tolerances={key:0.05 for key in first['headline_metrics']},
    note='Measured on the provided subset only; unrelated to the 33468-image full-suite table.')
path=root/'expected_metrics.json';path.write_text(json.dumps(expected,indent=2)+'\n')
verification=verify_reference(path,root/'repeat/summary.json')
(root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2),flush=True)
if not verification['passed']:raise SystemExit('Independent fixture repeat did not meet frozen tolerances')
