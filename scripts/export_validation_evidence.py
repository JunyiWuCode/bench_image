"""Publish portable measured evidence without machine paths or credentials."""
import argparse
import json
from pathlib import Path
import subprocess
from bench_image.config import PACKAGE
from bench_image.fixtures import verify_reference


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--native',type=Path,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.reference
    reference=json.loads((root/'payload/reference.json').read_text())
    expected=json.loads((root/'expected_metrics.json').read_text())
    verified=verify_reference(root/'expected_metrics.json',root/'repeat/summary.json')
    if not verified['passed']:raise ValueError('Independent repeat validation failed')
    runtime=json.loads(a.runtime.read_text());versions={}
    for key,python in runtime['python'].items():
        observed=json.loads(subprocess.check_output([python,str(PACKAGE/'probe.py')],text=True))
        versions[key]={k:v for k,v in observed.items() if k!='executable'}
    native={}
    for path in a.native:
        receipt=json.loads((path/'native_validation.json').read_text())
        if not receipt['complete'] or receipt['images']!=9:raise ValueError('Incomplete native generation validation')
        if not all(row['actual_nfe']==4 and row['pixel_identical'] for row in receipt['comparisons']):
            raise ValueError('Native adapter differs from recorded reference')
        backend=receipt['native_receipt']['config']['backend']
        native[backend]=dict(images=receipt['images'],schedule=receipt['native_receipt']['schedule'],
            comparisons=receipt['comparisons'],hardware='NVIDIA A100-SXM4-80GB',
            note='Measured exact pixel equality on this stack, not a cross-platform guarantee')
    result=dict(reference=reference,expected=expected,independent_repeat=verified,native_generation=native,
        observed_environments=versions,source_catalog=json.loads((PACKAGE/'data/resources.json').read_text()),
        fresh_controller_install=True,fresh_public_sources=True,
        all_fresh_ml_environments_installed=False,
        scorer_weights='Explicit existing official caches, reused read-only; see README for mutable legacy distributions',
        model_revisions={'flux1-raw2q':'2bcb04d9948e5d00ddd9972ff0318d62f4a33951',
                         'flux3-raw2q':'781ff93f92b84397ee0e62e02f052bce99549759'},
        raw_responses='Generated in each scorer run; reference ZIP contains fixed images and numeric expectations, not full-suite scores')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Portable measured evidence:',a.output,flush=True)


if __name__=='__main__':main()
