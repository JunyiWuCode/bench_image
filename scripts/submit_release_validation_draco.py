"""Optional CPU release packaging job, after every scorer/repeat succeeds."""
import argparse
import json
from pathlib import Path
import shlex
from submit_validation_draco import submit,ROOT

p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--reference-run',type=Path,required=True)
p.add_argument('--native',type=Path,nargs=2,required=True)
p.add_argument('--pup',type=Path,required=True)
p.add_argument('--dependency',type=int,required=True)
a=p.parse_args();root=a.reference.absolute()
receipt=root/'release_job.json'
if receipt.exists():raise ValueError('Release job already submitted; inspect live/accounting state before retrying')
runtime=json.loads(a.runtime.read_text());python=runtime['setup_python']
commands=[
    [python,'scripts/export_validation_evidence.py','--reference',root,'--runtime',a.runtime.absolute(),
     '--native',*[path.absolute() for path in a.native],'--pup',a.pup.absolute(),'--output','examples/validation.json'],
    [python,'scripts/build_reference_fixture.py','--reference-run',a.reference_run.absolute(),'--output',root,
     '--pack','examples/reference.zip'],
    [python,'scripts/validate_reference_archive.py','--archive','examples/reference.zip',
     '--reference-run',a.reference_run.absolute(),'--output',ROOT/'validation_runs/extracted_release'],
]
command=' && '.join(shlex.join(list(map(str,item))) for item in commands)
job=submit(command,'bench-image-release-check',root/'logs/release_%j.log',
    cpus=4,memory='32G',dependency=[a.dependency],time='00:20:00')
receipt.write_text(json.dumps(dict(job_id=job,dependency=a.dependency,commands=[[str(v) for v in item] for item in commands]),indent=2)+'\n')
print(job,flush=True)
