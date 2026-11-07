"""Small portable example images and strict diagnostic reference comparison."""
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from urllib.request import urlopen
import zipfile
from PIL import Image
from .protocols import read_jsonl,write_jsonl,output_path_for_record,validate_records

DEFAULT_URL='https://raw.githubusercontent.com/JunyiWuCode/bench_image/v0.1.0/examples/reference.zip'


def extract_example(output,archive=None):
    """Extract losslessly; reject traversal, links, wrong IDs/counts or resolution."""
    root=Path(output).expanduser().absolute()
    if root.exists() and any(root.iterdir()):raise ValueError('Example output must be empty')
    root.mkdir(parents=True,exist_ok=True)
    temporary=None
    if archive is None:
        temporary=tempfile.TemporaryDirectory(prefix='bench-image-example-')
        archive=Path(temporary.name)/'reference.zip'
        with urlopen(DEFAULT_URL,timeout=120) as source,archive.open('wb') as target:
            while chunk:=source.read(1024*1024):target.write(chunk)
    try:
        with zipfile.ZipFile(archive) as bundle:
            if sum(item.file_size for item in bundle.infolist())>2*1024**3:raise ValueError('Unexpectedly large example archive')
            for item in bundle.infolist():
                name=Path(item.filename)
                if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(item.external_attr>>16):
                    raise ValueError('Unsafe ZIP member')
            records=[json.loads(line) for line in bundle.read('records.jsonl').decode().splitlines() if line.strip()]
            validate_records(records)
            if any(row.get('reportable') for row in records):raise ValueError('Examples must not claim full benchmark scores')
            manifest=[]
            for row in records:
                target=output_path_for_record(root,row);target.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open(row['fixture_image']) as file,Image.open(file) as image:
                    image.load()
                    if image.size!=(row['width'],row['height']):raise ValueError('Fixture size mismatch')
                    image.convert('RGB').save(target,format='PNG')
                result=dict(row,image_path=str(target));manifest.append(result)
            for name in ('reference.json','generation_config.json','expected_metrics.json','selection.json'):
                (root/name).write_bytes(bundle.read(name))
            write_jsonl(root/'records.jsonl',records);write_jsonl(root/'generation_manifest.jsonl',manifest)
            (root/'generation_summary.json').write_text(json.dumps(dict(complete=True,record_count=len(records),
                note='Lossless WebP fixture decoded to identical RGB PNG pixels'),indent=2)+'\n')
    finally:
        if temporary is not None:temporary.cleanup()
    return dict(output=str(root),images=len(records),mode='fixture',benchmark_complete=False)


def verify_reference(expected,actual,*,atol=None):
    expected=json.loads(Path(expected).read_text());actual=json.loads(Path(actual).read_text())
    if not actual.get('selection_complete') or actual.get('evaluation_mode')!='fixture':
        raise ValueError('Expected a complete diagnostic fixture evaluation, not a partial/full benchmark report')
    if expected['image_count']!=actual['image_count']:raise ValueError('Fixture image count changed')
    if set(expected['headline_metrics'])!=set(actual['headline_metrics']):raise ValueError('Missing or unexpected metric columns')
    differences={};passed=True
    for name,reference in expected['headline_metrics'].items():
        observed=float(actual['headline_metrics'][name]);difference=abs(float(reference)-observed)
        tolerance=float(atol if atol is not None else expected.get('tolerances',{}).get(name,0.05))
        okay=math.isfinite(observed) and difference<=tolerance
        passed &= okay
        differences[name]=dict(expected=reference,actual=observed,absolute_difference=difference,tolerance=tolerance,passed=okay)
    return dict(passed=passed,metrics=differences,
        note='A matching fixture validates scoring consistency for these inputs; it is not a full-suite quality score or proof of universal correctness.')
