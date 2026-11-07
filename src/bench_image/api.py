"""Model-independent interface, analogous to AnyFlowBench for video."""
import json
import os
from pathlib import Path
from PIL import Image
from .config import cache_root,load_runtime
from .protocols import generation_samples,read_jsonl,write_jsonl,output_path_for_record,validate_records


def register_images(records,samples_dir,output_dir):
    """Attach existing lossless PNGs without copying or modifying source images."""
    source=Path(samples_dir).expanduser().absolute();root=Path(output_dir).expanduser().absolute()
    root.mkdir(parents=True,exist_ok=True);rows=[dict(row) for row in records];validate_records(rows)
    record_path=root/'records.jsonl'
    if record_path.exists() and read_jsonl(record_path)!=rows:raise ValueError('Output contains another sample selection')
    generated=[]
    existing={r['artifact_id']:r for r in read_jsonl(source/'generation_manifest.jsonl')} if (source/'generation_manifest.jsonl').exists() else {}
    for row in rows:
        original=output_path_for_record(source,row)
        with Image.open(original) as image:
            if image.format!='PNG' or image.size!=(row['width'],row['height']):raise ValueError(f'Invalid source PNG: {original}')
        target=output_path_for_record(root,row);target.parent.mkdir(parents=True,exist_ok=True)
        if source!=root:
            if target.exists() or target.is_symlink():
                if target.resolve()!=original.resolve():raise ValueError('Conflicting image destination')
            else:target.symlink_to(os.path.relpath(original,target.parent))
        generated.append(dict(existing.get(row['artifact_id'],row),image_path=str(target)))
    write_jsonl(record_path,rows);write_jsonl(root/'generation_manifest.jsonl',generated)
    (root/'generation_summary.json').write_text(json.dumps(dict(complete=True,record_count=len(rows),
        provenance='existing immutable PNGs; no inferred generation steps or NFE'),indent=2)+'\n')
    return root


class ImageBench:
    def __init__(self,*,cache_dir=None,runtime=None,devices='0',resolution=1024,seed=0):
        self.cache_dir=Path(cache_dir or cache_root()).expanduser().absolute()
        self.runtime=runtime;self.devices=devices;self.resolution=resolution;self.seed=seed

    def generation_samples(self,*,benchmarks=None,selection=None):
        from .protocols import BENCHMARKS
        source=load_runtime(self.runtime)['source_root'] if self.runtime else self.cache_dir/'sources'
        return generation_samples(source_root=source,benchmarks=benchmarks or BENCHMARKS,
            resolution=self.resolution,seed=self.seed,selection=selection)

    def evaluate(self,samples_dir,output_dir=None,*,model='model',records=None,benchmarks=None):
        from .scoring import evaluate,write_summary
        samples_dir=Path(samples_dir)
        if records is None:records=read_jsonl(samples_dir/'records.jsonl')
        root=register_images(records,samples_dir,output_dir or samples_dir)
        report=evaluate(root,load_runtime(self.runtime or self.cache_dir/'runtime.json'),
                        benchmarks=benchmarks,devices=self.devices,model_name=model)
        return write_summary(root) if report['selection_complete'] else report
