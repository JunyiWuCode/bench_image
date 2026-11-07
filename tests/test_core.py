import json
import io
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

from bench_image.protocols import BENCHMARKS,FIELDS,validate_records,generation_samples
from bench_image.generation import _lock,_native_backend
from bench_image.scoring import _device_tokens,_snapshot_runner,_compatible_configuration
from bench_image.fixtures import extract_example,verify_reference
from bench_image.setup import environment_plan
from bench_image.config import WORKERS,TOOLS
from PIL import Image,ImageChops


class CoreTests(unittest.TestCase):
    def test_global_readout_is_not_mislabeled_as_raw2q(self):
        self.assertEqual(_native_backend('Flux1FourStepPipeline',{'num_horizon_basis':128}),'flux1-global-readout')
        self.assertEqual(_native_backend('Flux1FourStepPipeline',{'gate_value':.25}),'flux1-raw2q')
        with self.assertRaises(ValueError):_native_backend('FluxPipeline',{})
    def test_runtime_asset_additions_do_not_hide_changed_assets(self):
        base=dict(model_name='test',assets={'judge':'original'})
        self.assertTrue(_compatible_configuration(base,dict(model_name='test',assets={'judge':'original','new_scorer':'weights'})))
        self.assertFalse(_compatible_configuration(base,dict(model_name='test',assets={'judge':'different'})))
        self.assertFalse(_compatible_configuration(base,dict(model_name='other',assets={'judge':'original'})))
    def test_protocol_columns_and_counts(self):
        from bench_image._legacy.official_suite import OFFICIAL_PROTOCOLS
        self.assertEqual(len(BENCHMARKS),11)
        self.assertEqual(len(FIELDS),14)
        self.assertEqual(sum(OFFICIAL_PROTOCOLS[n].images for n in BENCHMARKS),33468)
        self.assertIn('HPSv3',FIELDS)
        self.assertNotIn('HPSv3++',FIELDS)

    def test_fail_closed_repetition_and_full(self):
        rows=[dict(benchmark='geneval',artifact_id=f'geneval:{i}',prompt_index=0,sample_index=i,
                   width=1024,height=1024,prompt='two cats',reportable=False) for i in range(4)]
        validate_records(rows)
        with self.assertRaisesRegex(ValueError,'repetition'):validate_records(rows[:3])
        with self.assertRaisesRegex(ValueError,'full benchmark'):validate_records(rows,require_full=True)
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_records(rows+rows)
        with self.assertRaisesRegex(ValueError,'dimensions'):validate_records([dict(r,width=-16) for r in rows])
        with self.assertRaisesRegex(ValueError,'indices'):validate_records([dict(r,prompt_index=-1) for r in rows])

    def test_resume_config_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json'
            _lock(path,dict(steps=4));_lock(path,dict(steps=4))
            with self.assertRaisesRegex(ValueError,'incompatible'):_lock(path,dict(steps=8))
            self.assertEqual(json.loads(path.read_text()),dict(steps=4))
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_allocated_gpu_tokens(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-abc,GPU-def'}):
            self.assertEqual(_device_tokens('1,0'),['GPU-def','GPU-abc'])
            with self.assertRaises(ValueError):_device_tokens('2')
            with self.assertRaises(ValueError):_device_tokens('0,0')

    def test_zip_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as bundle:bundle.writestr('../outside','unsafe')
            with self.assertRaisesRegex(ValueError,'Unsafe'):extract_example(Path(d)/'out',archive)
            self.assertFalse((Path(d)/'outside').exists())

    def test_zip_restores_exact_rgb_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'fixture.zip'
            image=Image.new('RGB',(64,64),(27,91,188));buffer=io.BytesIO()
            image.save(buffer,format='WEBP',lossless=True)
            row=dict(benchmark='cvtg',artifact_id='cvtg:0:0',prompt_index=0,sample_index=0,
                width=64,height=64,prompt='A textured poster',reportable=False,fixture_image='image.webp')
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('records.jsonl',json.dumps(row)+'\n')
                bundle.writestr('image.webp',buffer.getvalue())
                for name in ('reference','generation_config','expected_metrics','selection'):
                    bundle.writestr(name+'.json','{}')
            root=Path(d)/'out'
            result=extract_example(root,archive)
            self.assertFalse(result['benchmark_complete'])
            self.assertEqual(result['images'],1)
            with Image.open(root/'cvtg/raw_images/000000_00.png') as decoded:
                self.assertIsNone(ImageChops.difference(image,decoded).getbbox())

    def test_reference_requires_complete_exact_columns(self):
        with tempfile.TemporaryDirectory() as d:
            expected=Path(d)/'expected.json';actual=Path(d)/'actual.json'
            expected.write_text(json.dumps(dict(image_count=106,headline_metrics={'GenEval':50},tolerances={'GenEval':.05})))
            value=dict(image_count=106,headline_metrics={'GenEval':50.01},selection_complete=True,evaluation_mode='fixture')
            actual.write_text(json.dumps(value));self.assertTrue(verify_reference(expected,actual)['passed'])
            value['headline_metrics']['GenEval']=50.1
            actual.write_text(json.dumps(value));self.assertFalse(verify_reference(expected,actual)['passed'])
            value['selection_complete']=False;actual.write_text(json.dumps(value))
            with self.assertRaises(ValueError):verify_reference(expected,actual)

    def test_scorer_attempt_logs_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            first=_snapshot_runner(Path(d));second=_snapshot_runner(Path(d))
            self.assertNotEqual(first,second)
            self.assertTrue(first.exists())
            self.assertNotIn('$OUTPUT/logs',first.read_text())
            self.assertIn('LOGS=',first.read_text())

    def test_paddle_model_identity_and_cvtg_spaces(self):
        text=(WORKERS/'score_cvtg.py').read_text()
        self.assertIn('recognized_text(raw, separator=" ")',text)
        loader=(WORKERS/'score_ocr.py').read_text()
        self.assertIn('en_PP-OCRv5_mobile_rec',loader)
        self.assertIn('PP-OCRv5_server_det',loader)

    def test_setup_isolated_and_paddle_not_downgraded(self):
        with tempfile.TemporaryDirectory() as d:
            plan=environment_plan(Path(d))
        self.assertGreaterEqual(len(plan),12)
        ocr=next(item for item in plan if 'CVTG_OCR_PYTHON' in item['recipe']['exports'])
        self.assertIn('paddleocr==3.3.3',ocr['recipe']['packages'])
        self.assertEqual(ocr['recipe']['paddle'],'3.3.1')
        for item in plan:self.assertTrue(item['prefix'].startswith(d))

    def test_tiif_full_and_diagnostic_modes_are_distinct(self):
        source='line_indices = sample_idx[attr_type]\ntemperature=1.0 # You may set it to 0 if you require stricter reproducibility.\nprompt_template = random.choice([raw_prompt, raw_prompt_1, raw_prompt_2])\n'
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/'source.py';full=Path(d)/'full.py';fixture=Path(d)/'fixture.py'
            original.write_text(source)
            command=[sys.executable,str(TOOLS/'prepare_tiif_pinned_evaluator.py'),'--source',str(original)]
            subprocess.run(command+['--output',str(full)],check=True,capture_output=True)
            subprocess.run(command+['--output',str(fixture),'--deterministic-fixture'],check=True,capture_output=True)
            self.assertIn('temperature=1.0',full.read_text())
            self.assertIn('random.choice(',full.read_text())
            self.assertIn('temperature=0.0, seed=0',fixture.read_text())
            self.assertIn('random.Random("fixture-v1:',fixture.read_text())


if __name__=='__main__':unittest.main()
