# bench_image

Generate images first, then score them with isolated, open-source evaluators.
This is the standalone image-benchmark integration used by the
[FAR AnyFlow2 FLUX branch](https://github.com/dc-ai-projects/FAR-Dev/tree/junyiwu/anyflow-v2-flux-unified).
The model-independent Python API and fixed-image replay example follow the
approach of [Anyflow_T2VA_Bench](https://github.com/Orannue/Anyflow_T2VA_Bench).

No closed API, HPSv2, HPSv3++, CoReBench or WeGenBench is part of this suite.
**HPSv3 means the original MizzenAI HPSv3 scorer on its official HPDv3 prompts.**

## Install

```bash
git clone https://github.com/JunyiWuCode/bench_image.git
cd bench_image
python3 -m venv .venv
.venv/bin/python -m pip install .
export BENCH_IMAGE_CACHE=/path/to/large/persistent/bench-image
.venv/bin/bench-image setup --dry-run
.venv/bin/bench-image setup
.venv/bin/bench-image doctor --gpu
```

Run downloads/environment creation on an allocated CPU compute node; run CUDA
extension builds and scoring on an allocated GPU node, **not a login node**.
Setup requires Linux x86_64, Conda, a CUDA-capable driver and matching `nvcc`
for the GenEval MMCV build. It creates **12 separate worker environments**
under the cache, in addition to the lightweight controller venv above. It
never installs incompatible scorers into a training environment.
Plan for several hundred GB of disk space for environments and scorer weights.
Accept gated-model licenses and authenticate HF outside the repository when
required. Never put credentials in a config or commit.

The single entry point installs environments, fetches pinned public upstream
sources, and downloads official assets. No private FAR-RL checkout is required.
Critical environment versions are in
[`environments.json`](src/bench_image/data/environments.json), and source/model
revisions in [`resources.json`](src/bench_image/data/resources.json).
Setup resumes matching installed recipes; a changed recipe requires a new
cache root. `--only ocr,clip` installs selected environments but intentionally
does **not** claim complete setup. `download --only sources` fetches only code.

An existing installation can supply a JSON runtime via `--runtime runtime.local.json`
(never commit that file). Its keys are `source_root`, `setup_python`, `python`
(the interpreter exports in the environment recipes), `assets` (the resource
catalog's exported paths), and optional `environment`. `${CACHE}` in paths is
expanded using `cache_root`. A runtime is explicit: the library never searches
other projects for environments or weights. `doctor` checks required paths,
public source revisions and installed metadata; `--gpu` additionally checks
CUDA. It is not itself a successful benchmark pass.

### Hardware and installation limitations

Generation uses one GPU per task. The bundled full scoring interface uses one
node with **eight 80-GB GPUs** for the TIIF 72B judge and WISE; smaller scorers
can run on one GPU. The diagnostic fixture routes smaller scorers to one GPU
even when all eight are allocated. Only one- or eight-GPU configurations are
accepted. Logical GPU indices honor the scheduler's CUDA mask; no Slurm
partition or cluster path is required by the library.

The fresh lightweight controller and fresh public-source checkouts have been
tested separately from the existing isolated GPU scorer environments. The
scoring validation reuses those previously validated environments/weights;
this does **not** establish that all 12 fresh worker environment recipes have been
reinstalled and exercised on every platform. FlashAttention wheels target the
specified Python/Torch/CUDA ABI. Adjust an explicit runtime for another ABI.
PaddleOCR remains **3.3.3 / Paddle GPU 3.3.1**, not a downgraded OCR stack.
The adopted CLIP-only environment also contains unused legacy Paddle packages;
they never perform OCR here and are not installed by the fresh CLIP recipe.
The exported environment metadata lists installed packages, not just imports.

Two original upstream distributions are not immutable: mPLUG is served from
ModelScope `master`, and OneIG CSD from its official Google Drive file ID.
Keep those download receipts and caches for repeat studies; a Git source pin
does not freeze a mutable model endpoint. DreamSim also downloads its official
released dependencies. If institutional networking blocks Google Drive,
download the [official CSD file](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view)
on an allowed machine and configure `ONEIG_CSD_MODEL`; do not substitute a
different style encoder. Existing legacy WISE runs used the official
ModelScope Qwen3.5-35B-A3B distribution; the fresh catalog pins its HF mirror.
Their byte identity has not been established, so mirror changes must be noted
when comparing scores.

## Protocols

All prompts are original benchmark prompts at **1024 × 1024**, seed base 0,
unless the caller explicitly requests a separate resolution experiment.
For prompt index `p` and repetition `s`, the image seed is `base + 1000*p + s`.
Original prompt IDs and complete official repetition groups are preserved.

| CLI name | Official prompts | Images/prompt | Images | Headline metric(s) |
| --- | ---: | ---: | ---: | --- |
| `geneval` | 553 | 4 | 2,212 | GenEval |
| `geneval2` | 800 | 1 | 800 | Soft-TIFA AM / GM |
| `dpgbench` | 1,065 | 4 | 4,260 | DPG-Bench |
| `tiif_short` | 2,538 | 1 | 2,538 | TIIF Short |
| `tiif_long` | 2,538 | 1 | 2,538 | TIIF Long |
| `cvtg` | 2,000 | 1 | 2,000 | Word Accuracy / NED / CLIPScore |
| `longtext_en` | 160 | 4 | 640 | LongText-EN |
| `oneig_en` | 1,120 | 4 | 4,480 | OneIG-EN Overall |
| `qwen_image_bench_en` | 1,000 | 1 | 1,000 | Qwen-Image-Bench-EN Overall |
| `hpsv3_official` | 12,000 | 1 | 12,000 | HPSv3 |
| `wise_verified` | 1,000 | 1 | 1,000 | WISE Overall |
| **Total** | | | **33,468** | **14 columns** |

The packaged source lock documents the evaluated protocol, not just each
benchmark's name. In particular:

- GenEval2 uses the official Qwen3-VL Soft-TIFA evaluator and AM/GM aggregation.
- DPG uses official dependency-aware mPLUG questions and four-image grids.
- TIIF Short/Long share the official local Qwen2.5-VL-72B judge and nine-group
  macro aggregation. Full runs retain upstream temperature 1 and random question
  template selection. The local service is an open model, not a closed API.
- CVTG joins PaddleOCR regions with spaces; CLIP uses the official ViT-L/14
  and `A photo depicts ` text prefix. WA/NED/CLIP are not interchangeable.
- OneIG combines its five native components, including both official style
  encoders and DreamSim; it is not a text-alignment-only score.
- Qwen uses the **English 1,000-prompt** protocol and official Q-Judger mapping
  `0/1/2/N/A → 0/60/100/excluded`, retaining raw judge responses.
- WISE uses the original verified prompts (no rewrite) and local open judge.

Headlines use the historical table's 100× scaling for fractional metrics;
HPSv3 stays in native units. CVTG CLIP is `100 × 2.5 × max(cosine, 0)`,
not a bounded accuracy percentage. Raw component units remain available in
per-benchmark summaries.

## Python API: bring your own generator

```python
from pathlib import Path
from bench_image import ImageBench
from bench_image.protocols import output_path_for_record, write_jsonl

bench = ImageBench(devices="0,1,2,3,4,5,6,7", resolution=1024, seed=0)
records = bench.generation_samples()
root = Path("runs/my_model")
write_jsonl(root / "records.jsonl", records)
for record in records:
    # Your own native pipeline; keep this exact prompt, size and per-image seed.
    image = native_generate(record)
    target = output_path_for_record(root, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")

# Invoke this only after generation has finished, in a separate scorer job.
report = bench.evaluate(root, model="my_model", records=records)
print(report["headline_metrics"])
```

`native_generate` is your model-specific function, not a function shipped by
this library. `generation_samples()` loads prompt metadata only: no model/GPU
allocation. A custom generator must save its own model/code revision,
inference configuration and actual NFE; the evaluator never guesses those
from a directory name. `selection={"geneval": [0, 1]}` together with
`benchmarks=["geneval"]` makes a diagnostic subset, never a full score.

## Native AnyFlow2 FLUX generation

The included generation adapter supports the published **native custom
`Flux1FourStepPipeline` and `Flux3FourStepPipeline` exports**. Other models
can use the API above. Unknown pipeline classes are rejected; ordinary FLUX
sampling is never substituted for an interval-conditioned FlowMap update.
The branch's newer FLUX.1 128-basis global-readout export shares the native
four-step pipeline contract; its configuration is recorded separately as
`flux1-global-readout`, not mislabeled Raw2Q. The pixel-parity GPU tests below
cover the two specified Raw2Q reference exports, not that newer checkpoint.

| Export family | Steps / actual NFE | Guidance | Shift | Last interval scale | Prompt |
| --- | --- | ---: | ---: | ---: | --- |
| FLUX.1 Raw2Q | 4 / 4 | 3.5 guidance embedding, not external CFG | exp(1.15) | 0.3 | Original raw |
| FLUX.3 Raw2Q | 4 / 4 | 1.0, guidance-distilled | 6.93 | 0.3 | Cached official PUP |

Both keep BF16 transformer/text encoders, **FP32 VAE and integration**, CPU
per-image generators and native timestep grids. FLUX.1 uses BCHW BF16 noise;
FLUX.3 uses token-major FP32 noise rounded through BF16. Receipts record raw
and shifted times and a forward-hook count of actual transformer NFE.

Review the pinned remote model code before enabling `--trust-remote-code`.

```bash
.venv/bin/bench-image samples --output runs/flux1/records.jsonl
.venv/bin/bench-image download --only sources
# Run this with the generation environment's Python on one GPU:
python -m bench_image generate --records runs/flux1/records.jsonl \
  --output runs/flux1 --model kimi000/Flux1-Dev-4Step-Raw2Q-3000 \
  --revision 2bcb04d9948e5d00ddd9972ff0318d62f4a33951 \
  --steps 4 --guidance 3.5 --trust-remote-code
```

Install this lightweight package in that environment with `pip install --no-deps .`
or expose `src` on `PYTHONPATH`. The FAR wrapper's `run` command handles the
interpreter handoff. The native export accepts one CPU Generator rather than
a generator list: calls preserve one image's native random sequence. Scale
through independent **single-GPU** shards, e.g. `--num-shards 256 --shard-id 0`
through 255; never change seeds by shard. Each completed PNG gets an append-only
receipt. Matching shards resume without regeneration. Incompatible configs or
orphan PNGs without receipts fail rather than silently reuse another model.

### FLUX.3 prompt upsampling

```bash
.venv/bin/bench-image download --only FLUX3_PUP_MODEL
# Use GENERATION_PYTHON on a GPU, with bench_image installed/visible:
python -m bench_image pup --records runs/flux3/records.jsonl \
  --model black-forest-labs/flux-3-image-klein-pup \
  --revision 3f3edb52b2c55de0d49feab28598b68f7f8e4830 \
  --output runs/flux3_pup/prompts.jsonl
python -m bench_image generate --records runs/flux3/records.jsonl \
  --output runs/flux3 --model kimi000/Flux3-Klein-4Step-Raw2Q-2000 \
  --revision 781ff93f92b84397ee0e62e02f052bce99549759 \
  --steps 4 --guidance 1 --prompt-map runs/flux3_pup/prompts.jsonl \
  --trust-remote-code
```

First create FLUX.3 records with `samples`. PUP uses the official `t2i` system
prompt, thinking disabled, greedy/temperature-0 decoding, generated-token
presence penalty 1.5 and max 3,584 new tokens. The included Transformers backend
preserves those constants, but does not promise token-identical output to
vLLM on different hardware. **Reuse one frozen caption cache across models**;
the cache is the reproducibility boundary. Records retain `prompt`; generation
receipts retain `model_input_prompt`. Judges always see original benchmark
text. FLUX.1 Raw2Q forbids PUP in this protocol; this preprocessing difference
must accompany cross-family comparisons. Raw FLUX.3 is an explicit ablation
(`--allow-raw-flux3`), not the reported default.

## Score and resume

```bash
.venv/bin/bench-image evaluate --run runs/flux1 \
  --devices 0,1,2,3,4,5,6,7 --model-name FLUX1-Raw2Q-3000
.venv/bin/bench-image collect --run runs/flux1
```

After a generation array succeeds, submit scorer jobs with an `afterok`
dependency. A scorer can select `--benchmarks geneval` etc.; TIIF Short/Long
are handled together to reuse their server. The final `collect` depends on
all scorers. `scripts/submit_validation_draco.py` is an **optional Draco-only**
example; the library itself does not require Slurm. No redundant image
preparation stage is required: the scoring boundary joins existing receipts
once and materializes each scorer's native layout.

Outputs include `records.jsonl`, image PNGs, `generation_config.json`, per-shard
generation manifests and actual NFE, `generation_manifest.jsonl`, raw scorer
responses/results, per-benchmark `complete.json`, `summary.json` and
`metrics_summary.csv`. Commands and failure logs are retained. Cached completed
scorers are reused only with compatible runtime/model settings. A partial image
set, incomplete official repetition group, missing metric, parse failure or
nonfinite value cannot become a completed full report. Failed judgments are
not zero-filled. Regeneration is unnecessary for a score-only retry.

## Reproduce the fixed-image example

The versioned [`examples/reference.zip`](examples/reference.zip) contains
**106 original-resolution images** selected by original prompt IDs across all
11 protocols. Repetition groups and category coverage are retained. Pixels
are stored losslessly as WebP to keep the archive below GitHub's single-file
limit; extraction restores **identical RGB PNG pixels**, without resizing.
It also contains prompts, seeds, model revision, native generation config,
selection IDs and measured diagnostic expectations.

```bash
.venv/bin/bench-image example --output runs/reference
.venv/bin/bench-image evaluate --run runs/reference \
  --devices 0,1,2,3,4,5,6,7 --model-name reference
.venv/bin/bench-image collect --run runs/reference
.venv/bin/bench-image verify --expected runs/reference/expected_metrics.json \
  --actual runs/reference/summary.json
```

`example` downloads the ZIP from tag `v0.1.0`, or accepts `--archive /path/reference.zip`.
These are images from FLUX.1 Raw2Q-3000, **not** cherry-picked model quality
comparisons and **not** a 33,468-image full benchmark. The expected numbers
are measured on the exact subset. A second independent scoring pass checks a
predeclared tolerance of **0.05 score points** per headline metric. GPU/kernel,
processor and VLM decoding changes can still affect results; agreement checks
scorer consistency for these inputs, not universal correctness.

**Diagnostic TIIF exception:** its official evaluator uses temperature 1 and
randomly chooses one of three question templates, so an unseeded nine-prompt
replay did not reproduce closely (Short 59.86 vs 49.86; Long 48.49 vs 55.53).
We retain that failed validation evidence. The ZIP diagnostic instead uses the
upstream evaluator's explicitly permitted **temperature 0**, request seed 0,
and a deterministic per-question choice among the same three official templates.
This applies only to subsets/fixtures. Full-suite TIIF keeps the original
stochastic protocol; its historical numbers are not overwritten by fixture
numbers. These two modes must not be mixed in a model-quality comparison.
Online serving still varied in the seeded temperature-0 test (Short
50.69/49.58, Long 54.14/51.92). The diagnostic client therefore uses offline
vLLM with in-process deterministic scheduling, one sequence, eager execution
and no prefix/chunked-prefill caching, as advised by
[vLLM's reproducibility documentation](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/).
It retains the same Qwen2.5-VL-72B weights, original messages, native parsing
and aggregation. Full-suite online scoring is unchanged. Failed earlier
replays remain recorded; neither tolerance nor sample selection is adjusted
to conceal them.

### Measured diagnostic reference

The complete measured table and independent replay evidence are published
alongside the final validated archive. Missing results are not filled from
papers or full-run scores.

### Historical full-suite measurements (not the ZIP subset)

These are previously completed runs of the same 33,468-image English protocol
on Draco A100-80GB GPUs, 1024px, seed base 0, native four-step sampling. They
are historical measurements, **not newly rerun full evaluations during package
installation validation**. Revisions are listed below. FLUX.3 used cached PUP;
FLUX.1 used raw prompts. No paper-reported number is substituted.

| Model | GenEval | GE2 AM | GE2 GM | DPG | TIIF S | TIIF L | CVTG WA | CVTG NED | CVTG CLIP | LongText | OneIG | Qwen EN | HPSv3 | WISE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLUX.1 Raw2Q-1000 | 66.07 | 69.51 | 21.16 | 84.46 | 63.95 | 66.32 | 51.34 | 72.44 | 76.07 | 36.21 | 36.50 | 42.08 | 11.10 | 43.70 |
| FLUX.1 Raw2Q-3000 | 67.23 | 70.17 | 21.32 | 84.91 | 64.23 | 67.42 | 56.89 | 76.63 | 77.34 | 42.56 | 39.34 | 41.44 | 11.03 | 45.10 |
| FLUX.3 Raw2Q-2000 + PUP | 89.48 | 90.44 | 66.58 | 87.04 | 85.22 | 82.81 | 92.35 | 96.34 | 79.00 | 95.50 | 55.39 | 57.53 | 10.37 | 80.20 |

Model revision links:

- [FLUX.1 Raw2Q-1000](https://huggingface.co/kimi000/Flux1-Dev-4Step-Raw2Q-1000/tree/bb8e739c259a4991c883fb55f39710ea0746efbd)
- [FLUX.1 Raw2Q-3000](https://huggingface.co/kimi000/Flux1-Dev-4Step-Raw2Q-3000/tree/2bcb04d9948e5d00ddd9972ff0318d62f4a33951)
- [FLUX.3 Raw2Q-2000](https://huggingface.co/kimi000/Flux3-Klein-4Step-Raw2Q-2000/tree/781ff93f92b84397ee0e62e02f052bce99549759)

This repository does not publish controlled latency comparisons. Generation
receipts contain elapsed work time, explicitly excluding model loading; those
are operational records, not warmup-controlled speed benchmarks.

## Development and attribution

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
git diff --check
```

The controller tests cover protocol/count contracts, fail-closed repetitions,
immutable resumes, allocated GPU masks, ZIP traversal, score completeness and
OCR configuration. GPU validation is separate from unit tests. Source import
provenance is in `data/provenance.json`; the importer is a guarded one-time
maintenance tool, not an installation prerequisite. Do not rerun it over the
maintained adapters.

See [NOTICE](NOTICE). The integration code is Apache-2.0; official code,
prompts, weights and generated-image model terms retain their upstream licenses.
