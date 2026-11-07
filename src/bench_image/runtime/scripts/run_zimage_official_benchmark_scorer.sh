#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 smoke|full BENCHMARK RUN_ROOT SOURCE_ROOT" >&2
  exit 2
fi

MODE=$1
BENCHMARK=$2
RUN_ROOT=$(realpath "$3")
SOURCE_ROOT=$(realpath "$4")
REPO_ROOT=${BENCH_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKER_ROOT=${BENCH_WORKER_ROOT:?set BENCH_WORKER_ROOT}
INPUT="$RUN_ROOT/scoring_inputs/$BENCHMARK.jsonl"
OUTPUT="$RUN_ROOT/scoring/$BENCHMARK"
LAYOUT="$RUN_ROOT/scorer_layouts/$BENCHMARK"
WORLD_SIZE=${BENCH_SCORER_WORLD_SIZE:-8}
IFS=',' read -r -a BENCH_ALLOCATED_DEVICES <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
gpu_for_rank() { printf '%s' "${BENCH_ALLOCATED_DEVICES[$1]}"; }
MODEL_NAME=${BENCH_EVAL_MODEL_NAME:-Z-Image-Base-cfg4-nfe50}
SETUP_PYTHON=${BENCH_SETUP_PYTHON:-python}

# Draco's ~/.tmp is a symlink to login-node-local /tmp and is therefore
# dangling on compute nodes.  vLLM/TorchInductor otherwise discovers that
# path through Python's tempfile machinery and fails while compiling kernels.
# Give every scorer job a node-local, job-unique cache root explicitly.
SCORER_CACHE_ROOT=${BENCH_SCORER_LOCAL_CACHE_ROOT:-/tmp/zimage-benchmark-${SLURM_JOB_ID:-manual}-$(hostname)}
mkdir -p "$SCORER_CACHE_ROOT"/{tmp,xdg,torchinductor,triton,pycache}
mkdir -p "$SCORER_CACHE_ROOT/xdg/torch/kernels"
export TMPDIR="$SCORER_CACHE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$SCORER_CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SCORER_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$SCORER_CACHE_ROOT/triton"
export PYTHONPYCACHEPREFIX="$SCORER_CACHE_ROOT/pycache"

[[ "$MODE" == smoke || "$MODE" == full ]] || { echo "mode must be smoke or full" >&2; exit 2; }
((WORLD_SIZE == 1 || WORLD_SIZE == 8)) || {
  echo "BENCH_SCORER_WORLD_SIZE must be 1 or 8" >&2
  exit 2
}
mkdir -p "$OUTPUT/logs"
PYTHONPATH="$REPO_ROOT/third_party/benchmark_image:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$SETUP_PYTHON" "$REPO_ROOT/scripts/materialize_zimage_official_scorer.py" \
  --records "$RUN_ROOT/records.jsonl" \
  --generation-manifest "$RUN_ROOT/generation_manifest.jsonl" \
  --image-root "$RUN_ROOT" --benchmark "$BENCHMARK" --input "$INPUT" \
  --layout-root "$LAYOUT" --model-name "$MODEL_NAME" \
  --layout-workers "${BENCH_LAYOUT_WORKERS:-8}" \
  >"$OUTPUT/logs/input_setup.log" 2>&1
[[ -s "$INPUT" ]] || { echo "missing scorer input: $INPUT" >&2; exit 1; }
EXPECTED_COUNT=$(grep -cve '^$' "$INPUT")
PARTIAL_ARGS=()
CVTG_PARTIAL_ARGS=()
if [[ "$MODE" == smoke ]]; then
  PARTIAL_ARGS=(--allow-partial)
  CVTG_PARTIAL_ARGS=(--allow_partial)
fi

run_shards() {
  local python=$1
  local worker=$2
  shift 2
  local pids=()
  for ((rank=0; rank<WORLD_SIZE; rank++)); do
    (
      export CUDA_VISIBLE_DEVICES=$(gpu_for_rank "$rank")
      "$python" "$worker" "$@" --rank "$rank" --world-size "$WORLD_SIZE"
    ) >"$OUTPUT/logs/rank_${rank}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == 0 ]]
}

case "$BENCHMARK" in
  hpsv2_official)
    PYTHON=${HPSV2_PYTHON:?set HPSV2_PYTHON}
    export PYTHONPATH="${HPSV2_SOURCE_ROOT:?set HPSV2_SOURCE_ROOT}:${HPSV2_SOURCE_ROOT}/hpsv2${PYTHONPATH:+:$PYTHONPATH}"
    run_shards "$PYTHON" "$WORKER_ROOT/score_hpsv2.py" \
      --results "$INPUT" --output-dir "$OUTPUT" \
      --batch-size "${BENCH_HPSV2_BATCH_SIZE:-32}" \
      --checkpoint-path "${HPSV2_CHECKPOINT:?set HPSV2_CHECKPOINT}" \
      --open-clip-pretrained "${HPSV2_OPEN_CLIP_PRETRAINED:-laion2B-s32B-b79K}" \
      --open-clip-pretrained-path "${HPSV2_OPEN_CLIP_PRETRAINED_PATH:?set HPSV2_OPEN_CLIP_PRETRAINED_PATH}"
    "$PYTHON" "$WORKER_ROOT/score_hpsv2.py" --merge \
      --output-dir "$OUTPUT" --world-size "$WORLD_SIZE" "${PARTIAL_ARGS[@]}"
    ;;
  geneval)
    PYTHON=${GENEVAL_PYTHON:?set GENEVAL_PYTHON}
    run_shards "$PYTHON" "$WORKER_ROOT/score_geneval.py" \
      --image-root "$LAYOUT/geneval" --output-dir "$OUTPUT" \
      --geneval-root "$SOURCE_ROOT/geneval" \
      --model-config "${GENEVAL_MODEL_CONFIG:?set GENEVAL_MODEL_CONFIG}" \
      --model-path "${GENEVAL_MODEL_PATH:?set GENEVAL_MODEL_PATH}"
    "$PYTHON" "$WORKER_ROOT/score_geneval.py" --merge \
      --output-dir "$OUTPUT" --world-size "$WORLD_SIZE" "${PARTIAL_ARGS[@]}"
    ;;
  geneval2)
    PYTHON=${GENEVAL2_PYTHON:?set GENEVAL2_PYTHON}
    GENEVAL2_MODEL=${GENEVAL2_MODEL:?set GENEVAL2_MODEL to the locked Qwen3-VL-8B snapshot}
    "$PYTHON" "$REPO_ROOT/scripts/prepare_geneval2_pinned_evaluator.py" \
      --source "$SOURCE_ROOT/geneval2/evaluation.py" --model-path "$GENEVAL2_MODEL" \
      --output "$OUTPUT/evaluation_pinned.py"
    run_shards "$PYTHON" "$WORKER_ROOT/score_geneval2.py" \
      --results "$INPUT" --output-dir "$OUTPUT" \
      --geneval2-root "$SOURCE_ROOT/geneval2" \
      --evaluator-script "$OUTPUT/evaluation_pinned.py" \
      --benchmark-data "$SOURCE_ROOT/geneval2/geneval2_data.jsonl"
    "$PYTHON" "$WORKER_ROOT/score_geneval2.py" --merge \
      --results "$INPUT" --output-dir "$OUTPUT" \
      --benchmark-data "$SOURCE_ROOT/geneval2/geneval2_data.jsonl" \
      --world-size "$WORLD_SIZE" "${PARTIAL_ARGS[@]}"
    ;;
  dpgbench)
    PYTHON=${DPGBENCH_PYTHON:?set DPGBENCH_PYTHON}
    DPG_RESOLUTION=$("$SETUP_PYTHON" - "$INPUT" <<'PY'
import json
import sys

sizes = {
    (int(row["height"]), int(row["width"]))
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
    for row in [json.loads(line)]
}
if len(sizes) != 1:
    raise RuntimeError(f"DPG requires one raw image size, got {sorted(sizes)}")
height, width = sizes.pop()
if height != width:
    raise RuntimeError(f"DPG raw images must be square, got {height}x{width}")
print(width)
PY
)
    "$PYTHON" -m accelerate.commands.launch --num_processes "$WORLD_SIZE" \
      "$SOURCE_ROOT/dpgbench/src/eval/dpgbench/step2_compute_dpg_bench.py" \
      --image_root_path "$LAYOUT/dpgbench" --resolution "$DPG_RESOLUTION" \
      --csv "$SOURCE_ROOT/dpgbench_questions/dpgbench.csv" \
      --res_path "$OUTPUT/results.txt" --pic_num 4 --vqa_model mplug \
      --mplug_local_path "${DPGBENCH_MPLUG_MODEL:?set DPGBENCH_MPLUG_MODEL}" \
      >"$OUTPUT/logs/scorer.log" 2>&1
    ;;
  cvtg)
    OCR_PYTHON=${CVTG_OCR_PYTHON:?set CVTG_OCR_PYTHON}
    CLIP_PYTHON=${CVTG_CLIP_PYTHON:?set CVTG_CLIP_PYTHON}
    pids=()
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
      (export CUDA_VISIBLE_DEVICES=$(gpu_for_rank "$rank"); "$OCR_PYTHON" "$WORKER_ROOT/score_cvtg.py" \
        --stage ocr --results "$INPUT" --output_dir "$OUTPUT" \
        --num_shards "$WORLD_SIZE" --shard_id "$rank" "${CVTG_PARTIAL_ARGS[@]}") \
        >"$OUTPUT/logs/ocr_${rank}.log" 2>&1 & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
      (export CUDA_VISIBLE_DEVICES=$(gpu_for_rank "$rank"); "$CLIP_PYTHON" "$WORKER_ROOT/score_cvtg.py" \
        --stage clip --results "$INPUT" --output_dir "$OUTPUT" \
        --num_shards "$WORLD_SIZE" --shard_id "$rank" "${CVTG_PARTIAL_ARGS[@]}") \
        >"$OUTPUT/logs/clip_${rank}.log" 2>&1 & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$CLIP_PYTHON" "$WORKER_ROOT/score_cvtg.py" --stage merge \
      --results "$INPUT" --output_dir "$OUTPUT" --num_shards "$WORLD_SIZE" \
      "${CVTG_PARTIAL_ARGS[@]}"
    ;;
  longtext_en)
    PYTHON=${LONGTEXT_PYTHON:?set LONGTEXT_PYTHON}
    # This worker uses underscore-form rank arguments for compatibility with
    # the released LongText adapter.
    pids=()
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
      (export CUDA_VISIBLE_DEVICES=$(gpu_for_rank "$rank"); "$PYTHON" "$WORKER_ROOT/score_longtext.py" \
        --results "$INPUT" --output_dir "$OUTPUT" --mode en \
        --rank "$rank" --world_size "$WORLD_SIZE") \
        >"$OUTPUT/logs/rank_${rank}.log" 2>&1 & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$PYTHON" "$WORKER_ROOT/score_longtext.py" --merge \
      --results "$INPUT" --output_dir "$OUTPUT" --mode en \
      --world_size "$WORLD_SIZE" "${CVTG_PARTIAL_ARGS[@]}"
    ;;
  tiif_short)
    VLLM_PYTHON=${TIIF_VLLM_PYTHON:?set TIIF_VLLM_PYTHON}
    TIIF_MODEL=${TIIF_MODEL:?set TIIF_MODEL to the locked Qwen2.5-VL-72B snapshot}
    TIIF_CLIENTS=${TIIF_CLIENTS:-16}
    TIIF_FIXTURE_ARGS=()
    if [[ "$MODE" == smoke ]]; then TIIF_FIXTURE_ARGS=(--deterministic-fixture); fi
    "$VLLM_PYTHON" "$REPO_ROOT/scripts/prepare_tiif_pinned_evaluator.py" \
      --source "$SOURCE_ROOT/tiif/eval/eval_with_vlm.py" \
      --output "$OUTPUT/eval_with_vlm_pinned.py" "${TIIF_FIXTURE_ARGS[@]}"
    TIIF_RAW_ROOT="$OUTPUT/raw/$MODEL_NAME"
    mkdir -p "$TIIF_RAW_ROOT"
    TIIF_EXPECTED_RESULTS=$((2 * EXPECTED_COUNT))
    TIIF_EXISTING_RESULTS=$(find "$TIIF_RAW_ROOT" -type f -name '*.json' 2>/dev/null | wc -l)
    if ((TIIF_EXISTING_RESULTS != TIIF_EXPECTED_RESULTS)); then
      if [[ "$MODE" == smoke ]]; then
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
        "$VLLM_PYTHON" "$REPO_ROOT/scripts/run_tiif_parallel_clients.py" \
          --evaluator "$OUTPUT/eval_with_vlm_pinned.py" \
          --source-root "$SOURCE_ROOT/tiif" \
          --image-dir "$LAYOUT/tiif" --output-dir "$OUTPUT/raw" \
          --sample-indices "$LAYOUT/tiif/sample_indices.json" \
          --work-dir "$OUTPUT/logs/tiif_clients" \
          --eval-model "$MODEL_NAME" --base-url http://127.0.0.1/unused \
          --model Qwen2.5-VL-72B-Instruct --workers 1 \
          >"$OUTPUT/logs/scorer.log" 2>&1
      else
      PORT=${TIIF_PORT:-18090}
      "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$TIIF_MODEL" --served-model-name Qwen2.5-VL-72B-Instruct \
        --tensor-parallel-size "$WORLD_SIZE" --port "$PORT" \
        >"$OUTPUT/logs/vllm_server.log" 2>&1 &
      server_pid=$!
      trap 'kill "$server_pid" 2>/dev/null || true' EXIT
      ready=0
      for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
        sleep 5
      done
      [[ "$ready" == 1 ]] || { echo "TIIF vLLM server failed to become ready" >&2; exit 1; }
      "$VLLM_PYTHON" "$REPO_ROOT/scripts/run_tiif_parallel_clients.py" \
        --evaluator "$OUTPUT/eval_with_vlm_pinned.py" \
        --source-root "$SOURCE_ROOT/tiif" \
        --image-dir "$LAYOUT/tiif" --output-dir "$OUTPUT/raw" \
        --sample-indices "$LAYOUT/tiif/sample_indices.json" \
        --work-dir "$OUTPUT/logs/tiif_clients" \
        --eval-model "$MODEL_NAME" \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --model Qwen2.5-VL-72B-Instruct --workers "$TIIF_CLIENTS" \
        >"$OUTPUT/logs/scorer.log" 2>&1
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
      trap - EXIT
      fi
    else
      echo "TIIF inference already complete: $TIIF_EXISTING_RESULTS/$TIIF_EXPECTED_RESULTS results" \
        >"$OUTPUT/logs/scorer.log"
    fi
    TIIF_EXISTING_RESULTS=$(find "$TIIF_RAW_ROOT" -type f -name '*.json' | wc -l)
    ((TIIF_EXISTING_RESULTS == TIIF_EXPECTED_RESULTS)) || {
      echo "TIIF result count mismatch: $TIIF_EXISTING_RESULTS != $TIIF_EXPECTED_RESULTS" >&2
      exit 1
    }
    (
      cd "$SOURCE_ROOT/tiif"
      "$VLLM_PYTHON" eval/summary_results.py --input_dir "$OUTPUT/raw"
      set +e
      "$VLLM_PYTHON" eval/summary_dimension_results.py \
        --input_excel "$OUTPUT/raw/result_summary.xlsx" \
        --output_txt "$OUTPUT/summary.txt"
      summary_status=$?
      set -e
      if ((summary_status != 0)); then
        if [[ "$summary_status" == 120 && -s "$OUTPUT/summary.txt" ]] && \
          grep -q 'Overall average score' "$OUTPUT/summary.txt"; then
          echo "TIIF accepted upstream Tee shutdown status 120 after complete summary"
        else
          exit "$summary_status"
        fi
      fi
    ) >>"$OUTPUT/logs/scorer.log" 2>&1
    ;;
  oneig_en)
    PYTHON=${ONEIG_PYTHON:?set ONEIG_PYTHON}
    ONEIG_SOURCE="$SOURCE_ROOT/oneig_en"
    ONEIG_IMAGES="$LAYOUT/oneig_en"
    ONEIG_WORK="$OUTPUT/workdir_pinned"
    mkdir -p "$ONEIG_WORK"
    ONEIG_DREAMSIM_CACHE=${ONEIG_DREAMSIM_CACHE:?set ONEIG_DREAMSIM_CACHE}
    for asset in dino_vitb16_pretrain.pth open_clip_vitb16_pretrain.pth.tar \
      clip_vitb16_pretrain.pth.tar ensemble_lora/adapter_model.safetensors; do
      [[ -s "$ONEIG_DREAMSIM_CACHE/$asset" ]] || {
        echo "missing OneIG DreamSim cache asset: $ONEIG_DREAMSIM_CACHE/$asset" >&2
        exit 1
      }
    done
    if [[ ! -e "$ONEIG_WORK/models" ]]; then
      ln -s "$ONEIG_DREAMSIM_CACHE" "$ONEIG_WORK/models"
    fi
    "$PYTHON" "$REPO_ROOT/scripts/prepare_oneig_pinned_source.py" \
      --source "$ONEIG_SOURCE" --output "$ONEIG_WORK/scripts" \
      --qwen-model "${ONEIG_QWEN_MODEL:?set ONEIG_QWEN_MODEL}" \
      --llm2clip-model "${ONEIG_LLM2CLIP_MODEL:?set ONEIG_LLM2CLIP_MODEL}" \
      --llm2clip-llm "${ONEIG_LLM2CLIP_LLM:?set ONEIG_LLM2CLIP_LLM}" \
      --llm2vec-code "${ONEIG_LLM2VEC_CODE:?set ONEIG_LLM2VEC_CODE}" \
      --clip-model "${ONEIG_CLIP_MODEL:?set ONEIG_CLIP_MODEL}" \
      --style-model "${ONEIG_STYLE_MODEL:?set ONEIG_STYLE_MODEL}"
    if [[ "$MODE" == smoke ]]; then
      (
        cd "$ONEIG_WORK"
        if ! compgen -G 'results/alignment_score_EN_*.csv' >/dev/null; then
          "$PYTHON" -m scripts.alignment.alignment_score --mode EN \
            --image_dirname "$ONEIG_IMAGES" --model_names "$MODEL_NAME" \
            --image_grid 2,2 --class_items anime human object
        fi
        if ! compgen -G 'results/text_score_EN_*.csv' >/dev/null; then
          "$PYTHON" -m scripts.text.text_score --mode EN \
            --image_dirname "$ONEIG_IMAGES/text" --model_names "$MODEL_NAME" \
            --image_grid 2,2
        fi
        if ! compgen -G 'results/reasoning_score_EN_*.csv' >/dev/null; then
          "$PYTHON" -m scripts.reasoning.reasoning_score --mode EN \
            --image_dirname "$ONEIG_IMAGES/reasoning" --model_names "$MODEL_NAME" \
            --image_grid 2,2
        fi
        if ! compgen -G 'results/style_score_EN_*.csv' >/dev/null; then
          "$PYTHON" -m scripts.style.style_score --mode EN \
            --image_dirname "$ONEIG_IMAGES/anime" --model_names "$MODEL_NAME" \
            --image_grid 2,2
        fi
        if ! compgen -G 'results/diversity_score_EN_*.csv' >/dev/null; then
          "$PYTHON" -m scripts.diversity.diversity_score --mode EN \
            --image_dirname "$ONEIG_IMAGES" --model_names "$MODEL_NAME" \
            --image_grid 2,2 --class_items anime human object text reasoning
        fi
      ) >"$OUTPUT/logs/scorer.log" 2>&1
    else
      ((WORLD_SIZE == 8)) || {
        echo "full OneIG scoring requires eight GPUs" >&2
        exit 1
      }
      mkdir -p "$ONEIG_WORK/results/far_alignment_shards"
      pids=()
      for ((rank=0; rank<WORLD_SIZE; rank++)); do
        shard_output=$(printf '%s/results/far_alignment_shards/global-shard-%02d-of-%02d.csv' \
          "$ONEIG_WORK" "$rank" "$WORLD_SIZE")
        (
          export CUDA_VISIBLE_DEVICES=$rank
          "$PYTHON" "$REPO_ROOT/scripts/run_oneig_alignment_shard.py" \
            --oneig-root "$ONEIG_WORK" --image-root "$ONEIG_IMAGES" \
            --model-name "$MODEL_NAME" --all-classes \
            --shard-index "$rank" --num-shards "$WORLD_SIZE" \
            --output "$shard_output"
        ) >"$OUTPUT/logs/alignment_rank_${rank}.log" 2>&1 & pids+=("$!")
      done
      failed=0
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      [[ "$failed" == 0 ]] || { echo "OneIG alignment shard failed" >&2; exit 1; }

      pids=()
      if ! compgen -G "$ONEIG_WORK/results/text_score_EN_*.csv" >/dev/null; then
        (cd "$ONEIG_WORK"; export CUDA_VISIBLE_DEVICES=0; "$PYTHON" -m scripts.text.text_score \
          --mode EN --image_dirname "$ONEIG_IMAGES/text" --model_names "$MODEL_NAME" \
          --image_grid 2,2) >"$OUTPUT/logs/text.log" 2>&1 & pids+=("$!")
      fi
      if ! compgen -G "$ONEIG_WORK/results/reasoning_score_EN_*.csv" >/dev/null; then
        (cd "$ONEIG_WORK"; export CUDA_VISIBLE_DEVICES=1; "$PYTHON" -m scripts.reasoning.reasoning_score \
          --mode EN --image_dirname "$ONEIG_IMAGES/reasoning" --model_names "$MODEL_NAME" \
          --image_grid 2,2) >"$OUTPUT/logs/reasoning.log" 2>&1 & pids+=("$!")
      fi
      if ! compgen -G "$ONEIG_WORK/results/style_score_EN_*.csv" >/dev/null; then
        (cd "$ONEIG_WORK"; export CUDA_VISIBLE_DEVICES=2; "$PYTHON" -m scripts.style.style_score \
          --mode EN --image_dirname "$ONEIG_IMAGES/anime" --model_names "$MODEL_NAME" \
          --image_grid 2,2) >"$OUTPUT/logs/style.log" 2>&1 & pids+=("$!")
      fi
      if ! compgen -G "$ONEIG_WORK/results/diversity_score_EN_*.csv" >/dev/null; then
        (cd "$ONEIG_WORK"; export CUDA_VISIBLE_DEVICES=3; "$PYTHON" -m scripts.diversity.diversity_score \
          --mode EN --image_dirname "$ONEIG_IMAGES" --model_names "$MODEL_NAME" \
          --image_grid 2,2 --class_items anime human object text reasoning) \
          >"$OUTPUT/logs/diversity.log" 2>&1 & pids+=("$!")
      fi
      failed=0
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      [[ "$failed" == 0 ]] || { echo "OneIG component scorer failed" >&2; exit 1; }
      "$PYTHON" "$REPO_ROOT/scripts/collect_oneig_official_scores.py" \
        --results-root "$ONEIG_WORK/results" --alignment-shards "$WORLD_SIZE" \
        --model-name "$MODEL_NAME" --output "$OUTPUT/summary.json" \
        >"$OUTPUT/logs/scorer.log" 2>&1
    fi
    ;;
  t2i_corebench)
    PYTHON=${COREBENCH_PYTHON:?set COREBENCH_PYTHON}
    COREBENCH_MODEL=${COREBENCH_MODEL:?set COREBENCH_MODEL to the locked Qwen3.5-9B snapshot}
    export PATH="$(dirname "$PYTHON"):$PATH"
    if [[ -n "${COREBENCH_CUDA_COMPAT_LIB:-}" ]]; then
      [[ -s "$COREBENCH_CUDA_COMPAT_LIB/libcuda.so.1" ]] || {
        echo "missing CoReBench CUDA compatibility library: $COREBENCH_CUDA_COMPAT_LIB/libcuda.so.1" >&2
        exit 1
      }
      export LD_LIBRARY_PATH="$COREBENCH_CUDA_COMPAT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    CORE_TASKS=(C-MI C-MA C-MR C-TR R-LR R-BR R-HR R-PR R-GR R-AR R-CR R-RR)
    CORE_RESULT_ROOT="$LAYOUT/t2i_corebench/$MODEL_NAME"
    CORE_EXISTING_RESULTS=0
    for task in "${CORE_TASKS[@]}"; do
      [[ -s "$CORE_RESULT_ROOT/${task}-Qwen3_5_9B.json" ]] && \
        CORE_EXISTING_RESULTS=$((CORE_EXISTING_RESULTS + 1))
    done
    if ((CORE_EXISTING_RESULTS == ${#CORE_TASKS[@]})); then
      echo "CoReBench inference already complete: $CORE_EXISTING_RESULTS/${#CORE_TASKS[@]} tasks" \
        >"$OUTPUT/logs/scorer.log"
    elif [[ "$MODE" == full && "$WORLD_SIZE" -gt 1 ]]; then
      "$PYTHON" "$REPO_ROOT/scripts/prepare_corebench_pinned_evaluator.py" \
        --source "$SOURCE_ROOT/t2i_corebench/evaluate.py" \
        --model-path "$COREBENCH_MODEL" --output "$OUTPUT/evaluate_pinned.py"
      groups=()
      for ((rank=0; rank<WORLD_SIZE; rank++)); do groups+=(""); done
      for ((index=0; index<${#CORE_TASKS[@]}; index++)); do
        rank=$((index % WORLD_SIZE))
        groups[$rank]="${groups[$rank]:+${groups[$rank]},}${CORE_TASKS[$index]}"
      done
      pids=()
      for ((rank=0; rank<WORLD_SIZE; rank++)); do
        [[ -n "${groups[$rank]}" ]] || continue
        (
          cd "$SOURCE_ROOT/t2i_corebench"
          export CUDA_VISIBLE_DEVICES=$rank
          PYTHONPATH="$SOURCE_ROOT/t2i_corebench${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON" "$OUTPUT/evaluate_pinned.py" \
            --output_path "$LAYOUT/t2i_corebench" --model "$MODEL_NAME" \
            --mllm Qwen3_5_9B --gen_eval_file "${groups[$rank]}"
        ) >"$OUTPUT/logs/core_rank_${rank}.log" 2>&1 & pids+=("$!")
      done
      failed=0
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      [[ "$failed" == 0 ]] || { echo "CoReBench scorer shard failed" >&2; exit 1; }
    else
      "$PYTHON" "$REPO_ROOT/scripts/prepare_corebench_pinned_evaluator.py" \
        --source "$SOURCE_ROOT/t2i_corebench/evaluate.py" \
        --model-path "$COREBENCH_MODEL" --output "$OUTPUT/evaluate_pinned.py"
      (
        cd "$SOURCE_ROOT/t2i_corebench"
        PYTHONPATH="$SOURCE_ROOT/t2i_corebench${PYTHONPATH:+:$PYTHONPATH}" \
          "$PYTHON" "$OUTPUT/evaluate_pinned.py" --output_path "$LAYOUT/t2i_corebench" \
          --model "$MODEL_NAME" --mllm Qwen3_5_9B \
          --gen_eval_file 'C-MI,C-MA,C-MR,C-TR,R-LR,R-BR,R-HR,R-PR,R-GR,R-AR,R-CR,R-RR'
      ) >"$OUTPUT/logs/scorer.log" 2>&1
    fi
    ;;
  qwen_image_bench_en)
    PYTHON=${QJUDGER_PYTHON:?set QJUDGER_PYTHON}
    run_shards "$PYTHON" "$WORKER_ROOT/score_qwen_image_bench.py" \
      --results "$INPUT" --output-dir "$OUTPUT" \
      --qwen-image-bench-root "${QWEN_IMAGE_BENCH_ROOT:?set QWEN_IMAGE_BENCH_ROOT}" \
      --model "${QJUDGER_MODEL:?set QJUDGER_MODEL}" --backend pt \
      --max-batch-size 24 --max-new-tokens 4096
    "$PYTHON" "$WORKER_ROOT/score_qwen_image_bench.py" --merge \
      --results "$INPUT" --output-dir "$OUTPUT" \
      --qwen-image-bench-root "$QWEN_IMAGE_BENCH_ROOT" --model "$QJUDGER_MODEL" \
      --world-size "$WORLD_SIZE" "${PARTIAL_ARGS[@]}"
    ;;
  wise_verified)
    PYTHON=${WISE_PYTHON:?set WISE_PYTHON}
    WISE_MODEL=${WISE_MODEL:?set WISE_MODEL to the Qwen3.5-35B-A3B snapshot}
    [[ -f "$WISE_MODEL/config.json" ]] || {
      echo "missing WISE judge config.json: $WISE_MODEL" >&2
      exit 1
    }
    export PATH="$(dirname "$PYTHON"):$PATH"
    if [[ -n "${COREBENCH_CUDA_COMPAT_LIB:-}" ]]; then
      [[ -s "$COREBENCH_CUDA_COMPAT_LIB/libcuda.so.1" ]] || {
        echo "missing CUDA compatibility library: $COREBENCH_CUDA_COMPAT_LIB/libcuda.so.1" >&2
        exit 1
      }
      export LD_LIBRARY_PATH="$COREBENCH_CUDA_COMPAT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    PORT=${WISE_PORT:-$((18000 + SLURM_JOB_ID % 1000))}
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$WISE_MODEL" --served-model-name Qwen3.5-35B-A3B \
      --host 127.0.0.1 --port "$PORT" \
      --tensor-parallel-size "$WORLD_SIZE" \
      --gpu-memory-utilization "${WISE_GPU_MEMORY_UTILIZATION:-0.90}" \
      --max-model-len "${WISE_MAX_MODEL_LEN:-8192}" \
      >"$OUTPUT/logs/vllm_server.log" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null || true' EXIT
    ready=0
    for _ in $(seq 1 240); do
      if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        ready=1
        break
      fi
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 5
    done
    [[ "$ready" == 1 ]] || {
      echo "WISE Qwen3.5-35B-A3B vLLM server failed to become ready" >&2
      exit 1
    }
    WISE_SOURCE="$SOURCE_ROOT/wise_verified"
    WISE_IMAGES="$LAYOUT/wise_verified/images"
    for spec in \
      'cultural_common_sense_verified.json:cultural_common_sense' \
      'spatio-temporal_reasoning_verified.json:spatio-temporal_reasoning' \
      'natural_science_verified.json:natural_science'; do
      json_name=${spec%%:*}
      stem=${spec#*:}
      "$PYTHON" "$WISE_SOURCE/vllm_eval.py" \
        --json_path "$WISE_SOURCE/data_verified/$json_name" \
        --image_dir "$WISE_IMAGES" --output_dir "$OUTPUT/$stem" \
        --api_key EMPTY --api_base "http://127.0.0.1:$PORT/v1" \
        --model Qwen3.5-35B-A3B \
        --result_full "$OUTPUT/${stem}_full_results.json" \
        --result_scores "$OUTPUT/${stem}_scores_results.jsonl" \
        --max_workers "${WISE_MAX_WORKERS:-96}" \
        >"$OUTPUT/logs/${stem}.log" 2>&1
    done
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    trap - EXIT
    WISE_PARTIAL_ARGS=()
    [[ "$MODE" == smoke ]] && WISE_PARTIAL_ARGS=(--allow-partial)
    "$SETUP_PYTHON" "$REPO_ROOT/scripts/collect_wise_verified_scores.py" \
      --input "$INPUT" \
      --score-file "$OUTPUT/cultural_common_sense_scores_results.jsonl" \
      --score-file "$OUTPUT/spatio-temporal_reasoning_scores_results.jsonl" \
      --score-file "$OUTPUT/natural_science_scores_results.jsonl" \
      --output "$OUTPUT/summary.json" "${WISE_PARTIAL_ARGS[@]}" \
      >"$OUTPUT/logs/summary.log" 2>&1
    ;;
  hpsv3_official)
    PYTHON=${HPSV3_PYTHON:?set HPSV3_PYTHON}
    HPSV3_BASE_MODEL=${HPSV3_BASE_MODEL:?set HPSV3_BASE_MODEL to the locked Qwen2-VL snapshot}
    HPSV3_CHECKPOINT=${HPSV3_CHECKPOINT:?set HPSV3_CHECKPOINT}
    if [[ "${BENCH_STAGE_HPSV3_CHECKPOINT:-1}" == 1 ]]; then
      LOCAL_TMP_ROOT=${BENCH_LOCAL_TMP_ROOT:-/tmp}
      HPSV3_STAGE_DIR=$(mktemp -d "$LOCAL_TMP_ROOT/zimage-hpsv3.XXXXXX")
      trap 'rm -rf -- "$HPSV3_STAGE_DIR"' EXIT
      STAGED_CHECKPOINT="$HPSV3_STAGE_DIR/HPSv3.safetensors"
      echo "staging HPSv3 checkpoint: $HPSV3_CHECKPOINT -> $STAGED_CHECKPOINT"
      cp -- "$HPSV3_CHECKPOINT" "$STAGED_CHECKPOINT"
      [[ "$(stat -Lc %s "$STAGED_CHECKPOINT")" == "$(stat -Lc %s "$HPSV3_CHECKPOINT")" ]] || {
        echo "staged HPSv3 checkpoint size mismatch" >&2
        exit 1
      }
      echo "staged HPSv3 checkpoint bytes: $(stat -Lc %s "$STAGED_CHECKPOINT")"
      HPSV3_CHECKPOINT=$STAGED_CHECKPOINT
    fi
    "$PYTHON" "$REPO_ROOT/scripts/prepare_hpsv3_pinned_config.py" \
      --source "$SOURCE_ROOT/hpsv3_scorer/hpsv3/config/HPSv3_7B.yaml" \
      --model-path "$HPSV3_BASE_MODEL" --output "$OUTPUT/HPSv3_7B_pinned.yaml"
    run_shards "$PYTHON" "$WORKER_ROOT/score_hpsv3_official.py" \
      --results "$INPUT" --output-dir "$OUTPUT" \
      --hpsv3-root "$SOURCE_ROOT/hpsv3_scorer" \
      --config-path "$OUTPUT/HPSv3_7B_pinned.yaml" \
      --checkpoint-path "$HPSV3_CHECKPOINT" \
      --expected-count "$EXPECTED_COUNT"
    "$PYTHON" "$WORKER_ROOT/score_hpsv3_official.py" --merge \
      --output-dir "$OUTPUT" --world-size "$WORLD_SIZE" \
      --expected-count "$EXPECTED_COUNT"
    if [[ -n "${HPSV3_STAGE_DIR:-}" ]]; then
      rm -rf -- "$HPSV3_STAGE_DIR"
      trap - EXIT
    fi
    ;;
  *)
    echo "scorer wrapper not yet defined for: $BENCHMARK" >&2
    exit 2
    ;;
esac

echo "official scorer completed: $BENCHMARK ($MODE)"
