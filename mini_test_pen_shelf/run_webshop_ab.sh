#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f /data2/myl/miniconda3/etc/profile.d/conda.sh ]; then
    # conda activation may read variables that nounset considers missing.
    set +u
    source /data2/myl/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-skillRL}"
    set -u
fi

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export MODEL_PATH="${MODEL_PATH:-/data2/myl/home_configs/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

OUT_ROOT="${OUT_ROOT:-$ROOT/mini_test_pen_shelf/output_webshop}"
BASELINE_DIR="$OUT_ROOT/baseline"
TEMPLATE_DIR="$OUT_ROOT/template"
REPORT_PREFIX="$OUT_ROOT/ab_report"
MAX_STEPS="${MAX_STEPS:-15}"
HISTORY_LENGTH="${HISTORY_LENGTH:-8}"
THINK_BUDGET="${THINK_BUDGET:-640}"
ACTION_BUDGET="${ACTION_BUDGET:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.72}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TASK_LIMIT="${TASK_LIMIT:-0}"

if [ -d "$OUT_ROOT" ] && find "$OUT_ROOT" -mindepth 1 -maxdepth 1 | grep -q .; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$ROOT/mini_test_pen_shelf/backups"
    mv "$OUT_ROOT" "$ROOT/mini_test_pen_shelf/backups/output_webshop_$stamp"
fi
mkdir -p "$BASELINE_DIR" "$TEMPLATE_DIR"

COMMON=(
    --max_steps "$MAX_STEPS"
    --history_length "$HISTORY_LENGTH"
    --think_budget "$THINK_BUDGET"
    --action_budget "$ACTION_BUDGET"
    --max_model_len "$MAX_MODEL_LEN"
    --gpu_mem_util "$GPU_MEM_UTIL"
    --task_limit "$TASK_LIMIT"
)

echo "===== WebShop baseline: fixed 5 categories x 2 tasks ====="
python -u -m mini_test_pen_shelf.run_webshop_mini \
    --variant baseline --outdir "$BASELINE_DIR" "${COMMON[@]}" \
    2>&1 | tee "$BASELINE_DIR/run.log"

echo "===== WebShop max-score template: identical fixed tasks ====="
python -u -m mini_test_pen_shelf.run_webshop_mini \
    --variant template --outdir "$TEMPLATE_DIR" "${COMMON[@]}" \
    2>&1 | tee "$TEMPLATE_DIR/run.log"

python -m mini_test_pen_shelf.compare_webshop_ab \
    --template "$TEMPLATE_DIR/summary.json" \
    --baseline "$BASELINE_DIR/summary.json" \
    --out_prefix "$REPORT_PREFIX"

echo "Readable report: $REPORT_PREFIX.html"
