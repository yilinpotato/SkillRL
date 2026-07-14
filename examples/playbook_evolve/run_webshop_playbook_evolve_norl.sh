#!/usr/bin/env bash
set -x
set -euo pipefail
# =============================================================================
# WebShop CoSkill / Skill Tree evolution (frozen model, no RL)
#
# Comparison standard follows run_alfworld_playbook_evolve_norl.sh:
#   train_data_size=12, val_data_size=32, group_size=6 -> 72 episodes/group
#   100 groups -> 7200 episodes, checkpoint every 2 groups.  With four GPUs,
#   default topology is 2 data-parallel workers x tensor parallel size 2.
# WebShop-specific contract follows examples/grpo_trainer/run_webshop_skills.sh:
#   small 1000-product simulator, max_steps=15, prompt=8192, response=4096.
# =============================================================================

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# Stable physical-device ordering; the Python driver snapshots a disjoint GPU
# group per spawned vLLM worker to prevent KV-cache allocation collisions.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 超算默认双卡；本地共享 3090 只允许空闲的 GPU 0。
if [ -d /GLOBALFS/hit_wxia_1 ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "0" ]; then
        echo "Local shared-server launcher only permits CUDA_VISIBLE_DEVICES=0." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=0
    GPU0_ACTIVE_PIDS=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)
    if [ -n "$GPU0_ACTIVE_PIDS" ]; then
        echo "GPU 0 is in use by PID(s): $GPU0_ACTIVE_PIDS. Refusing to start." >&2
        exit 1
    fi
    NUM_VISIBLE_GPUS=1
fi
# Four-GPU policy:
#   auto (default): DP=2 x TP=2.  It keeps the prior two-worker task split
#   (6+6 base tasks, 72 rollouts/group) while using four GPUs for vLLM.
#   dp4: DP=4 x TP=1, the throughput-oriented option.  It keeps the same total
#   rollout count but changes sampling scheduling, so use it only for a new
#   seed-controlled run rather than bitwise continuation of an existing DP=2 run.
#   manual: honor explicit DATA_PARALLEL_WORKERS / TENSOR_PARALLEL_SIZE /
#   PIPELINE_PARALLEL_SIZE values.
VLLM_PARALLEL_TOPOLOGY="${VLLM_PARALLEL_TOPOLOGY:-auto}"
case "$VLLM_PARALLEL_TOPOLOGY" in
    auto)
        if [ "$NUM_VISIBLE_GPUS" -ge 4 ]; then
            DEFAULT_DP=2
            DEFAULT_TP=2
        else
            DEFAULT_DP="$NUM_VISIBLE_GPUS"
            DEFAULT_TP=1
        fi
        DEFAULT_PP=1
        ;;
    dp2_tp2)
        if [ "$NUM_VISIBLE_GPUS" -lt 4 ]; then
            echo "VLLM_PARALLEL_TOPOLOGY=dp2_tp2 requires four visible GPUs." >&2
            exit 1
        fi
        DEFAULT_DP=2
        DEFAULT_TP=2
        DEFAULT_PP=1
        ;;
    dp4)
        if [ "$NUM_VISIBLE_GPUS" -lt 4 ]; then
            echo "VLLM_PARALLEL_TOPOLOGY=dp4 requires four visible GPUs." >&2
            exit 1
        fi
        DEFAULT_DP=4
        DEFAULT_TP=1
        DEFAULT_PP=1
        ;;
    manual)
        DEFAULT_DP="$NUM_VISIBLE_GPUS"
        DEFAULT_TP=1
        DEFAULT_PP=1
        ;;
    *)
        echo "Unknown VLLM_PARALLEL_TOPOLOGY=$VLLM_PARALLEL_TOPOLOGY (use auto, dp2_tp2, dp4, or manual)." >&2
        exit 1
        ;;
esac
DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$DEFAULT_DP}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TP}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-$DEFAULT_PP}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
REQUIRED_GPUS=$((DATA_PARALLEL_WORKERS * TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
if [ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]; then
    echo "Need $REQUIRED_GPUS visible GPUs for DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE; only $NUM_VISIBLE_GPUS are visible." >&2
    exit 1
fi
ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-$CUDA_VISIBLE_DEVICES}"
# CUDA Graph capture pays back over long A800 training runs; retain eager mode
# by default for the shared 3090 smoke-test path where startup memory is tighter.
if [ -d /GLOBALFS/hit_wxia_1 ]; then
    DEFAULT_VLLM_ENFORCE_EAGER=0
else
    DEFAULT_VLLM_ENFORCE_EAGER=1
fi
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-$DEFAULT_VLLM_ENFORCE_EAGER}"
if [ "$VLLM_ENFORCE_EAGER" != "0" ] && [ "$VLLM_ENFORCE_EAGER" != "1" ]; then
    echo "VLLM_ENFORCE_EAGER must be 0 or 1." >&2
    exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
if [ -d /GLOBALFS/hit_wxia_1 ]; then
    RUN_ENV="超算 (supercomputer)"
    export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
else
    RUN_ENV="本地3090 (local)"
    export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/skillrl_data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
fi

export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

# The WebShop repository does not always commit its 5.3GB data directory.
# Prefer an explicit WEBSHOP_DATA_DIR, then this checkout, then a sibling
# Skill0 checkout used on the current machine.
if [ -z "${WEBSHOP_DATA_DIR:-}" ]; then
    for candidate in \
        "$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data" \
        "$(dirname "$PROJECT_ROOT")/Skill0/agent_system/environments/env_package/webshop/webshop/data" \
        "/data2/myl/Skill0/agent_system/environments/env_package/webshop/webshop/data"
    do
        if [ -f "$candidate/items_shuffle_1000.json" ] \
            && [ -f "$candidate/items_ins_v2_1000.json" ] \
            && [ -f "$candidate/items_human_ins.json" ] \
            && [ -d "$(dirname "$candidate")/search_engine/indexes" ]; then
            WEBSHOP_DATA_DIR="$candidate"
            break
        fi
    done
fi
if [ -z "${WEBSHOP_DATA_DIR:-}" ]; then
    echo "WebShop assets not found. Set WEBSHOP_DATA_DIR to the data directory beside a populated search_engine/indexes directory." >&2
    exit 1
fi
export WEBSHOP_DATA_DIR

PROJECT_NAME="verl_agent_webshop"
EXPERIMENT_NAME="qwen3-4b_skill_tree_evolve_norl_coskill_standard"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-12}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-6}"
TOTAL_GROUPS="${TOTAL_GROUPS:-100}"
MAX_EPISODES="${MAX_EPISODES:-7200}"
CHECKPOINT_EVERY_GROUPS="${CHECKPOINT_EVERY_GROUPS:-2}"
RESUME="${RESUME:-0}"
if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
    echo "RESUME must be 0 (new run) or 1 (continue from summary_partial.json)." >&2
    exit 1
fi
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-0}"
# Keep a compact audit trail of actual model thinking without writing every
# prompt/trajectory.  Group 1 plus each tenth group yields 11 full episodes in a
# standard 100-group run; these files do not feed learning or evaluation.
THINK_TRACE_SAMPLES_PER_GROUP="${THINK_TRACE_SAMPLES_PER_GROUP:-1}"
THINK_TRACE_EVERY_GROUPS="${THINK_TRACE_EVERY_GROUPS:-10}"
# Match both GRPO WebShop paths: an 8,192-token prompt budget plus a 4,096-token
# response budget.  CoSkill's two-stage generator reserves the latter as
# 3,840 thinking tokens + 256 action tokens.
PROMPT_CHAR_LIMIT="${PROMPT_CHAR_LIMIT:-24000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THINK_BUDGET="${THINK_BUDGET:-3840}"
ACTION_BUDGET="${ACTION_BUDGET:-256}"
if [ $((THINK_BUDGET + ACTION_BUDGET)) -gt "$MAX_TOKENS" ]; then
    echo "THINK_BUDGET + ACTION_BUDGET must not exceed MAX_TOKENS" >&2
    exit 1
fi

echo "Run environment: $RUN_ENV"
echo "CACHE_ROOT: $CACHE_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "vLLM topology: $VLLM_PARALLEL_TOPOLOGY (DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE; GPUs=$REQUIRED_GPUS)"
echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS:-0} (0 means each worker's actual rollout batch size)"
echo "vLLM enforce_eager: $VLLM_ENFORCE_EAGER (0 enables CUDA Graphs after warm-up)"
echo "WebShop data: $WEBSHOP_DATA_DIR"
echo "Rollout standard: train=$TRAIN_DATA_SIZE val=$VAL_DATA_SIZE group_size=$GROUP_SIZE groups=$TOTAL_GROUPS max_episodes=$MAX_EPISODES"
echo "Resume: $RESUME (requires checkpoint-consistent summary_partial.json in OUTPUT_DIR)"
echo "Token standard: prompt<=8192, response=$MAX_TOKENS (think=$THINK_BUDGET action=$ACTION_BUDGET)"
echo "Thought audit: $THINK_TRACE_SAMPLES_PER_GROUP samples at group 1 and every $THINK_TRACE_EVERY_GROUPS groups"
echo "Outputs: $OUTPUT_DIR"

python3 -u -m examples.playbook_evolve.run_webshop_evolve \
    --outdir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --webshop_file_path "$WEBSHOP_DATA_DIR/items_shuffle_1000.json" \
    --webshop_attr_path "$WEBSHOP_DATA_DIR/items_ins_v2_1000.json" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE" \
    --group_size "$GROUP_SIZE" \
    --total_groups "$TOTAL_GROUPS" \
    --max_episodes "$MAX_EPISODES" \
    --max_steps 15 \
    --seed 0 \
    --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
    --rollout_worker_gpus "$ROLLOUT_WORKER_GPUS" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
    --checkpoint_every_groups "$CHECKPOINT_EVERY_GROUPS" \
    --resume "$RESUME" \
    --history_length 8 \
    --prompt_char_limit "$PROMPT_CHAR_LIMIT" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    --think_budget "$THINK_BUDGET" \
    --action_budget "$ACTION_BUDGET" \
    --temperature 1.0 \
    --gpu_mem_util 0.8 \
    --skills_json memory_data/webshop/claude_style_skills.json \
    --retrieval_mode template \
    --top_k 6 \
    --enable_hierarchy 1 \
    --stable_cycles_l1 3 \
    --stable_cycles_l2 5 \
    --success_l1 0.7 \
    --demote_threshold 0.3 \
    --min_calls 10 \
    --enable_coskill 1 \
    --enable_skill_tree 1 \
    --enable_skill_tree_evolve 1 \
    --enable_failure_analysis 1 \
    --max_new_skills 3 \
    --skill_tree_evolve_min_samples 6 \
    --capacity_watermark 50000 \
    --perf_watermark 0.6 \
    --min_samples 16 \
    --loop_threshold 3 \
    --log_trajectories "$LOG_TRAJECTORIES" \
    --think_trace_samples_per_group "$THINK_TRACE_SAMPLES_PER_GROUP" \
    --think_trace_every_groups "$THINK_TRACE_EVERY_GROUPS" \
    "$@" 2>&1 | tee "$OUTPUT_DIR/driver.log"
