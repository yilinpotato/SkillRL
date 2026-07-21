set -x
set -euo pipefail
# =============================================================================
# 独立 Skill Tree 进化 driver（无 verl / 无 Ray / 无 FSDP）
# -----------------------------------------------------------------------------
# 冻结小模型（vLLM 只加载一份）在 ALFWorld 上 rollout → TracesPool → 水位线触发
# CoSkillCloudLoop（失败诊断 + skill tree 从零生成/层次化细化 + 可选 skill 蒸馏）→
# 进化后的 agent skill tree 写回共享 skill_lib，下一局即注入。
#
# 所有运行条件对齐 examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh，
# 仅去掉 RL 权重训练（无 Ray、无 FSDP、无第二份模型、无反向传播/checkpoint）。
# 复用其环境探测头（超算 vs 本地 3090 的缓存/数据/输出根目录、模型/DEEPSEEK）。
# =============================================================================

# Default remains the frozen no-RL driver.  An explicit external ``rl=1`` (or
# ``RL=1``) switches to the Ray/GRPO skill-tree curriculum without changing the
# familiar entry command.  TREE_RL_ORDER=root|leaf selects the curriculum
# direction in the delegated launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
# `python3 -m examples...` below resolves the `examples` package off the
# current working directory, not off this script's location — must cd here
# (matching run_coskill_tree_rl.sh) or it 404s when launched from elsewhere.
cd "$PROJECT_ROOT"
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE

RL_MODE="${rl:-${RL:-0}}"
if [[ "$RL_MODE" != "0" && "$RL_MODE" != "1" ]]; then
    echo "rl/RL must be 0 (default no-RL) or 1 (Ray skill-tree RL)." >&2
    exit 1
fi
if [[ "$RL_MODE" == "1" ]]; then
    exec bash "$SCRIPT_DIR/../grpo_trainer/run_coskill_tree_rl.sh" alfworld "$@"
fi

# 强制离线
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# vLLM v1 需 spawn（agent_vllm 内也设了，这里兜底）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Keep physical-device enumeration stable across the parent, its data-parallel
# workers, and vLLM EngineCore grandchildren.  Each worker is masked to one
# entry by the Python driver before it is spawned.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Opt-in single-GPU mode for an ALFWorld no-RL job.  It keeps the global
# rollout group at 72 and changes only where the frozen vLLM replica runs.
# If the scheduler exposes several devices, use the first entry in its existing
# CUDA_VISIBLE_DEVICES mask rather than assuming that physical GPU 0 is ours.
COSKILL_ONE_GPU="${COSKILL_ONE_GPU:-0}"
if [[ "$COSKILL_ONE_GPU" != "0" && "$COSKILL_ONE_GPU" != "1" ]]; then
    echo "COSKILL_ONE_GPU must be 0 or 1." >&2
    exit 1
fi

# 关键：stdout 被下面的 `| tee` 管道重定向后，Python 默认切到【全缓冲】（不是行缓冲），
# 我们自己的 print() 会攒够几 KB 才真正写出——vLLM 走 logging 模块会立刻刷出来，两者
# 混在一起看，就像是"跑完 vLLM 日志后卡住"，其实是自己的进度 print 被缓冲憋住了。
# PYTHONUNBUFFERED=1（等价 python3 -u）强制无缓冲，进度 print 立刻可见。
export PYTHONUNBUFFERED=1

# GPU selection.  The self-contained Docker image is an isolated allocation,
# not the shared local-3090 policy.  Discover every GPU exposed by Docker if
# CUDA_VISIBLE_DEVICES was not explicitly set; one vLLM replica is then bound
# to each selected device by the Python driver.
IS_CONTAINER=0
if [[ "${COSKILL_CONTAINER:-0}" == "1" || -f /.dockerenv || -f /run/.containerenv ]]; then
    IS_CONTAINER=1
fi
if [[ "$IS_CONTAINER" == "1" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        DETECTED_GPU_COUNT="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
        if [[ "$DETECTED_GPU_COUNT" -le 0 ]]; then
            echo "No CUDA GPUs are visible inside this container." >&2
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES="$(awk -v n="$DETECTED_GPU_COUNT" 'BEGIN {for (i=0;i<n;i++) printf "%s%d", (i?",":""), i}')"
    fi
    NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
elif [ -d /GLOBALFS/hit_wxia_1 ]; then
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
if [[ "$COSKILL_ONE_GPU" == "1" ]]; then
    SELECTED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
    if [[ -z "$SELECTED_GPU" ]]; then
        echo "COSKILL_ONE_GPU=1 but CUDA_VISIBLE_DEVICES is empty." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
    NUM_VISIBLE_GPUS=1
    # An explicit mode must not inherit stale multi-GPU variables from a prior
    # shell.  The mode owns these three topology values by design.
    DATA_PARALLEL_WORKERS=1
    ROLLOUT_WORKER_GPUS="$SELECTED_GPU"
    TENSOR_PARALLEL_SIZE=1
else
    DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$NUM_VISIBLE_GPUS}"
    ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-$CUDA_VISIBLE_DEVICES}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
fi

if ! [[ "$DATA_PARALLEL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DATA_PARALLEL_WORKERS must be a positive integer." >&2
    exit 1
fi
if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE must be a positive integer." >&2
    exit 1
fi
if [[ "$DATA_PARALLEL_WORKERS" -gt 1 && "$TENSOR_PARALLEL_SIZE" -ne 1 ]]; then
    echo "ALFWorld data-parallel workers each own one vLLM replica; use TENSOR_PARALLEL_SIZE=1 when DATA_PARALLEL_WORKERS>1." >&2
    exit 1
fi
if [[ "$DATA_PARALLEL_WORKERS" -gt 1 ]]; then
    REQUIRED_GPUS="$DATA_PARALLEL_WORKERS"
else
    REQUIRED_GPUS="$TENSOR_PARALLEL_SIZE"
fi
if [[ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]]; then
    echo "Need $REQUIRED_GPUS visible GPU(s) for DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE; only $NUM_VISIBLE_GPUS visible." >&2
    exit 1
fi

# 0 lets the Python driver use the actual largest batch handled by each vLLM
# replica (single GPU: 72).  This bounds scheduler warm-up without reducing or
# repartitioning the real rollout batch.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
if ! [[ "$VLLM_MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then
    echo "VLLM_MAX_NUM_SEQS must be a non-negative integer." >&2
    exit 1
fi
# CUDA Graph capture improves long-running A800/container decode throughput.
# Keep eager as the default only for the shared local single-GPU path, where
# its startup-memory overhead is more likely to matter.
if [[ "$IS_CONTAINER" == "1" || -d /GLOBALFS/hit_wxia_1 ]]; then
    DEFAULT_VLLM_ENFORCE_EAGER=0
else
    DEFAULT_VLLM_ENFORCE_EAGER=1
fi
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-$DEFAULT_VLLM_ENFORCE_EAGER}"
if [[ "$VLLM_ENFORCE_EAGER" != "0" && "$VLLM_ENFORCE_EAGER" != "1" ]]; then
    echo "VLLM_ENFORCE_EAGER must be 0 or 1." >&2
    exit 1
fi

# FlashInfer is opt-in and validated before any worker is spawned.  It is not
# an attention-backend override: dense Qwen continues to use vLLM FlashAttn.
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/configure_vllm_acceleration.sh"

# ── 自动判断运行环境：超算 vs 本地3090（与训练脚本一致）─────────────────────────
if [[ "$IS_CONTAINER" == "1" ]]; then
    RUN_ENV="Docker container"
    export CACHE_ROOT="${CACHE_ROOT:-/models/.cache}"
    export DATA_ROOT="${DATA_ROOT:-/opt/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-/outputs}"
elif [ -d /GLOBALFS/hit_wxia_1 ]; then
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
echo "Run environment detected: $RUN_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "data_parallel_workers: $DATA_PARALLEL_WORKERS"
echo "rollout_worker_gpus: ${ROLLOUT_WORKER_GPUS:-<auto>}"
echo "vLLM tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
echo "single-GPU mode: $COSKILL_ONE_GPU"
echo "vLLM enforce_eager: $VLLM_ENFORCE_EAGER"
echo "vLLM FlashInfer sampler: $VLLM_USE_FLASHINFER_SAMPLER"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"

# 云端大模型（与训练脚本一致）：DeepSeek V4 Flash
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
# export DEEPSEEK_API_KEY=...   # 需在环境里提供

PROJECT_NAME="verl_agent_alfworld"
EXPERIMENT_NAME="qwen3-4b_skill_tree_evolve_norl_v8"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"

# Match 100 GRPO trainer steps approximately:
# train_data_size(12) × group_size(6) × 100 = 7200 rollout episodes.
# Override with MAX_EPISODES=... or a later --max_episodes CLI argument.
MAX_EPISODES="${MAX_EPISODES:-7200}"
# Match one GRPO rollout batch by default: train_data_size(12) × group_size(6).
BATCH_ROLLOUT_SIZE="${BATCH_ROLLOUT_SIZE:-72}"
if ! [[ "$BATCH_ROLLOUT_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_ROLLOUT_SIZE must be a positive integer." >&2
    exit 1
fi
MAX_WORKER_BATCH=$(((BATCH_ROLLOUT_SIZE + DATA_PARALLEL_WORKERS - 1) / DATA_PARALLEL_WORKERS))
if [[ "$VLLM_MAX_NUM_SEQS" -gt 0 && "$VLLM_MAX_NUM_SEQS" -lt "$MAX_WORKER_BATCH" ]]; then
    echo "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS is smaller than the largest per-replica rollout batch=$MAX_WORKER_BATCH." >&2
    exit 1
fi
EFFECTIVE_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
if [[ "$EFFECTIVE_VLLM_MAX_NUM_SEQS" == "0" ]]; then
    EFFECTIVE_VLLM_MAX_NUM_SEQS="$MAX_WORKER_BATCH"
fi
echo "vLLM max_num_seqs: $VLLM_MAX_NUM_SEQS (effective largest replica limit: $EFFECTIVE_VLLM_MAX_NUM_SEQS)"
# Save lightweight experiment checkpoints every N rollout groups.  This does not
# force a cloud update; CoSkill still follows the paper trigger/watermark logic.
CHECKPOINT_EVERY_GROUPS="${CHECKPOINT_EVERY_GROUPS:-2}"
# Long runs otherwise dump every step's full prompt into trajectories/, which is
# useful for debugging but creates huge IO. Metrics/raw traces/cloud_io are still
# written, so this does not affect rollout decisions or CoSkill updates.
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-0}"

# CI/container smoke check: validate the resolved launch contract without
# importing vLLM, allocating CUDA memory, modifying an output, or contacting
# the cloud backend.  It is deliberately opt-in and cannot affect a real run.
if [[ "${COSKILL_LAUNCHER_DRY_RUN:-0}" == "1" ]]; then
    echo "CoSkill no-RL launcher dry run passed: benchmark=alfworld env=$RUN_ENV GPUs=$NUM_VISIBLE_GPUS DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE batch=$BATCH_ROLLOUT_SIZE max_num_seqs=$EFFECTIVE_VLLM_MAX_NUM_SEQS DATA_ROOT=$DATA_ROOT"
    exit 0
fi

python3 -u -m examples.playbook_evolve.run_playbook_evolve \
    --outdir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    `# 环境 / 采样：正式测试用全量数据，覆盖全部 6 类 task_type，每类的全部 game。` \
    `# num_games=-1 表示不抽样、不限量，跑该 task_type 在 split 下的全部 game。` \
    `# max_steps=40 / seed=0 / group_size≈env.rollout.n=6，与训练脚本一致。` \
    --task_types "pick_and_place_simple,look_at_obj_in_light,pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep,pick_two_obj_and_place" \
    --num_games -1 \
    --group_size 6 \
    --split train \
    --max_steps 40 \
    --seed 0 \
    `# epochs=1：全量数据已跑遍每个 game(×group_size 次)，一轮覆盖足够全面；` \
    `# 需要更多云端迭代轮次再调大(会把全量数据集重新跑一遍)。` \
    --epochs 1 \
    --max_episodes "$MAX_EPISODES" \
    --batch_rollout_size "$BATCH_ROLLOUT_SIZE" \
    --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
    --rollout_worker_gpus "$ROLLOUT_WORKER_GPUS" \
    --checkpoint_every_groups "$CHECKPOINT_EVERY_GROUPS" \
    `# NO_HIS 模板本身没有记忆，靠 history_length 条最近 obs+action 弥补；调大到 8` \
    --history_length 8 \
    `# vLLM：max_prompt6144+max_response4096=10240; 冻结推理 gpu_mem_util 可给高` \
    --max_model_len 10240 \
    --max_tokens 4096 \
    --think_budget 3500 \
    --temperature 1.0 \
    --gpu_mem_util 0.8 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
    `# 记忆 / 技能：与训练脚本 skills_only_memory.* 对齐` \
    --skills_json memory_data/alfworld/claude_style_skills.json \
    `# 初始 SkillRL 匹配机制：template/关键词 task_type 检测；top_k=6 只限制 general skills，` \
    `# task-specific skills 默认取该 task_type 下全部技能。需要语义检索时命令行显式覆盖。` \
    --retrieval_mode template \
    --top_k 6 \
    --enable_hierarchy 1 \
    --stable_cycles_l1 3 \
    --stable_cycles_l2 5 \
    --success_l1 0.7 \
    --demote_threshold 0.3 \
    --min_calls 10 \
    `# 闭环开关：重新启用扁平 skill bullets；contrastive_distill 生成 dyn_ 补丁，` \
    `# prompt 同时注入 General/Task-specific/Mistakes。消融实验显式传 0 覆盖。` \
    --enable_coskill 1 \
    --enable_skill_tree 1 \
    --enable_skill_tree_evolve 1 \
    --enable_failure_analysis 1 \
    --max_new_skills 3 \
    --skill_tree_evolve_min_samples 6 \
    `# 轨迹池水位线：与 env.traces_pool.* 对齐` \
    --capacity_watermark 50000 \
    --perf_watermark 0.6 \
    --min_samples 16 \
    --loop_threshold 3 \
    --log_trajectories "$LOG_TRAJECTORIES" \
    "$@" 2>&1 | tee "$OUTPUT_DIR/driver.log"
