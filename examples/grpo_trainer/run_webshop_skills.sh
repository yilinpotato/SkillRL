#!/usr/bin/env bash
set -x
set -euo pipefail
ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then
    shift  # Remove engine so $@ only contains extra params.
fi
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Enable more verbose logging
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG

# export WANDB_API_KEY=""
# ── 自动判断运行环境：超算 vs 本地3090（与 ALFWorld 启动脚本一致）──────────────
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
if [ -d /GLOBALFS/hit_wxia_1 ]; then
    RUN_ENV="超算 (supercomputer)"
    export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
    DEFAULT_RAY_NUM_CPUS=56
else
    RUN_ENV="本地3090 (local)"
    export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/skillrl_data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
    DEFAULT_RAY_NUM_CPUS="$(nproc)"
fi

# Respect an explicit CUDA_VISIBLE_DEVICES/GPUS/GPU setting.  Otherwise use at
# most two GPUs, matching the A800 reference while remaining runnable on 1x3090.
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
fi
NUM_VISIBLE_GPUS=${NUM_VISIBLE_GPUS:-1}
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"

# Small model (actor, trained locally)
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
# Large model (SkillUpdater skill generation via DeepSeek API)
export SKILL_UPDATER_BACKEND="deepseek"
# export DEEPSEEK_API_KEY=""
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# All run outputs (checkpoints, updated skills, and the training log) are collected here.
PROJECT_NAME="verl_agent_webshop"
EXPERIMENT_NAME="grpo_qwen3_4b_webshop_skills_dynamic_lora"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"
echo "Run environment detected: $RUN_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Ray resources: CPUs=$RAY_NUM_CPUS GPUs=$NUM_VISIBLE_GPUS"

# Per-step training metrics are appended here as one JSON object per line.
export JSONL_PATH="$OUTPUT_DIR/metrics.jsonl"

# Per-episode rollout trajectories (prompt/response/reward) for human inspection.
# Exported before Python starts so the Ray actors inherit it (same as JSONL_PATH).
export TRAJECTORY_DUMP_DIR="$OUTPUT_DIR/trajectories"

num_cpus_per_env_worker="${ENV_WORKER_CPUS:-0.35}"
# 4,096-token WebShop trajectories can exceed an A800's activation budget with
# the previous 6-sample micro-batch. Conservative defaults keep the formal
# 12×6 rollout batch while reducing per-GPU backward memory.
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
export REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
export FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-True}"
export FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-True}"

# Do not call `ray start` / `ray stop`: some cluster environments combine a
# Ray release with a newer Click whose Sentinel cannot be deep-copied by the
# Ray CLI. main_ppo calls Python's ray.init() itself, which avoids that broken
# command-line entry point and receives the CPU limit below via ray_init.*.
ray_init_args=("ray_init.num_cpus=$RAY_NUM_CPUS")
if [ -n "${RAY_ADDRESS:-}" ]; then
    # Ray will connect to the scheduler-provided cluster. Passing local
    # resource declarations in this mode is rejected by ray.init().
    ray_init_args=()
    echo "Using existing Ray cluster at RAY_ADDRESS=$RAY_ADDRESS"
fi

train_data_size="${TRAIN_DATA_SIZE:-12}"
val_data_size="${VAL_DATA_SIZE:-32}"
group_size="${GROUP_SIZE:-6}"

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "$DATA_ROOT" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_ROOT/text/train.parquet \
    data.val_files=$DATA_ROOT/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-8192} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-4096} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$FSDP_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$FSDP_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$FSDP_PARAM_OFFLOAD \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.history_length=8 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/webshop/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_skills_from_train=True \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','jsonl'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='grpo_qwen3_4b_webshop_skills_dynamic_lora' \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$NUM_VISIBLE_GPUS \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=1200 \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_TRAINING_STEPS:-150} \
    "${ray_init_args[@]}" \
    trainer.val_before_train=False $@ 2>&1 | tee "$OUTPUT_DIR/training.log"
