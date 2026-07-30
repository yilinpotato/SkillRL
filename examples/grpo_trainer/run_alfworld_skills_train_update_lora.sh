set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PRIVATE_ENV_FILE="${SKILLRL_ENV_FILE:-$PROJECT_ROOT/.env}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE

ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Enable more verbose logging
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG


# export WANDB_API_KEY=""
# Small model (actor, trained locally)
export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
# Large model (SkillUpdater skill generation via DeepSeek API)
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
# export DEEPSEEK_API_KEY=""
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# All run outputs (checkpoints, updated skills, and the training log) are collected here.
PROJECT_NAME="${PROJECT_NAME:-verl_agent_alfworld}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_4b_skills_dynamic_lora_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"

# Per-step training metrics are appended here as one JSON object per line.
export JSONL_PATH="$OUTPUT_DIR/metrics.jsonl"
# Canonical, nested group records shared with CoSkill and Skill0.  The native
# metrics.jsonl remains unchanged for existing tooling.
export COMPARISON_METRICS_PATH="$OUTPUT_DIR/comparison_metrics.jsonl"

# Per-episode rollout trajectories (prompt/response/reward) for human inspection.
# Exported BEFORE `ray start` so the Ray actors inherit it (same as JSONL_PATH).
export TRAJECTORY_DUMP_DIR="$OUTPUT_DIR/trajectories"

num_cpus_per_env_worker=0.35 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# GPU ownership is governed by the scheduler-provided CUDA_VISIBLE_DEVICES.
# For a local one-card speed test, set CUDA_VISIBLE_DEVICES to one *idle*
# physical GPU (for example, 1).  Ray and the trainer must receive the same
# count; the historical hard-coded 2 made a single-card launch inconsistent.
if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    DEFAULT_RAY_NUM_CPUS=56
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${LOCAL_GPU:-0}}"
    DEFAULT_RAY_NUM_CPUS="$(nproc)"
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        echo "Local SkillRL ALFWorld launcher accepts exactly one GPU; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    GPU_ACTIVE_PIDS=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)
    if [[ -n "$GPU_ACTIVE_PIDS" ]]; then
        echo "Requested local GPU $CUDA_VISIBLE_DEVICES is in use by PID(s): $GPU_ACTIVE_PIDS. Refusing to start." >&2
        exit 1
    fi
fi
NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$NUM_VISIBLE_GPUS}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"
if ! [[ "$N_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || [[ "$N_GPUS_PER_NODE" -ne "$NUM_VISIBLE_GPUS" ]]; then
    echo "N_GPUS_PER_NODE=$N_GPUS_PER_NODE must equal visible GPU count=$NUM_VISIBLE_GPUS." >&2
    exit 1
fi

# Do not invoke `ray stop` / `ray start`: a shared host may already have a
# Ray session, and some supported Ray/Click combinations fail in the CLI.
# main_ppo calls Python's ray.init(); it inherits CUDA_VISIBLE_DEVICES and the
# trainer GPU count above, while this option only bounds local CPU workers.
ray_init_args=("ray_init.num_cpus=$RAY_NUM_CPUS")
if [[ -n "${RAY_ADDRESS:-}" ]]; then
    ray_init_args+=("ray_init.address=$RAY_ADDRESS")
fi

train_data_size=12  # Minimal test (divisible by 1)
val_data_size=32    # Minimal test
group_size=6        # GRPO group size (trajectories per prompt)
# Keep the ALFWorld prompt history aligned with the completed comparison runs
# and the WebShop launcher.  The Hydra base recipe defaults to 2, so this must
# be explicit here rather than relying on an external override.
history_length="${HISTORY_LENGTH:-8}"
# Completed comparison runs use 100 GRPO updates.  Keep this overridable for
# ablations, but do not silently revert the default to the older 150-step run.
total_training_steps="${TOTAL_TRAINING_STEPS:-100}"

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=36 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=40 \
    env.history_length=$history_length \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_skills_from_train=True \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','jsonl'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    ++trainer.comparison_metrics_jsonl_path="$COMPARISON_METRICS_PATH" \
    ++trainer.comparison_method=skillrl \
    ++trainer.comparison_benchmark=alfworld \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=1200 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_training_steps \
    "${ray_init_args[@]}" \
    trainer.val_before_train=False $@ 2>&1 | tee "$OUTPUT_DIR/training.log"
