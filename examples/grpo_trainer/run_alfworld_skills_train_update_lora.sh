set -x
ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Enable more verbose logging
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG

# 强制离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export RAY_IGNORE_HTTP_PROXY=1
export ALFWORLD_DATA="${ALFWORLD_DATA:-/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/cache/alfworld}"

export HF_HOME=/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/cache/hf
export HF_DATASETS_CACHE=/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/cache/datasets
export TRANSFORMERS_CACHE=/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/cache/hf

# export WANDB_API_KEY=""
# Small model (actor, trained locally)
export CACHE_ROOT="${CACHE_ROOT:-/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/cache}"
export HF_HOME="${HF_HOME:-/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/cache/huggingface}"
export MODEL_PATH="${MODEL_PATH:-/XYAIFS00/HDD_POOL/hit_wxia/hit_wxiaxy_1/myl/model/Qwen3-4B-Thinking-2507}"
# Large model (SkillUpdater skill generation via DeepSeek API)
export SKILL_UPDATER_BACKEND="deepseek"
# export DEEPSEEK_API_KEY=""
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# All run outputs (checkpoints, updated skills, and the training log) are collected here.
PROJECT_NAME="verl_agent_alfworld"
EXPERIMENT_NAME="grpo_qwen2.5_7b_skills_dynamic_lora"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"

num_cpus_per_env_worker=0.5 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# Restart Ray with full CPU/GPU access to avoid resource starvation from previous crashed runs
ray stop --force 2>/dev/null || true
ray start --head --num-cpus=56 --num-gpus=2
sleep 3

train_data_size=12  # Minimal test (divisible by 1)
val_data_size=32    # Minimal test
group_size=6        # GRPO group size (trajectories per prompt)

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
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=36 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
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
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen2.5_7b_skills_dynamic_lora' \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=1200 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False $@ 2>&1 | tee "$OUTPUT_DIR/training.log"
