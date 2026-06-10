set -x
ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# it is incompatible with vLLM's CuMemAllocator memory pool (sleep mode) and
# makes vLLM abort at init (see pytorch issue 147851).

# Enable more verbose logging
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG

# 强制离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export RAY_IGNORE_HTTP_PROXY=1



# export WANDB_API_KEY=""
# Small model (actor, trained locally)
export CACHE_ROOT="${CACHE_ROOT:-/data2/myl/home_configs/.cache}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
# Skill-retrieval embedding model (dedicated 0.6B encoder, NOT the 4B actor —
# loading the 4B LM as an encoder caused CPU-RAM OOM). ~0.6B, light on memory.
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B}"
# Large model (CoSkill CloudAnalyzer contrastive distillation via DeepSeek API)
export SKILL_UPDATER_BACKEND="deepseek"
# export DEEPSEEK_API_KEY=""
# CoSkill cloud model: DeepSeek V4 Flash
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

# All run outputs (checkpoints, updated skills, and the training log) are collected here.
PROJECT_NAME="verl_agent_alfworld"
EXPERIMENT_NAME="grpo_qwen3-4b_co_skill"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"

num_cpus_per_env_worker=0.35 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# Auto-detect how many GPUs this machine has so the script uses all of them.
# Respect CUDA_VISIBLE_DEVICES if the user set it, otherwise ask nvidia-smi.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)
fi
# Fallback to 1 if detection fails.
NUM_GPUS=${NUM_GPUS:-1}
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
echo "Detected $NUM_GPUS GPU(s); training will use all of them."

# Auto-detect CPU count too (was hard-coded to 48).
NUM_CPUS=$(nproc 2>/dev/null || echo 8)

# Restart Ray with full CPU/GPU access to avoid resource starvation from previous crashed runs
ray stop --force 2>/dev/null || true
ray start --head --num-cpus="$NUM_CPUS" --num-gpus="$NUM_GPUS"
sleep 3

train_data_size=2  # Minimal test (divisible by 2)
val_data_size=2    # Minimal test
group_size=2       # Minimal parallelism

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
    `# 原值: data.max_prompt_length=4096 (降低以省显存/降低OOM)` \
    data.max_prompt_length=2048 \
    `# 原值: data.max_response_length=4096 -> 2048; 激进测试: 512 (clip_ratio~0.85说明大量冗余thinking, 砍长度直接降gen时间)` \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    `# 原值: rollout.log_prob_micro_batch_size_per_gpu=8 (降低以省前向显存)` \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    `# 新增: 解决FSDP->vLLM首次权重同步OOM的根因。` \
    `# load_format默认dummy时, sharding_manager.base_sync_done=False, 首次同步会summon整个4B base模型全参(~8G)与vLLM(~11.8G)叠加撞顶24G。` \
    `# 改safetensors让vLLM启动即加载真实base权重(base_sync_done=True), 之后仅逐层summon LoRA小参数, 避免全参峰值。` \
    actor_rollout_ref.rollout.load_format=safetensors \
    `# 配合load_format=safetensors使用: 逐层summon LoRA参数而非一次性拉全模型 (默认False)。该键已在schema中, 用普通赋值不加+前缀。` \
    actor_rollout_ref.rollout.layered_summon=True \
    `# 原值: gpu_memory_utilization=0.5。注意: 不能设太低(如0.3), vLLM权重7.73G+前向峰值2.81G≈10.55G, util*24G需大于此值否则KV cache为负直接报错` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    `# 原值: enforce_eager=True; 激进测试: False 开CUDA graph消除CPU launch瓶颈(实测gen时GPU仅22%util,CPU单核97%卡在算子dispatch)` \
    actor_rollout_ref.rollout.enforce_eager=False \
    `# 原值: free_cache_engine=False (改True: 训练阶段释放vLLM KV cache, 缓解权重同步时OOM)` \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    `# 原值: ref.log_prob_micro_batch_size_per_gpu=4 (降低以省前向显存)` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    `# 原值: env.max_steps=50; 激进测试: 15 (失败episode常走满50步, 砍上限直接少~70%生成步数)` \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_skills_from_train=True \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    +env.skills_only_memory.enable_coskill=True \
    +env.skills_only_memory.enable_hierarchy=True \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path="$EMBEDDING_MODEL_PATH" \
    +env.skills_only_memory.stable_cycles_l1=3 \
    +env.skills_only_memory.stable_cycles_l2=5 \
    +env.skills_only_memory.success_l1=0.7 \
    +env.skills_only_memory.demote_threshold=0.3 \
    +env.skills_only_memory.enable_internalize=False \
    +env.traces_pool.capacity_watermark=50000 \
    +env.traces_pool.perf_watermark=0.6 \
    +env.traces_pool.min_samples=8 \
    +env.traces_pool.loop_threshold=3 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='qwen3-4b_co_skill' \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=1200 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    `# 原值: total_epochs=150; 激进测试: 2 epoch (train_data_size=2 => 每epoch 1步, 只为看一步能否跑完)` \
    trainer.total_epochs=2 \
    trainer.val_before_train=False $@ 2>&1 | tee "$OUTPUT_DIR/training.log"
