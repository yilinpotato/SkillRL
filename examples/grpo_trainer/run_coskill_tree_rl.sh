#!/usr/bin/env bash
# CoSkill skill-tree GRPO: progressive root/leaf internalization on Ray.
#
# Usage (normally reached through rl=1 on a no-RL launcher):
#   bash examples/grpo_trainer/run_coskill_tree_rl.sh alfworld
#   TREE_RL_ORDER=leaf bash examples/grpo_trainer/run_coskill_tree_rl.sh webshop
#
# This script intentionally accepts only a 2- or 4-A800 allocation.  It does
# not run `ray stop` or `ray start`: main_ppo owns its Ray lifecycle, so a job
# never destroys a different user's Ray session on a shared cluster.

set -euo pipefail

BENCHMARK="${1:-}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
case "$BENCHMARK" in
    alfworld|webshop) ;;
    *)
        echo "Usage: $0 {alfworld|webshop} [Hydra overrides...]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE
cd "$PROJECT_ROOT"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export RAY_IGNORE_HTTP_PROXY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
    RUN_ENV="超算 (supercomputer)"
    export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    DEFAULT_RAY_NUM_CPUS=56
else
    echo "CoSkill tree RL is configured for a scheduled 2/4-A800 allocation, not the shared local 3090." >&2
    echo "Request 2 or 4 A800 GPUs and set CUDA_VISIBLE_DEVICES to that allocation." >&2
    exit 1
fi

NUM_GPUS="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
if [[ "$NUM_GPUS" != "2" && "$NUM_GPUS" != "4" ]]; then
    echo "Need exactly 2 or 4 visible A800 GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)." >&2
    exit 1
fi
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$NUM_GPUS}"
if [[ "$N_GPUS_PER_NODE" != "$NUM_GPUS" ]]; then
    echo "N_GPUS_PER_NODE must equal the allocated visible GPU count ($NUM_GPUS)." >&2
    exit 1
fi

TREE_RL_ORDER="${TREE_RL_ORDER:-root}"
if [[ "$TREE_RL_ORDER" != "root" && "$TREE_RL_ORDER" != "leaf" ]]; then
    echo "TREE_RL_ORDER must be root or leaf, got: $TREE_RL_ORDER" >&2
    exit 1
fi

# Experiment contract: every CoSkill Tree-RL update contains exactly
# 12 distinct WebShop/ALFWorld goals × 6 GRPO samples = 72 rollouts,
# regardless of whether FSDP uses two or four A800s.  Only the dispatch is
# sharded more finely on four GPUs.  Ray/verl requires the *expanded* rollout
# batch (not the number of base goals) to divide across ranks.
DEFAULT_TRAIN_DATA_SIZE=12
DEFAULT_PPO_MINI_BATCH=36
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-$DEFAULT_TRAIN_DATA_SIZE}"
GROUP_SIZE="${GROUP_SIZE:-6}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-32}"
ROLLOUTS_PER_STEP=$((TRAIN_DATA_SIZE * GROUP_SIZE))
if (( ROLLOUTS_PER_STEP % NUM_GPUS != 0 )); then
    echo "Expanded rollout batch TRAIN_DATA_SIZE×GROUP_SIZE=${ROLLOUTS_PER_STEP} must be divisible by $NUM_GPUS Ray/FSDP ranks." >&2
    exit 1
fi
if (( VAL_DATA_SIZE % NUM_GPUS != 0 )); then
    echo "VAL_DATA_SIZE=$VAL_DATA_SIZE must be divisible by $NUM_GPUS Ray/FSDP ranks." >&2
    exit 1
fi

# Keep the *global* GRPO/PPO geometry invariant across two and four ranks:
# 72 rollout samples -> two global PPO mini-batches of 36 -> nine global
# micro-batches of four samples per mini-batch.  With two ranks this is
# 18 samples/rank and micro=2; with four ranks it is 9 samples/rank and must
# be micro=1.  Keeping micro=2 on four ranks causes FSDP's creation-time
# assertion: normalized mini-batch 9 is not divisible by micro-batch 2.
if (( NUM_GPUS == 4 )); then
    DEFAULT_PPO_MICRO_BATCH_PER_GPU=1
else
    DEFAULT_PPO_MICRO_BATCH_PER_GPU=2
fi
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$DEFAULT_PPO_MINI_BATCH}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_PPO_MICRO_BATCH_PER_GPU}"
if (( PPO_MINI_BATCH_SIZE <= 0 || PPO_MICRO_BATCH_SIZE_PER_GPU <= 0 )); then
    echo "PPO_MINI_BATCH_SIZE and PPO_MICRO_BATCH_SIZE_PER_GPU must be positive." >&2
    exit 1
fi
if (( ROLLOUTS_PER_STEP % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "ROLLOUTS_PER_STEP=$ROLLOUTS_PER_STEP must be divisible by PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE." >&2
    exit 1
fi
if (( PPO_MINI_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE must be divisible by $NUM_GPUS Ray/FSDP ranks." >&2
    exit 1
fi
PPO_MINI_BATCH_PER_GPU=$((PPO_MINI_BATCH_SIZE / NUM_GPUS))
if (( PPO_MINI_BATCH_PER_GPU % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
    echo "Per-rank PPO mini-batch=$PPO_MINI_BATCH_PER_GPU must be divisible by PPO_MICRO_BATCH_SIZE_PER_GPU=$PPO_MICRO_BATCH_SIZE_PER_GPU." >&2
    exit 1
fi
PPO_GRAD_ACCUM_STEPS=$((PPO_MINI_BATCH_PER_GPU / PPO_MICRO_BATCH_SIZE_PER_GPU))
PPO_GLOBAL_MICRO_BATCH=$((NUM_GPUS * PPO_MICRO_BATCH_SIZE_PER_GPU))

MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
if [[ "$BENCHMARK" == "alfworld" ]]; then
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
    MAX_STEPS="${MAX_STEPS:-40}"
    SKILLS_JSON="${SKILLS_JSON:-memory_data/alfworld/claude_style_skills.json}"
    ENV_NAME="alfworld/AlfredTWEnv"
    RETRIEVAL_MODE="${RETRIEVAL_MODE:-template}"
    PROJECT_NAME="verl_agent_alfworld"
else
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
    MAX_STEPS="${MAX_STEPS:-15}"
    SKILLS_JSON="${SKILLS_JSON:-memory_data/webshop/claude_style_skills.json}"
    ENV_NAME="Webshop"
    RETRIEVAL_MODE="${RETRIEVAL_MODE:-template}"
    PROJECT_NAME="verl_agent_webshop"
fi

export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"
export ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
export REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
# Keep DP=4 and TP=1.  This optimization compacts only already-finished
# trajectories before vLLM generation; it does not alter model parallelism,
# rollouts-per-step, prompts, rewards, or PPO geometry.
export COMPACT_FINISHED_TRAJECTORIES="${COMPACT_FINISHED_TRAJECTORIES:-True}"
case "$COMPACT_FINISHED_TRAJECTORIES" in
    True|False) ;;
    *)
        echo "COMPACT_FINISHED_TRAJECTORIES must be True or False, got: $COMPACT_FINISHED_TRAJECTORIES" >&2
        exit 1
        ;;
esac
# A complete Tree-RL run needs a usable cloud client.  Check it before Ray and
# vLLM reserve the A800s; --probe is opt-in because it makes one tiny API call.
export CLOUD_BOOTSTRAP_CHECK="${CLOUD_BOOTSTRAP_CHECK:-1}"
export CLOUD_BOOTSTRAP_PROBE="${CLOUD_BOOTSTRAP_PROBE:-0}"
case "$CLOUD_BOOTSTRAP_CHECK" in 0|1) ;; *) echo "CLOUD_BOOTSTRAP_CHECK must be 0 or 1" >&2; exit 1;; esac
case "$CLOUD_BOOTSTRAP_PROBE" in 0|1) ;; *) echo "CLOUD_BOOTSTRAP_PROBE must be 0 or 1" >&2; exit 1;; esac
# Match the frozen CoSkill WebShop path.  This is a soft character guard used
# only to compact oldest complete history records; the hard prompt limit stays
# data.max_prompt_length=8192 tokens.
export WEBSHOP_PROMPT_CHAR_LIMIT="${WEBSHOP_PROMPT_CHAR_LIMIT:-24000}"

# Curriculum gates: a layer first receives normal on-policy GRPO, then is
# hidden for an independent on-policy probe.  Passing the probe permanently
# removes only that layer; failing restores it and continues GRPO.
export TREE_RL_MIN_UPDATES="${TREE_RL_MIN_UPDATES:-5}"
export TREE_RL_MIN_TRAIN_EPISODES="${TREE_RL_MIN_TRAIN_EPISODES:-24}"
export TREE_RL_TRAIN_SUCCESS_THRESHOLD="${TREE_RL_TRAIN_SUCCESS_THRESHOLD:-0.7}"
export TREE_RL_MIN_PROBE_EPISODES="${TREE_RL_MIN_PROBE_EPISODES:-24}"
export TREE_RL_PROBE_SUCCESS_THRESHOLD="${TREE_RL_PROBE_SUCCESS_THRESHOLD:-0.7}"
export TREE_RL_STATE_SAVE_FREQ="${TREE_RL_STATE_SAVE_FREQ:-1}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_coskill_tree_rl_${TREE_RL_ORDER}_${NUM_GPUS}xa800}"
OUTPUT_DIR="${RL_OUTPUT_DIR:-${OUTPUT_DIR:-$OUTPUT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR" "$DATA_ROOT/text"
export JSONL_METRICS_DIR="$OUTPUT_DIR"

if [[ "$BENCHMARK" == "webshop" ]]; then
    if [[ -z "${WEBSHOP_DATA_DIR:-}" ]]; then
        for candidate in \
            "$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data" \
            "$(dirname "$PROJECT_ROOT")/Skill0/agent_system/environments/env_package/webshop/webshop/data" \
            "/data2/myl/Skill0/agent_system/environments/env_package/webshop/webshop/data"
        do
            if [[ -f "$candidate/items_shuffle_1000.json" && -f "$candidate/items_ins_v2_1000.json" \
                  && -f "$candidate/items_human_ins.json" \
                  && -d "$(dirname "$candidate")/search_engine/indexes" ]]; then
                export WEBSHOP_DATA_DIR="$candidate"
                break
            fi
        done
    fi
    if [[ -z "${WEBSHOP_DATA_DIR:-}" ]]; then
        echo "WebShop assets missing. Set WEBSHOP_DATA_DIR to the populated data directory." >&2
        exit 1
    fi
fi

if [[ "$CLOUD_BOOTSTRAP_CHECK" == "1" ]]; then
    cloud_check_args=(
        --environment "$BENCHMARK"
        --skills-json "$SKILLS_JSON"
    )
    if [[ "$CLOUD_BOOTSTRAP_PROBE" == "1" ]]; then
        cloud_check_args+=(--probe)
    fi
    echo "Checking cloud bootstrap (probe=$CLOUD_BOOTSTRAP_PROBE) before allocating Ray/vLLM..."
    python3 scripts/check_cloud_bootstrap.py "${cloud_check_args[@]}"
else
    echo "WARNING: CLOUD_BOOTSTRAP_CHECK=0; this run can silently skip cloud evolution if its credential is unavailable." >&2
fi

echo "CoSkill tree RL: benchmark=$BENCHMARK environment=$RUN_ENV"
echo "GPU allocation: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ranks=$NUM_GPUS"
echo "vLLM topology: DP=$NUM_GPUS TP=1 PP=1 (unchanged)"
echo "Active trajectory compaction: $COMPACT_FINISHED_TRAJECTORIES (only completed rows are excluded from future vLLM calls)"
echo "GRPO rollout: train_data_size=$TRAIN_DATA_SIZE group_size=$GROUP_SIZE total=$ROLLOUTS_PER_STEP (fixed across GPU counts)"
echo "PPO geometry: global_mini=$PPO_MINI_BATCH_SIZE per_rank_mini=$PPO_MINI_BATCH_PER_GPU micro_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU global_micro=$PPO_GLOBAL_MICRO_BATCH accumulation=$PPO_GRAD_ACCUM_STEPS"
echo "Validation: val_data_size=$VAL_DATA_SIZE test_freq=$TEST_FREQ val_before_train=$VAL_BEFORE_TRAIN"
echo "Tree curriculum: order=$TREE_RL_ORDER train>=${TREE_RL_MIN_TRAIN_EPISODES}@${TREE_RL_TRAIN_SUCCESS_THRESHOLD} probe>=${TREE_RL_MIN_PROBE_EPISODES}@${TREE_RL_PROBE_SUCCESS_THRESHOLD}"
echo "Output: $OUTPUT_DIR"

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$DATA_ROOT" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

ppo_args=(
    algorithm.adv_estimator=grpo
    "data.train_files=$DATA_ROOT/text/train.parquet"
    "data.val_files=$DATA_ROOT/text/test.parquet"
    "data.train_batch_size=$TRAIN_DATA_SIZE"
    "data.val_batch_size=$VAL_DATA_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=left
    data.return_raw_chat=True

    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    "actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$VLLM_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
    algorithm.use_kl_in_reward=False

    "env.env_name=$ENV_NAME"
    env.seed=0
    "env.max_steps=$MAX_STEPS"
    env.history_length=8
    "++env.webshop.prompt_char_limit=$WEBSHOP_PROMPT_CHAR_LIMIT"
    "env.rollout.n=$GROUP_SIZE"
    "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
    +env.use_skills_only_memory=True
    "+env.compact_finished_trajectories=$COMPACT_FINISHED_TRAJECTORIES"
    "+env.skills_only_memory.skills_json_path=$SKILLS_JSON"
    "+env.skills_only_memory.retrieval_mode=$RETRIEVAL_MODE"
    +env.skills_only_memory.top_k=6
    +env.skills_only_memory.enable_dynamic_update=True
    +env.skills_only_memory.update_skills_from_train=True
    +env.skills_only_memory.skill_update_freq=1
    +env.skills_only_memory.enable_coskill=True
    +env.skills_only_memory.enable_hierarchy=True
    +env.skills_only_memory.enable_playbook=True
    +env.skills_only_memory.enable_playbook_evolve=True
    +env.skills_only_memory.enable_failure_analysis=True
    +env.skills_only_memory.playbook_evolve_min_samples=6
    +env.skills_only_memory.max_new_skills=3
    +env.skills_only_memory.stable_cycles_l1=3
    +env.skills_only_memory.stable_cycles_l2=5
    +env.skills_only_memory.success_l1=0.7
    +env.skills_only_memory.demote_threshold=0.3
    +env.skills_only_memory.min_calls=10
    +env.skills_only_memory.enable_internalize=False
    +env.skills_only_memory.enable_tree_rl_internalize=True
    "+env.skills_only_memory.tree_rl_order=$TREE_RL_ORDER"
    "+env.skills_only_memory.tree_rl_min_updates=$TREE_RL_MIN_UPDATES"
    "+env.skills_only_memory.tree_rl_min_train_episodes=$TREE_RL_MIN_TRAIN_EPISODES"
    "+env.skills_only_memory.tree_rl_train_success_threshold=$TREE_RL_TRAIN_SUCCESS_THRESHOLD"
    "+env.skills_only_memory.tree_rl_min_probe_episodes=$TREE_RL_MIN_PROBE_EPISODES"
    "+env.skills_only_memory.tree_rl_probe_success_threshold=$TREE_RL_PROBE_SUCCESS_THRESHOLD"
    "+env.skills_only_memory.tree_rl_state_save_freq=$TREE_RL_STATE_SAVE_FREQ"
    +env.dump_raw_trajectories=False
    +env.traces_pool.capacity_watermark=50000
    +env.traces_pool.perf_watermark=0.6
    +env.traces_pool.min_samples=16
    +env.traces_pool.loop_threshold=3

    trainer.critic_warmup=0
    trainer.logger=['console','jsonl']
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.default_local_dir=$OUTPUT_DIR"
    "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.ray_wait_register_center_timeout=1200
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.total_epochs=$TOTAL_TRAINING_STEPS"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN"
)

if [[ "$BENCHMARK" == "webshop" ]]; then
    ppo_args+=(
        ++env.webshop.use_small=True
        ++env.webshop.human_goals=False
    )
fi
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    ppo_args+=("ray_init.num_cpus=$RAY_NUM_CPUS")
else
    echo "Using scheduler-provided Ray cluster: RAY_ADDRESS=$RAY_ADDRESS"
fi

{
    echo "timestamp=$(date -Is)"
    for key in BENCHMARK RUN_ENV CUDA_VISIBLE_DEVICES NUM_GPUS N_GPUS_PER_NODE MODEL_PATH DATA_ROOT OUTPUT_ROOT OUTPUT_DIR TRAIN_DATA_SIZE GROUP_SIZE VAL_DATA_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU PPO_MINI_BATCH_PER_GPU PPO_GLOBAL_MICRO_BATCH PPO_GRAD_ACCUM_STEPS MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_STEPS WEBSHOP_PROMPT_CHAR_LIMIT TEST_FREQ VAL_BEFORE_TRAIN TREE_RL_ORDER TREE_RL_MIN_UPDATES TREE_RL_MIN_TRAIN_EPISODES TREE_RL_TRAIN_SUCCESS_THRESHOLD TREE_RL_MIN_PROBE_EPISODES TREE_RL_PROBE_SUCCESS_THRESHOLD TREE_RL_STATE_SAVE_FREQ COMPACT_FINISHED_TRAJECTORIES CLOUD_BOOTSTRAP_CHECK CLOUD_BOOTSTRAP_PROBE; do
        echo "$key=${!key}"
    done
    echo "ROLLOUTS_PER_STEP=$ROLLOUTS_PER_STEP"
    [[ "$BENCHMARK" == "webshop" ]] && echo "WEBSHOP_DATA_DIR=$WEBSHOP_DATA_DIR"
} > "$OUTPUT_DIR/run_config.env"
printf '%s\n' "${ppo_args[@]}" > "$OUTPUT_DIR/ppo_args.txt"

TEE_ARGS=()
if [[ -f "$OUTPUT_DIR/training.log" ]]; then
    TEE_ARGS=(-a)
    printf '\n===== CoSkill tree RL resume: %s =====\n' "$(date -Is)" >> "$OUTPUT_DIR/training.log"
fi
python3 -u -m verl.trainer.main_ppo "${ppo_args[@]}" "$@" 2>&1 | tee "${TEE_ARGS[@]}" "$OUTPUT_DIR/training.log"
