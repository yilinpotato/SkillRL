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

# 强制离线
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# vLLM v1 需 spawn（agent_vllm 内也设了，这里兜底）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 关键：stdout 被下面的 `| tee` 管道重定向后，Python 默认切到【全缓冲】（不是行缓冲），
# 我们自己的 print() 会攒够几 KB 才真正写出——vLLM 走 logging 模块会立刻刷出来，两者
# 混在一起看，就像是"跑完 vLLM 日志后卡住"，其实是自己的进度 print 被缓冲憋住了。
# PYTHONUNBUFFERED=1（等价 python3 -u）强制无缓冲，进度 print 立刻可见。
export PYTHONUNBUFFERED=1

# GPU selection.  By default this no-RL batch rollout uses two data-parallel
# workers on the two-A800 server: each worker gets one GPU and one full vLLM
# replica.  If the user already set CUDA_VISIBLE_DEVICES, respect it; otherwise
# GPU=... can select a single device, and GPUS=0,1 can select multiple devices.
if [ -n "${GPUS:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
elif [ -n "${GPU:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_VISIBLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ "${NUM_VISIBLE_GPUS:-0}" -ge 2 ]; then
        export CUDA_VISIBLE_DEVICES="0,1"
        NUM_VISIBLE_GPUS=2
    elif [ "${NUM_VISIBLE_GPUS:-0}" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES="0"
        NUM_VISIBLE_GPUS=1
    else
        NUM_VISIBLE_GPUS=1
    fi
fi
NUM_VISIBLE_GPUS=${NUM_VISIBLE_GPUS:-1}
[ "$NUM_VISIBLE_GPUS" -lt 1 ] && NUM_VISIBLE_GPUS=1
DEFAULT_DP=$(( NUM_VISIBLE_GPUS < 2 ? NUM_VISIBLE_GPUS : 2 ))
DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$DEFAULT_DP}"
ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
DEFAULT_TP=1
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TP}"

# ── 自动判断运行环境：超算 vs 本地3090（与训练脚本一致）─────────────────────────
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
echo "Run environment detected: $RUN_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "data_parallel_workers: $DATA_PARALLEL_WORKERS"
echo "rollout_worker_gpus: ${ROLLOUT_WORKER_GPUS:-<auto>}"
echo "vLLM tensor_parallel_size: $TENSOR_PARALLEL_SIZE"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"

# 云端大模型（与训练脚本一致）：DeepSeek V4 Flash
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
# export DEEPSEEK_API_KEY=...   # 需在环境里提供

PROJECT_NAME="verl_agent_alfworld"
EXPERIMENT_NAME="qwen3-4b_skill_tree_evolve_norl_v6"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"

# Match 100 GRPO trainer steps approximately:
# train_data_size(12) × group_size(6) × 100 = 7200 rollout episodes.
# Override with MAX_EPISODES=... or a later --max_episodes CLI argument.
MAX_EPISODES="${MAX_EPISODES:-7200}"
# Match one GRPO rollout batch by default: train_data_size(12) × group_size(6).
BATCH_ROLLOUT_SIZE="${BATCH_ROLLOUT_SIZE:-72}"
# Save lightweight experiment checkpoints every N rollout groups.  This does not
# force a cloud update; CoSkill still follows the paper trigger/watermark logic.
CHECKPOINT_EVERY_GROUPS="${CHECKPOINT_EVERY_GROUPS:-2}"
# Long runs otherwise dump every step's full prompt into trajectories/, which is
# useful for debugging but creates huge IO. Metrics/raw traces/cloud_io are still
# written, so this does not affect rollout decisions or CoSkill updates.
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-0}"

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
