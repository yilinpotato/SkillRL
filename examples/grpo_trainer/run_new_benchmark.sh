#!/usr/bin/env bash
# Shared environment settings; pass method/hardware settings as explicit Hydra overrides.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${BENCHMARK:?Set searchqa_live or scienceworld}"
: "${MODEL_PATH:?Set the SFT checkpoint, not the unfine-tuned base}"
: "${DATA_DIR:?Directory containing train/dev/test.parquet}"
: "${MAX_PROMPT_LENGTH:?Set a common supported context budget; overflow fails rather than truncates}"
: "${N_GPUS_PER_NODE:?Use scheduler allocation or an explicitly verified idle GPU}"
case "$BENCHMARK" in
  searchqa_live) steps=15; : "${TAVILY_API_KEY:?Set Tavily credentials outside Git}" ;;
  scienceworld) steps=100 ;;
  *) echo 'Unknown benchmark' >&2; exit 1 ;;
esac
extra=()
if [[ -n "${SKILLS_PATH:-}" ]]; then
  extra+=(+env.use_skills_only_memory=True "+env.skills_only_memory.skills_json_path=$SKILLS_PATH" +env.skills_only_memory.retrieval_mode=template +env.skills_only_memory.top_k=6)
fi
# No preset small skills or invented domain skills. Supply an explicitly selected bank if required.
python -m verl.trainer.main_ppo \
  "env.env_name=$BENCHMARK" "env.max_steps=$steps" env.history_length=0 \
  "data.train_files=$DATA_DIR/train.parquet" "data.val_files=$DATA_DIR/${EVAL_SPLIT:-dev}.parquet" \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" data.max_response_length=4096 \
  data.truncation=error data.filter_overlong_prompts=False data.return_raw_chat=True \
  "actor_rollout_ref.model.path=$MODEL_PATH" "trainer.n_gpus_per_node=$N_GPUS_PER_NODE" \
  "${extra[@]}" "$@"
