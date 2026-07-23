#!/usr/bin/env bash
# Source after loading .env, before touching CUDA/model assets.
# PROJECT_ROOT is required; the marker prevents nested launcher wrappers from
# sending duplicate probe requests for one invocation.

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "preflight_cloud_api.sh requires PROJECT_ROOT." >&2
  return 2
fi
if [[ "${COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE:-0}" == "1" ]]; then
  return 0
fi

CLOUD_BOOTSTRAP_PROBE="${CLOUD_BOOTSTRAP_PROBE:-${CLOUD_PROBE:-1}}"
if [[ "$CLOUD_BOOTSTRAP_PROBE" != "0" && "$CLOUD_BOOTSTRAP_PROBE" != "1" ]]; then
  echo "CLOUD_BOOTSTRAP_PROBE/CLOUD_PROBE must be 0 or 1." >&2
  return 2
fi
if [[ "${SKILL_UPDATER_BACKEND:-deepseek}" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required before CoSkill artifact generation." >&2
  echo "Export DEEPSEEK_API_KEY or set it in ${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}." >&2
  return 2
fi

cloud_check_args=(--environment alfworld --skills-json "$PROJECT_ROOT/memory_data/alfworld/claude_style_skills.json")
if [[ "$CLOUD_BOOTSTRAP_PROBE" == "1" ]]; then
  cloud_check_args+=(--probe)
fi
echo "Checking cloud API before CUDA/model setup (probe=$CLOUD_BOOTSTRAP_PROBE)..."
python "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py" "${cloud_check_args[@]}"
unset cloud_check_args
export COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE=1
