#!/usr/bin/env bash
# Load a user-owned runtime .env without leaking its values through `set -x`.
#
# Callers must set PROJECT_ROOT.  PRIVATE_ENV_FILE can override the default;
# launchers expose this as COSKILL_ENV_FILE.  Values in this file deliberately
# override inherited environment variables, then launcher defaults apply only
# when a value remains unset.

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "load_private_env.sh requires PROJECT_ROOT." >&2
    return 2
fi

_private_env_file="${PRIVATE_ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$_private_env_file" ]]; then
    _private_env_restore_xtrace=0
    case "$-" in
        *x*)
            _private_env_restore_xtrace=1
            set +x
            ;;
    esac

    # .env is intentionally a shell-format, user-owned configuration file.
    # `set -a` exports its assignments to Python, Ray, and vLLM children.
    set -a
    # shellcheck disable=SC1090
    source "$_private_env_file"
    set +a

    if [[ "$_private_env_restore_xtrace" == "1" ]]; then
        set -x
    fi
    echo "[private-env] Loaded $_private_env_file; its values override inherited environment variables." >&2
fi

unset _private_env_file _private_env_restore_xtrace
