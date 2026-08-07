#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="model,alfworld,webshop"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash prepare.sh [--assets LIST] [--dry-run]

LIST is a comma-separated subset of: model,alfworld,webshop,parquet.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --assets)
            [[ "$#" -ge 2 ]] || { echo "--assets requires a value" >&2; exit 2; }
            ASSETS="$2"
            shift 2
            ;;
        --assets=*)
            ASSETS="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

export CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data/verl-agent}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$ROOT/data/alfworld}"
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$ROOT/agent_system/environments/env_package/webshop/webshop/data}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Thinking-2507}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/$MODEL_ID}"
WEBSHOP_ROOT="$(dirname "$WEBSHOP_DATA_DIR")"

contains_asset() {
    [[ ",$ASSETS," == *",$1,"* ]]
}

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

for item in ${ASSETS//,/ }; do
    case "$item" in
        model|alfworld|webshop|parquet) ;;
        *)
            echo "unsupported asset: $item" >&2
            exit 2
            ;;
    esac
done

if contains_asset model; then
    if [[ -f "$MODEL_PATH/config.json" ]]; then
        echo "model ready: $MODEL_PATH"
    else
        run python - "$MODEL_ID" "$CACHE_ROOT/modelscope/hub" <<'PY'
import sys
from modelscope import snapshot_download

snapshot_download(sys.argv[1], cache_dir=sys.argv[2])
PY
    fi
fi

if contains_asset alfworld; then
    if [[ -f "$ALFWORLD_DATA/logic/alfred.pddl" && -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
        echo "ALFWorld ready: $ALFWORLD_DATA"
    else
        run alfworld-download --data-dir "$ALFWORLD_DATA" -f
    fi
fi

if contains_asset webshop; then
    run mkdir -p "$WEBSHOP_DATA_DIR"
    declare -A FILE_IDS=(
        [items_shuffle_1000.json]=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
        [items_ins_v2_1000.json]=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
        [items_human_ins.json]=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
    )
    for name in "${!FILE_IDS[@]}"; do
        if [[ ! -f "$WEBSHOP_DATA_DIR/$name" ]]; then
            run gdown "https://drive.google.com/uc?id=${FILE_IDS[$name]}" \
                -O "$WEBSHOP_DATA_DIR/$name"
        fi
    done

    if ! python -c 'import spacy; spacy.load("en_core_web_sm")' >/dev/null 2>&1; then
        run python -m spacy download en_core_web_sm
    fi

    if ! compgen -G "$WEBSHOP_ROOT/search_engine/indexes/segments_*" >/dev/null; then
        command -v java >/dev/null 2>&1 || {
            echo "Java 11 is required to build the WebShop index" >&2
            exit 1
        }
        run mkdir -p "$WEBSHOP_ROOT/search_engine/resources" \
            "$WEBSHOP_ROOT/search_engine/indexes"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "+ build WebShop 1,000-product Lucene index"
        else
            (
                cd "$WEBSHOP_ROOT/search_engine"
                python convert_product_file_format.py
                python -m pyserini.index.lucene \
                    --collection JsonCollection \
                    --input resources \
                    --index indexes \
                    --generator DefaultLuceneDocumentGenerator \
                    --threads 1 \
                    --storePositions --storeDocvectors --storeRaw
            )
        fi
    fi
fi

if contains_asset parquet; then
    run mkdir -p "$DATA_ROOT/text"
    run python -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$DATA_ROOT" \
        --train_data_size 12 \
        --val_data_size 32
fi

echo "requested assets prepared: $ASSETS"
