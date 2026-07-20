# syntax=docker/dockerfile:1.7
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_APT_MIRROR=https://mirrors.aliyun.com/ubuntu
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
        /etc/apt/sources.list.d/nvidia*.list \
    && sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git build-essential \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Keep build archives out of the image history.  A normal COPY followed by rm
# leaves the archive in a lower layer, needlessly doubling the registry upload.
RUN --mount=type=bind,from=coskill_assets,source=.,target=/build-assets,readonly \
    mkdir -p /opt/conda/envs/skillRL \
    && tar -xzf /build-assets/skillRL.tar.gz -C /opt/conda/envs/skillRL \
    && PATH=/opt/conda/envs/skillRL/bin:$PATH \
        /opt/conda/envs/skillRL/bin/conda-unpack

# conda-pack can restore an obsolete conda Faiss package after the newer pip
# faiss-cpu files.  Overlay the host's tested runtime copy, just as the fixed
# trajectory ablation image does, so full and shared main images agree.
RUN --mount=type=bind,from=coskill_assets,source=.,target=/build-assets,readonly \
    site=/opt/conda/envs/skillRL/lib/python3.10/site-packages \
    && rm -rf "$site/faiss" "$site"/faiss-*.egg-info \
              "$site"/faiss_cpu-*.dist-info "$site/faiss_cpu.libs" \
    && tar -xzf /build-assets/faiss-runtime-overlay.tar.gz -C "$site" \
    && /opt/conda/envs/skillRL/bin/python -c 'import faiss; assert faiss.__version__ == "1.13.2"'

ENV PATH=/opt/conda/envs/skillRL/bin:$PATH \
    CONDA_DEFAULT_ENV=skillRL \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    FAISS_OPT_LEVEL=AVX2 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    COSKILL_CONTAINER=1 \
    PROJECT_ROOT=/workspace/CoSkill \
    ALFWORLD_DATA=/opt/data/alfworld \
    DATA_ROOT=/opt/data/verl-agent \
    OUTPUT_ROOT=/outputs \
    MODEL_PATH=/models/Qwen3-4B-Thinking-2507 \
    WEBSHOP_DATA_DIR=/workspace/CoSkill/agent_system/environments/env_package/webshop/webshop/data

WORKDIR /workspace/CoSkill
COPY --from=coskill_source . /workspace/CoSkill

# The packed environment contains editable pointers to the host SkillRL/Skill0
# checkouts.  Remove those pointers and bind verl to this CoSkill source tree.
RUN python -m pip uninstall -y verl skyrl-gym >/dev/null 2>&1 || true \
    && python -m pip install --no-deps -e /workspace/CoSkill

RUN --mount=type=bind,from=coskill_assets,source=.,target=/build-assets,readonly \
    mkdir -p /opt/data \
    && tar -xzf /build-assets/alfworld-data.tar.gz -C /opt/data

# The 12/32 prepared GRPO parquet files are a runtime contract for every
# Tree-RL entrypoint.  Keep them in the image, not in an ignored source-tree
# cache, so a fresh server can run preflight without a hidden host mount.
RUN --mount=type=bind,from=coskill_assets,source=.,target=/build-assets,readonly \
    tar -xzf /build-assets/prepared-verl-data.tar.gz -C /opt/data \
    && test -f /opt/data/verl-agent/text/train.parquet \
    && test -f /opt/data/verl-agent/text/test.parquet

RUN --mount=type=bind,from=coskill_assets,source=.,target=/build-assets,readonly \
    tar -xzf /build-assets/webshop-small-data.tar.gz \
        -C /workspace/CoSkill/agent_system/environments/env_package/webshop/webshop

# The build script always creates this archive.  It is empty by default and
# can optionally carry the Apache-2.0 Qwen weights for a fully offline image.
RUN --mount=type=bind,from=coskill_assets,source=.,target=/build-assets,readonly \
    mkdir -p /models \
    && tar -xzf /build-assets/model-optional.tar.gz -C /models

RUN mkdir -p /outputs /models \
    && chmod +x docker/coskill/entrypoint.sh examples/grpo_trainer/run_coskill_tree_rl_smoke.sh

ENTRYPOINT ["/workspace/CoSkill/docker/coskill/entrypoint.sh"]
CMD ["preflight"]
