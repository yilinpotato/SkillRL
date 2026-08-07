# CoSkill 新服务器 Docker 部署与八卡 Tree-RL 训练

适用镜像：

```text
export IMAGE=crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data-20260721-fix9-dotenv-api-preflight-20260727-r6
```

本文只覆盖 ALFWorld Tree-RL 的八卡 `root` / `leaf` 运行。每个训练步固定为
`12 个任务实例 × 6 个 GRPO 样本 = 72 rollouts`。

## 1. 前置检查

```bash
docker version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi -L
```

每台服务器需要向容器暴露恰好八张已分配 GPU。`--gpus` 中是宿主机物理 GPU 编号，
容器内部会重编号为逻辑 `cuda:0..7`。

## 2. 下载镜像、建立默认运行目录与预下载模型

默认运行目录是 `$HOME/coskill-run`：

```bash
export REGISTRY='crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com'
export IMAGE="$REGISTRY/yilinpotato/coskill:skillrl-cu128-data-20260721-fix9-dotenv-api-preflight-20260727-r6"
export RUN_ROOT="$HOME/coskill-run"

mkdir -p "$RUN_ROOT/models" "$RUN_ROOT/outputs"
docker login --username=yilinpotato "$REGISTRY"
docker pull "$IMAGE"
```

模型默认存放位置：

```text
宿主机：$RUN_ROOT/models/Qwen3-4B-Thinking-2507
容器内：/models/Qwen3-4B-Thinking-2507
```

首次预下载模型：

```bash
docker run --rm \
  -v "$RUN_ROOT/models:/models" \
  "$IMAGE" shell -lc '
    modelscope download --model Qwen/Qwen3-4B-Thinking-2507 \
      --local_dir /models/Qwen3-4B-Thinking-2507 --max-workers 8
    test -f /models/Qwen3-4B-Thinking-2507/config.json
    compgen -G "/models/Qwen3-4B-Thinking-2507/*.safetensors" >/dev/null
  '
```

此下载可中断后重复执行。训练阶段使用只读模型挂载与
`MODEL_AUTO_DOWNLOAD=0`，不会重新下载。

## 3. 写入云端配置

在 `$RUN_ROOT/.env` 创建下列内容。

```bash
DEEPSEEK_API_KEY='replace-with-secret'
DEEPSEEK_MODEL='deepseek-v4-flash'
SKILL_UPDATER_BACKEND='deepseek'
```

训练容器将此文件以只读方式挂载到 `/workspace/CoSkill/.env`。fix9 使用容器内
Python 解析 allowlist 字段，避免 Docker `--env-file` 将 shell 引号带入 API key。
不要使用 `--env-file "$RUN_ROOT/.env"`。



可以同时直接在终端export：
```bash
export DEEPSEEK_API_KEY='replace-with-secret'
export DEEPSEEK_MODEL='deepseek-v4-flash'
export SKILL_UPDATER_BACKEND='deepseek'
```

## 4. 八卡预检

```bash
docker run --rm \
  --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e CLOUD_BOOTSTRAP_PROBE=1 \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -v "$RUN_ROOT/.env:/workspace/CoSkill/.env:ro" \
  -v "$RUN_ROOT/models:/models:ro" \
  -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" preflight
```

预检应确认模型、数据、八张 GPU 和云端 API。正式 `alfworld-root` / `alfworld-leaf`
启动时还会再次执行强制真实 API probe；probe 失败会在 CUDA、Ray、vLLM 和模型加载前退出。

## 5. 两台八卡服务器并行运行 root 与 leaf

两个实验独立运行：

| 服务器 | `TASK` | `RUN_ID` | 容器输出目录 |
| --- | --- | --- | --- |
| Server A | `alfworld-root` | `alfworld_root_002` | `/outputs/alfworld_root_002` |
| Server B | `alfworld-leaf` | `alfworld_leaf_002` | `/outputs/alfworld_leaf_002` |



### Server A：root

```bash
export RUN_ID=alfworld_root_002
export CONTAINER="coskill-$RUN_ID"

docker run -d --name "$CONTAINER" \
  --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e TREE_RL_USE_ALL_8=1 \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -e RL_OUTPUT_DIR=/outputs/$RUN_ID \
  -e RESUME_SKILL_TREE_STATE=disable \
  -v "$RUN_ROOT/.env:/workspace/CoSkill/.env:ro" \
  -v "$RUN_ROOT/models:/models:ro" \
  -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" alfworld-root trainer.resume_mode=disable

docker logs -f "$CONTAINER"
```

### Server B：leaf

```bash
export RUN_ID=alfworld_leaf_002
export CONTAINER=coskill-$RUN_ID

docker run -d --name "$CONTAINER" \
  --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e TREE_RL_USE_ALL_8=1 \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -e RL_OUTPUT_DIR=/outputs/$RUN_ID \
  -e RESUME_SKILL_TREE_STATE=disable \
  -v "$RUN_ROOT/.env:/workspace/CoSkill/.env:ro" \
  -v "$RUN_ROOT/models:/models:ro" \
  -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" alfworld-leaf trainer.resume_mode=disable

docker logs -f "$CONTAINER"
```


## 7. 运行、恢复和停止

查看日志：

```bash
docker logs -f "$CONTAINER"
tail -f "$RUN_ROOT/outputs/$RUN_ID/training.log"
```


停止容器：

```bash
docker stop "$CONTAINER"
```

恢复：保留 `$RUN_ROOT/outputs/$RUN_ID`，使用相同 `TASK`、`RUN_ID`、模型与挂载。
若容器仍存在，直接：

```bash
docker start -ai "$CONTAINER"
```
