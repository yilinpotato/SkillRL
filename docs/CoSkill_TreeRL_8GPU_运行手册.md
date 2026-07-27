# CoSkill 新服务器 Docker 部署与八卡 Tree-RL 训练

适用镜像：

```text
crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data-20260721-fix9-dotenv-api-preflight-20260727
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
export IMAGE="$REGISTRY/yilinpotato/coskill:skillrl-cu128-data-20260721-fix9-dotenv-api-preflight-20260727"
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

在 `$RUN_ROOT/.env` 创建下列内容。不要把真实 key 写入仓库、命令历史或日志。

```bash
DEEPSEEK_API_KEY='replace-with-secret'
DEEPSEEK_MODEL='deepseek-v4-flash'
SKILL_UPDATER_BACKEND='deepseek'
```

训练容器将此文件以只读方式挂载到 `/workspace/CoSkill/.env`。fix9 使用容器内
Python 解析 allowlist 字段，避免 Docker `--env-file` 将 shell 引号带入 API key。
不要使用 `--env-file "$RUN_ROOT/.env"`。

读取优先级：

1. `COSKILL_CONTAINER_DOTENV` 指定的已挂载文件；
2. `/workspace/CoSkill/.env`；
3. `/run/secrets/coskill.env`。

也可以不用 `.env`，改用宿主机 `export DEEPSEEK_API_KEY`、
`export DEEPSEEK_MODEL`、`export SKILL_UPDATER_BACKEND`，并在 `docker run` 中传入
相应 `-e` 参数；两种方式不要混用同名变量。

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
| Server A | `alfworld-root` | `alfworld_root_001` | `/outputs/alfworld_root_001` |
| Server B | `alfworld-leaf` | `alfworld_leaf_001` | `/outputs/alfworld_leaf_001` |

两台服务器必须使用相同镜像、模型、数据、`MAX_STEPS`、`TOTAL_TRAINING_STEPS` 和其他
超参数；必须使用不同 `RUN_ID`。不得共享 checkpoint、`skill_lib/skills_tree_rl_latest.json`
或输出目录。

### Server A：root

```bash
export TASK='alfworld-root'
export RUN_ID='alfworld_root_001'
export CONTAINER="coskill-$RUN_ID"

docker run -d --name "$CONTAINER" \
  --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e TREE_RL_USE_ALL_8=1 \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -e RL_OUTPUT_DIR="/outputs/$RUN_ID" \
  -v "$RUN_ROOT/.env:/workspace/CoSkill/.env:ro" \
  -v "$RUN_ROOT/models:/models:ro" \
  -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" "$TASK"

docker logs -f "$CONTAINER"
```

### Server B：leaf

```bash
export TASK='alfworld-leaf'
export RUN_ID='alfworld_leaf_001'
export CONTAINER="coskill-$RUN_ID"

docker run -d --name "$CONTAINER" \
  --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e TREE_RL_USE_ALL_8=1 \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -e RL_OUTPUT_DIR="/outputs/$RUN_ID" \
  -v "$RUN_ROOT/.env:/workspace/CoSkill/.env:ro" \
  -v "$RUN_ROOT/models:/models:ro" \
  -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" "$TASK"

docker logs -f "$CONTAINER"
```

八卡仍为 72 rollouts。由于全局 36 mini-batch 不能被 8 个 FSDP rank 整除，八卡
使用一个全局 72 mini-batch；二/四卡使用两个全局 36 mini-batch。因此八卡与二/四卡
数据和方法相同，但优化器 mini-batch 几何不同，学习曲线不应合并作严格对照。

## 6. 可选：使用 50 环境步

默认 ALFWorld `MAX_STEPS=40`。若 root 与 leaf 均需使用 50 步，在两条 `docker run`
中都加入：

```bash
-e MAX_STEPS=50
```

50 步会提高失败轨迹的最大生成 token 和运行时间，也改变环境终止上限。只可与同样
`MAX_STEPS=50` 的实验比较。

## 7. 运行、恢复和停止

查看日志：

```bash
docker logs -f "$CONTAINER"
tail -f "$RUN_ROOT/outputs/$RUN_ID/training.log"
```

启动日志应包含：

```text
[cloud-bootstrap] ... "status": "ok"
8-GPU all-in-one mode enabled
GRPO rollout: train_data_size=12 group_size=6 total=72
PPO geometry: global_mini=72
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

若容器已被删除，重新执行第 5 节对应的 `docker run` 命令；launcher 会从同一输出目录
恢复模型、优化器、dataloader 和 `skill_lib/skills_tree_rl_latest.json`。

不要删除输出目录。root 与 leaf 不能相互恢复。

## 8. 输出与指标

每个 `$RUN_ROOT/outputs/$RUN_ID` 包含：

| 路径 | 内容 |
| --- | --- |
| `run_config.env` | 实际解析后的 GPU、步数、batch 和恢复配置 |
| `ppo_args.txt` | 传给 Hydra/verl 的完整参数 |
| `training.log` | 训练过程日志 |
| `metrics.jsonl` | 每步训练、验证、token、云端和树指标 |
| `skill_lib/skills_tree_rl_latest.json` | 每步保存的技能树控制状态 |
| checkpoint 目录 | LoRA、优化器和 dataloader 恢复状态 |

主要指标字段：

| 字段 | 含义 |
| --- | --- |
| `episode/<task>_success_rate` | 当前训练 batch 的分任务成功率 |
| `val/<task>_success_rate` | 验证集分任务成功率 |
| `tokens/small_model/by_task_type/<task>/{prompt,response,total}` | 当前训练步的分任务小模型 token |
| `tokens/small_model/by_task_type/<task>/total_cumulative` | 分任务小模型累计 token |
| `tokens/large_model/by_task_type/<task>/total` | 可直接归因到该任务树演化的云端 token |
| `tokens/large_model/mixed/total` | 跨任务诊断/对比蒸馏 token；不均摊到单任务 |

## 9. 不改变实验语义的加速项

| 项目 | 用法 |
| --- | --- |
| 预下载模型 | 使用第 2 节的共享 `$RUN_ROOT/models` |
| 复用 parquet | 默认 `PREPARE_DATA=auto` |
| CUDA Graph、FlashAttention、chunked prefill | 默认开启 |
| 完成轨迹压缩 | `COMPACT_FINISHED_TRAJECTORIES=True` 默认开启 |
| root/leaf 跨服务器并行 | 使用独立八卡服务器与独立输出目录 |
| 减少验证开销 | 增大 `TEST_FREQ`；训练梯度不变，但中间验证点变少 |

不要修改 `TRAIN_DATA_SIZE`、`GROUP_SIZE`、72 rollout、`MAX_RESPONSE_LENGTH`、云端更新开关或
树证据预算来换取速度；这些会改变实验方法或降低对照公平性。
