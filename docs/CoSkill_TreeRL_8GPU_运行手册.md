# CoSkill Tree-RL：八卡运行手册

适用镜像：

```text
crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data-20260721-fix9-dotenv-api-preflight-20260727
```

适用任务：`alfworld-root`、`alfworld-leaf`、`webshop-root`、`webshop-leaf`。

## 实验不变量

| 项目 | 正式默认值 |
| --- | --- |
| 训练任务数 | 12 |
| 每任务 GRPO 样本数 | 6 |
| 每训练步 rollout | 72 |
| ALFWorld 最大环境步数 | 40（可显式设为 50） |
| 验证任务数 | 32 |
| 总训练步数 | 100 |
| 云端检查 | 每次启动真实 API probe，失败立即退出 |
| 日志 | console + `metrics.jsonl`；W&B 默认关闭 |

八卡 all-in-one 模式使用 8 个 FSDP rank。72 条 rollout 不变，但 PPO 全局
mini-batch 为 72；二/四卡模式为两个 36 的 mini-batch。因此八卡与二/四卡的
**数据、rollout、reward、prompt 和树机制相同，但优化器 mini-batch 几何不同**。
学习曲线不要混在同一严格对照图中。

## 两台八卡服务器：root 与 leaf 并行

这是推荐的并行方式。两台服务器各自运行完整 8 卡实验：

| 服务器 | 容器任务 | 输出目录 |
| --- | --- | --- |
| Server A | `alfworld-root` | `/outputs/alfworld_root_8gpu` |
| Server B | `alfworld-leaf` | `/outputs/alfworld_leaf_8gpu` |

两者必须使用：同一镜像 tag、同一模型目录、同一数据版本、相同 `MAX_STEPS`、
`TOTAL_TRAINING_STEPS`、随机种子和其他 launcher 参数。两者必须使用不同
`RL_OUTPUT_DIR`；不要共享 checkpoint、`skill_lib/skills_tree_rl_latest.json` 或
输出目录。

### 方式 A：直接 export 云端配置

在各自服务器执行。不要同时挂载 `/workspace/CoSkill/.env`，否则挂载文件会覆盖
同名 export。

```bash
export IMAGE='crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data-20260721-fix9-dotenv-api-preflight-20260727'
export DEEPSEEK_API_KEY='replace-with-secret'
export DEEPSEEK_MODEL='deepseek-v4-flash'
export SKILL_UPDATER_BACKEND='deepseek'
export MODEL_HOST_DIR='/absolute/path/to/Qwen3-4B-Thinking-2507-parent'
export OUTPUT_HOST_DIR='/absolute/path/to/outputs'
mkdir -p "$OUTPUT_HOST_DIR"
```

Server A：

```bash
nohup docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e DEEPSEEK_API_KEY -e DEEPSEEK_MODEL -e SKILL_UPDATER_BACKEND \
  -e TREE_RL_USE_ALL_8=1 \
  -e RL_OUTPUT_DIR=/outputs/alfworld_root_8gpu \
  -v "$MODEL_HOST_DIR:/models:ro" \
  -v "$OUTPUT_HOST_DIR:/outputs" \
  "$IMAGE" alfworld-root \
  > "$OUTPUT_HOST_DIR/alfworld_root_8gpu.launch.log" 2>&1 &
```

Server B：

```bash
nohup docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -e DEEPSEEK_API_KEY -e DEEPSEEK_MODEL -e SKILL_UPDATER_BACKEND \
  -e TREE_RL_USE_ALL_8=1 \
  -e RL_OUTPUT_DIR=/outputs/alfworld_leaf_8gpu \
  -v "$MODEL_HOST_DIR:/models:ro" \
  -v "$OUTPUT_HOST_DIR:/outputs" \
  "$IMAGE" alfworld-leaf \
  > "$OUTPUT_HOST_DIR/alfworld_leaf_8gpu.launch.log" 2>&1 &
```

`--gpus` 中的编号必须替换为调度器实际分配给该容器的八张物理卡。容器内部会将
它们重编号为逻辑 `cuda:0..7`。

### 方式 B：挂载根目录 `.env`

`.env` 使用普通 shell dotenv 格式即可，例如：

```text
DEEPSEEK_API_KEY='replace-with-secret'
DEEPSEEK_MODEL='deepseek-v4-flash'
SKILL_UPDATER_BACKEND='deepseek'
```

在上面的命令中删除三个 `-e DEEPSEEK...` / `-e SKILL...`，并加入：

```bash
-v /absolute/path/to/.env:/workspace/CoSkill/.env:ro
```

容器读取优先级：

1. `COSKILL_CONTAINER_DOTENV` 指定的路径；
2. `/workspace/CoSkill/.env`；
3. `/run/secrets/coskill.env`。

容器入口使用 Python allowlist 解析这三个路径，不使用 Docker `--env-file`；读取后
不会再把该挂载 `.env` 作为 shell 文件二次解析。

## 手动设置 50 环境步

在 root 与 leaf 的两条命令中都加入：

```bash
-e MAX_STEPS=50
```

或在非容器启动前：

```bash
export MAX_STEPS=50
```

这会增加最长失败轨迹的生成时间和小模型 token，并改变任务终止上限。所有需要严格
比较的实验臂必须同时使用 50；不能与 40-step 历史结果直接合并。

## 非容器八卡启动

根目录已有 `.env` 时，launcher 会加载它。只用 export 且不希望本地 `.env` 覆盖时，
显式禁止 shell dotenv 加载：

```bash
cd /path/to/CoSkill
export COSKILL_ENV_FILE=/dev/null
export DEEPSEEK_API_KEY='replace-with-secret'
export DEEPSEEK_MODEL='deepseek-v4-flash'
export SKILL_UPDATER_BACKEND='deepseek'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_STEPS=50
export RL_OUTPUT_DIR="$PWD/outputs/alfworld_root_8gpu"
mkdir -p "$(dirname "$RL_OUTPUT_DIR")"
nohup bash examples/grpo_trainer/run_coskill_tree_rl_8xa800.sh \
  > "$RL_OUTPUT_DIR.launch.log" 2>&1 &
```

leaf 运行只需在同一命令前加入：

```bash
export TREE_RL_ORDER=leaf
```

并使用不同的 `RL_OUTPUT_DIR`。

## 单次启动的执行顺序

1. 加载 export 或挂载 dotenv。
2. 真实云端 API probe；失败时停止，尚未初始化 CUDA、Ray、vLLM 或模型。
3. 检查八张可见 GPU、模型、ALFWorld/WebShop 数据和 parquet 行数。
4. 恢复同一输出目录的模型/优化器 checkpoint 与 `skills_tree_rl_latest.json`；新实验使用新目录。
5. 每训练步生成 72 条多步环境轨迹。
6. 记录环境成功、动作有效性、逐任务小模型 token，并写入 trace pool。
7. 对轨迹进行 loop filter、obs delta、prefix-tree/consensus 编码；调用云端进行技能树、playbook 与失败分析更新。
8. 将更新后的技能树用于下一训练步的检索；当前已开始的 rollout 不会被中途改变。
9. 用 GRPO 更新 LoRA；tree controller 按 root 或 leaf 策略、训练成功率和独立 probe 决定层级内部化。
10. 按频率验证、写入 checkpoint、树状态和 JSONL metrics。

## 输出与检查

```bash
tail -f "$OUTPUT_HOST_DIR/alfworld_root_8gpu.launch.log"
tail -f "$OUTPUT_HOST_DIR/alfworld_root_8gpu/training.log"
```

启动日志应依次出现：真实 cloud probe `status: ok`、`8-GPU all-in-one mode enabled`、
`GRPO rollout: ... total=72`、`PPO geometry: global_mini=72`。

每个输出目录至少包含：

| 文件/目录 | 内容 |
| --- | --- |
| `run_config.env` | 本次实际解析后的环境、GPU、步数和 batch 参数 |
| `ppo_args.txt` | 传给 Hydra/verl 的完整参数 |
| `training.log` | 训练日志 |
| `metrics.jsonl` | 每步训练、验证、token、云端与树指标 |
| `skill_lib/skills_tree_rl_latest.json` | 每步保存的树控制状态，恢复时必须与 checkpoint 同目录 |
| checkpoint 目录 | LoRA/optimizer/dataloader 恢复状态 |

关键指标：

| 指标 | 含义 |
| --- | --- |
| `episode/<task>_success_rate` | 当前训练 batch 的任务成功率 |
| `val/<task>_success_rate` | 验证集任务成功率 |
| `tokens/small_model/by_task_type/<task>/{prompt,response,total}` | 本训练步每任务小模型 token |
| `tokens/small_model/by_task_type/<task>/total_cumulative` | 每任务小模型累计 token |
| `tokens/large_model/by_task_type/<task>/total` | 可直接归因到该任务树演化的云端 token |
| `tokens/large_model/mixed/total` | 跨任务诊断/对比蒸馏云端调用；不应均摊到单一任务 |

## 不改变方法的加速项

| 项目 | 当前状态/操作 | 是否改变实验语义 |
| --- | --- | --- |
| CUDA Graph、FlashAttention、chunked prefill | 已默认开启 | 否 |
| 完成轨迹从后续 vLLM batch 移除 | `COMPACT_FINISHED_TRAJECTORIES=True` 默认 | 否 |
| 预生成并复用固定 parquet | 保持 `PREPARE_DATA=auto` | 否 |
| 节点本地模型/数据缓存 | 将 `/models` 和数据放高速本地盘 | 否 |
| 两台服务器并行 root/leaf | 独立输出目录 | 否 |
| 降低验证频率 | 例如 `TEST_FREQ=10` | 不改训练梯度，但减少中间验证点 |

不要为了加速修改 `TRAIN_DATA_SIZE`、`GROUP_SIZE`、`MAX_RESPONSE_LENGTH`、云端更新开关、
树证据预算或 72 rollout；这些会改变方法或使比较失去公平性。
