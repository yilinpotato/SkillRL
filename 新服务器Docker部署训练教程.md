# CoSkill 新服务器 Docker 部署与训练



## 1. 前置检查


```bash
docker version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi -L
```



## 2. 下载镜像与模型

```bash
export REGISTRY=crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com
# 固定使用已验证的版本；不要填无 tag 的仓库地址（它会请求不存在的 latest）。
export IMAGE=$REGISTRY/yilinpotato/coskill:skillrl-cu128-data-20260721-fix5
export RUN_ROOT=$HOME/coskill-run
mkdir -p "$RUN_ROOT/models" "$RUN_ROOT/outputs"

docker login --username=yilinpotato "$REGISTRY"
docker pull "$IMAGE"
```

第一次只下载模型（可中断后重复执行）：

```bash
docker run --rm -v "$RUN_ROOT/models:/models" "$IMAGE" shell -lc '
modelscope download --model Qwen/Qwen3-4B-Thinking-2507 \
  --local_dir /models/Qwen3-4B-Thinking-2507 --max-workers 8
test -f /models/Qwen3-4B-Thinking-2507/config.json
compgen -G "/models/Qwen3-4B-Thinking-2507/*.safetensors" >/dev/null
'
```

在 `$RUN_ROOT/.env` 写入deepseek密钥：

```bash
DEEPSEEK_API_KEY='your-key'
DEEPSEEK_MODEL='deepseek-v4-flash'
```



## 3. 预检

4 卡服务器：

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  --env-file "$RUN_ROOT/.env" -e MODEL_AUTO_DOWNLOAD=0 \
  -v "$RUN_ROOT/models:/models" -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" preflight
```

## 4. 常见启动错误

### `Failed to find C compiler` / `torch._inductor.exc.InductorError`

这是 vLLM 初始化时，Triton 要即时编译 GPU 内核，但旧镜像没有 `gcc/g++/make`。
请确认平台导入的镜像是本教程的 **`skillrl-cu128-data-20260721-fix5`**，而不是旧的
`fix1`/`fix2`/`fix3`/`fix4` 或缓存中的默认标签。`fix5` 已在镜像中安装并自检了编译工具链；不要在训练容器
中临时 `apt install`，这样会破坏可复现性。

若平台只能通过图形界面导入镜像，镜像 URL 填完整的
`crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data-20260721-fix5`，
内部镜像名可填 `coskill:rlfix5`。导入后先运行上一节 `preflight`；通过后才启动训练。

### `missing required asset: .../skillrl_data/verl-agent/text/train.parquet`

这是旧薄镜像遗漏固定的 GRPO parquet 所致。`fix5` 将 `train=12`、`test=32` 的
parquet 打包在 `/opt/data/verl-agent`，并在预检中验证行数。不要通过挂载旧宿主机
仓库来绕过；重新导入 `fix5` 后运行 `preflight` 即可。

### 容器内缺少源码或云端 key

`/workspace/CoSkill/examples/playbook_evolve` 已包含在镜像中。不要把宿主机旧仓库挂载到
`/workspace/CoSkill`，只挂载 `/models` 与 `/outputs`。DeepSeek key 不会打进镜像，必须用
`--env-file "$RUN_ROOT/.env"` 在运行时注入。

## 5. 四个 Tree-RL 实验：四台服务器并行

四个实验是彼此独立的对照组，应使用相同镜像、模型版本、72 rollout/step 和云端模型，
但不共用训练输出或技能树状态：

| 服务器 | `TASK` | `RUN_ID` |
|---|---|---|
| A | `alfworld-root` | `alfworld_root_001` |
| B | `alfworld-leaf` | `alfworld_leaf_001` |
| C | `webshop-root` | `webshop_root_001` |
| D | `webshop-leaf` | `webshop_leaf_001` |

在每台服务器执行同一份命令，只修改前三行。例如服务器 B：

```bash
export TASK=alfworld-leaf
export RUN_ID=alfworld_leaf_001
export CONTAINER=coskill-$RUN_ID

docker run -d --name "$CONTAINER" --gpus '"device=0,1,2,3"' --ipc=host \
  --env-file "$RUN_ROOT/.env" -e MODEL_AUTO_DOWNLOAD=0 \
  -e RL_OUTPUT_DIR=/outputs/$RUN_ID \
  -v "$RUN_ROOT/models:/models:ro" \
  -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" "$TASK"

docker logs -f "$CONTAINER"
```

两卡机器改为 `--gpus '"device=0,1"'`；每个 Tree-RL 实验均可使用 2 或 4 卡，
启动器会保持同一全局训练几何和 `12×6=72` 条 rollout。不要在同一组 GPU 上再启动
第二个 Tree-RL 容器。8 卡单机可以用 `TREE_RL_GPU_SLOT=0`、`1` 划分成两组四卡，
但四台独立服务器更容易保证日志和资源隔离。

所有服务器可以使用同一镜像和相同模型权重；模型目录只读挂载即可。每台服务器必须有
不同的 `$RUN_ROOT/outputs/$RUN_ID`。云端 key 可共用，但四个任务同时触发技能演化时
可能受 API 限流影响；最好为每台服务器配置独立 key，或至少监控 `cloud-bootstrap` 和
API 429/限流日志。

中断后在**同一台服务器**用相同的 `TASK`、`RUN_ID` 和输出挂载重启，会恢复 checkpoint、
数据加载器和 `skills_tree_rl_latest.json`。查看状态用 `docker logs -f "$CONTAINER"`；
停止用 `docker stop "$CONTAINER"`，不要删除输出目录。

## 6. 运行固定轨迹消融

它与 RL 训练独立，建议使用另一个输出目录：

```bash
docker run -d --name coskill-ablation --gpus '"device=0,1,2,3"' --ipc=host \
  --env-file "$RUN_ROOT/.env" -e MODEL_AUTO_DOWNLOAD=0 \
  -e AB_ROOT=/outputs/alfworld_ablation_001 \
  -v "$RUN_ROOT/models:/models" -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" alfworld-ablation --phase all
```

结果在 `$RUN_ROOT/outputs/alfworld_ablation_001/ablation_summary.json` 和 CSV。
消融可按 `--phase manifests|bootstrap|artifacts|evaluate|all` 续跑；固定 manifest 和
冻结轨迹会被保留，不能拿 RL 训练目录复用。

## 7. 运行 ALFWorld noRL

noRL 使用同一镜像，但入口是 `alfworld-norl`。它可使用容器分配的全部 GPU，保持
全局 `72 rollout/group` 与 `7200 episode = 100 group`；4 卡为每卡 18 条、8 卡为每卡
9 条，均使用 DP、TP=1：

```bash
docker run -d --name coskill-alfworld-norl --gpus all --ipc=host \
  --env-file "$RUN_ROOT/.env" -e MODEL_AUTO_DOWNLOAD=0 \
  -e MAX_EPISODES=7200 -e BATCH_ROLLOUT_SIZE=72 \
  -v "$RUN_ROOT/models:/models:ro" -v "$RUN_ROOT/outputs:/outputs" \
  "$IMAGE" alfworld-norl
```
