# CoSkill Tree-RL and Ablation Docker

该镜像固化当前 `skillRL` Conda 环境、CoSkill 代码、ALFWorld 文本环境、固定
`train=12/test=32` GRPO parquet，以及 WebShop 1000 商品数据和对应 Lucene 索引。
Tree-RL 的四个任务共用同一镜像：`alfworld-root`、`alfworld-leaf`、
`webshop-root`、`webshop-leaf`。每个任务独占
2、4 或 8 张 GPU；单个实验使用 DP=2/4、TP=PP=1。干净的单张 80 GiB A800/A100
也可通过 `TREE_RL_ALLOW_SINGLE_GPU=1` 正式运行。rollout 始终为 `12×6=72`，
验证集、采样和全局 PPO 几何不会随卡数变化。

同一镜像也提供 `alfworld-ablation`。它运行固定任务、固定轨迹的 ALFWorld
表示/压缩消融，仍使用独立的评估协议，**不会**混入或改变四个 Tree-RL 任务。

冻结模型的端云协同基线也可以直接运行：`alfworld-norl`、`webshop-norl`。它们
可以使用 1、2、4 或 8 张容器可见 GPU；全局 rollout 仍为 72，GPU 数只改变数据并行
分片。Tree-RL 的 8 卡规则不同：每个实验仍只使用一个 4 卡 slot。

若只需确认新服务器的 GPU、Ray、vLLM、FSDP、数据和技能树链路能完整贯通，可用
`alfworld-smoke` 或 `webshop-smoke`。它**只允许一张 GPU**，只跑一个极小更新，并启用
CPU offload；输出固定在 `/outputs/smoke/`，绝不能作为实验结果或恢复正式训练。

基础镜像默认使用 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`。当前构建只安装
已经打包好的 Python/CUDA wheel，不调用 `nvcc`，因此无需下载体积很大的 CUDA
开发工具层。若后续加入必须现场编译的 CUDA 扩展，可临时恢复 devel 基础镜像：

```bash
CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 \
  bash docker/coskill/build_image.sh
```

构建阶段默认使用阿里云 Ubuntu apt 镜像，并移除当前构建不需要的 NVIDIA apt
开发仓库；这只影响 `bash/git/build-essential` 等通用系统包的下载来源，不改变
CUDA runtime。可通过 `UBUNTU_APT_MIRROR` 切换为其他 Ubuntu 镜像。构建默认使用
`DOCKER_BUILD_NETWORK=host`，避免 rootless 默认网络无法直连软件源；该选项仅作用
于镜像构建，不改变训练容器的运行网络或实验结果。

## 构建

```bash
conda activate skillRL
cd /path/to/CoSkill
IMAGE_TAG=coskill:skillrl-cu128-data \
  bash docker/coskill/build_image.sh
```

若当前账号没有 `/var/run/docker.sock` 权限，可由管理员加入 `docker` 组；临时使用
sudo 时设置 `DOCKER_USE_SUDO=1`。构建需要较大磁盘空间，因为会打包约 14GB 的
真实 Conda 环境和约 2.1GB 的 ALFWorld 文本数据。

默认不把 7.6GB 模型权重写入镜像；启动时会从 ModelScope 下载到挂载的
`/models`。完全离线镜像可在构建时设置：

```bash
INCLUDE_MODEL=1 MODEL_SOURCE=/path/to/Qwen3-4B-Thinking-2507 \
  bash docker/coskill/build_image.sh
```

生成的 `docker/coskill/assets/` 很大且已被 Git 忽略。镜像不会复制 `.env`、
API key、训练输出或历史 checkpoint。

构建时会额外保存当前 `skillRL` 的 Faiss 运行时覆盖包。它修复 `conda-pack` 在
同时存在旧 conda Faiss 与新版 `faiss-cpu` 时可能恢复错误 Python wrapper 的问题；
不改变默认的 template 检索策略，也不改变训练样本或超参数。

## 与固定轨迹消融镜像共用层

固定轨迹 ALFWorld 消融保持自己的入口和评估协议，不能与主训练逻辑合并；但它和
主实验使用同一份 `skillRL` 环境及 ALFWorld 数据。消融基础镜像已推到 ACR 后，
可用下面的薄主镜像复用其约 7GB 压缩环境层，避免两个标签重复上传。此做法只改变
容器分层和上传时间，不改变四个 Tree-RL 任务的代码、采样、rollout=72 或超参数。

```bash
BASE_IMAGE=crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:alfworld-fixed-ablation-skillrl-cu128 \
IMAGE_TAG=crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data \
  bash docker/coskill/build_from_ablation.sh

docker push crpi-6gyywp4rhk17pb91.cn-guangzhou.personal.cr.aliyuncs.com/yilinpotato/coskill:skillrl-cu128-data
```

薄镜像仍包含当前 CoSkill 代码、WebShop 1000 商品数据和固定 GRPO parquet；父镜像
包含已验证的 Conda 环境和 ALFWorld 数据。保留独立的 `coskill:skillrl-cu128-data`
完整镜像，它适合没有消融基础镜像的离线节点。

## 预检与运行

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  --env-file .env \
  -v /path/to/models:/models \
  -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data preflight

docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  --env-file .env \
  -v /path/to/models:/models \
  -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data alfworld-leaf
```

将最后一个参数替换为其他三个任务即可。四个任务不能在同一组 4 卡上并发；
应串行运行，或为每个容器申请独立的 4 卡节点。相同输出目录会自动恢复模型、
优化器、dataloader 和 `skills_tree_rl_latest.json`。新实验请通过
`RL_OUTPUT_DIR=/outputs/<unique-name>` 使用独立目录。

单张 80 GiB A800/A100 的正式 Tree-RL 也保持 `12×6=72` rollout、`val=32`、
response=4096 和原奖励/采样配置；它不是 smoke。必须显式确认，避免把共享或小显存
卡误当正式资源：

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  --env-file .env -e TREE_RL_ALLOW_SINGLE_GPU=1 \
  -e RL_OUTPUT_DIR=/outputs/alfworld_leaf_1xa800 \
  -v /path/to/models:/models -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data alfworld-leaf
```

单卡的 FSDP micro-batch/gradient accumulation 会随 rank 数自动调整，但全局 PPO
mini-batch、72 条 rollout 和训练语义不变；它只会比多卡慢。24/40 GiB 卡只能使用
`alfworld-smoke` / `webshop-smoke`，不可将 smoke 的 `2×2=4` rollout 用于正式比较。

例如运行 ALFWorld noRL（8 卡时会启动 8 个 TP=1 的 vLLM 数据并行 worker）：

```bash
docker run --rm --gpus all --ipc=host \
  --env-file .env -e MODEL_AUTO_DOWNLOAD=0 \
  -e MAX_EPISODES=7200 -e BATCH_ROLLOUT_SIZE=72 \
  -v /path/to/models:/models -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data alfworld-norl
```

单卡 Tree-RL 链路测试（建议先用 ALFWorld；容器内逻辑 GPU 0 对应宿主机指定的 GPU）：

```bash
docker run --rm --gpus '"device=1"' --ipc=host \
  --env-file .env -e MODEL_AUTO_DOWNLOAD=0 -e CLOUD_BOOTSTRAP_PROBE=1 \
  -e MODEL_PATH=/models/Qwen3-0.6B \
  -v /path/to/models:/models:ro -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data alfworld-smoke
```

它使用 `2 goals × 2 samples = 4` rollout、最多 4 环境步、512 response token 和一个
GRPO 更新；正式入口的 `12×6=72` rollout 与超参数完全不受此命令影响。24 GiB 显卡必须
使用 Qwen3-0.6B（先下载到挂载目录）；4B FSDP actor 与独立 vLLM 引擎不能在这类单卡上
共存。40 GiB 及以上显存可尝试 4B smoke，正式 4B 训练仍要求 2/4/8 卡。

若希望预检真实调用一次云端 API，可加 `-e CLOUD_BOOTSTRAP_PROBE=1`；密钥始终
通过 `--env-file` 或容器 secret 注入，不要写入镜像。

## vLLM CUDA Graph 与 FlashInfer sampler

Tree-RL 以及超算/容器中的 noRL rollout 默认允许 CUDA Graph；这是 vLLM 的
`enforce_eager=False` 路径，不改变 prompt、奖励、rollout=72 或 token 预算。首次
启动日志应出现 `Capturing CUDA graphs` 和 `Graph capturing finished`。若短 smoke 因
显存或驱动组合失败，可显式传 `-e VLLM_ENFORCE_EAGER=1`（仅 noRL）回退 eager。

镜像保留原生 vLLM sampler 作为默认，以免已在跑的随机训练曲线被不同 CUDA sampler
悄悄改变。若需要单独测试 FlashInfer 的 decode 吞吐，先把它安装到输出挂载中的
隔离 overlay（不会修改镜像或 `/opt/conda/envs/skillRL`）：

```bash
docker run --rm --ipc=host \
  -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data install-flashinfer
```

随后为正式容器同时增加如下环境变量：

```bash
-e COSKILL_ENABLE_FLASHINFER_SAMPLER=1 \
-e COSKILL_FLASHINFER_OVERLAY=/outputs/flashinfer-cu128
```

只有日志出现 `Using FlashInfer for top-p & top-k sampling.` 才算真正启用。每一种
GPU 架构第一次使用会编译并缓存 kernel，首次启动较慢正常；应在缓存后比较稳态速度。
不要把启用与未启用 sampler 的随机训练曲线作为严格可重复的同一条曲线比较，也不要
设置 `VLLM_ATTENTION_BACKEND=FLASHINFER`：这里优化的是 sampler，稠密 Qwen 注意力
仍使用 vLLM 默认 Flash Attention。

单卡（例如本地 5070）可做镜像、模型、数据和 CUDA 预检；若挂载 Qwen3-0.6B，也可运行
上述不可报告的端到端 smoke。4B 则不能在 24 GiB 单卡上运行 Tree-RL：

```bash
docker run --rm --gpus all --ipc=host \
  -e PREFLIGHT_ALLOW_SINGLE_GPU=1 \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -v /path/to/models:/models \
  -v /path/to/outputs:/outputs \
  coskill:skillrl-cu128-data preflight
```

固定轨迹消融使用同一镜像，但明确使用自己的入口和输出根目录：

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  --env-file .env -v /path/to/models:/models -v /path/to/outputs:/outputs \
  -e MODEL_AUTO_DOWNLOAD=0 \
  -e AB_ROOT=/outputs/alfworld_fixed_trajectory_ablation/run_001 \
  coskill:skillrl-cu128-data alfworld-ablation --phase all
```

`alfworld-ablation` 不使用 GRPO 的 72 rollout/step 配置；它的 bootstrap 与每个
评估臂固定为各 36 条轨迹，详见 `ablation_summary.json`。

容器只分配一张 A800 时同样可运行该入口：把 `--gpus '"device=0"'`（或调度器分配的
单卡）替代四卡列表即可。它会使用 DP=1、TP=1，仍保持 bootstrap=36 与每臂评估=36；
只降低速度，不改变固定任务、采样 seed 或统计口径。

如果节点分配了 8 张卡，不要把单个实验改成 8-rank（全局 mini-batch 36 无法
无损分到 8 rank）。镜像会按 `TREE_RL_GPU_SLOT=0|1` 切成两个互不重叠的 4 卡组，
可同时跑两个不同任务：

```bash
# 容器 A 使用前四卡
docker run ... -e TREE_RL_GPU_SLOT=0 coskill:skillrl-cu128-data alfworld-root
# 容器 B 使用后四卡
docker run ... -e TREE_RL_GPU_SLOT=1 coskill:skillrl-cu128-data webshop-root
```

启动器读取实际显存：80/96GB 的 A800/A100/H20 使用高吞吐 vLLM 配置；40GB
A100 自动降低 KV-cache 比例，并在两卡时采用更小的等价 micro-batch 以避免 OOM。
这些调整不改变 rollout 数、样本、响应预算、奖励或全局 mini-batch。
