# 单卡 Tree-RL 容器测试报告

测试日期：2026-07-21。测试目标是验证容器能否在不改变正式实验配置的前提下，完整走通
CoSkill Tree-RL 的部署链路。此报告中的 smoke 结果不能作为模型效果或 token 效率实验结果。

## 测试环境

- Docker：rootful Docker，数据根目录位于 `/data2/docker`；未改动公共 daemon 配置。
- GPU：宿主机 NVIDIA RTX 3090 24 GiB 的 **1 号卡**，容器中映射为 `cuda:0`。宿主机 0 号卡全程未使用。
- 运行镜像：`skillrl-cu128-data-20260721-fix7`，digest
  `sha256:ac53b719ad6ce1398f2412e70e2d90467822b4f7530f12dae084c8a7f53f277e`。
- smoke 模型：ModelScope 的 `Qwen/Qwen3-0.6B`，挂载到 `/models/Qwen3-0.6B`。
- 云端：本次设置 `CLOUD_BOOTSTRAP_CHECK=0`、`CLOUD_BOOTSTRAP_PROBE=0`，因此没有请求
  DeepSeek，也没有验证真实云端凭证；镜像本身不会携带 API key。

## 结果

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 镜像拉取与 rootful GPU 映射 | 通过 | `docker run --gpus 'device=1' ... nvidia-smi -L` 只显示宿主机 1 号 RTX 3090。 |
| 数据和入口预检 | 通过 | ALFWorld、WebShop 小数据/Lucene、12/32 prepared parquet、所有声明入口和输出目录均通过。 |
| Ray + ALFWorld 初始化 | 通过 | 8,810 个 ALFWorld game 索引完成；配置校验通过。 |
| FSDP actor + LoRA | 通过 | 0.6B actor（601.10M 参数）完成初始化。 |
| vLLM | 通过 | safetensors 加载、CUDA graph capture、动作生成均完成。 |
| CoSkill 轨迹/技能状态 | 通过 | 4 条 raw trace、`skills_tree_rl_latest.json` 和 15 条 debug artifact 已写出。 |
| 一次 GRPO 更新与 checkpoint | 通过 | 容器以 exit code 0 退出，`global_step_1` 中含 actor、optimizer、extra state 与 LoRA adapter。 |
| 可比较训练指标 | 不适用 | 该 smoke 使用 0.6B、`2×2=4` rollout、4 环境步、512 response token。 |

输出根目录：`/home/myl/coskill-smoke/outputs/smoke/coskill_tree_rl_smoke_alfworld_06b_20260721/`。

## 关键运行数据

`metrics.jsonl` 只有一条 step 1 记录：

- `perf/time_per_step=414.401s`，其中测试阶段 349.548s；这是小模型首次部署验证，包含环境/验证开销。
- `tokens/small_model/total=26477`，与 `perf/total_num_tokens=26477` 一致；云端 token 为 0。
- `episode/success_rate=0`，`episode/valid_action_ratio=0.9375`，严格 valid rate 为 0.375。
- `coskill/pool/total_added=4`；本轮不足 `playbook_evolve_min_samples=6`，所以没有触发云端树进化，
  `coskill/tree_rl/enabled=0` 是预期现象，不是失败。

0.6B 并非正式模型，且有不少动作格式/质量不足，零成功率不能解释正式 4B 的表现。

## 发现与修复

1. 初始 4B 单卡 smoke 确实失败：FSDP actor 初始化后仅余 6.59 GiB，而 vLLM 的 0.45 利用率
   需要 10.6 GiB。问题是同卡共存的显存边界，不是数据、Ray 或 prompt bug。
2. 单卡诊断脚本现在会在 24 GiB + 4B 的组合上提前给出可读错误，建议挂载 0.6B；40 GiB 以上
   才可尝试 4B smoke。正式 4B Tree-RL 的 2/4/8 卡要求保持不变。
3. 初次容器 smoke 错把运行 batch 的 `2/2` 当成镜像 prepared parquet 的行数，已改为显式复用
   固定 `12/32` prepared corpus；该问题已在 fix7 中修复。
4. 文档中此前“单卡 smoke 可用”与“单卡只能预检”相互矛盾，现已统一为：24 GiB 单卡可用 0.6B
   做完整链路诊断，不能用 4B 做完整 Tree-RL。

## 复现命令

先下载模型一次：

```bash
sudo docker run --rm --entrypoint bash \
  -v "$HOME/coskill-smoke/models/Qwen3-0.6B:/models/Qwen3-0.6B" "$IMAGE" \
  -lc 'modelscope download --model Qwen/Qwen3-0.6B --local_dir /models/Qwen3-0.6B --max-workers 4'
```

再运行 smoke（将 `IMAGE` 替换为当前文档推荐 tag）：

```bash
sudo docker run --rm --gpus '"device=1"' --ipc=host \
  --env-file "$HOME/coskill-smoke/.env" \
  -e MODEL_AUTO_DOWNLOAD=0 -e MODEL_PATH=/models/Qwen3-0.6B \
  -v "$HOME/coskill-smoke/models/Qwen3-0.6B:/models/Qwen3-0.6B:ro" \
  -v "$HOME/coskill-smoke/outputs:/outputs" \
  "$IMAGE" alfworld-smoke
```

要验证云端，只在确认 `.env` 内已有有效 `DEEPSEEK_API_KEY` 后额外设置
`-e CLOUD_BOOTSTRAP_PROBE=1`；不要将 key 写进镜像或日志。
