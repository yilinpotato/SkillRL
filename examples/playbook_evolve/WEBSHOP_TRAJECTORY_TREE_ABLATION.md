# WebShop 轨迹树压缩消融

入口：

```bash
examples/playbook_evolve/run_webshop_trace_compression_one_step_ablation.sh
```

## 实验单位

- 环境：WebShop 1000-product 小数据集。
- 一次训练步（一个 rollout group）：从训练 split `[500, num_goals)` 固定采样
  12 个不同目标，每个目标生成 6 个副本，共 72 条轨迹。
- 每条轨迹最多 15 个环境动作；动作只能是 `search[...]` 或 `click[...]`。
- capture 阶段使用正常 CoSkill 技能提示，但显式关闭云端更新，因此只产生一份
  不会被任何实验臂修改的 `shared/raw_traces.jsonl`。
- `compression_on` 和 `compression_off` 必须使用同一份 raw trace SHA-256。

WebShop 的任务类别来自技能检索器，不保证每类数量相同。实验会记录每类的轨迹数、
满分成功率和平均 `task_score`，但不会把它伪装成 ALFWorld 的“每类 12 条”设计。

## 两个实验臂

- `compression_on`：loop filter、observation delta、trajectory prefix tree 和
  consensus prefix 全开；云端只接收自包含的 tree codec。
- `compression_off`：四项全关；云端接收逐条完整 observation 轨迹。

两臂各自执行一次相同的 CoSkill 云端更新链。由于没有更新后的第二轮 rollout，
成功率与 `task_score` 是共享 capture 的诊断值，在两臂中完全相同；本实验比较的是
轨迹证据大小、实际云端 prompt token、缓存命中/未命中 token 和费用。

## 启动

单卡（本地共享机只能使用 GPU1）：

```bash
export CUDA_VISIBLE_DEVICES=1
export WEBSHOP_DATA_DIR=/path/to/webshop/data
nohup bash examples/playbook_evolve/run_webshop_trace_compression_one_step_ablation.sh \
  --phase all > webshop_trace_ablation.log 2>&1 &
```

超算双卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DATA_PARALLEL_WORKERS=2
export WEBSHOP_DATA_DIR=/path/to/webshop/data
nohup bash examples/playbook_evolve/run_webshop_trace_compression_one_step_ablation.sh \
  --phase all > webshop_trace_ablation.log 2>&1 &
```

脚本默认自动令 `DATA_PARALLEL_WORKERS` 等于可见 GPU 数，但总 rollout 数仍为 72。
API key 可由项目 `.env` 或 `export DEEPSEEK_API_KEY=...` 提供。启动时会先执行云端
preflight，再加载 CUDA 和本地模型。

分阶段恢复使用同一个 `AB_ROOT`：

```bash
export AB_ROOT=/path/to/existing/run
bash examples/playbook_evolve/run_webshop_trace_compression_one_step_ablation.sh --phase capture
bash examples/playbook_evolve/run_webshop_trace_compression_one_step_ablation.sh --phase arms
bash examples/playbook_evolve/run_webshop_trace_compression_one_step_ablation.sh --phase report
```

已完成的 capture 或 arm 会由文件和配置检查后复用；配置变化必须换新目录。

## 主要输出

- `run_config.json`：完整协议、模型/并行设置和价格快照。
- `capture/capture_integrity.json`：12×6 采样完整性、动作/步数、分类成功率与
  `task_score`。
- `shared/raw_traces.jsonl`：两个臂共同使用的唯一原始轨迹。
- `arms/*/compressed_batch.json`：各臂本地压缩产物。
- `arms/*/cloud_io/call_audit.json`：每次云端调用的真实 provider usage。
- `compression_comparison.json`：证据、provider token 和费用差值。
- `token_waterfall.json` / `.csv`：Raw → loop filter → obs delta →
  prefix tree context → actual cloud prompt。
- `metrics.jsonl`、`cost_comparison.csv`、`summary.json`：统一汇总。

费用使用报告中固化的 DeepSeek V4 Flash 公开价格快照重算，不等同于查询账号账单。
