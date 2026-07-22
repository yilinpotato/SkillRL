# Repository Guidelines

## 角色与仓库边界

CoSkill 是三组实验中的“云端经验分析 + 端侧技能树/轨迹压缩”方案。它既能运行冻结模型的 no-RL 闭环，也能通过 Ray/GRPO 运行逐层技能内化；不要把它写成 SkillRL 或 Skill0 的复制品。对比实验必须固定模型、数据 split、任务数、最大步数、采样温度、验证集和 rollout 数，并在提交说明中明确本次改动影响的是 ALFWorld、WebShop、no-RL 还是 Tree-RL。

## 目录与职责

- `agent_system/`：环境适配、prompt、技能记忆、云端分析和多轮 rollout。WebShop 与 ALFWorld 的假设不可互相迁移。
- `verl/`：通用 PPO/GRPO、Ray、FSDP 和指标代码。只有通用行为才放这里。
- `examples/playbook_evolve/`：CoSkill 主入口；`run_*_playbook_evolve_norl.sh` 为冻结 no-RL 流程，传入 `rl=1` 才委托 Tree-RL。
- `examples/grpo_trainer/`、`recipe/`：GRPO 配置及其他基线入口。
- `memory_data/`、`skillrl_data/`：技能初始文件和数据资产；不得把成功轨迹、oracle 或验证答案写回种子文件。
- `tests/`：按源目录镜像放回归测试，例如 `tests/agent_system/environments/`。`outputs/`、checkpoint、日志、`__pycache__/` 都是生成物。

## 常用开发、测试与运行命令

```bash
conda activate skillRL
pip install -e .
python -m pytest -q tests/agent_system/environments/test_webshop_prompt_contract.py
python -m pytest -q tests/agent_system/trainer/test_token_traffic_metrics.py
ruff check agent_system verl examples

# 主实验入口（默认 no-RL）
bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh
bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh
# 同一入口切换 Tree-RL
rl=1 bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh
```

先运行与修改点最接近的 pytest；parser、metrics、resume 或 prompt 改动必须新增小型确定性回归测试。GPU/Ray/vLLM 测试是集成测试，只有依赖、数据和显存齐全时才运行。`pyproject.toml` 配置 Ruff，Python 使用四空格、`snake_case` 函数/变量、`PascalCase` 类；新公共函数加类型标注。

## 协议、指标与可比性

WebShop 输出必须遵守 `<think>...</think><action>...</action>`；不要只改 observation 展示而不同时检查 prompt、projection、环境动作与轨迹记录。ALFWorld 的可执行动作必须来自当前 admissible commands；CoSkill 的 `valid_action` 是奖惩口径，`strict_valid_action` 与 `non_strict_valid_action` 是诊断口径。WebShop 同时保留严格和非严格比率，`relaxed_*` 是历史兼容别名。

新增指标时保留旧字段，并写入主 `metrics.jsonl`/`group_metrics.jsonl`，使用既有命名空间：`episode/...`、`validation/...`、`tokens/...`、`coskill/...`。小模型 token 是实际 prompt 加生成 token；不要把 `perf/total_num_tokens`、累计 token、云端 API usage 混为同一口径。恢复运行前先确认已有 `metrics.jsonl`、`summary_partial.json` 和技能库 checkpoint，不能通过删除旧输出“解决”对齐问题。

## GPU、数据与安全

本地共享 3090 只可使用空闲的物理 GPU 1；先检查 `nvidia-smi`。超算按 launcher/调度器分配使用 2 或 4 张 A800，不要在 worker 内重写继承的 `CUDA_VISIBLE_DEVICES`。模型、数据和 API key 一律通过环境变量或 launcher 参数提供；不得提交 token、绝对个人路径、checkpoint、生成轨迹、原始输出或无关 dirty 文件。

## 提交与审查

近期提交使用简短祈使式标题：`Fix ...`、`Add ...`、`Align ...`。一个提交只处理一个可验证行为，附上测试。PR/实验交接应说明：影响环境与入口、是否改变训练结果、数据/rollout/验证设置、指标 schema 变化、精确测试命令、GPU 前提和恢复方式；涉及 prompt 或技能树时同时说明没有注入 oracle/手写成功答案。
