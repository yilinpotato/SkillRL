# Repository Guidelines

## 角色与仓库边界

SkillRL 是论文复现基线：端侧模型使用检索到的技能并通过 GRPO/RL 更新，不应混入 CoSkill 的云端技能树、云端 API 分析或 no-RL 云端闭环。所有比较先对齐 ALFWorld/WebShop 数据 split、模型权重、最大 action 步数、response budget、温度、训练 rollout 数和验证频率；不要为了修复 OOM 或加速而静默改变这些实验变量。

## 目录与职责

- `agent_system/`：环境 manager、ALFWorld/WebShop projection、prompt、技能检索和多轮 rollout。环境解析逻辑必须与对应 prompt 协议一起改。
- `verl/`：PPO/GRPO、Ray worker、FSDP、reward、checkpoint 与通用指标。
- `examples/grpo_trainer/`：本仓库主训练入口。WebShop 使用 `run_webshop_skills.sh`；ALFWorld 的技能训练更新 LoRA 使用 `run_alfworld_skills_train_update_lora.sh`。
- `recipe/`：Hydra/训练配方；`memory_data/` 和 `skillrl_data/` 是数据与技能资产。
- `tests/`：按源目录镜像的 pytest；输出目录、Ray 临时文件、checkpoint、日志和 `__pycache__/` 不属于源码。

## 开发、测试与训练

```bash
conda activate skillRL
pip install -e .
python -m pytest -q tests/agent_system/environments/test_webshop_prompt_contract.py
python -m pytest -q tests/trainer/ppo/test_metric_utils.py
ruff check agent_system verl examples

bash examples/grpo_trainer/run_webshop_skills.sh
bash examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh
```

先跑与修改位置对应的 CPU pytest，再考虑完整训练。Ray、FSDP、vLLM 和真实 WebShop/ALFWorld 都是集成测试：记录所用 conda 环境、GPU 数、模型路径和数据根目录，不能把“脚本成功启动”当作训练正确性。恢复训练时使用 launcher 支持的 checkpoint 参数，并保留已有 `metrics.jsonl`、验证记录和 checkpoint；不要手动合并或清空指标文件。

## 协议、奖励与指标

WebShop 输出采用严格双段协议：恰好一个 `<think>...</think>` 后接一个 `<action>...</action>`，think 必须在 action 前。Qwen Thinking 的开 `<think>` 可能由 chat template 放在 prompt；处理这类兼容时必须保留真实 sampled tokens 给 PPO，并单独记录用于环境解析的完整 transcript。ALFWorld 动作必须在当前 admissible commands 中执行。

`is_action_valid` 会参与 invalid-action penalty；改动它之前必须查清 projection、环境 manager、`ray_trainer.py` 和轨迹 dump 的完整数据流。指标应保留旧字段，并稳定写入 `metrics.jsonl` 和 `group_metrics.jsonl`。比较 token 时使用相同定义：`tokens/small_model/{prompt,response,total}` 或 `perf/total_num_tokens` 的单步 active decision 口径，绝不能把累计流量、padding 或 FSDP 多次前反向吞吐混入。

## 代码、测试与审查

Python 使用四空格、`snake_case` 函数/变量、`PascalCase` 类；为新公开接口添加类型标注。遵循 `pyproject.toml` 的 Ruff 配置（300 字符行宽）。新增 parser、validity、reward、resume 或指标功能时，增加 `test_<behavior>.py` 与 `test_<expected_outcome>()`，覆盖合法、缺标签、截断及错误动作输入。

提交标题使用简短祈使式，如 `Fix WebShop ...`、`Align ALFWorld metrics ...`。每个提交只包含一个可复现变更及测试。PR/交接必须列出入口、环境、训练超参数是否改变、验证结果、GPU 资源、checkpoint 恢复方式和新增字段。不得提交 API key、个人绝对路径、模型缓存、checkpoint、outputs、轨迹或无关 dirty 文件。

## 共享 GPU 规则

本地共享 3090 只能在 GPU 0 空闲时使用，并先检查 `nvidia-smi`。超算使用调度器分配的 2/4 张 A800；不要让多个 Ray/vLLM 副本落在同一物理卡，也不要在子进程中改写父进程继承的 `CUDA_VISIBLE_DEVICES`。
