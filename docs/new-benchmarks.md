# SearchQA 与 ScienceWorld 接入说明

两个环境已注册到原生 `make_envs`，使用本仓库原有多轮 rollout、GRPO、奖励与指标链路。新增的 `messages` 观察字段保留 system/user/assistant 角色；旧环境继续使用原来的单条 user 提示，ALFWorld、WebShop 和原 SearchR1 分支不变。

## 协议与方法边界

| 项目 | SearchQA | ScienceWorld |
|---|---|---|
| 环境名 | `searchqa_live` | `scienceworld` |
| 固定协议 | live-search-v4-search-answer | scienceworld-react-observation-v1 |
| 默认动作预算 | 15 decisions、8次搜索、每次5条 Tavily snippets | 100 decisions、无 simplifications |
| 可执行动作 | `search[...]` / `answer[...]`，没有 Open | 完整命令交给模拟器，不用动作示例白名单过滤 |
| 输入 | 问题一次、完整实际动作/观察历史 | reset/step 原始观察、完整实际动作/观察历史 |
| 终局奖励 | EM，0/1 | 成功达到100分，0/1 |
| 额外指标 | EM、token F1 | 原始score、归一化task_score |

这复用项目已确认的环境协议，不声称实时联网版是原始固定片段 SearchQA，也不声称完整复现 SwiftSage。不额外拼接 look/inventory/valid actions，不强制最近8组，不在历史中回放模型思考。参考答案只存环境侧，不能进入模型消息。格式恢复仅提取思考之外唯一完整 action；真实 sampled tokens 仍由原 rollout 保留。

`benchmark_protocol.py` 从3090上当前CoSkill的三个文件抽取纯协议函数；原文件快照SHA256见 `benchmark-protocol-source.json`。未导入CoSkill技能树、云端分析、学习闭环。SkillRL/Skill0各自继承本仓库 `SearchEnvironmentManager` 的技能记忆初始化，保留 `retrieval_memory`，因此原有更新/内化接口仍可访问。重置时读取当前记忆，评测步骤本身不新增技能。

没有自动生成小技能或借用其他领域技能。可以用 `SKILLS_PATH` 显式加载已确认的本领域技能库；不传时为空技能基线，不应将其直接报告成完整的SkillRL/Skill0方法实验。方法侧动态更新或内化日程需通过原有Hydra配置显式开启，环境接入不替代这些实验设计。

## 数据与运行

使用相同的SearchQA train/dev/test JSONL和ScienceWorld冻结manifest，不能重新随机抽另一套测试题。ScienceWorld每个实例在reset时再次检查是否属于声明的官方variation split；训练拒绝dev/test数据。SearchQA转换时检查ID和问题重复，三个集合两两不交叉。

```bash
python -m examples.data_preprocess.prepare_new_benchmarks --benchmark searchqa_live --train "$TRAIN_JSONL" --dev "$DEV_JSONL" --test "$TEST_JSONL" --output "$NEW_DATA_DIR"
python -m examples.data_preprocess.prepare_new_benchmarks --benchmark scienceworld --train "$SCIENCE_MANIFEST" --output "$NEW_DATA_DIR"

# 只检查真实环境，不调用模型。Search smoke消耗一次搜索API请求。
CUDA_VISIBLE_DEVICES= python -m scripts.smoke_new_benchmarks --benchmark scienceworld
CUDA_VISIBLE_DEVICES= python -m scripts.smoke_new_benchmarks --benchmark searchqa_live

# 设置BENCHMARK、MODEL_PATH(SFT checkpoint)、DATA_DIR、MAX_PROMPT_LENGTH、N_GPUS_PER_NODE。
# 可选SKILLS_PATH、EVAL_SPLIT。凭据通过环境变量提供，不能写入Git。
bash examples/grpo_trainer/run_new_benchmark.sh algorithm.adv_estimator=grpo <原方法和硬件配置>
```

开发验证使用dev，最终冻结模型后才指定 `EVAL_SPLIT=test`；禁止用最终test进行动态技能更新。启动器不重写GPU可见性，不自动选卡，不启动时修改系统参数。SFT学习率1e-4不等于RL学习率；此接入不改既有优化器设置。

环境历史完整保留，框架必须配置明确可承载的 `MAX_PROMPT_LENGTH`。超出直接报错，不静默截断。响应上限4096与现有项目协议一致。批次内已经结束的实例不再执行动作，也不重复发放奖励。当前manager串行驱动CPU环境，优先保障协议一致；大批次吞吐需另行测量，不声称完成多卡性能优化。

依赖：项目原有训练环境、ScienceWorld 1.2.3与Java、SearchQA使用Tavily凭据。没有安装或修改共享训练依赖。环境smoke、CPU回归与实际模型benchmark分数是不同证据；完整GRPO训练另行安排GPU，不能把接入测试当成方法结果。

## 2026-10-24 验证与部署

- CPU回归：`python -m pytest -q tests/agent_system/environments/test_new_benchmarks.py`，13 passed（两个仓库分别运行）。
- 真环境：ScienceWorld 1.2.3重置、官方test variation核对、look around执行和关闭均通过；SearchQA各一次真实Tavily搜索及固定答案提交通过。
- 原生rollout实际预处理检查保留消息角色，旧环境仍使用原single-user格式。
- 尚未运行完整GPU GRPO或模型benchmark评分；本次没有占用GPU。
- 3090部署使用独立检出目录 `coskill-workspace/skillrl-bench-integration` 和 `coskill-workspace/skill0-bench-integration`，保留归档仓库里的既有未提交修改。
- 同一份数据位于工作区 `datasets/searchqa-native-20261024`（100/100/100）和 `datasets/scienceworld-native-20261024`（120/117/120）。ScienceWorld直接复用原冻结manifest，dev数量由官方split可用样本决定，没有补造。
