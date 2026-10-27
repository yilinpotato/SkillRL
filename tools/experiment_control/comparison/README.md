> 下方命令从 tools/experiment_control 运行。执行代码已经迁入 Co-Skill，数据与历史记录仍保存在 train-alfworld-data。

# 两组独立的三方法对照

正式代码位置：Skill0=`yilinpotato/Skill0:main`；SkillRL=`yilinpotato/SkillRL:main`；CoSkill=`yilinpotato/SkillRL:Co-Skill`。大规模实验转到用户后续提供的超算执行，详见[仓库与执行约定](https://github.com/yilinpotato/train-alfworld-data/blob/main/EXPERIMENT_WORKSPACE.md)。

用户于2026-10-24确认同时保留CoSkill no-RL和Tree-RL，分别比较。

| 对照组 | 方法 | 控制变量 |
|---|---|---|
| `norl_primary` | Skill0、SkillRL、CoSkill no-RL | 同一SFT起点、数据、初始技能来源、生成参数、环境协议、交互预算、dev/test；CoSkill参数冻结 |
| `all_rl` | Skill0、SkillRL、CoSkill Tree-RL | 上述设置，加公共RL优化器、LoRA、奖励、惩罚和更新预算 |

两个组独立保存运行目录、模型和结果，不混用checkpoint、不合并成一条主结果。Skill0/SkillRL若使用同一次运行作为两表共同参照，必须明确写出相同run ID，不得当成两个独立随机重复。

## 新的公共标准

机器可读配置为 [standard_v1.json](standard_v1.json)，这是新的受控实验设置，不是历史实验真实配置，也不是论文官方推荐超参数。

- 同一benchmark使用相同的具体SFT权重和tokenizer校验值。SFT学习率仍为1e-4；RL学习率单独统一为1e-6。
- RL：GRPO、LoRA32/64/all-linear、KL0.01、entropy0.001、invalid penalty0.1、无warmup、constant scheduler、mini-batch36、PPO epoch1。
- 训练温度1.0，dev/test温度0.4并采样；top-p1，输出4096 tokens。相同种子用于配对，不宣称不同推理后端能产生逐token相同结果。
- 每次12个任务×6条rollout，目标100次更新、7200条训练轨迹；dev32题，每360条训练轨迹验证，包含训练前验证。实际完成量和额外技能probe开销单独记录。
- ALFWorld涵盖相同的全部任务类别，关闭Skill0独有的dense reward。初始general skill预算6；Skill0内化日程还须检查实际首轮注入ID，不能只看top_k。
- 历史按benchmark：ALFWorld/WebShop保留原8组；SearchQA/ScienceWorld保留完整实际动作与观察历史。
- 上下文容量由实验显式指定，三方法使用相同容量；溢出报错，禁止静默截断。这个容量是模型/框架约束，不是benchmark原生规则。
- 训练和技能更新不得读取最终test反馈。无对应领域技能时不自动制造技能，也不能将空技能普通GRPO称为完整Skill0/SkillRL。

## 生成可审查的配置

```bash
python comparison/resolve.py --group all_rl --method skill0 --benchmark alfworld --prompt-tokens 16384 > skill0-plan.json
python comparison/resolve.py --group norl_primary --method coskill_norl --benchmark searchqa_live --prompt-tokens 16384 > coskill-norl-plan.json
python -m unittest discover -s comparison -p 'test_*.py'
```

`prompt-tokens=16384`只是调用示例，必须与部署的模型上下文和另外两种方法一致。生成文件包含公共设置、独立输出命名、Hydra覆盖参数和阻塞项。RL覆盖参数应追加在原方法启动参数之后；仍需要原来的模型、数据、硬件及方法专属参数。解析器不启动训练，不重写GPU可见性，也不改历史入口。

`--require-ready`仅检查当前已知的驱动缺口；即使返回0，也不表示数据/权重/技能及真实训练已经核验。运行前必须核对生成结果里的 `required_external_evidence`，并保存最终解析后的配置。

## 实际启动入口（2026-10-26）

三个方法仓库现在都提供 `examples/grpo_trainer/run_controlled_comparison.sh`，共同调用本仓库的 `comparison/launch.py`。Skill0/SkillRL 的包装脚本固定各自方法；CoSkill 用 `--method coskill_norl` 或 `--method coskill_tree_rl` 选择版本。历史脚本的默认行为不变。

```bash
export COMPARISON_ROOT=/path/to/SkillRL-CoSkill/tools/experiment_control
bash examples/grpo_trainer/run_controlled_comparison.sh   --group all_rl --benchmark searchqa_live   --model /path/to/merged/Search-SFT-v2   --skills /path/to/train-derived/skills.json   --train-data /path/to/train.parquet --dev-data /path/to/dev.parquet   --prompt-tokens 16384 --output-root /path/to/results --run-id seed0
```

上述命令准备配置和采样清单；加 `--execute` 后启动原生 `verl.trainer.main_ppo`。CoSkill no-RL 使用 `--group norl_primary --method coskill_norl`，并指定已部署的同一个 SFT 模型服务 `--base-url ... --served-model ...`；它调用新增的 `examples.playbook_evolve.run_controlled_norl`，不建立优化器、不自行启动 GPU 服务。必须确认该服务加载的权重与 `--model` 对应；模型名相同本身不能证明权重相同。

保存内容包括 `resolved_config.yaml`、`comparison.json`、1200 条任务的 `scheduled_train.parquet`、32 条固定验证任务的 `fixed_dev.parquet`、数据/技能/tokenizer 校验值和运行状态。每个任务展开六次，目标 7200 条训练轨迹。不同组与方法目录隔离，已有 run-id 拒绝覆盖。`--smoke-updates` 可缩小测试，产物明确标记 smoke，不能混入正式结果。

- SearchQA/ScienceWorld 原生 manager 读取每一行真实 `env_kwargs`，保留角色化完整历史。CoSkill 的批量文本快速路径也保留相同角色，不再压成单条 user 消息。
- ALFWorld 受控模式将共享 reset seed 传到 worker；WebShop 要求 parquet 明确给出 `env_kwargs.goal_index`，train 使用 >=500、dev 使用 <500 的原生分区。无参数调用仍保留原来的抽样行为。ALFWorld 的固定 seed 已用真实环境验证；还应保存实际 gamefile，以检查不同安装版本的数据是否一致。
- no-RL 每组完成后才接入原有 TracesPool/云端演化；每五组固定 dev、评测温度独立设置为 0.4。dev 使用训练技能库的副本，评测轨迹不加入学习池。记录 provider usage，缺失时保持 null，不伪造零 token。
- 新环境的 Tree-RL 使用原有 HierarchicalSkillLib 与课程状态机，按 SearchQA 或 ScienceWorld 官方任务类型归因，保留 probe/内化机制。额外 probe 是方法开销，须单独记录，不能把 7200 条主训练轨迹当成全部调用费用。
- 三个原生 PPO 入口的 Hydra 配置解析、各自新环境测试及 CoSkill no-RL 调度测试需通过后再启动训练。CPU 检查不等于 GPU 优化器/vLLM 全链路验证。

## 仍需要满足的实验前提

ScienceWorld 不在当前已发布的 Search/ALFWorld/WebShop 三套 SFT v2 中；启动器要求带领域来源记录的 SFT checkpoint，禁止默默换用底座或 Search 模型。Skill0 没有真实初始领域技能时不能进行内化；不自动编造小技能，只有显式 `--allow-empty-skill-ablation` 才允许空技能消融，并写入报告。

GPU 完整训练本轮未执行，既有 SFT 产物没有修改。上述代码接通与环境/调度测试不表示新的正式 benchmark 分数已经产生。与旧 DeepSeek 实验的差异见 [deepseek-baseline-audit-20261026.md](https://github.com/yilinpotato/train-alfworld-data/blob/main/comparison/deepseek-baseline-audit-20261026.md)。
