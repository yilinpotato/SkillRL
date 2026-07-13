# CoSkill：ALFWorld 与 WebShop 运行及实验指南

本文档说明 CoSkill 冻结端侧模型、端云协同技能演化实验的实际使用方式：环境准备、启动命令、全部公开运行参数、功能开关、输出、恢复、评测边界和排错。它面向实验运行，不展开具体源码设计；模块架构请查看 项目架构.md、co-skill design.md 和 docs/。

## 1. 两条主链路

| 数据集 | 启动脚本 | Python driver | 环境动作 |
| --- | --- | --- | --- |
| ALFWorld | examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh | examples.playbook_evolve.run_playbook_evolve | 家庭文本操作，如 go to、take、heat |
| WebShop | examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh | examples.playbook_evolve.run_webshop_evolve | search[...]、click[...] |

两条链路均为 no-RL / frozen-model 实验：端侧 Qwen 只负责 rollout，不执行 LoRA、PPO、GRPO 或权重更新。学习状态保存在外部 skill_lib；下一 rollout group 读取新快照。

~~~text
端侧模型 rollout
  -> TracesPool 压缩成功/失败轨迹
  -> 满足触发条件时调用云端
  -> 新 bullet skill 与按任务族 skill tree 写入 skill_lib
  -> 下一 group 使用最新技能库
~~~

云端的 successful rollout reference 仅指本次运行中已由环境判定成功的历史 rollout；不是 oracle demonstration、ground-truth action plan，不读取 PDDL、游戏文件、专家计划或隐藏环境状态。

## 2. 前置条件

### 2.1 Python 与依赖

推荐统一环境：

~~~bash
conda activate skillRL
cd /data2/myl/CoSkill
pip install -e .
~~~

需要可用的 PyTorch/CUDA、vLLM、openai Python 包及 ALFWorld/WebShop 依赖。根目录 README.md 的 Installation 和 Environment Setup 章节提供基础安装步骤。

启动脚本会设定 HF_DATASETS_OFFLINE、TRANSFORMERS_OFFLINE、HF_HUB_OFFLINE 为 1。因此模型、数据和可选 Embedding 模型必须提前下载到本地，运行中不会联网下载。

### 2.2 模型与云端 API

默认端侧模型由 MODEL_PATH 指定，通常是 Qwen3-4B-Thinking-2507。可覆盖为本地模型目录：

~~~bash
export MODEL_PATH=/path/to/model
~~~

完整端云闭环还需要：

~~~bash
export SKILL_UPDATER_BACKEND=deepseek
export DEEPSEEK_MODEL=deepseek-v4-flash
export DEEPSEEK_API_KEY='...'
~~~

没有 DEEPSEEK_API_KEY 时 rollout、环境、轨迹池与指标仍会运行，但云端蒸馏、失败诊断和 tree 演化会跳过；不能将此结果称为完整 CoSkill。不要把 API key、PAT 或其他密钥写入脚本、Git、README 或日志。

### 2.3 数据

ALFWorld 默认读取 ALFWORLD_DATA；启动脚本默认值是 CACHE_ROOT/alfworld。正式脚本覆盖六类 train split：

- pick_and_place_simple
- look_at_obj_in_light
- pick_clean_then_place_in_recep
- pick_heat_then_place_in_recep
- pick_cool_then_place_in_recep
- pick_two_obj_and_place

WebShop 需要 WEBSHOP_DATA_DIR，目录必须包含：

~~~text
items_shuffle_1000.json
items_ins_v2_1000.json
items_human_ins.json
../search_engine/indexes/
~~~

WebShop 脚本会依次尝试当前 CoSkill、相邻 Skill0 和本机预设数据路径；失败时必须显式设置 WEBSHOP_DATA_DIR。只有商品 JSON、没有 search_engine/indexes 时不能运行。

## 3. 环境探测、GPU 和输出目录

两个主脚本以 /GLOBALFS/hit_wxia_1 判断运行环境。

| 场景 | 默认 GPU | CACHE_ROOT | DATA_ROOT | OUTPUT_ROOT |
| --- | --- | --- | --- | --- |
| 超算 | 0,1；两个 data-parallel vLLM worker | /GLOBALFS/hit_wxia_1/.cache | HOME/data/verl-agent | 项目 outputs/ |
| 本地共享 3090 | 仅 GPU 0，且必须空闲 | HOME/.cache | 项目 skillrl_data/verl-agent | 项目 skillrl_outputs/ |

本地脚本会检查 GPU 0 是否有计算进程，非空闲即退出。不要绕过这个保护。超算正式运行默认使用两张卡；单卡 smoke test 可显式覆盖：

~~~bash
CUDA_VISIBLE_DEVICES=0 DATA_PARALLEL_WORKERS=1 ROLLOUT_WORKER_GPUS=0 \
  bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh
~~~

两个 CoSkill driver 会在 `multiprocessing spawn` **之前**向每个 rollout worker 写入仅含其分配 GPU group 的 `CUDA_VISIBLE_DEVICES`，并在 worker ready 日志中打印该值；随后父进程恢复原始可见列表。worker 只校验并继承这个 GPU group mask，不会在子进程中再次重设它，以免 vLLM EngineCore 在不同 CUDA 可见性命名空间中把第二个副本重新映射到 GPU 0。若继承值不匹配，driver 会在加载模型前失败。所有入口同时设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，保证编号稳定。

### 4.2 CoSkill WebShop 四卡并行（不改变 rollout 规模）

WebShop launcher 按调度器分配的 `CUDA_VISIBLE_DEVICES` 自动检查可见 GPU 数。若可见卡至少为四张，`VLLM_PARALLEL_TOPOLOGY=auto` 默认采用 `DP=2 × TP=2 × PP=1`：两个常驻 rollout worker 分别继承 `0,1`、`2,3`（或调度器给出的前四张卡），每个 worker 仍处理 6 个基础任务、每组仍合计 72 条 rollout。模型、任务、种子、token 预算、云端更新边界均不变；只改变 vLLM 的模型分片方式。

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh
~~~

`VLLM_PARALLEL_TOPOLOGY=dp4` 使用 `DP=4 × TP=1`，常常有更高吞吐，但改变了采样调度，不应用于与已经启动的 DP=2 运行做逐 token 续跑。`manual` 模式允许设置 `DATA_PARALLEL_WORKERS`、`TENSOR_PARALLEL_SIZE` 和 `PIPELINE_PARALLEL_SIZE`；pipeline parallel 默认 1，因为 Qwen3-4B 通常不会从 PP 获得额外吞吐。每个 worker 必须获得互不重叠的 GPU group；不足 `DP×TP×PP` 张可见卡时，launcher 会在启动模型前失败。

`VLLM_MAX_NUM_SEQS=0`（默认）会把每个 worker 的 vLLM scheduler 上限设为它实际处理的 rollout batch（DP=2 时为 36，DP=4 时为 18），而非 vLLM 的 1024-request 默认预热。所有实际请求数、rollout、token 预算和 action protocol 保持不变；此开关只消除从不发生的空余调度容量和对应的 sampler warm-up。显式提高该值可用于吞吐实验，但不能低于该 worker batch。

WebShop rollout 已按每个环境 step 批量调用 vLLM：整批先生成 think，再整批续写 action。后者是严格双段协议的一部分，不能合并为单次自由生成而不改变策略分布。vLLM V1 默认启用 prefix caching，第二阶段可以复用第一阶段 chat prompt 的块级 KV 前缀。超算 launcher 默认 `VLLM_ENFORCE_EAGER=0`，使 A800 在 warm-up 后使用 CUDA Graph 提升长生成 decode 吞吐；该开关不修改 prompt、采样参数、token/rollout 预算或环境接口。若短本地 smoke 出现 CUDA Graph 显存问题，可显式设 `VLLM_ENFORCE_EAGER=1`。

常用环境变量：

| 变量 | 用途 |
| --- | --- |
| PROJECT_ROOT | 项目根目录，通常无需设置 |
| CACHE_ROOT、DATA_ROOT、OUTPUT_ROOT | 覆盖自动探测目录 |
| MODEL_PATH | 端侧模型目录 |
| OUTPUT_DIR | 本次运行的完整输出目录 |
| CUDA_VISIBLE_DEVICES | 可见 GPU 列表 |
| DATA_PARALLEL_WORKERS | rollout worker 数 |
| ROLLOUT_WORKER_GPUS | worker GPU 列表，例如 0,1 |
| TENSOR_PARALLEL_SIZE | 仅 ALFWorld launcher 使用，默认每 worker 1 |
| ALFWORLD_DATA | ALFWorld 数据根目录 |
| WEBSHOP_DATA_DIR | WebShop 数据目录 |
| MAX_EPISODES、TOTAL_GROUPS、GROUP_SIZE | 脚本层面的规模覆盖 |
| LOG_TRAJECTORIES | 1 时保存完整 prompt 和轨迹；磁盘占用显著增加 |
| THINK_TRACE_SAMPLES_PER_GROUP、THINK_TRACE_EVERY_GROUPS | WebShop 保存少量完整 episode（最多 15 步）的实际 `<think>`/`<action>` 审计样本；默认每 10 group 取 1 局，另含 group 1 |

每次新实验必须用新的 OUTPUT_DIR。metrics.jsonl 和 group_metrics.jsonl 是追加写入；同一目录重新以 resume=0 启动会混入旧记录。

## 4. 快速开始

### 4.1 ALFWorld 正式学习

~~~bash
conda activate skillRL
cd /data2/myl/CoSkill
export DEEPSEEK_API_KEY='...'
export OUTPUT_DIR=/GLOBALFS/hit_wxia_1/myl/CoSkill/outputs/alfworld/run_001
bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh
~~~

脚本默认：六类 train 游戏、每个 game group_size=6 次 rollout、每 group 72 局、最多 7200 局、单局 40 步、temperature=1.0、response=4096 token。

本地最小 smoke test：

~~~bash
MAX_EPISODES=2 BATCH_ROLLOUT_SIZE=1 DATA_PARALLEL_WORKERS=1 \
ROLLOUT_WORKER_GPUS=0 LOG_TRAJECTORIES=1 \
OUTPUT_DIR=/data2/myl/smoke_outputs/coskill_alfworld \
bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh
~~~

它只检查环境、模型、协议和输出，不应用于比较成功率。

### 4.2 WebShop 正式学习

~~~bash
conda activate skillRL
cd /data2/myl/CoSkill
export DEEPSEEK_API_KEY='...'
export WEBSHOP_DATA_DIR=/path/to/webshop/data
export OUTPUT_DIR=/GLOBALFS/hit_wxia_1/myl/CoSkill/outputs/webshop/run_001
bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh
~~~

默认：train_data_size=12、group_size=6，即 72 局/group；total_groups=100，至多 7200 局；单局 15 步。脚本对齐的 token 设置是名义 prompt 不超过 8192 token、response 4096 token，其中 thinking=3840、action=256。端侧 agent 会按实际 tokenizer 在首阶段前压缩超长 chat prompt，并为强制 `</think>` 闭合预留少量 token；因此第二阶段 action prompt 始终保留 256-token 动作空间，不会越过 `max_model_len=12288`。这不改变 response/rollout 预算，只在输入超过上下文窗口时压缩中间的旧上下文，同时保留任务开头和最新 observation/action 后缀。

本地最小 smoke test：

~~~bash
TRAIN_DATA_SIZE=1 GROUP_SIZE=1 TOTAL_GROUPS=1 MAX_EPISODES=1 \
DATA_PARALLEL_WORKERS=1 ROLLOUT_WORKER_GPUS=0 \
MAX_TOKENS=64 THINK_BUDGET=48 ACTION_BUDGET=16 LOG_TRAJECTORIES=1 \
OUTPUT_DIR=/data2/myl/smoke_outputs/coskill_webshop \
bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh
~~~

## 5. CoSkill 功能开关

| 开关 | 默认 | 含义 | 关闭后的效果 |
| --- | ---: | --- | --- |
| --enable_coskill | 1 | 注入 general/task/mistakes bullet skills，云端产生 dyn_ patch | 不注入 bullet，停止 patch 蒸馏 |
| --enable_skill_tree | 1 | 将当前任务族 skill tree 注入端侧 prompt | 不注入 tree；tree 演化也关闭 |
| --enable_skill_tree_evolve | 1 | 云端按任务族生成、保留或细化 tree | 可保留已有 tree，但不再更新 |
| --enable_failure_analysis | 1 | 云端批量诊断失败轨迹 | tree 可进化，但缺少结构化失败诊断 |
| --enable_hierarchy | 1 | 启用 bullet skill 的 L0/L1/L2 生命周期 | 退化为普通 SkillsOnlyMemory 行为 |
| --no_thinking | false | 关闭 thinking | 更快，通常更弱 |
| --nowait | false | 抑制 Wait/Hmm 等回溯式生成 | 只改变生成约束 |

ALFWorld driver 还接受兼容别名 --enable_playbook_evolve；它与
--enable_skill_tree_evolve 指向同一个开关。新实验统一使用后者，避免配置记录出现两种名称。

默认 skills JSON 提供静态 bullet skills。当前正式 ALFWorld/WebShop driver 的默认 JSON 都没有 skill_trees 或 task_playbooks：新运行在第一次云端更新前不会注入手写 seed playbook。云端产生的 tree 会保存在本次输出的 skill_lib 快照中。

静态 bullet skills 仍是显式实验先验。若主张架构贡献，所有实验臂必须使用相同模型、初始 skills、任务接口、token 预算、任务清单与种子，只切换 CoSkill 功能开关。

## 6. 检索与生命周期参数

### 6.1 检索

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| --skills_json | 数据集对应 claude_style_skills.json | 初始 bullet skill bank |
| --retrieval_mode | ALFWorld driver 为 embedding，WebShop 为 template；两个 launcher 均显式 template | template 为规则式任务族检索；embedding 为跨类语义排序 |
| --embedding_model_path | 无 | embedding 模式使用的本地路径/模型 ID；默认 Qwen3-Embedding-0.6B |
| --top_k | 6 | 注入的 general skills 数；template 下 task-specific skills 默认不由此截断 |

固定检索模式、skills JSON 和 top_k，才能进行公平对比。Embedding 模式额外占用模型/显存，且离线时必须已有本地权重。

### 6.2 生命周期

| 参数 | 默认 | 含义 |
| --- | ---: | --- |
| --stable_cycles_l1 | 3 | L0 到 L1 的稳定周期要求 |
| --stable_cycles_l2 | 5 | L1 到 L2 的稳定周期要求 |
| --success_l1 | 0.7 | L0 晋升 L1 的使用成功率阈值 |
| --demote_threshold | 0.3 | 样本充分且低于该值时降级/弃用 |
| --min_calls | 10 | 做生命周期判断前的最少使用次数 |

冻结模型不会把 L2 自动写入模型权重；internalized=0 是 no-RL 运行的预期状态，不代表生命周期功能未工作。

## 7. 轨迹池、云端更新与参数

每局结束后，成功/失败轨迹进入 TracesPool。相邻 observation 会压缩；相同且无进展的重复动作可按阈值折叠。

触发优先级：

1. 任一任务族近期失败率达到性能水位线；
2. 某任务族近期成功率明显下降，或低成功率下停滞；
3. 累积压缩轨迹 token 达到容量水位线；
4. 若给出 --cloud_update_every N，则按每 N 个 group 强制更新。

| 参数 | 默认 | 说明 |
| --- | ---: | --- |
| --capacity_watermark | 50000 | 未导出压缩轨迹达到该近似 token 数时触发 |
| --perf_watermark | 0.6 | 某任务族近期失败率达到该值时触发 |
| --min_samples | 16 | 判断性能/趋势前的样本下限 |
| --loop_threshold | 3 | 连续相同无进展动作达到该次数时折叠 |
| --cloud_update_every | 0 | 大于 0 时按 group 边界强制更新；0 只用水位线 |
| --max_new_skills | 3 | 每云端周期最多新增 bullet skill 数 |
| --skill_tree_evolve_min_samples | 6 | 某任务族进化 tree 前所需成功+失败样本数 |
| --coskill_debug | 0 | 1 时落盘更多 patch/debug 信息 |

弱任务族持续停滞时可能每 group 都触发云端更新。这是功能正常但成本高的状态；应单独报告云端 token、调用次数和耗时。

## 8. 全部 driver 参数

### 8.1 共同参数

| 分类 | 参数 | ALFWorld / WebShop Python 默认 | 说明 |
| --- | --- | --- | --- |
| 随机性 | --seed | 0 / 0 | 环境与采样种子 |
| 单局限制 | --max_steps | 40 / 15 | 单局最大环境步数 |
| 模型 | --model_path | 无 / 必填 | vLLM 模型目录 |
| 模型 | --gpu_mem_util | 0.8 / 0.8 | vLLM 显存利用率上限 |
| 模型 | --max_model_len | 10240 / 6768 | launcher 的 WebShop 实际覆盖为 12288 |
| 模型 | --max_tokens | 4096 / 768 | launcher 的 WebShop 实际覆盖为 4096 |
| 模型 | --think_budget | 3500 / 640 | launcher 的 WebShop 实际覆盖为 3840 |
| 模型 | --temperature | 1.0 / 1.0 | rollout 采样温度 |
| 技能 | --skills_json | 数据集 JSON | 初始技能库 |
| 技能 | --retrieval_mode、--top_k | 见第 6 节 | 检索配置 |
| 技能 | --enable_hierarchy 等 | 见第 5、6 节 | 生命周期与端云开关 |
| 云端 | --capacity_watermark 等 | 见第 7 节 | 触发与蒸馏配置 |
| 记录 | --log_trajectories | 1 / 0 | 是否保存每步完整 prompt/轨迹 |
| 输出 | --outdir | 必填 | 独立输出目录 |

Python driver 默认值可能被 launcher 覆盖。正式复现实验应记录最终 shell 命令，而不是只记录 Python 默认值。

### 8.2 ALFWorld 专用参数

| 参数 | 默认 | 说明 |
| --- | ---: | --- |
| --task_types | 六类任务字符串 | 逗号分隔的任务类型 |
| --fixed_games_manifest | 无 | 固定 game 清单 JSON；推荐用于严格 A/B |
| --num_games | -1 | 每类 game 数；小于等于 0 表示该 split 全部 |
| --group_size | 6 | 同一 game 的 rollout 次数 |
| --batch_rollout_size | 1 | 同步 batch 大小；launcher 正式设为 72 |
| --data_parallel_workers | 1 | 常驻 rollout 进程数 |
| --rollout_worker_gpus | 无 | 逗号分隔 GPU，例如 0,1 |
| --tensor_parallel_size | 1 | 单个 vLLM 实例的 tensor parallel 度 |
| --sample | false | num_games 大于 0 时按物体均匀抽样 |
| --sample_seed | 0 | 抽样种子 |
| --split | train | 数据 split |
| --epochs | 1 | 完整游戏池重复次数 |
| --max_episodes | 0 | 总局数硬上限；0 为不额外限制 |
| --checkpoint_every_groups | 0 | 每 N group 写 checkpoint；launcher 设为 2 |
| --history_length | 8 | prompt 历史窗口 |
| --resume | 0 | 1 时从同一输出目录恢复 |

fixed_games_manifest 可以固定 ON/OFF 两臂的 game 文件。driver 会检查路径、task_type 与游戏可解性。

### 8.3 WebShop 专用参数

| 参数 | 默认 | 说明 |
| --- | ---: | --- |
| --train_data_size | 12 | 每个 group 的不同购物任务数 |
| --val_data_size | 32 | 记录比较设定；当前 no-RL driver 不执行 validation rollout |
| --group_size | 6 | 每个任务的 rollout 数；group 总局数为 train_data_size 乘 group_size |
| --total_groups | 100 | 最大 rollout group 数 |
| --max_episodes | 0 | 总局数硬上限；0 时按 total_groups 计算 |
| --repeat_stop_threshold | 6 | 同一动作连续达到该次数时提前结束该局 |
| --webshop_file_path | 必填 | items_shuffle_1000.json 路径 |
| --webshop_attr_path | 必填 | items_ins_v2_1000.json 路径 |
| --data_parallel_workers | 2 | worker 数；不能超过可见 GPU 或 train_data_size |
| --rollout_worker_gpus | 无 | 逗号分隔 GPU 列表 |
| --checkpoint_every_groups | 2 | checkpoint 周期 |
| --history_length | 8 | 端侧历史窗口 |
| --prompt_char_limit | 13000 | launcher 实际覆盖为 24000 字符 |
| --action_budget | 128 | launcher 实际覆盖为 256 |

WebShop 强制 think_budget 加 action_budget 不超过 max_tokens。val_data_size 不是当前 driver 生成的 held-out 成绩。

## 9. 输出、成功标准与 valid action

正常 OUTPUT_DIR 包含：

~~~text
driver.log
metrics.jsonl
group_metrics.jsonl
summary_partial.json
summary.json
coskill_status.json
traces_pool/raw_traces.jsonl
traces_pool/batch_*.json
cloud_io/
skill_lib/skills_step*.json
skill_lib/skills_latest_*.json
checkpoints/step*.json
trajectories/                 仅 log_trajectories=1
~~~

ALFWorld 的 won 由环境 won 判定。WebShop 仅 terminal task_score 等于 1.0 时将 won 视为 true；连续 task_score 是部分匹配信息，不能替代严格成功率。

为了兼容历史指标并诊断协议问题，每步、每局与每组保存两种有效动作口径：

| 字段 | 含义 |
| --- | --- |
| valid_action | 宽松/历史口径。ALFWorld：模型直接给出可提取、命中 admissible action 的动作并有闭合 think；WebShop：可提取 action 块。 |
| strict_valid_action | 恰好一个完整 think 块和 action 块，think 在前，并满足环境的直接动作要求。 |
| execution_source | ALFWorld：direct、salvaged、fallback；WebShop：direct 或 malformed。 |

ALFWorld 中 episode/valid_action_ratio 为宽松口径，另有 episode/strict_valid_action_ratio。WebShop 为兼容既有日志，episode/valid_action_ratio 保持历史严格口径；另写入 episode/strict_valid_action_ratio 与 episode/relaxed_valid_action_ratio。

WebShop 的第一步同样注入检索到的静态技能：首步常常决定搜索 query，不能因为尚无 history 而退化为无技能模板。CoSkill 的 no-RL driver 与其 GRPO 环境管理器均采用该规则，且已与 SkillRL、Skill0 的 WebShop GRPO 路径对齐。该注入不包含手写 seed playbook，也不读取 oracle；新运行的任务树仍为空，只有云端从训练期 successful rollout 分析得到的树才会被写入并注入。启用 `LOG_TRAJECTORIES=1` 时，每步 JSON/TXT 额外写入 `raw_model_output`，可直接判断 malformed 是模型未产出动作块，还是后续投影/环境执行问题。

WebShop 默认开启 Qwen thinking：prompt 要求恰好一个 `<think>...</think>` 后接一个 `<action>...</action>`；driver 先批量生成最多 3,840 token 的 think，再批量生成最多 256 token 的 action。为便于检查真实推理而不写出全部 trajectory，launcher 默认在 group 1 和之后每 10 个 group 各保存 1 局完整 episode（最多 15 步）到 `thinking_samples.txt` 与 `thinking_samples.jsonl`；逐步包含 observation、完整 think、action、投影后执行动作、有效性和环境得分。这些文件只复制已经生成、已经执行的模型输出，不参与 TracesPool、云端更新、奖励或指标。设 `THINK_TRACE_SAMPLES_PER_GROUP=0` 可关闭。

常见指标前缀：

| 指标 | 含义 |
| --- | --- |
| coskill/pool/* | 已收集/待导出轨迹、压缩 token、折叠循环数 |
| coskill/cloud/* | 云端更新、patch、prompt/completion token |
| coskill/skilllib/L0,L1,L2 | bullet skill 生命周期层数量 |
| coskill/skill_tree/* | 诊断、tree 进化、tree 更新和 tree 数量 |
| coskill/timing/* | 导出、蒸馏、tree 进化耗时 |
| experiment/cloud_round_used | 当前局使用的云端技能版本轮次 |

状态为 running 时 summary_partial.json 是中途结果，不是最终结论。

## 10. Checkpoint、恢复和公平评测

ALFWorld 的 --resume 1 会读取同一 OUTDIR 的 summary_partial.json 与 skill_lib/skills_latest_checkpoint.json；没有 checkpoint 时回退到 skills_latest_rollout.json。恢复前必须保持游戏池、manifest、种子、模型、技能 JSON 与核心参数不变。

~~~bash
OUTPUT_DIR=/path/to/existing_run \
bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh --resume 1
~~~

恢复不是新实验。新实验必须创建新的 OUTPUT_DIR。WebShop 会写 checkpoint 与最终 skill snapshot；中断恢复前应核对对应版本的 summary_partial、skill_lib 和 launcher 参数。

最终 held-out 评测必须：

1. 加载训练结束后的 skills_latest_final.json；
2. 使用预先固定且未参与云端更新的 ALFWorld manifest 或 WebShop goal-id 集合；
3. 关闭 tree 演化、失败分析和 bullet 更新路径；
4. 固定模型、初始条件、最大步数、token 预算、动作接口和随机种子；
5. 不把 held-out 轨迹写回训练技能库。

建议消融：

| 臂 | enable_coskill | enable_skill_tree | enable_skill_tree_evolve |
| --- | ---: | ---: | ---: |
| Frozen baseline | 0 | 0 | 0 |
| Bullet only | 1 | 0 | 0 |
| Tree only | 0 | 1 | 1 |
| Full CoSkill | 1 | 1 | 1 |

## 11. 常见问题

### 云端未更新

检查 DEEPSEEK_API_KEY、driver.log 中的 CloudAnalyzer、watermark fired、skipping。样本不足或未达到水位线时不更新是正常的。需要固定更新边界时，所有比较臂使用相同的 --cloud_update_every N。

### WebShop 找不到数据

显式设置 WEBSHOP_DATA_DIR，并同时检查三份商品 JSON 与 search_engine/indexes。复制商品 JSON 而遗漏索引目录不能运行。

### 本地 GPU 0 被拒绝

共享服务器保护正常生效。运行 nvidia-smi --id=0，等待 GPU 0 无计算进程后再启动；不要改用其他本地 GPU 绕过规则。

### 超算 vLLM EngineCore 在 KV cache 初始化时 OOM

若日志显示两个 EngineCore 的 OOM，且一张 80GB 卡只剩不足 1GB、两个 vLLM 进程合计已占约 79GB，首先确认使用的是包含上述“spawn 前单卡 mask、worker 内不重设”的版本。该症状是两个 data-parallel 副本争用同一张卡，不是 WebShop reward、动作协议或 4,096-token 预算本身造成的。修复后 worker 日志必须分别显示 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`（或调度器分配的两张卡）；若不匹配会在模型加载前报出明确错误。不要通过缩小 rollout 数、response budget 或 PPO batch 来掩盖此绑定错误。

### valid action 比率低

同时查看宽松和严格比率及 execution_source：

- 宽松高、严格低：通常是重复标签、标签顺序或协议格式问题；
- 两者都低且 ALFWorld salvaged 高：模型未直接给出 admissible action；
- WebShop malformed 高：模型没有给出可提取 action 块；
- success 高但有效率低：安全回退可能维持了执行，不能只报告成功率。

设 LOG_TRAJECTORIES=1 后，可在 trajectories 的 prompt 与 trajectory 文件中逐步检查。长训练默认关闭以节省磁盘。

### 没有 summary.json

运行中断或仍在运行时只有 summary_partial.json 是正常的。检查 driver.log、coskill_status.json、checkpoints 和 skill_lib，再决定恢复或新开目录。

## 12. 正式结果归档

每个正式实验至少归档：

- Git commit；
- 完整 launcher 命令和所有环境变量覆盖；
- 模型、数据、WebShop 索引、skills JSON 版本/路径；
- summary.json、coskill_status.json、group_metrics.jsonl；
- skills_latest_final.json；
- 云端 token、调用次数、耗时；
- held-out manifest 或 goal-id 清单；
- 随机种子、GPU 数、vLLM/CUDA/Python 版本。

不要归档任何 API key、GitHub PAT、W&B key 或其他机密。
