# CoSkill：论文方法与系统说明

本文档面向论文撰写、答辩陈述和方法复现说明，描述 CoSkill 在 ALFWorld 与 WebShop 上的端云协同技能学习系统：使用了什么技术、每个功能解决什么问题、端侧与云端如何协作、一次完整运行如何流动，以及实验结论应如何界定。

本文讨论当前 CoSkill 的冻结端侧模型版本：端侧不更新模型权重，学习状态写入外部技能库。运行参数、命令与排错请看 `COSKILL运行与实验指南.md`；第 4、5 节提供与当前代码一致的轨迹池和云端分析实现级说明，便于论文方法、附录和系统复核使用。

## 1. 方法定位

语言智能体在长时序交互任务中会产生大量成功与失败轨迹。直接把整段轨迹塞回 prompt 会带来三个问题：

1. 原始 observation 和重复动作高度冗余，token 成本随任务长度增长；
2. 小模型难以从少量零散失败中识别真正的决策错误；
3. 模型权重更新昂贵且慢，不适合每次环境失误后立即修正。

CoSkill 将学习过程拆分为高频端侧执行与低频云端归纳。端侧小模型只处理当前任务并快速行动；云端大模型只在积累到足够经验或检测到性能问题时，对压缩后的成功/失败轨迹做归纳。归纳结果不直接改写端侧权重，而是写入一个可检索、可演化、可审计的技能库，并在后续任务中作为紧凑上下文使用。

因此，CoSkill 的核心主张是：

> 通过端云分工、轨迹压缩与分层技能库，将高成本的经验归纳从逐步决策中解耦，使冻结的小语言模型能够在不更新权重的条件下，通过外部技能状态持续改进。

在本文当前实验中，CoSkill 不应被表述为纯端侧 RL 或权重内化方法。它是外部技能记忆持续演化的系统；其优势来源包括端云协作、轨迹处理和技能表示。

## 2. 系统总览

CoSkill 由四个逻辑组件组成：

| 组件 | 所在位置 | 输入 | 输出 | 主要作用 |
| --- | --- | --- | --- | --- |
| 端侧执行器 | 本地 GPU / 边缘侧 | 当前任务、观测、可行动作、历史、检索技能 | 环境动作与原始轨迹 | 高频决策与执行 |
| 轨迹池 | 本地父进程 | 成功/失败原始轨迹 | 压缩轨迹批次、触发信号 | 清洗、压缩、统计、决定何时调用云端 |
| 云端分析器 | 云端大模型 API | 压缩成功/失败样本、分叉信息、当前技能状态 | 新技能补丁、失败诊断、技能树更新 | 低频高认知归纳 |
| 分层技能库 | 本地持久化状态 | 初始 skills、云端补丁、使用结果 | 端侧检索上下文、版本快照、生命周期状态 | 在端云之间保存、调度和评估知识 |

整体闭环如下：

~~~text
环境 reset
  -> 任务描述 + 当前观测 + 可行动作
  -> 技能库检索对应 bullet skills 与任务族 skill tree
  -> 端侧小模型生成 think + action
  -> 动作解析、合法性检查、环境执行
  -> RawTrace 写入轨迹池
  -> 轨迹池检查容量、失败率和成功趋势
  -> 若触发：压缩批次送云端分析
  -> 云端生成 patch / 诊断 / skill tree 版本
  -> 更新技能库并落盘
  -> 下一 rollout group 读取新技能库
~~~

端侧 rollout 可以使用多个 GPU worker 并行。当前 driver 中，worker 在一个 rollout group 内独立执行；主进程汇总所有 worker 轨迹，在 group 边界进行云端更新、保存技能库，然后将最新快照交给下一 group。换言之，当前实现保证了同一 group 内技能状态固定、组与组之间技能状态可演化，便于记录版本与进行比较。

## 3. 端侧：冻结小模型执行与提示词组成

### 3.1 端侧输入

每一步端侧小模型看到的信息只来自可见环境接口和已获准的外部技能状态：

- 当前任务文本；
- 当前 observation；
- 当前环境提供的可行动作或可见点击项；
- 有限长度的近期 observation-action 历史；
- 当前任务相关的 general skills、task-specific skills、mistakes to avoid；
- 当前任务族已经由云端生成的 skill tree。

ALFWorld 使用环境提供的 admissible commands。WebShop 使用当前页面的可见操作，生成 search[...] 或 click[...]。两者都不向端侧提供游戏文件、PDDL、专家计划、隐藏物体位置、正确动作序列或测试集答案。

### 3.2 思考与行动协议

端侧输出遵循统一双段协议：

~~~text
<think>...</think>
<action>...</action>
~~~

其中 think 块包含当前步推理，action 块包含唯一待执行动作。WebShop 的 action 为 search[...] 或 click[...]；ALFWorld action 必须与当前 admissible action 精确匹配后才算直接合法。

为了避免思考过长导致没有行动，端侧采用 thinking 与 action 的两阶段预算控制：先生成 reasoning，达到 budget 或结束 reasoning 后再确保输出行动段。该机制的目标是提高协议完成率，而不是向模型提供额外环境信息。

### 3.3 动作执行与两种有效性口径

环境执行必须安全：模型输出不完整时，ALFWorld 可以从原始生成文本中恢复一个已出现在 admissible action 集合中的动作；仍无法恢复时采用确定性安全回退。WebShop 会将可提取的 action 交给环境，由环境验证页面状态。

输出中同时记录两种口径：

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| valid action，宽松/历史口径 | 动作块可提取；ALFWorld 还要求直接命中 admissible action 并带闭合 think | 与早期日志和执行质量比较 |
| strict valid action | 唯一完整 think 块、唯一完整 action 块且顺序正确；ALFWorld 还要求直接命中 admissible action，WebShop 的页面级合法性由环境执行时验证 | 衡量协议遵从 |
| execution source | direct、salvaged、fallback 或 malformed | 区分模型直接能力与执行保护 |

成功率与有效动作率必须同时报告。高成功率而低直接有效率可能说明安全恢复在帮助系统继续执行，不能被解释为模型本身完全遵从行动协议。

### 3.4 端侧的学习状态

当前 no-RL CoSkill 端侧模型权重始终冻结。端侧的适应来自下一 group prompt 中注入的更新技能，而不是梯度更新。因此，端侧状态应被描述为 external skill-conditioned policy，而不是已内化到参数中的新能力。

## 4. 轨迹池：从原始交互到可分析经验

轨迹池是一个运行在端侧主进程中的流式状态机，而不是训练完成后离线挑选若干失败案例的脚本。它接收所有 episode 的 RawTrace，在接收时立即清洗、压缩、分桶并更新水位线；云端只接收导出的 CompressedBatch。因而，轨迹处理不会阻塞逐步环境交互，且成功与失败具有同等的记录资格。

### 4.1 原始轨迹契约与本地持久化

每个 episode 在进入池之前被规范为下列 RawTrace。`outcome` 优先使用调用方给出的严格成功/失败；缺失时仅以 `episode_reward > 0` 推断成功。WebShop 的连续 `task_score` 被作为元数据保留，但不改变其严格成功定义（只有 `task_score = 1.0` 为 success）。

~~~text
RawTrace = {
  traj_uid, task, task_type, outcome, episode_reward,
  steps: [{step, observation, action, reward}, ...],
  meta: {skill_ids_used, task_score, ...}
}
~~~

接收一条轨迹时，原始 RawTrace 同步追加到 `OUTPUT_DIR/traces_pool/raw_traces.jsonl`，用于事后核对端侧究竟看到了什么、生成了什么。这个原始日志不参与 token 水位线计算；写盘失败被捕获为告警而不会中断训练。压缩后的记录则保留：`traj_uid`、任务和任务族、严格 outcome、奖励、WebShop `task_score`、压缩步骤、被删除的循环步数，以及本 episode 注入的 `skill_ids_used`。

内存中有两类按 `task_type` 分开的双端队列：成功桶和失败桶。每类默认最多保存 200 条压缩轨迹，超出上限时最早记录自动淘汰，避免长期运行时内存无界增长。另一组长度为 `recent_window`（类默认 20）的 outcome 队列只保存 success/failure 标签，用于近期性能统计；它与待导出样本桶相互独立。

### 4.2 流式清洗：无进展循环过滤

循环过滤发生在差分之前，直接依据原始 observation。对于每一步，维护前一步 observation、当前连续动作值和连续长度。仅当下面三个条件同时满足时，连续长度才增加：

~~~text
action_t == action_{t-1}
observation_t == observation_{t-1}
reward_t <= 0
~~~

达到 `loop_threshold` 后的重复步骤被丢弃；当前实验脚本使用 `loop_threshold = 3`，因此前两次重复仍保留，从第三次同一无进展重复开始删除。任何动作改变、observation 改变或正奖励都会重新开始计数。被删除的步数写入该条压缩轨迹的 `dropped_loops`，并累加到池级 `total_dropped_loops`，而不是静默删除。

这个算子只删除“相同动作 + 相同状态 + 无正反馈”的尾部重复，不会删除动作相同但页面/房间状态发生变化的合法重试，也不会利用未来奖励或隐藏环境状态判断某步是否有价值。因此它是可在线执行的局部清洗规则，不是事后根据成败重写轨迹。

### 4.3 自适应 observation 差分与 token 估计

对清洗后的每一步，池将 observation 按非空行切分并去除行首尾空白。令上一观测行集合为 \(P\)，当前观测行集合为 \(C\)，则候选差分为按当前顺序输出的 `+line`（\(C \setminus P\)）以及按上一观测顺序输出的 `-line`（\(P \setminus C\)）；完全相同则记为 `(no change)`。动作和 reward 单独保留，避免差分操作掩盖端侧实际执行了什么。

系统并不总使用差分。只有同时满足以下条件才将候选差分写入 `obs_delta`：

1. 完整 observation 长度至少 400 个字符；
2. 候选差分不是 `(no change)`；
3. 差分长度不超过完整 observation 的 50%，即至少节省一半字符数。

否则 `obs_delta` 直接保存完整、去除首尾空白后的 observation，并标记 `obs_is_full = true`；使用差分时标记为 `false`。这避免对短 observation 或几乎全部变化的长 observation 制造难以阅读的 `+/-` 碎片。池级 token 计数采用不依赖 tokenizer 的近似 `max(1, 字符数 // 4)`，按每条压缩步骤的 `obs_delta` 与 `action` 相加；它只用于容量触发，不应被报告成模型 tokenizer 的精确 token 数。

需要区分两个长度控制层次：池中压缩轨迹保留全部清洗后的步骤，以便日后导出和审计；构造云端 prompt 时才对每条样本最多渲染 12 个未折叠步骤、每个 observation/delta 最多 400 个字符。前者保护可复核性，后者控制单次云端请求成本。

### 4.4 前缀树、决策分叉与成功共识

一次导出将当前 success/failure 桶中的动作序列插入前缀树。每个节点以动作文本为键，保存：经过该节点的总次数 `count`、成功经过数 `n_success`、失败经过数 `n_failure`，以及子节点字典。因而，一个拥有多个子节点的节点表示共享历史后的决策分叉；云端 prompt 采用深度优先顺序最多展示 6 个分叉点，每个点最多展示 5 个候选动作及其成功/失败计数。

与前缀树不同，**共识前缀只从成功轨迹计算**：对所有成功动作序列逐位取最长完全一致前缀。导出全池时它来自本批全部成功轨迹；失败诊断和每个 task tree 演化阶段会重新在对应 task_type 的成功子集上计算。共识前缀意味着“当前端侧已经反复做到的共同开场”，在云端文本中以一行摘要折叠，而不是再次逐步重复。第一个不同动作之后的轨迹仍完整保留，因此折叠不会抹掉成功/失败的关键分叉。

前缀树是批次级动作统计结构，并不等同于环境状态图或专家计划：它既可能含多个 task_type，也不包含不可见状态。它的作用仅是向大模型指出“在相同动作历史后，不同选择的经验性 outcome 统计”。

### 4.5 分桶、触发水位线与导出语义

池在每次 `add_trace` 后更新压缩 token 计数、待导出轨迹数和按任务族的近期 outcome 窗口。默认类参数为容量 50,000 近似 token、失败率阈值 0.6、最小样本数 8、窗口 20；当前 ALFWorld/WebShop 主启动脚本显式将 `min_samples` 设为 16，其余前三个主阈值分别保持 50,000、0.6 和 3。`should_trigger()` 的检查顺序和条件如下：

| 优先级 | trigger_reason | 条件 | 作用 |
| --- | --- | --- | --- |
| 1 | `performance_watermark` | 某 task_type 近期样本数不少于 `min_samples`，且失败率 \(\ge 0.6\) | 快速处理明显困难的任务族 |
| 2 | `success_decline` | 样本可分成前后两半，后半成功率比前半低至少 0.05 | 检测能力退化 |
| 2 | `success_stagnation` | 前后半成功率绝对差不超过 0.05，且后半成功率低于 0.95 | 在未接近满分时处理停滞 |
| 3 | `capacity_watermark` | 自上次导出以来的近似 token 数不少于 50,000 | 定期批处理积累的经验 |

趋势判断同样要求近期样本数不少于 `max(min_samples, 4)`；奇数窗口的前半取 `floor(n/2)`，后半取其余样本。只要一个任务族触发性能或趋势条件，就会先于容量条件导出。

`export_batch()` 默认合并所有待导出的任务族，构造：成功/失败样本列表、`batch_id`、触发原因、共识前缀、上述统计量和前缀树，并将 JSON 写到 `OUTPUT_DIR/traces_pool/batch_<timestamp>_<id>.json`。导出后，已导出任务族的 success/failure 桶清空，待导出 token 与待导出条数归零；**近期 outcome 窗口不清空**，从而保留跨批次性能上下文并减少触发抖动。原始 JSONL 也不清空。

为了使消融或固定任务实验在相同样本边界调用云端，driver 可传入 `force_reason`（例如 `episode_interval_K` 或 `group_interval_K`）。这会绕过上述水位线判断，但不会改变 RawTrace、压缩或导出格式。固定间隔与自适应水位线是两种不同的实验协议，论文中必须分别说明。

### 4.6 轨迹池端到端伪代码

~~~text
for each completed episode r:
    append r to raw_traces.jsonl
    outcome <- r.outcome, otherwise (r.episode_reward > 0)
    clean_steps, n_drop <- filter_repeated_no_progress(r.steps)
    diff_steps <- adaptive_line_delta(clean_steps)
    append compressed trace to success[task_type] or failure[task_type]
    append outcome to recent_outcomes[task_type]
    pending_tokens += approx_tokens(diff_steps)

at each rollout-group boundary:
    fire, reason <- forced interval OR pool.should_trigger()
    if fire:
        batch <- pool.export_batch(reason)
        run cloud analysis over batch
~~~

上述流程中的成功轨迹来自同一系统在学习阶段实际取得的 rollout；轨迹池既不读取测试答案，也不替换失败轨迹为专家轨迹。

## 5. 云端：对比蒸馏、失败诊断与技能树演化

云端使用大模型 API 执行低频经验归纳。云端看到的是压缩批次、任务族、任务文本、动作、可见 observation 变化、环境成功判据与当前技能库，不接触环境隐藏状态。

### 5.1 云端编排、后端与调用顺序

端侧主进程只在 group 边界调用 `CoSkillCloudLoop.maybe_update(...)`。该函数的顺序是固定的：

~~~text
触发/强制触发
  -> export_batch（先将压缩批次落盘）
  -> 惰性创建 CloudAnalyzer
  -> [enable_coskill] contrastive_distill
  -> ingest_patches + advance_lifecycle
  -> [enable_playbook_evolve] diagnose_failures（整批一次）
  -> 按 task_type 逐个 evolve_playbook
  -> 保存 skill_lib/skills_step<N>.json
  -> 覆盖写 coskill_status.json
~~~

`CloudAnalyzer` 在第一次实际触发时才读取 API 配置。默认后端为 DeepSeek-compatible OpenAI client，要求 `DEEPSEEK_API_KEY`，默认基址为 `https://api.deepseek.com/v1`、默认模型为 `deepseek-v4-flash`；设定 `SKILL_UPDATER_BACKEND=azure` 时改用 Azure OpenAI，要求 Azure key 和 endpoint，默认模型名为 `o3`。三类调用统一采用单个 user message 的 `chat.completions.create`，并将 `max_completion_tokens` 固定为 4096。

如果 API key 缺失、客户端初始化失败或某次调用抛出异常，训练进程不会崩溃：对应分析返回空结果，旧技能库不被改写。一个需要如实说明的实现细节是，压缩批次在创建客户端之前已经被 `export_batch()` 写入 `traces_pool/batch_*.json`，且内存待导出桶已清空；当前实现不会自动将该批重新入队。因此恢复这类失败时应以落盘批次为依据，不能把“跳过云端调用”误报为已经完成一次技能更新。

两个开关彼此独立：`enable_coskill` 控制 bullet skill 对比蒸馏与生命周期推进；`enable_playbook_evolve` 控制失败诊断和 task-specific skill tree 演化。树演化还受 `enable_failure_analysis` 和每个任务族的 `playbook_evolve_min_samples` 约束，主脚本设为 6。也就是说，可以运行 tree-only、bullet-only 或二者同时运行的消融，而不能把“云端被调用”简单等同于“三种产物均已更新”。

### 5.2 成功-失败对比蒸馏

云端首先比较同一任务族的成功与失败 rollout：

1. 成功轨迹在何处做出了有效决策；
2. 失败轨迹从哪一步偏离；
3. 这种差异是偶发实例细节还是可复用规律；
4. 应当形成何种短、可执行的规则，避免重复已有技能。

输出为动态 skill patch。patch 通常包含标题、适用范围、任务族、原则、触发时机；结构化字段可表达动作流和应避免的行为。每轮最多新增的 patch 数量受 max_new_skills 控制，避免技能库无界膨胀。

实现中，调用前先扫描 general 与 task-specific 两个技能表中已有的 `dyn_<整数>` ID，取最大编号加一作为本轮起点。云端即使返回自定义 ID，安装前也会按返回顺序重新编号为连续 dyn ID。输出经 JSON 数组解析并规范化后，最终只取前 `max_new_skills` 条（主脚本为 3 条）安装；每条还会附加 `evidence = {from_success, from_failure}`，记录其来自本批的多少成功/失败样本。

安装时，`scope = task_specific` 且任务族有效的 patch 进入相应的 `task_specific_skills[task_type]`；其余进入 `general_skills`。重复或空 skill ID 会被拒绝，新 patch 的 lifecycle 初始化为 L0，并使 embedding 检索缓存失效。随后本轮调用一次 `advance_lifecycle(modified_ids)`：新技能的稳定周期清零，未被修改的技能才累计稳定周期。该过程是外部技能库状态更新，不是对端侧语言模型的梯度更新。

### 5.3 批量失败诊断

对失败轨迹，云端生成结构化诊断，例如错误目标、错误顺序、无效动作、状态误读、低效探索、循环或过早终止。诊断包含：

- 根因与简短证据；
- 可防止失败的规则；
- 当前 skill tree 的缺口；
- 应在 tree 哪一节添加或细化规则。

诊断不是监督标签，不参与模型权重反向传播；它是产生可审计技能编辑的中间产物。

一次 `diagnose_failures` 调用先按 `task_type` 对失败与成功样本分别分组。每个有失败样本的任务族最多提供前 6 条失败轨迹和前 3 条成功参照；失败样本获得稳定的 `task_type#序号` 引用。随后解析出的诊断按模型输出中的 `task_type` 再次分组，供该任务族的 tree 编辑使用。当前解析器验证“JSON 数组中的字典”这一结构，但不以隐藏答案校验自然语言根因；因此论文报告应将诊断视为可审计的模型归纳，而不是自动获得的真值标签。

### 5.4 按任务族技能树

与扁平 bullet skills 不同，skill tree 为每个任务族维护一个按层级组织的决策指南。云端根据新成功/失败样本和诊断选择：

- **rewrite**：当前任务族没有 tree 时从当前经验建立第一版；
- **refine**：只修改被证据指出的分支；
- **keep**：当前 tree 已足够且没有可避免的新失败。

树的内容要求在任务族内泛化，不允许硬编码特定 sampled entity、布局、商品 ID 或测试实例答案。它应说明目标理解、子目标顺序、状态判断、关键前置条件、常见失败与停止判据。端侧只会读取当前检测到任务族对应的 tree。

具体地说，整批失败诊断只调用一次；之后编排器把本批样本重新按任务族分开。某个任务族的 success 与 failure 总数低于 `playbook_evolve_min_samples` 时直接跳过，不产生 tree 调用。达到阈值时，云端看到该任务族当前正在注入端侧的完整 tree 文本（第一版则为无 tree）、最多 4 条成功轨迹、最多 5 条失败轨迹和最多 8 条诊断。返回 `keep` 或空 tree 不调用安装接口；其他返回将生成下一版本。

### 5.5 云端大模型的提示词构造与输出解析

云端调用不是把原始日志直接交给大模型，而是采用受约束的“经验证据 + 结构化输出”协议。所有云端请求都先加入一个简短的环境契约，限定该环境的动作空间、成功条件和禁止跨数据集臆测的行为：

- ALFWorld：动作只能从当前轨迹所示的 household action 语义中理解，成功是目标对象状态/操作完成；
- WebShop：动作是 search[query] 或 click[visible text]，只有 task score 为 1.0 的终止购买算成功；
- 云端被明确要求不把另一数据集的房间、容器、购物或商品选项规则迁移到当前环境。

在此基础上，当前系统使用三类云端提示词。它们的输入、输出与解析规则如下。

| 云端任务 | 输入证据 | 强制输出 | 解析后用途 |
| --- | --- | --- | --- |
| 对比蒸馏 | 成功样本、失败样本、决策分叉、已有技能标题 | JSON 数组；每项含技能标题、scope、task type、原则、触发时机 | 生成动态 bullet skill patch |
| 失败诊断 | 按任务族分组的失败轨迹、同族成功 rollout reference、共识前缀、决策分叉 | JSON 数组；每项含根因、证据、纠正规则、tree 缺口、patch 位置、置信度 | 为 tree 编辑提供可审计依据 |
| tree 演化 | 当前 tree、成功/失败轨迹、失败诊断、共识前缀 | JSON 对象；action 为 keep/refine/rewrite，另含完整 tree、层级、critique、changelog | 更新该任务族 tree 版本 |

#### 5.5.1 对比蒸馏提示词

对比蒸馏提示词提供压缩后的成功与失败轨迹，而不是完整环境文件。它最多渲染 5 条成功和 6 条失败样本；每条样本在折叠共识前缀后最多保留 12 个 action-observation 对，每个 observation/delta 最多 400 字符。系统同时提供少量决策分叉，即同一共同前缀之后不同动作分支的成功/失败计数；前缀树最多输出 6 个分叉点，每点最多 5 个子动作。

云端被要求从“成功与失败的差异”归纳短小、可执行、可泛化的规则，并看到当前技能标题以避免同义重复。输出限定为 JSON 数组；每个 patch 的原则最多两句、触发时机最多一句。这一约束的目的有二：

1. 使新技能适合 4B 端侧模型阅读，避免云端输出长篇解释占满 prompt；
2. 将云端结果限制为可检索的技能条目，而不是未经验证的自由文本策略。

收到响应后，系统从响应文本的第一个 `[` 到最后一个 `]` 尝试 JSON 解析，因此模型在 JSON 前后附带少量解释时仍可能被恢复。不是字典或没有 `title` 的项目会丢弃；有效项目会重新分配连续的 dyn_ ID、补齐 `principle`/`when_to_apply` 兼容字段，并附上本轮成功/失败样本数量作为 evidence。若 task-specific patch 的任务族不属于本轮已观察到的任务族，系统会将其改为 general 或映射到唯一已观察到的任务族，避免产生不可检索的 ALL 技能桶。为控制端侧上下文，兼容的冗长字段 `trigger`、`action_flow` 与 `avoid` 在安装前会移除；若模型只给旧式字段，则由它们拼接出 `principle`。解析失败、API 异常或空输出时，本轮不写入新 patch，旧技能库保持不变。

#### 5.5.2 失败诊断提示词

失败诊断按任务族构造。对于每个失败轨迹，提示词给出稳定的引用编号；同族成功 rollout 只作为“观察到的成功行为参照”，并明确声明它们不是 oracle demonstration 或 ground-truth action plan。诊断 prompt 对每个任务族最多渲染 6 个带引用的失败样本和 3 个成功样本，并在该任务族成功共识前缀之后开始展示步骤。

为了降低冗余，成功轨迹的共识起始动作会折叠。提示词要求云端把注意力放在分叉后的失败原因，并对每条失败只给出一个主要因果原因。结构化诊断字段包括：

- failure type：错误目标、错误顺序、低效探索、循环、过早停止、无效动作、状态误读、放弃或其他；
- root cause 与来自 action/observation 的简短 evidence；
- 可防止该错误的 corrective rule；
- 当前 tree 的 skill tree gap；
- patch location，即应修改的 tree 标题层级或新分支位置；
- confidence。

响应解析同样从第一个 `[` 到最后一个 `]` 只接收 JSON 数组中的字典；无法解析时返回空诊断，tree 演化仍可依据轨迹本身进行，系统不会用模型随意文本更新技能库。

#### 5.5.3 技能树演化提示词

tree 演化提示词把“当前端侧实际看到的 tree”原样提供给云端，同时给出新的成功/失败证据、共识前缀和已解析诊断。提示词明确要求：

- tree 只服务当前 goal family，不能混入无关任务族；
- 从重复证据与常识性任务前置条件中归纳，但不能写入具体采样实体、布局、商品 ID 或数据集答案；
- 优先修正当前 tree 自己导致的模糊、矛盾或过度具体表述；
- 仅在证据显示端侧未理解某一分支时增加层级，避免无依据地加深整棵树；
- 返回完整新 tree，而非只返回局部 diff，以便端侧下一个 group 直接使用确定版本。

解析器从第一个 `{` 到最后一个 `}` 尝试读取 JSON 对象。若 `action` 不属于 keep/refine/rewrite，当前实现会在 tree 文本非空时按 `refine` 处理、tree 文本为空时按 `keep` 处理；JSON/API 失败才返回 `None` 并保留旧 tree。这样云端错误不会清空已有技能状态，但这也意味着论文应报告结构化解析成功率，而不能把 prompt 中的格式要求误写成硬性的语义验证。每次成功编辑会保存 critique 和 changelog，支持后续分析“tree 是否被端侧忽略、误读或自身误导”。

#### 5.5.4 资源控制、落盘与论文含义

三类提示词都进行样本数、单条轨迹步数和 observation delta 长度截断；成功共识前缀被折叠，已有技能只传标题而不重复全文。发给云端的原始 prompt、解析后的 patch、诊断和 tree 编辑结果都落在 `cloud_io`，以便审计云端究竟看到了什么。当前实现保存的是 prompt 与**解析后产物**，不保证保存服务端返回的原始自由文本；因此如需逐字复核模型回答，应额外启用响应日志，而不能把已有产物误称为完整 API transcript。

论文中应强调：提示词中的环境契约是接口说明，不是任务答案；成功 rollout reference 是系统自己在学习阶段获得的经验，不是专家示范。建议报告每类云端调用的 prompt/completion token、解析成功率、空输出/失败次数、patch 接受数和 tree 的 keep/refine/rewrite 分布，以说明云端模块的有效性与成本。

### 5.6 Tree 安装、版本化与端侧回写

当一个 task tree 需要安装时，`update_playbook(task_type, content, level, meta)` 同时更新内存中的 `task_playbooks` 与将被 `save_skills()` 持久化的 `skills['skill_trees']`。版本号在该任务族内单调递增；记录保存完整 Markdown、层级、critique、changelog、最多 12 个本轮失败摘要、任务范围和更新时间。

Markdown tree 会被纯字符串解析器按标题深度（`#` 到 `######`）解析为树。每个节点以“标题路径 slug”作为稳定 ID，记录标题、层级、正文 hash、创建版本、最后修改版本、稳定版本数、调用次数、成功次数、deprecated 与 internalized 标记。新版本中标题路径相同且正文 hash 不变的节点继承统计并令 `stable_versions + 1`；正文变化的节点重置稳定计数；消失节点不再进入新版本。这使“整棵 Markdown 由云端返回”仍能在本地保留逐节点的生命周期证据。

下一个 group 读取最新 `skill_lib/skills_step<N>.json`。端侧按当前检测任务族只注入匹配 tree；调用 `record_playbook_usage` 时，所有未 deprecated、未 internalized 的注入节点共享一次调用与成功归因。这个归因是 prompt-level shared attribution，不能解释为已经定位到某一条 tree 规则的单独因果贡献。

### 5.7 可审计产物、计数器与最小复核集

每个正式实验应至少保留以下文件；它们分别回答“端侧做了什么”“云端看了什么”“云端的结构化结果是什么”“下一组实际加载了什么”。

| 位置 | 产物 | 内容与作用 |
| --- | --- | --- |
| `traces_pool/raw_traces.jsonl` | 原始轨迹追加日志 | 每局的原始 observation、动作、奖励与元数据 |
| `traces_pool/batch_*.json` | CompressedBatch | 清洗/差分后的样本、统计、共识前缀、前缀树和触发原因 |
| `cloud_io/distill_prompt_*.txt` | 对比蒸馏输入 | 云端实际收到的成功/失败样本、分叉和已有标题 |
| `cloud_io/patches_*.json` | 已解析且裁剪后的 patch | 实际可被安装的 bullet skills |
| `cloud_io/diagnose_prompt_*.txt`、`diagnoses_*.json` | 诊断输入与解析结果 | 失败引用、成功参照与按任务族分组的诊断 |
| `cloud_io/evolve_skill_tree_*_call*.txt` | tree 演化输入 | 当前 tree、证据与诊断 |
| `cloud_io/skill_tree_*_v*.json`、`*_debug.txt` | 已安装 tree 版本与人类可读调试信息 | 版本、节点状态、共识前缀、诊断和与前版的文本 diff |
| `skill_lib/skills_step<N>.json` | 下一组使用的技能快照 | bullet skills、tree 与生命周期状态 |
| `coskill_status.json` | 最近一次健康快照 | 触发原因、池统计、层计数、云端 token 与耗时 |

云端分析器累计服务端 `usage.prompt_tokens` 与 `usage.completion_tokens`；这些是 API 报告的精确调用 token，与轨迹池的字符近似 token 不是同一指标。`coskill_status.json` 和 group metrics 还记录池累计接收数、待导出数、循环删除数、云端调用/patch 数、诊断/演化调用数、tree 数量，以及导出、蒸馏和 tree 演化耗时。报告结果时应同时给出“调用了几次”和“实际安装了多少 patch/tree 版本”，因为一次触发可能因开关关闭、样本不足、`keep` 或 API 失败而不产生同一种更新。

## 6. 分层技能库：表示、检索与生命周期

技能库是端云共享的持久状态。它由三类内容组成：

1. **初始 bullet skills**：general skills、task-specific skills、common mistakes；
2. **动态 bullet skills**：云端产生的 dyn_ patch；
3. **任务族 skill trees**：云端从本次 rollout 经验生成和细化的层级指南。

### 6.1 检索

CoSkill 支持两种检索设置：

| 模式 | 方法 | 特点 |
| --- | --- | --- |
| template | 由任务文本识别任务族，取对应技能 | 无额外 embedding 模型，稳定且低延迟 |
| embedding | 按任务描述与技能文本的语义相似度排序 | 可跨任务族检索，但需要 embedding 模型和额外资源 |

每个 episode 只注入有限数量的 general skills；task-specific skills 和 tree 根据配置加入。这样端侧看到的是经过筛选的经验，而不是整个轨迹数据库。

### 6.2 生命周期

动态技能带有使用次数、使用时成功率、稳定周期、层级、弃用和内化状态。当前层次为：

| 层 | 含义 | 当前冻结模型实验中的解释 |
| --- | --- | --- |
| L0 | 新产生、需要验证的热技能 | 可立即注入端侧 prompt |
| L1 | 多周期稳定的技能 | 仍可作为外部技能使用 |
| L2 | 长期稳定或受保护的技能 | 在 no-RL 中仍是外部状态；不等同于已写入权重 |
| deprecated | 长期低效技能 | 从检索候选中移除 |
| internalized | 已由其他权重训练流程固化的标记 | 当前冻结 CoSkill 实验通常为 0 |

初始静态 skills 在当前库中作为受保护的基础知识处理；动态 patch 先进入 L0。每次云端周期后，系统依据稳定周期、使用成功率与最小调用次数推进或降级。必须注意：当前 no-RL 实验不执行技能到参数的实际固化，因此不应把 L2 结果描述为模型权重内化。

## 7. 完整运行流程

以下是一个正式两 GPU、72 局/group 的典型过程。

1. **初始化**
   加载端侧模型、初始 skills JSON、空的任务族 tree 状态、轨迹池和云端配置。每个 rollout worker 绑定一张 GPU。

2. **构造 rollout group**
   ALFWorld 从游戏池采样；WebShop 从训练 goal 池构造任务。每个基础任务复制 group_size 次，以提供同一任务下的多条采样轨迹。

3. **端侧交互**
   每步按任务、观测、可行动作、历史和检索技能构造 prompt。端侧生成 think/action，解析后交给环境。记录结果、有效动作口径、执行来源和技能 ID。

4. **组内状态固定**
   同一 group 使用同一份 skill_lib 快照。这样某一条 rollout 的结果不会在同组内立即改变其他并行 rollout 的技能条件。

5. **主进程汇总**
   worker 返回 episode 结果；主进程写入原始轨迹、更新任务族成功统计，记录技能被使用后的结果。

6. **轨迹池判断**
   主进程检查性能、趋势和容量条件。未触发时直接保存必要的指标并开始下一 group。

7. **云端周期（若触发）**
   导出压缩批次；云端先做对比蒸馏，再做失败诊断和任务族 tree 的 keep/refine/rewrite。主进程将 patch 和 tree 写回 skill_lib，推进生命周期，并保存新版本。

8. **下一个 group**
   worker 重新加载最新 skill_lib 快照。新动态技能和对应任务族 tree 从此时开始影响端侧 prompt。

9. **结束与冻结评测**
   训练/演化完成后保存 skills_latest_final。最终评测加载该冻结快照，在未参与更新的 held-out 实例上运行，禁止云端继续写回技能库。

## 8. ALFWorld 与 WebShop 的适配

### 8.1 共同部分

两套数据集共用端云闭环、轨迹池、对比蒸馏、失败诊断、技能树、层级技能库、云端 token 记录和双 valid-action 观测。共同使用同一类 think/action 输出协议，使输出诊断可比较。

### 8.2 ALFWorld

ALFWorld 是长时序家庭操作环境。端侧需在可行动作集合中选择精确文本命令，因此可以衡量模型是否直接命中 admissible action；系统另外记录 salvage 与 fallback，避免执行保护被误认为模型直接能力。

任务族包括拾取放置、双物体、光照观察、清洗、加热和冷却。tree 重点学习对象搜索覆盖、容器前置条件、状态转换顺序、数量跟踪和终止判断。

### 8.3 WebShop

WebShop 是带商品属性和购买流程的交互环境。端侧在搜索、结果浏览、商品页面、选项选择和购买步骤中行动。任务严格成功要求最终购买满足全部约束；连续 task score 仅描述部分匹配。

WebShop 的 skill tree 和 bullet skill 不应继承 ALFWorld 的房间、容器、操作命令等规则。云端 prompt 含环境契约，要求只从 WebShop 可见页面、动作与评分中归纳策略。

## 9. 论文实验与结论边界

### 9.1 必须区分的结果

论文中至少区分：

- 演化阶段的在线 rollout 表现；
- 训练结束后冻结 skill_lib 的 held-out 评测；
- 无 CoSkill、bullet only、tree only、full CoSkill 的消融；
- 成功率、任务长度、宽松/严格有效动作率；
- 云端 token、调用次数、延迟；
- 每个任务族的结果，而不只给总体均值。

在线演化过程中，不同 group 的任务组成可能不同，且技能状态不断变化。因此首组到后续组的上升只能作为学习过程迹象；证明泛化需使用固定未见实例和冻结技能库。

### 9.2 公平性要求

所有对照组应固定：

- 端侧模型与模型版本；
- 环境接口，包括 ALFWorld admissible actions；
- 起始 bullet skills；
- 任务/goal 清单、随机种子、group 规模、最大步数；
- prompt/response token 预算、温度、history length；
- 云端模型、更新触发策略和最大 patch 数（若比较云端臂）；
- 评测时的冻结 skill_lib。

如果所有组不共享 admissible actions、初始 skills 或任务清单，得到的差异不能仅归因于 CoSkill 架构。

### 9.3 非作弊边界

CoSkill 可以使用当前 observation、任务文本、可行动作、历史、自己的成功/失败 rollout 和已形成技能；这些属于交互过程中的可见信息。

下列内容不得输入端侧或云端：测试任务答案、游戏 PDDL、专家计划、隐藏对象位置、预先提供的正确动作序列、由 held-out 测试轨迹产生的技能更新。调试目录中若存在读取游戏文件的工具，必须与论文主链路隔离，且不用于报告结果。

## 10. 论文写作建议

论文方法部分可按如下结构组织：

1. **问题定义**：冻结小模型在长时序交互中如何利用外部经验持续适应；
2. **系统概览**：端侧执行器、轨迹池、云端分析器、分层技能库四部分；
3. **端侧策略条件化**：任务、观测、历史、技能树与 bullet skills 如何构成决策上下文；
4. **轨迹压缩与触发**：循环过滤、差分、成功共识前缀、性能/趋势/容量水位线；
5. **云端归纳**：成功-失败对比、失败诊断、patch 与任务族 tree；
6. **技能调度**：检索、使用归因、生命周期和版本化；
7. **实验协议**：数据集适配、冻结 held-out 评测、消融、成本与有效动作指标；
8. **局限性**：云端成本、任务分布依赖、外部技能而非权重内化、在线组间结果不等于泛化。

一张方法图建议画成四个模块的闭环：边缘端侧执行产生 RawTrace；轨迹池压缩和触发；云端分析返回 patch/tree；分层技能库向端侧提供检索上下文。图中应明确云端只收到压缩 rollout，而非环境 oracle。

## 11. 需要报告的可审计证据

为使论文结果可复核，建议附录或开源材料包含：

- 最终技能库及每个 tree 的版本历史；
- 每个云端周期的诊断、patch 和 token 统计；
- 每类任务成功/失败轨迹样例；
- 宽松与严格 valid-action 指标及 execution source 分布；
- 固定 held-out manifest 或 WebShop goal-id 列表；
- 各消融臂的完整配置与 Git commit；
- 端侧 rollout 时间、云端时间和总成本；
- 没有将 held-out 轨迹回写技能库的运行日志或配置证据。

这些材料将系统的性能结果、学习过程与外部调用成本分开呈现，避免只以最终成功率掩盖协议、成本或评测泄漏问题。
