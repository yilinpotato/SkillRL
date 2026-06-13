# CoSkill 架构设计：云端分析 + 端侧执行的解耦式智能体内存与技能系统

> 本文档描述在现有 SkillRL（基于 verl-agent / GRPO）之上，构建一个**完整闭环**的设计方案：
> 云端冻结大模型负责经验蒸馏，端侧可调小模型负责执行，中间由层次化 Skill Lib 调度，
> 轨迹池负责收集、压缩与触发。本文档先给出完整框架，再给出落地到现有代码的实现路径。
>
> **实现状态（2026-06 更新）**：M1~M4 全部已落地并通过语法/逻辑自检——轨迹池、
> 云端正负对比蒸馏、层次化 Skill Lib、**以及 Skill2param 闲时 RL 固化**。原文档曾把
> M4 标为"暂不启用"，现已实现完整通路（见 §6.3、§13）。本轮还做了若干质量修复：
> 原始逐 episode 轨迹 dump、自适应差分压缩、action/think 解析分离、初始技能 L2+protected、
> 云端技能精简化、技能增删的证据门槛提高。详见 §13「本轮实现增量」。

---

## 0. 现状与目标

### 0.1 当前已实现（SkillRL 基线）

| 已有能力 | 对应代码 |
|---|---|
| 扁平技能库（general / task_specific / common_mistakes），模板 + embedding 检索 | `agent_system/memory/skills_only_memory.py` |
| 失败轨迹 → 新技能（DeepSeek / Azure LLM） | `agent_system/memory/skill_updater.py` |
| 训练 / 验证阶段触发技能更新，写回 env 内存并落盘 | `verl/trainer/ppo/ray_trainer.py`（`_update_skills_from_validation/training`、`_collect_failed_trajectories`） |
| 离线对比蒸馏生成初始技能库（成功 + 失败） | `skill_generation/alfworld.py` 等 |
| GRPO + LoRA 的端侧 RL | `examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh` |

### 0.2 闭环缺口（本设计要补齐的部分）

1. **轨迹记录池（Traces Pool）缺失**：当前只有 `_collect_failed_trajectories` 这种临时收集，没有
   独立的池、没有差分压缩（状态差分 / 前缀树合并 / 死循环过滤）、没有水位线触发机制。
2. **层次化 Skill Lib 缺失**：当前技能库是扁平的，没有 L0/L1/L2 冷热分层、没有生命周期管理、
   没有按更新频率与成功率的晋升/降级逻辑。
3. **云端分析器不完整**：当前 `SkillUpdater` 只做失败归因，没有真正的**正负对比蒸馏**输出
   结构化 Skill Patch（触发条件 / 核心动作流 / 规避清单），也没有批量并行蒸馏。
4. **Skill2param 内化缺失**：Global skills 没有在闲时通过 RL 固化进 SLM 参数（LoRA 固化）。

### 0.3 设计目标

- **低 Token 开销**：绝大多数动作走参数级"肌肉记忆"，不消耗 Prompt Token；热技能仅注入少量 Context。
- **强适应性**：遇到现有技能无法解决的困境时，异步唤醒云端蒸馏，快速下发新技能补丁。
- **解耦**：云端（高认知负荷、低频）与端侧（高频执行）解耦，执行周期绝不停机等待云端。

---

## 1. 四大核心组件

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CoSkill 闭环生态                                 │
│                                                                        │
│   ┌─────────────────┐   Skill Patches   ┌──────────────────────────┐  │
│   │ ① 云端分析器      │ ───────────────▶  │ ② 层次化 Skill Lib        │  │
│   │ Remote Frozen LLM│                   │   L0 极热 / L1 温 / L2 极冷 │  │
│   │ 正负对比蒸馏       │ ◀───────────────  │   冷热分层调度枢纽          │  │
│   └─────────────────┘  压缩差分轨迹包      └──────────────────────────┘  │
│           ▲                                    │ hot         │ cold     │
│           │ 上传(异步触发)                       │ skills      │ skills   │
│   ┌───────┴─────────┐                          ▼ (Context)   ▼ (RL)     │
│   │ ④ 轨迹记录池      │   raw traces      ┌──────────────────────────┐  │
│   │ Traces Pool      │ ◀───────────────  │ ③ 端侧执行器              │  │
│   │ 清洗/压缩/水位线   │                   │ Local Tunable SLM        │  │
│   └─────────────────┘ ───挂载 Context──▶ │ 执行 + 闲时 RL 固化        │  │
│                                          └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

| 模块 | 基本功能 | 输入 | 输出 |
|---|---|---|---|
| **① 云端分析器** (Remote Frozen LLM) | 高认知负荷，经验提取与并行蒸馏；从无序动作序列抽象出可泛化的结构化 Skill Patch | 预处理压缩后的轨迹 Batch（含典型成功与失败） | 结构化技能（触发条件 / 动作流 / 规避清单） |
| **② 层次化 Skill Lib** | 云端与端侧的多级语义缓存；技能全生命周期增删改查，按更新频率与成功率冷热分层 | 云端下发的 Skill Patch | 向端侧提供热技能（Context）+ 冷技能（用于 RL） |
| **③ 端侧执行器** (Local Tunable SLM) | 混合内化 + RL 的轻量执行引擎；按热技能与参数肌肉记忆生成动作，更新时通过 RL 固化 | 任务状态 State + 热技能 Context | 环境交互动作流 Actions + 新原始轨迹 |
| **④ 轨迹记录池** (Traces Pool) | 收集试错轨迹，本地清洗 + 启发式前置压缩，监控端侧状态决定何时唤醒云端 | SLM 试错产生的所有动作与状态（含噪声） | 清洗压缩后的差分轨迹包 |

---

## 2. 数据契约（Schema）

闭环中所有跨组件传递的数据都需要稳定 schema。以下为核心结构。

### 2.1 原始轨迹 RawTrace（端侧 → 轨迹池）

```jsonc
{
  "traj_uid": "uuid",
  "task": "put a clean mug in coffeemachine",
  "task_type": "clean",
  "outcome": "success | failure",
  "episode_reward": 1.0,
  "steps": [
    {"step": 0, "observation": "<full env text>", "action": "go to sinkbasin 1", "reward": 0.0}
  ],
  "meta": {"model_version": "lora_step_120", "skill_ids_used": ["gen_003", "dyn_007"]}
}
```

### 2.2 压缩差分轨迹包 CompressedBatch（轨迹池 → 云端）

```jsonc
{
  "batch_id": "uuid",
  "trigger_reason": "capacity_watermark | performance_watermark",
  "task_type": "clean",
  "success_samples": [ /* 见下 DiffTrace */ ],
  "failure_samples": [ /* 见下 DiffTrace */ ],
  "stats": {"n_success": 12, "n_failure": 30, "avg_success_rate": 0.28},
  "prefix_tree": { /* 多条轨迹公共前缀合并结果，见 §5.2 */ }
}
```

`DiffTrace`（单条压缩轨迹，只记录状态增量）：

```jsonc
{
  "traj_uid": "uuid",
  "outcome": "failure",
  "steps": [
    {"action": "go to fridge 1", "obs_delta": "+fridge 1 is closed"},
    {"action": "open fridge 1", "obs_delta": "+open fridge 1. you see a egg 1, a apple 2"}
  ],
  "dropped_loops": 3
}
```

### 2.3 技能补丁 SkillPatch（云端 → Skill Lib）

在现有 skill 字段（`skill_id/title/principle/when_to_apply`）基础上扩展为结构化补丁：

```jsonc
{
  "skill_id": "dyn_012",
  "title": "Open Receptacle Before Search",
  "scope": "general | task_specific",
  "task_type": "clean",
  "trigger": "环境条件：目标物体可能在封闭容器中且 admissible actions 含 open",
  "action_flow": ["定位候选容器", "对每个 closed 容器执行 open", "再检索目标物体"],
  "avoid": ["在未 open 容器时反复 examine", "对同一容器重复 open"],
  "principle": "...",       // 兼容旧字段
  "when_to_apply": "...",   // 兼容旧字段
  "evidence": {"from_success": 8, "from_failure": 21},
  "lifecycle": { /* 见 §4.1 */ }
}
```

> **兼容性**：`principle` 由 `trigger + action_flow` 拼接生成，保证旧的
> `format_for_prompt()` 与检索逻辑无需改动即可消费新补丁。

---

## 3. 组件 ④：轨迹记录池 (Traces Pool)

轨迹池是闭环的"心脏起搏器"——它在后台静默收集、压缩，并决定何时唤醒云端。

### 3.1 职责

1. **收集**：端侧 SLM 每个 episode 结束后，将 RawTrace 推入池（成功与失败都收）。
2. **流式清洗 + 前置压缩**（在收集时即时做，降低后续云端输入成本）：
   - **状态差分法**：只记录相邻 observation 的增量 `obs_delta`，去掉每步重复的环境全文。
   - **前缀树合并**：把同 task_type 的多条轨迹按动作序列建前缀树，合并公共起始动作。
   - **死循环过滤**：检测重复 action（同一 action 连续 N 次无 reward 变化 / 无 obs_delta）并丢弃。
3. **监控 + 触发**（异步双轨）：
   - **容量水位线**：累计高质量 token / 轨迹数达到阈值 → 打包上传。
   - **表现水位线**：某 task_type 连续失败率超阈值（现有技能无法解困）→ 立即打包上传。

### 3.2 异步双轨触发机制

执行周期**绝不停机等待**云端。触发上传后，端侧 SLM 继续用现有 Skill Lib 运行其他任务；
云端蒸馏在后台进行，完成后通过 Skill Lib 异步下发。

```
水位线监控（每 episode 后检查）:
  if pool.token_count >= CAPACITY_WATERMARK:        # 容量轨：积累足够经验
      trigger_update(reason="capacity_watermark")
  if pool.recent_failure_rate(task_type) >= PERF_WATERMARK and pool.has_min_samples():
      trigger_update(reason="performance_watermark") # 表现轨：陷入困境
  # trigger_update 是非阻塞的：打包 → 入云端队列 → 立即返回
```

### 3.3 数据结构与接口（新增 `agent_system/memory/traces_pool.py`）

```python
class TracesPool:
    def __init__(self, capacity_watermark=50_000, perf_watermark=0.6,
                 min_samples=8, loop_threshold=3): ...

    def add_trace(self, raw_trace: dict) -> None:
        """收集并即时压缩；更新水位线计数。"""

    def _diff_compress(self, steps: list) -> list:
        """状态差分：相邻 observation 求增量。"""

    def _filter_loops(self, steps: list) -> tuple[list, int]:
        """死循环过滤，返回 (cleaned_steps, dropped_count)。"""

    def _merge_prefix_tree(self, traces: list) -> dict:
        """前缀树合并公共动作前缀。"""

    def should_trigger(self) -> tuple[bool, str | None]:
        """返回 (是否触发, 原因)。"""

    def export_batch(self) -> dict:
        """导出 CompressedBatch，并清空已导出部分。"""
```

> **落地映射**：现有 `ray_trainer._collect_failed_trajectories` 与
> `_parse_conversation_to_steps` 的解析逻辑可直接复用为 `add_trace` 的输入适配层；
> 区别是 TracesPool 同时收成功样本（对比蒸馏需要），并做差分压缩与水位线判断。

---

## 4. 组件 ②：层次化 Skill Lib（调度枢纽）

这是整个系统避免高频 RL 开销的核心引擎。它是云端与端侧之间的**多级语义缓存**。

### 4.1 三层冷热结构

| 层 | 名称 | 内容 | 形态 | 下发方式 |
|---|---|---|---|---|
| **L0** | 极热缓冲层 | 新提炼的 Env/Task-specific skills | Context（文本） | 热更新，立即注入 Prompt |
| **L1** | 温数据演化层 | 多周期稳定、高通用、高成功率的技能 | 候选 LoRA / Context | 编译为轻量 LoRA 下发 |
| **L2** | 极冷标记层 | 长期多环境验证的基础规律（Global skills） | 待固化队列 | 闲时 RL 内化进参数 |

每个技能携带 `lifecycle` 元数据，驱动晋升 / 降级：

```jsonc
"lifecycle": {
  "layer": "L0",
  "created_step": 120,
  "last_modified_step": 120,
  "stable_cycles": 0,        // 连续未被修改的更新周期数
  "call_count": 0,           // 被检索注入次数
  "success_when_used": 0,    // 使用该技能时成功的次数
  "success_rate": null,      // success_when_used / call_count
  "promoted_from": null
}
```

### 4.2 晋升 / 降级规则

```
每个更新周期结束后，对每个技能评估：

L0 → L1 (温化)：
    stable_cycles >= STABLE_CYCLES_L1 (默认 3)
    且 success_rate >= SUCCESS_L1 (默认 0.7)
    且 call_count >= MIN_CALLS (默认 20)
    → 标记为 L1 候选，触发 LoRA 编译

L1 → L2 (晋升 Global)：
    在 L1 经历 >= STABLE_CYCLES_L2 (默认 5) 个周期仍稳定
    且跨 >= MIN_TASK_TYPES (默认 3) 个 task_type 均高成功率
    → 标记 promote_to_global，进入待固化队列

降级 (任意层 → 回炉/淘汰)：
    success_rate < DEMOTE_THRESHOLD (默认 0.3) 且 call_count 充分
    → 降级一层；若已在 L0 且持续低效 → 标记 deprecated，从注入中剔除
```

### 4.3 调度接口

- **对端侧执行（热路径）**：`get_hot_skills(task) -> Context`，只返回 L0 + 尚未固化的 L1 文本技能，
  数量受 `top_k` 限制 → 低 Token。**已固化进参数的 L2 技能不再注入**（避免重复消耗 Token）。
- **对端侧 RL（冷路径）**：`get_cold_skills() -> list`，返回 L2 待固化队列，供 Skill2param 内化。
- **接收云端补丁**：`ingest_patches(patches)`，按 scope 落到 L0，登记 lifecycle。
- **生命周期推进**：`advance_lifecycle(usage_stats)`，每周期更新计数并执行晋升/降级。

### 4.4 落地映射

`SkillsOnlyMemory` 扩展为 `HierarchicalSkillLib`（或在其内部增加 layer 维度）：
- 现有 `general_skills / task_specific_skills` 保留为存储后端；新增 `lifecycle` 字段（向后兼容，缺省视为 L0）。
- `retrieve()` 增加按 layer 过滤：热路径排除已固化技能。
- 新增 `usage tracking`：端侧每次用某技能并拿到 reward 时回写 `call_count / success_when_used`。
- `add_skills()` 升级为 `ingest_patches()`，写入 lifecycle 元数据。


---

## 5. 组件 ①：云端分析器（Remote Frozen LLM）

承担高认知负荷的离线归纳，把无序动作序列抽象为可泛化的结构化 Skill Patch。

### 5.1 正负对比蒸馏（Contrastive Distillation）

现有 `SkillUpdater.analyze_failures` 只看失败。云端分析器升级为**同时吃成功 + 失败**：

```
输入：CompressedBatch（success_samples + failure_samples + stats + prefix_tree）
推理：
  1. 对比分析：成功轨迹"做对了什么"vs 失败轨迹"在哪一步偏离"。
  2. 归因：失败的共性根因（如"未 open 容器就检索"）。
  3. 抽象：把"成功路径 - 失败路径"的差异提炼成结构化 SkillPatch。
输出：SkillPatch[]（trigger / action_flow / avoid 三段式，见 §2.3）
```

### 5.2 前缀树驱动的归因

轨迹池上传的 `prefix_tree` 把同 task_type 多条轨迹的公共动作前缀合并，分叉点即"决策分歧点"。
云端在分叉点对比"走向成功的分支"与"走向失败的分支"，归因更精准、Token 更省。

```
        [start]
           │ go to fridge 1
        [open?]
        ╱       ╲
   open fridge   examine fridge   ← 分叉点：成功侧 open，失败侧反复 examine
   (success x8)  (failure x21)    ← SkillPatch: "Open Receptacle Before Search"
```

### 5.3 批量并行蒸馏

按 task_type 分桶，多个桶并行调用云端 LLM（asyncio / 线程池），各自产出补丁后合并去重。
这是低频、可容忍延迟的离线作业，不阻塞端侧执行。

### 5.4 落地映射

在 `skill_updater.py` 基础上新增 `CloudAnalyzer`（或重构 `SkillUpdater`）：
- 复用现有 DeepSeek / Azure 客户端与 token 计费逻辑。
- `analyze_failures` → `contrastive_distill(compressed_batch)`，prompt 升级为正负对比模板，
  输出 §2.3 的结构化补丁；保留 `principle/when_to_apply` 拼接以兼容旧 `format_for_prompt`。
- 复用 `_reassign_dyn_ids` 保证 ID 不冲突。

---

## 6. 组件 ③：端侧执行器（Local Tunable SLM）

混合"内化（参数肌肉记忆）+ RL（固化）"的轻量执行引擎。这部分**基线已基本具备**
（Qwen3-4B + GRPO + LoRA），闭环只需补上两条数据通路。

### 6.1 执行（热路径，已具备）

每个任务从 Skill Lib 取热技能 Context（`get_hot_skills`），注入 prompt（现有 `format_for_prompt`），
SLM 生成动作流。Global skills 已固化进参数，不再注入 → 低 Token。

### 6.2 新增通路 A：轨迹回流

每个 episode 结束，把 RawTrace（含 skill_ids_used）推入 TracesPool。
现有 rollout_loop 已收集 `total_batch_list / episode_rewards / success / traj_uid`，
只需在 episode 收尾处增加一次 `traces_pool.add_trace(...)`。

### 6.3 新增通路 B：Skill2param 闲时 RL 固化

```
触发时机：更新周期、或系统判断 L2 待固化队列非空且端侧空闲时。
做法：
  1. 从 Skill Lib 取 cold_skills（L2 待固化队列）。
  2. 用这些技能构造/采样一批任务，正常跑 GRPO + LoRA（现有训练通路）。
  3. 训练目标：让 SLM 在【不注入这些技能 Context】的情况下也能复现技能行为
     → 技能从 Prompt "吃"进参数。
  4. 固化完成 → Skill Lib 把这些技能标记 internalized=True，热路径不再注入。
绝不在每步触发；只在 update phase 或显式空闲信号时启动，避免高频 RL 开销。
```

### 6.4 落地映射

- 通路 A：`rollout_loop.py` episode 收尾 + `ray_trainer` 已有的轨迹解析逻辑。
- 通路 B：复用现有 `examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh` 的 LoRA 训练，
  增加"固化模式"——构造 cold_skills 任务集、训练后回写 `internalized` 标记。


---

## 7. 两阶段闭环时序

### 7.1 阶段一：执行周期 (Execution Phase)

新经验数据的"孵化"阶段（生命周期中的"演化 Evolution"）。端侧主导，轨迹池后台静默收集。

```
for each task in stream:
    ctx   = skill_lib.get_hot_skills(task)        # L0+L1 文本技能，低 Token
    traj  = slm.run(task, ctx)                    # 参数肌肉记忆 + 热技能执行
    skill_lib.record_usage(traj.skill_ids, traj.reward)   # 回写使用统计
    traces_pool.add_trace(traj)                   # 流式压缩收集
    fire, reason = traces_pool.should_trigger()   # 双轨水位线
    if fire:
        cloud_queue.put(traces_pool.export_batch(), reason)  # 非阻塞，立即返回
    # 端侧继续下一个任务，绝不停机等待云端
```

### 7.2 阶段二：更新周期 (Update Phase)

系统的重构与认知升级（"提取 Abstraction" + "形态 Operation 固化"）。

```
1. 上传 + 云端对比提炼：
     batch = cloud_queue.get()
     patches = cloud_analyzer.contrastive_distill(batch)   # 正负对比 → SkillPatch[]

2. Skill Lib 多级冷热调度：
     skill_lib.ingest_patches(patches)        # 新补丁落 L0，立即热下发解困
     skill_lib.advance_lifecycle(usage_stats) # L0→L1→L2 晋升 / 降级
     # L1: 稳定+高通用 → 编译候选 LoRA
     # L2: 长期多环境验证 → 进入待固化队列

3. 端侧参数内化（闲时 RL 固化）：
     if skill_lib.has_cold_skills() and system_idle():
         cold = skill_lib.get_cold_skills()
         slm.internalize(cold)                 # GRPO+LoRA 把 Global skills 吃进参数
         skill_lib.mark_internalized(cold)     # 热路径不再注入 → 进一步降 Token

# 至此：探索 → 抽象 → 内化，完整技能生命周期闭环。
```

### 7.3 解耦保证

| 维度 | 云端（更新周期） | 端侧（执行周期） |
|---|---|---|
| 频率 | 低频，水位线触发 | 高频，每任务 |
| 延迟容忍 | 高（离线、可排队） | 低（实时执行） |
| 通信 | 异步队列，非阻塞 | 不等待云端，用现有 Skill Lib |
| 资源 | 大模型 API | 本地 SLM + LoRA |

---

## 8. 落地实现路径（分阶段）

按"风险递增、可独立验证"排序，每阶段可单独跑通再进下一步。

### M1. 轨迹记录池（最小闭环基础）
- 新增 `agent_system/memory/traces_pool.py`（§3.3 接口）。
- 复用 `ray_trainer._parse_conversation_to_steps / _detect_task_type_from_input` 做输入适配。
- 在 `rollout_loop.py` episode 收尾处接入 `add_trace`（成功 + 失败都收）。
- 实现状态差分、死循环过滤、前缀树合并、双轨水位线。
- **验证**：跑一次训练，确认池能正确收集、压缩并在阈值处触发 `export_batch`。

### M2. 云端对比蒸馏
- 在 `skill_updater.py` 基础上新增 `CloudAnalyzer.contrastive_distill(batch)`，
  prompt 升级为正负对比模板，输出 §2.3 结构化补丁。
- 复用现有 DeepSeek 客户端、token 计费、`_reassign_dyn_ids`。
- **验证**：喂入 M1 导出的 batch，确认产出含 trigger/action_flow/avoid 的补丁，且 `principle` 兼容旧 prompt。

### M3. 层次化 Skill Lib
- 扩展 `SkillsOnlyMemory` → 增加 `lifecycle` 字段 + layer 维度（向后兼容）。
- 实现 `ingest_patches / record_usage / advance_lifecycle / get_hot_skills / get_cold_skills`。
- `retrieve()` 热路径排除 `internalized=True`。
- 将 `ray_trainer` 现有 `_update_skills_from_*` 改为：导出 batch → CloudAnalyzer → ingest_patches → advance_lifecycle。
- **验证**：多周期训练后观察 lifecycle 计数、L0→L1→L2 晋升日志、被固化技能停止注入。

### M4. Skill2param 闲时固化
- 新增"固化模式"训练入口：取 cold_skills 构造任务集，GRPO+LoRA 训练后回写 `internalized`。
- 接入 update phase / 空闲信号触发。
- **验证**：固化后撤掉技能 Context，SLM 仍能复现该技能行为，且 Token 开销下降。

### M5. 全闭环联调
- 串起执行周期 ↔ 更新周期，异步队列解耦，端到端跑通。
- **验证**：观测成功率随周期上升、注入 Token 随固化下降、云端调用频率受水位线控制。

---

## 9. 关键配置项（`env.skills_only_memory.*` 扩展）

```
# --- 轨迹池 ---
+env.traces_pool.capacity_watermark=50000     # 容量水位线（token）
+env.traces_pool.perf_watermark=0.6           # 表现水位线（失败率）
+env.traces_pool.min_samples=16               # 触发表现轨的最小样本数（原 8，提高以稳）
+env.traces_pool.loop_threshold=3             # 死循环判定的重复次数

# --- 层次化 Skill Lib ---
+env.skills_only_memory.enable_hierarchy=True
+env.skills_only_memory.stable_cycles_l1=3    # L0→L1 稳定周期阈值
+env.skills_only_memory.stable_cycles_l2=5    # L1→L2 稳定周期阈值
+env.skills_only_memory.success_l1=0.7        # L1 成功率阈值
+env.skills_only_memory.demote_threshold=0.2  # 降级阈值（原 0.3，降低以少删技能）
+env.skills_only_memory.min_calls=40          # 淘汰/降级所需最小调用证据（原默认 20）

# --- 云端分析器（沿用现有） ---
+env.skills_only_memory.max_new_skills=3
# SKILL_UPDATER_BACKEND / DEEPSEEK_* 环境变量不变

# --- Skill2param 固化（M4，已实现） ---
+env.skills_only_memory.enable_internalize=True   # 开启闲时 RL 固化
+env.skills_only_memory.internalize_freq=10       # 每 N 步检查 cold skills 并固化
+env.skills_only_memory.internalize_max_episodes=8  # 单次固化最多用多少条成功 episode
+env.dump_raw_trajectories=True                   # 开启原始轨迹 dump（固化数据源）

# --- 输出目录（统一落到 OUTPUT_DIR，见 §11.2 与 §14） ---
# traces_pool/ skill_lib/ cloud_io/ raw_episodes/ 由代码基于 trainer.default_local_dir 自动创建
```

---

## 10. 与现有代码的兼容性原则

1. **SkillPatch 向后兼容**：始终生成 `principle / when_to_apply`，旧 `format_for_prompt()` 与
   embedding 检索无需改动。
2. **lifecycle 缺省安全**：无 lifecycle 字段的旧技能默认视为 L0、未固化，照常注入。
3. **渐进开关**：`enable_hierarchy / enable_internalize` 默认关闭时，系统退化为现有 SkillRL 行为。
4. **复用而非重写**：轨迹解析、ID 分配、LLM 客户端、LoRA 训练通路全部复用现有实现。

---

## 11. 本期落地决策（已拍板）

> 以下为本次实现采用的决策，区别于 §0.3 的长期设计目标。

1. **入口脚本**：在 `examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh` 上修改，不新建脚本。
2. **统一输出目录**：所有新产物都落到该脚本已设置的 `OUTPUT_DIR`（独立文件夹），便于查看且不污染原有数据：
   - 轨迹池：`OUTPUT_DIR/traces_pool/`
   - 技能库（含 lifecycle 的演化版本）：`OUTPUT_DIR/skill_lib/`
   - 云端导出的 CompressedBatch / 补丁：`OUTPUT_DIR/cloud_io/`
3. **轨迹池流式压缩**：后台进行——状态差分法（只记录环境变化增量 `obs_delta`）+ 前缀树合并（整合多条轨迹的公共起始动作）+ 死循环动作过滤，大幅降低云端输入成本。
4. **reward 归因**：本期用**共享归因**（episode 成功则该 episode 注入的所有技能 `success_when_used += 1`）。更细的 credit assignment 见 §12 TODO。
5. **检索用 LLM**：技能 embedding 检索默认与 SFT 模型一致（即 `MODEL_PATH`），不再单独指定 embedding 模型。
6. **云端模型**：使用 **DeepSeek V4 Flash**（`SKILL_UPDATER_BACKEND=deepseek`，`DEEPSEEK_MODEL=deepseek-v4-flash`）。
7. **Skill2param RL 固化（M4）已实现**：原计划因算力暂缓，现已落地完整通路（复用
   `update_actor` 做行为克隆，数据源为原始轨迹 dump）。默认开关 `enable_internalize`，
   详见 §13.1。算力紧张时可关闭。
8. **L1 编译轻量级 LoRA 下发：本期不做**，见 §12 TODO；本期 L1 仅以"加权/优先 Context"形态下发。
9. **云端队列形态**：单机进程内 queue 起步；落盘到 `OUTPUT_DIR/cloud_io/` 兼作断点续训与人工查看。

---

## 12. TODO（后续迭代）

- [ ] **更细的 reward 归因**：替换共享归因，做 per-skill credit assignment（如按技能在轨迹中实际命中的步贡献分摊，或反事实对比）。
- [ ] **L1 → 轻量级 LoRA 编译下发**：把多周期稳定的 L1 技能编译为一组独立轻量 LoRA 下发端侧，替代当前的加权 Context 形态。
- [x] **Skill2param 闲时 RL 固化（M4）联调**：已实现（§13.1）。复用 update_actor 做行为克隆，
  数据源为原始轨迹 dump。待大规模算力到位后做收敛性/Token 下降的端到端验证。
- [ ] **分布式云端队列**：单机 queue → Redis / 文件队列，支持多端侧并发上传。

---

## 13. 本轮实现增量（2026-06，相对原设计的落地与修正）

### 13.1 Skill2param 闲时 RL 固化（M4，已实现）

原文档 §11.7 标为"暂不启用"，现已实现完整通路：

- **复用 `update_actor` 而非新建 SFT trainer**：行为克隆 = advantage≡正常数的 PPO 特例。
  固化阶段构造特殊 batch（prompt=去技能注入的观测，response=成功轨迹动作，advantage=正），
  直接调 `self.actor_rollout_wg.update_actor(batch)`，自动复用 FSDP/offload/LoRA。
- **数据来源 = 原始轨迹 dump**（见 §13.2）：从 dump 里筛"used cold skill 且 won"的 episode。
- **触发**：fit 循环每 `internalize_freq` 步检查 `has_cold_skills()`，非空则固化。
- **收尾**：`mark_internalized()` 标记，热路径 `retrieve()` 不再注入该技能；落盘
  `skill_lib/skills_step{N}_internalized.json`。
- 代码：`ray_trainer._internalize_cold_skills / _load_internalize_episodes / _build_internalize_batch`。
- 开关：`enable_internalize`、`internalize_freq`、`internalize_max_episodes`、`internalize_adv`。

### 13.2 原始逐 episode 轨迹 dump（固化数据源）

`rollout_loop.py` 新增：每个 rollout 把每条 episode 的完整逐步轨迹各写一个 JSON。

- 路径：`OUTPUT_DIR/raw_episodes/rollout_<NNNN>/ep_<idx>_<uid>.json`
- 每步字段：`step/active/observation/raw_response/think/action/action_text/model_output/
  env_action/is_action_valid/reward/done/won/next_observation`，外加 `prompt_token_ids/
  response_token_ids`（供固化免重 tokenize）。
- 关键正确性：动作快照在 `envs.step()` 前取（projection 会就地改写）；`think/action`
  用 `<think>`/`<action>` 正则分离。
- 开关：`+env.dump_raw_trajectories=True`（默认关，关时零开销）。

### 13.3 自适应差分压缩

原 `_diff_compress` 无条件对所有观测求 `+/-` 增量，导致短观测也被拆成零散增量行、
大模型难读。现改为**只在划算时差分**：观测长度 ≥ `diff_min_obs_chars`(400) 且差分能省
≥ `diff_min_savings`(0.5) 才用增量，否则存完整观测原文，并以 `obs_is_full` 标记。
云端 prompt 标签相应区分 `obs:`(完整)/`delta:`(增量)。

### 13.4 action/think 解析分离（bug 修复）

`_parse_conversation_to_steps` 原先把 assistant **整段输出**（含 `<think>` 长思维链）塞进
`action` 字段，导致 traces 里 action 时而是大段 CoT、时而为空，污染对比蒸馏。新增
`_extract_action_from_response()`：优先取 `<action>` 内容，无标签则剥离 think 后取末行。
CoSkill ingest 与 legacy SkillUpdater 两条路径共用此修复。

### 13.5 技能稳定性强化

- **初始 gen_* 技能 = L2 + protected**：启动即进固化队列且永不 demote/deprecate
  （`hierarchical_skill_lib.__init__`）。
- **云端技能精简**：CloudAnalyzer prompt 改为输出与初始种子一致的 4 字段
  （`title/scope/principle/when_to_apply`），principle ≤30 词，去掉冗长的
  `action_flow/avoid` 数组——4B 小模型更易读。
- **提高增删证据门槛**：`min_samples` 8→16（触发云端更新需更多近期轨迹）、
  `min_calls` 默认 20→40（淘汰/降级需更多调用证据）、`demote_threshold` 0.3→0.2
  （成功率需更低才淘汰）。缓解技能 churn（此前 30 步废弃 18 条）。

---

## 14. Output 目录数据导览（怎么看）

所有产物落在 `OUTPUT_DIR`（默认 `outputs/verl_agent_alfworld/grpo_qwen3-4b_co_skill/`）。

| 文件/目录 | 内容 | 怎么看 |
|---|---|---|
| `metrics.jsonl` | 每 step 一行 JSON 的训练指标 | `tail -f`；认 `training/global_step`、`episode/success_rate`、`val/success_rate`、`actor/{entropy_loss,kl_loss,grad_norm}`、`timing_s/step`、`coskill/*` |
| `training.log` | 全量 stdout/stderr | step 卡住/报错看这里；jsonl 只在 step 结束才写 |
| `config.yaml` | 展开后的完整生效配置 | 核对实际参数（比 .sh 更准，override 已合并） |
| `coskill_status.json` | 最新一次 CoSkill 周期总览（覆盖写） | 快速看技能库健康：`skill_lib`(L0/L1/L2/internalized/deprecated)、`cloud`(更新次数/token)、`pool` |
| `skill_lib/skills_step{N}.json` | 第 N 步技能库快照 | 看每条技能 `lifecycle`(layer/call_count/success_rate)；对比不同 N 看演化 |
| `skill_lib/skills_step{N}_internalized.json` | 固化后的技能库快照 | 看哪些技能 `internalized=True`（已进参数、不再注入） |
| `cloud_io/patches_*.json` | 每次云端 DeepSeek 返回的技能补丁 | 看"某次更新具体增/改了什么技能" |
| `traces_pool/raw_traces.jsonl` | 每条原始轨迹一行（压缩前的解析结果） | 看 task/outcome/steps（action 已是分离后的纯动作） |
| `traces_pool/batch_*.json` | 触发更新时导出给云端的压缩批 | 看喂给蒸馏的 success/failure 样本 + 前缀树；`obs_is_full` 标记完整观测 |
| `raw_episodes/rollout_<N>/ep_*.json` | 逐 episode 完整原始轨迹（需开 dump_raw） | 看单条 episode 每步的 observation/think/action/reward/won；Skill2param 的数据源 |
| `latest_checkpointed_iteration.txt` | 最近 checkpoint 步数 | 续训定位 `global_step_{N}/` |

**典型排查顺序**：进度/动力学看 `metrics.jsonl` → 报错看 `training.log` → 技能为何变看
`coskill_status.json`(总览) + `cloud_io/patches_*`(具体改动) + `skill_lib/skills_step*`(演化)
→ 想读完整轨迹看 `raw_episodes/`。

**关键 coskill 指标**：
- `coskill/skilllib/{L0,L1,L2,internalized,deprecated}` — 各层技能数
- `coskill/cloud/{total_updates,total_patches,large_model_total_tokens}` — 云端调用与花费
- `coskill/internalize/{last_step,n_skills,n_samples,seconds}` — Skill2param 固化进度
- `coskill/pool/{total_added,...}` — 轨迹池状态
