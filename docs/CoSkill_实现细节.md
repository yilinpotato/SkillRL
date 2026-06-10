# CoSkill 实现细节文档（Implementation Details）

> 本文档记录 CoSkill 闭环在代码中**实际实现**的细节，配合 `CoSkill_架构设计.md`（设计蓝图）阅读。
> 重点回答：每个组件实现到什么程度、数据在组件间长什么样、前缀树如何喂给模型、
> 注入给执行模型的技能文本是什么样式。

更新日期：2026-06-10。对应分支 `Co-Skill`。

---

## 0. 实现状态总览

| 组件 / 里程碑 | 设计章节 | 状态 | 代码位置 |
|---|---|---|---|
| ④ 轨迹池 TracesPool | §3 | ✅ 已实现 | `agent_system/memory/traces_pool.py` |
| ① 云端分析器 CloudAnalyzer | §5 | ✅ 已实现（单桶，未并行） | `agent_system/memory/cloud_analyzer.py` |
| ② 层次化 Skill Lib | §4 | ✅ 已实现 | `agent_system/memory/hierarchical_skill_lib.py` |
| ③ 端侧执行器 — 轨迹回流 | §6.2 | ✅ 已实现 | `verl/trainer/ppo/ray_trainer.py` |
| ③ 端侧执行器 — 结构化技能注入 | §6.1 | ✅ 已实现（本次补齐） | `agent_system/memory/skills_only_memory.py` |
| 闭环触发接线 | §7 | ✅ 已实现 | `verl/trainer/ppo/ray_trainer.py` |
| 训练指标 metrics.jsonl | — | ✅ 已实现 | `verl/utils/tracking.py` |
| ③ Skill2param 闲时 RL 固化 | §6.3 / M4 | ⛔ 未实现（接口预留，开关默认关） | — |
| L1 → 轻量级 LoRA 编译下发 | §4.1 / §12 | ⛔ 未实现（本期以 Context 形态） | — |
| 云端批量并行蒸馏 | §5.3 | ⛔ 未实现（当前单批顺序） | — |
| 更细 reward 归因 | §12 | ⛔ 未实现（当前共享归因） | — |

> ✅ = 代码已落地并通过离线测试；⛔ = 有意未做，原因见各节与设计文档 §11/§12。

---

## 1. 端到端数据流（一次水位线触发的完整生命）

```
端侧 rollout（每个训练 step 的 batch）
   │  ray_trainer._coskill_ingest_batch_to_pool(batch)
   │    · tokenizer 解码 prompts/responses
   │    · _parse_conversation_to_steps → steps[{observation, action}]
   │    · token_level_scores>0 判定 success/failure
   ▼
RawTrace（§2.1） ── TracesPool.add_trace ──▶ 即时压缩
   │    · _filter_loops   死循环过滤
   │    · _diff_compress  状态差分 → obs_delta
   │    · 落盘 traces_pool/raw_traces.jsonl
   ▼
同时：HierarchicalSkillLib.record_usage（共享归因，回写 call_count / success_when_used）
   ▼
TracesPool.should_trigger() ── 双轨水位线 ──▶ (fire, reason)
   │  fire=True
   ▼
TracesPool.export_batch(reason)
   │    · 聚合 success_samples / failure_samples
   │    · _merge_prefix_tree 建前缀树
   │    · 落盘 traces_pool/batch_<ts>_<id>.json
   ▼
CompressedBatch（§2.2）── CloudAnalyzer.contrastive_distill ──▶ DeepSeek V4 Flash
   │    · _build_contrastive_prompt（成功/失败/前缀树分叉点）
   │    · _parse_patches + _normalize_patches
   │    · 落盘 cloud_io/patches_<id>.json
   ▼
SkillPatch[]（§2.3）── HierarchicalSkillLib.ingest_patches ──▶ 落 L0
   ▼
HierarchicalSkillLib.advance_lifecycle ── 晋升/降级 ──▶ L0/L1/L2 迁移
   │    · 落盘 skill_lib/skills_step<N>.json
   ▼
下一轮 rollout：retrieve() 注入新技能（format_for_prompt 含 Do/Avoid）
```

所有产物统一落到 `OUTPUT_DIR/{traces_pool,cloud_io,skill_lib}/` 与 `OUTPUT_DIR/metrics.jsonl`。

---

## 2. 轨迹池 TracesPool 实现细节

`agent_system/memory/traces_pool.py`

### 2.1 状态差分压缩（_diff_compress / _line_delta）

逐行集合差分：把相邻两步 observation 按行切分，只保留**新增行 `+`** 与**消失行 `-`**，
完全相同则记 `(no change)`。首步全部视为新增。

输入（RawTrace 原始两步 observation）：
```
step0: "room\nfridge 1 is closed"
step1: "fridge 1 is closed"          # "room" 这行消失了
```
输出（DiffTrace.steps）：
```jsonc
{"action": "go to fridge 1",    "obs_delta": "+room | +fridge 1 is closed", "reward": 0}
{"action": "examine fridge 1",  "obs_delta": "-room",                       "reward": 0}
```
效果：环境全文不再每步重复，只剩"这一步改变了什么"，云端输入 token 大幅下降。

### 2.2 死循环过滤（_filter_loops）

同一 action **连续重复** 且 **obs 无变化** 且 **reward≤0** 时，从第 `loop_threshold` 次起丢弃。
返回 `(cleaned_steps, dropped_count)`，`dropped_count` 写入 DiffTrace.`dropped_loops` 供云端知情。

上例失败轨迹连做 3 次 `examine fridge 1`，`loop_threshold=2` 时丢弃 2 步，
DiffTrace 里只剩 `dropped_loops: 2` 的标记。

### 2.3 前缀树合并（_merge_prefix_tree）—— 喂给云端的核心结构

把同批多条轨迹按 **action 序列** 合并成一棵树。每个节点记录该动作被走过多少次、
其中多少条最终成功 / 失败。**分叉点（children > 1）即"决策分歧点"**。

真实输出（1 条成功 + 2 条失败，三条都以 `go to fridge 1` 开头）：

```jsonc
{
  "action": "<root>", "count": 0, "n_success": 0, "n_failure": 0,
  "children": {
    "go to fridge 1": {                         // 公共前缀，3 条都走
      "action": "go to fridge 1",
      "count": 3, "n_success": 1, "n_failure": 2,
      "children": {
        "open fridge 1": {                      // ← 分叉：成功侧
          "n_success": 1, "n_failure": 0,
          "children": { "take mug 1 from fridge 1": { "n_success": 1, ... } }
        },
        "examine fridge 1": {                    // ← 分叉：失败侧
          "n_success": 0, "n_failure": 2, "children": {}
        }
      }
    }
  }
}
```

一眼可见：在 `go to fridge 1` 之后，选 `open` 的全成功、选 `examine` 的全失败 → 决策分歧点。

### 2.4 双轨水位线（should_trigger）

- **表现轨**（优先）：任一 task_type 近期失败率 ≥ `perf_watermark` 且近期样本数 ≥ `min_samples`。
  近期失败率用按 task_type 的滑动窗口 `_recent_outcomes`（`recent_window` 默认 20）计算。
- **容量轨**：累计压缩后估算 token 数 ≥ `capacity_watermark`（token 估算 ≈ 字符数/4）。

返回 `(True, "performance_watermark" | "capacity_watermark")` 或 `(False, None)`。

### 2.5 落盘

- `traces_pool/raw_traces.jsonl`：每条 RawTrace 追加一行（人工查看用，不参与触发）。
- `traces_pool/batch_<时间戳>_<batch_id前8位>.json`：每次 export 的完整 CompressedBatch。

---

## 3. 云端分析器 CloudAnalyzer 实现细节

`agent_system/memory/cloud_analyzer.py`

### 3.1 前缀树如何喂给模型（_format_forks）

前缀树 JSON 不直接塞进 prompt（太冗长），而是先抽取**分叉节点**压成一行行自然语言。
深度优先遍历，遇到 `children > 1` 的节点就输出一条，最多 `max_forks`（默认 6）条：

真实输出（喂进 prompt 的 `DECISION FORKS` 段）：
```
After [go to fridge 1]: 'open fridge 1' (succ=1,fail=0); 'examine fridge 1' (succ=0,fail=2)
```

格式：`After [到分叉点的动作路径]: '分支动作' (succ=N,fail=M); ...`。
模型据此直接看到"哪一步、选哪个动作导致成功 vs 失败"。

### 3.2 对比蒸馏 prompt 结构（_build_contrastive_prompt）

喂给 DeepSeek V4 Flash 的 prompt 含五段：
1. `SUCCESSFUL TRAJECTORIES` —— 成功 DiffTrace（action + obs_delta，最多 5 条）
2. `FAILED TRAJECTORIES` —— 失败 DiffTrace（最多 6 条，含 `dropped N looping actions` 提示）
3. `DECISION FORKS` —— §3.1 的分叉点文本
4. `TASK — Contrastive Analysis` —— 三步指令：对比→归因→抽象，并附去重提示（现有技能标题）
5. 输出 schema + 一个 few-shot 例子

### 3.3 输出补丁规范化（_normalize_patches）

模型返回 JSON 数组后：
- **重分配 dyn_ ID**：`_next_dyn_index` 扫描现有 `dyn_NNN` 取最大值+1，杜绝 ID 冲突。
- **补兼容字段**：若模型没给 `principle`，自动用 `trigger + "Steps: " + action_flow` 拼出，
  保证旧检索/格式化逻辑可消费。`when_to_apply` 缺省用 `trigger`。
- **附 evidence**：`{from_success, from_failure}` 记录该补丁基于多少正负样本。

最终落盘 `cloud_io/patches_<batch_id前8位>.json`。

### 3.4 后端与容错

- 后端经 `SKILL_UPDATER_BACKEND` 选择，本期 `deepseek` + `DEEPSEEK_MODEL=deepseek-v4-flash`。
- `openai` 客户端在 `__init__` 内延迟导入；缺 `DEEPSEEK_API_KEY` 会抛错。
- **训练侧已容错**：`ray_trainer` 构造 CloudAnalyzer 失败时打印告警、跳过本轮蒸馏、
  **不中断训练**（压缩 batch 已落盘，可事后补蒸馏）。

---

## 4. 层次化 Skill Lib 实现细节

`agent_system/memory/hierarchical_skill_lib.py`（继承 `SkillsOnlyMemory`）

### 4.1 lifecycle 元数据

每个技能挂一个 `lifecycle`（`_ensure_lifecycle` 保证缺省安全，旧技能默认 L0）：

```jsonc
"lifecycle": {
  "layer": "L0",              // L0 极热 / L1 温 / L2 极冷
  "created_cycle": 0,
  "last_modified_cycle": 0,
  "stable_cycles": 0,         // 连续未被修改的更新周期数
  "call_count": 0,            // 被检索注入次数
  "success_when_used": 0,     // 注入且 episode 成功的次数
  "success_rate": null,       // success_when_used / call_count
  "internalized": false,      // 是否已 Skill2param 固化（固化后热路径不注入）
  "task_types_seen": []       // 出现过的 task_type 集合（L1→L2 跨域判据）
}
```

### 4.2 接收补丁（ingest_patches）

按 `scope` 落库：`task_specific` + 有 `task_type` → 进 `task_specific_skills[task_type]`；
否则进 `general_skills`。新补丁一律 `layer=L0`，记 `created_cycle`，并失效 embedding 缓存。

### 4.3 使用统计（record_usage）—— 共享归因

本期实现的是**共享归因**：一个 episode 注入的所有技能，episode 成功则各自 `success_when_used += 1`、
`call_count += 1`。`ray_trainer` 在收集每条轨迹时，对该任务重新 `retrieve` 得到 `injected_skill_ids`
再回写。更细的 per-skill credit assignment 列在设计文档 §12 TODO。

### 4.4 生命周期推进（advance_lifecycle）

每个更新周期 `cycle += 1`，逐技能评估：
- 本周期被修改的技能 `stable_cycles=0`；否则 `stable_cycles += 1`。
- **降级**：`success_rate < demote_threshold` 且 `call_count >= min_calls` → 降一层；L0 仍低效 → `deprecated`。
- **L0→L1**：`stable_cycles >= stable_cycles_l1` 且 `success_rate >= success_l1` 且 `call_count >= min_calls`。
- **L1→L2**：`stable_cycles >= stable_cycles_l2` 且 `len(task_types_seen) >= min_task_types_l2`。

返回事件摘要 `{to_l1, to_l2, demoted, deprecated}` 并打印。

### 4.5 热路径检索（retrieve）—— 过量请求 + 分层过滤

为避免"先取 top_k 再过滤"把结果数打到 top_k 以下：先数出被过滤（`internalized` / `deprecated`）
的技能数 `n_filtered`，向父类请求 `top_k + n_filtered`，过滤后再截断回 `top_k`。
返回结果额外带 `injected_skill_ids`，供 `record_usage` 回写。

### 4.6 冷路径（get_cold_skills / mark_internalized）

`get_cold_skills` 返回 L2 且未固化的技能（待 Skill2param 队列）；`mark_internalized` 标记固化完成。
本期 M4 未启用，这两个接口预留给后续算力到位时联调。

---

## 5. 端侧执行器：注入给模型的技能样式（本次补齐的关键）

`agent_system/memory/skills_only_memory.py::format_for_prompt` + `_format_skill_lines`

### 5.1 问题与修复

云端补丁最有价值的是 `action_flow`（核心动作流）与 `avoid`（规避清单），但旧 `format_for_prompt`
只渲染 `principle / when_to_apply`，导致结构化字段**进了检索向量却没进 Prompt**。本次新增
`_format_skill_lines`，把这两个字段显式渲染为 `Do:` / `Avoid:` 行。

### 5.2 真实注入样式（template 模式，含一条结构化补丁）

```
### General Principles
- **Explore**: search once

### Clean Skills
- **Open Before Search**: open then search
  Do: locate containers → open each closed one → then search
  Avoid: examine closed container repeatedly; open same container twice
```

- `Do:` 行由 `action_flow` 用 `→` 连接，给模型清晰的动作序列。
- `Avoid:` 行由 `avoid` 用 `;` 连接，明确规避项。
- 旧技能（只有 principle/when_to_apply）渲染不变 —— `_format_skill_lines` 向后兼容：
  无结构化字段时退回 `- **title**: principle` + `_Apply when: ..._`。

### 5.3 检索向量也含结构化字段（_skill_to_text）

embedding 检索时 `_skill_to_text` 已把 `trigger / action_flow(Steps:) / avoid(Avoid:)` 一并编码，
新补丁的语义匹配以全字段为准，而非只靠 title。

### 5.4 embedding 检索的工程细节

- **强制 CPU**：`SentenceTransformer(..., device=os.environ.get("SKILL_EMBED_DEVICE","cpu"))`，
  避免检索模型抢占训练卡显存（曾导致 vLLM 加载 OOM）。
- **增量缓存**：`_skill_vec_store` 按 `(skill_id, text_hash)` memo 每条技能向量，
  `ingest_patches` 后重建只编码新增/改动的技能，其余复用 —— 即便用重型编码器也廉价。

---

## 6. 闭环触发接线（ray_trainer）

`verl/trainer/ppo/ray_trainer.py`

- `_coskill_ingest_batch_to_pool(batch)`：解码 batch → RawTrace → `TracesPool.add_trace`；
  同时对每条轨迹 `retrieve` 得 `injected_skill_ids` 并 `record_usage`（共享归因）。
- `_update_skills_coskill(batch)`：收集 → `should_trigger` → 命中则 `export_batch` →
  `CloudAnalyzer.contrastive_distill` → `ingest_patches` → `advance_lifecycle` →
  落盘 `skill_lib/skills_step<N>.json`。
- `fit()` 循环按 `+env.skills_only_memory.enable_coskill` 在 CoSkill 路径与旧
  `_update_skills_from_training` 之间切换；触发频率沿用 `skill_update_freq`（默认 `test_freq`）。
- 兼容性：`enable_coskill=False` 或 `enable_hierarchy=False` 时，系统退化为原 SkillRL 行为。

---

## 7. 训练指标 metrics.jsonl

`verl/utils/tracking.py` 新增 `jsonl` 后端（加入 `supported_backend`）。

- 每个训练 step 追加一行 JSON 到 `OUTPUT_DIR/metrics.jsonl`（`JSONL_METRICS_DIR` 环境变量指定目录）。
- numpy / tensor 标量自动 `.item()` 转 float，非标量丢弃。
- 启用方式：`trainer.logger=['console','jsonl']`（run 脚本已配），与 console 并存。

样式（每行一个 step）：
```jsonl
{"step": 1, "train/loss": 0.5, "val/clean_success_rate": 0.30}
{"step": 2, "train/loss": 0.4, "val/clean_success_rate": 0.34}
```

---

## 8. 未实现部分（与设计文档对齐）

| 项 | 现状 | 触发条件 / 说明 |
|---|---|---|
| Skill2param 闲时 RL 固化（M4） | 接口预留，`enable_internalize` 默认 False | 算力到位后联调（设计 §11.7） |
| L1 → 轻量级 LoRA 编译下发 | 本期以加权/优先 Context 形态 | 设计 §12 TODO |
| 云端批量并行蒸馏（§5.3） | 当前单批顺序调用 | 多 task_type 分桶并行，设计 §5.3 |
| 更细 reward 归因 | 当前共享归因 | per-skill credit assignment，设计 §12 |

---

## 9. 如何验证（无 GPU 也可跑）

1. **TracesPool 单测**：构造成功/失败 RawTrace，验证 `obs_delta`、`dropped_loops`、前缀树分叉、水位线触发。
2. **CloudAnalyzer 无网测试**：`_format_forks`、`_parse_patches`、`_normalize_patches` 不需网络即可验证。
3. **HierarchicalSkillLib 生命周期**：多周期 `record_usage` + `advance_lifecycle`，观察 L0→L1 晋升与 `layer_counts`。
4. **format_for_prompt**：喂一条带 `action_flow/avoid` 的技能，确认输出含 `Do:` / `Avoid:` 行（§5.2）。
5. **真实 rollout**：跑训练，检查 `OUTPUT_DIR/{traces_pool,cloud_io,skill_lib}/` 产物与 `metrics.jsonl`。

---

## 10. 可观测性 / Debug 配置

为便于检查问题、查效率、debug，闭环运行时产出三类可观测信息，全部默认开启且零侵入
（`coskill_debug` 除外，默认关）。

### 10.1 指标接入 metrics.jsonl

`ray_trainer.fit()` 每个训练 step 把 `_coskill_metrics()` 的结果并入 `metrics.jsonl`
（需 `enable_coskill=True`）。可直接画曲线：

| 指标键 | 含义 |
|---|---|
| `coskill/pool/total_added` | 轨迹池累计收集的轨迹数 |
| `coskill/pool/pending_tokens` | 自上次导出以来累计的压缩 token（逼近容量水位线） |
| `coskill/pool/total_dropped_loops` | 累计被死循环过滤丢弃的动作数 |
| `coskill/pool/n_task_types` | 当前池中 task_type 数 |
| `coskill/skilllib/{L0,L1,L2,internalized,deprecated}` | 各层技能数（看分层演化） |
| `coskill/cloud/total_updates` | 云端蒸馏触发总次数 |
| `coskill/cloud/total_patches` | 云端累计产出补丁数 |
| `coskill/cloud/large_model_total_tokens` | 云端累计 token 消耗（看成本） |
| `coskill/timing/export_seconds` | 上次压缩导出耗时 |
| `coskill/timing/distill_seconds` | 上次云端调用耗时（看云端慢不慢） |
| `coskill/last_trigger_reason` | 0=未触发 / 1=容量水位线 / 2=表现水位线 |

### 10.2 健康快照 coskill_status.json

每次触发更新后覆盖式写 `OUTPUT_DIR/coskill_status.json`，一个文件看全貌：
pool stats + skill_lib 分层计数 + cloud 摘要 + timing + global_step。人工排查首选。

### 10.3 调试开关 coskill_debug

`+env.skills_only_memory.coskill_debug=True`（默认 False）。开启时：
- 每个 step 打印 `[CoSkill][dbg] step=N pool={...} trigger=(fire,reason)`，
  实时看池子涨势与是否触发。
- 触发蒸馏后把每条补丁的 `skill_id/title/scope/trigger/action_flow/avoid`
  dump 到 `cloud_io/patches_step<N>_debug.json`，核对云端到底产出了什么。

关闭时完全静默，不影响正常训练。

### 10.4 排查路径速查

| 想查什么 | 看哪里 |
|---|---|
| 轨迹有没有正常收集、压缩比 | `traces_pool/raw_traces.jsonl` + `metrics.jsonl` 的 pool 指标 |
| 水位线为什么没触发 | `coskill_debug=True` 的 `[dbg]` 行（看 pending_tokens / 失败率） |
| 云端到底产出了什么补丁 | `cloud_io/patches_<id>.json`（含 debug dump） |
| 技能分层有没有演化 | `metrics.jsonl` 的 `coskill/skilllib/*` 曲线 + `skill_lib/skills_step<N>.json` |
| 云端慢 / 贵 | `coskill/timing/distill_seconds` + `coskill/cloud/large_model_total_tokens` |
| 整体健康度快照 | `coskill_status.json` |
