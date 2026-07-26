# ALFWorld 技能树深度消融 V4

V4 用来比较原始扁平技能 L0 与同一棵技能树逐层扩展得到的 L1–L5。它不再进行在线生长，也不把各深度当成彼此独立生成的树。

## 固定协议

- 外部轨迹池按任务类型分层抽样，每类固定 12 条：6 条成功、6 条失败。
- L1–L5 的每一次云端生成都看到该任务相同的全部 12 条轨迹。
- 展示完整 step 和完整 observation，不折叠共识前缀，不使用轨迹压缩。
- observation 明确标为执行当前 action 前的状态；下一条 observation 明确标为上一 action 后、下一 action 前的状态。终止后状态未记录时明确禁止推断。
- L1 从证据生成；L2 只能在已验收 L1 中插入内容；依此扩展到 L5。父树的每个非空行及其顺序必须原样保留。
- 新增最深层标题必须给出真实 `traj_uid + step` 证据。引用不存在、缺少引用或违反 ALFWorld 通用动作契约时，自动携带验证错误重试。
- 没有标题节点数上限，也没有每任务树字符数上限。`--tree_max_completion_tokens` 只是单次 API 返回的技术上限，不是树内容的目标或截断阈值。
- 本地评测默认使用 `--local_max_model_len 16384`，在保留 4096-token 响应预算后可提供约 12288 个输入 token。任何 context guard 裁剪都会写入 metrics，并把对应臂标为 `N.A.`，禁止把缺失技能树的结果当成有效成绩。
- L0 是原始 SkillRL 扁平技能基线，不参与树扩展。
- L0–L5 共用同一个独立验证 manifest；默认每类 3 个游戏、每游戏 12 次 rollout，即每个臂 216 个 episode。

V4 保证深层树包含浅层树的全部信息并增加有证据的新辅助，但不人为保证成功率单调上升；成功率仍由独立验证决定。

## 单卡运行

```bash
cd /data2/myl/CoSkill
export DEEPSEEK_API_KEY='由环境注入，不要写入仓库'
export EXTERNAL_RAW_TRACES='/path/to/raw_traces.jsonl'
export CUDA_VISIBLE_DEVICES=1
export AB_ROOT="$PWD/skillrl_outputs/alfworld_skill_tree_depth_v4/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AB_ROOT"
nohup bash examples/playbook_evolve/run_alfworld_skill_tree_depth_v4_1xa800.sh \
  --phase all >"$AB_ROOT/run.log" 2>&1 &
```

## 双卡运行

`CUDA_VISIBLE_DEVICES` 填调度器实际分配的两张物理卡：

```bash
cd /data2/myl/CoSkill
export DEEPSEEK_API_KEY='由环境注入，不要写入仓库'
export EXTERNAL_RAW_TRACES='/path/to/raw_traces.jsonl'
export CUDA_VISIBLE_DEVICES=0,1
export AB_ROOT="$PWD/skillrl_outputs/alfworld_skill_tree_depth_v4/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AB_ROOT"
nohup bash examples/playbook_evolve/run_alfworld_skill_tree_depth_v4_2xa800.sh \
  --phase all >"$AB_ROOT/run.log" 2>&1 &
```

## 分阶段与续跑

同一个 `AB_ROOT` 可反复执行；已有且完整的产物会被复用。

```bash
# 只做证据选择、L0 和 L1–L5 逐层生成
bash examples/playbook_evolve/run_alfworld_skill_tree_depth_v4_1xa800.sh --phase prepare

# 只评测已准备好的 L0–L5
bash examples/playbook_evolve/run_alfworld_skill_tree_depth_v4_1xa800.sh --phase evaluate

# 重新汇总已有结果
bash examples/playbook_evolve/run_alfworld_skill_tree_depth_v4_1xa800.sh --phase summary
```

## 主要输出

- `frozen/initial_evidence.jsonl`：实际选中的 72 条轨迹。
- `frozen/initial_evidence.selection.json`：分层抽样和完整证据协议审计。
- `artifacts/skill_level_l*/artifact_manifest.json`：各层可用性及失败任务。
- `artifacts/skill_level_l*/generation_status.json`：逐任务验证、重试次数及失败原因。
- `artifacts/skill_level_l*/tree_increment_accounting.json`：父树与子树的字符、节点及逐节点 token 增量。
- `artifacts/skill_level_l*/cloud_io/`：实际云端 prompt 和 response；云端 token 调用审计同时写入对应 `artifact_manifest.json`。
- `generation_metrics.jsonl`、`generation_metrics_by_task.jsonl`：树生成 token 与结构增量。
- `metrics.jsonl`、`metrics_by_task.jsonl`：各臂整体及分任务成功率和本地模型 token。
- `arms/*/group_metrics.jsonl` 与 `summary.json`：包含 `context_guard` 裁剪次数、裁剪 token 数及协议有效性。
- `ablation_summary.json`、`ablation_summary.csv`、`skill_level_by_task.csv`：最终汇总。

若外部 `raw_traces.jsonl` 没有源游戏 ID，程序无法自动证明它与验证游戏不重合；正式实验应从数据准备阶段保证训练证据与验证 manifest 隔离。
