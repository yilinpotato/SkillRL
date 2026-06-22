# Mini Test — ALFWorld 小模型策略测试台

一个**自包含**的迷你测试环境，用来回答：**给 Qwen3-4B-Thinking 写一个自然语言策略 template，能不能让它在 ALFWorld 上更稳地完成「搬运类」任务？**

它**绕过** verl / Ray / GRPO / LoRA 整套训练管线，直接调用最底层的纯文本环境
`AlfredTWEnv`（TextWorld 变体，**零渲染、零 GPU 占用**），只用 vLLM 加载 4B 小模型逐步决策，
对比「有策略 template」vs「无策略 baseline」的成功率、步数、动作合法率。

支持三类任务：
1. **pen→shelf**（最初的单一任务，pick_and_place 的子集）
2. **pick_and_place_simple**（通用：任意 object → 任意 receptacle）
3. **pick_two_obj_and_place**（把两个同类 object 搬到同一容器）

---

## 一、整体架构（数据流）

```
                    ┌─────────────────────────────────────────────┐
                    │  env_utils.py                               │
   ALFWORLD_DATA ──▶│  · 按 task_type 筛游戏 find_games_by_type   │
   (game.tw-pddl)   │  · 解析目标物/容器  extract_task_target     │──┐
                    │  · 构造单环境       make_single_env         │  │
                    └─────────────────────────────────────────────┘  │
                                                                      ▼
   ┌──────────────┐   每步 prompt   ┌──────────────────┐   原始文本   ┌──────────────┐
   │ strategy.py  │───playbook────▶│ run_generic.py / │────────────▶│ agent_vllm.py│
   │ (策略模板)   │                │ run_mini_test.py │             │ (vLLM+Qwen3) │
   └──────────────┘                │  主循环：        │◀────────────│              │
                                   │  注入状态/解析/   │ text+截断flag└──────────────┘
   ┌──────────────┐   合法性校验    │  环境step/记录    │
   │ projection.py│◀───────────────│                  │──▶ env.step(action) ──▶ AlfredTWEnv
   │ (共用,不改)  │                └──────────────────┘
   └──────────────┘                       │
                                          ▼ 每步/每局
                    ┌──────────────────────────────────────────┐
                    │ trajectory_logger.py  → output_*/         │
                    │   game_NN_WIN/FAIL_Ksteps_trajectory.txt  │ 观察/思考/动作
                    │   game_NN_..._prompts.txt                 │ 每步完整 prompt
                    │   game_NN_..._.json / summary.json        │ 结构化+run汇总
                    └──────────────────────────────────────────┘
                                          │
                    ┌─────────────────────▼──────────────────────┐
                    │ compare_ab.py → output_ab/<tag>_report.txt │ A/B 对比表
                    └────────────────────────────────────────────┘
```

驱动脚本 `run.sh` / `run_ab.sh` / `run_all_ab.sh` 负责设环境变量、选 GPU、串起上面的流程。

---

## 二、各文件功能 & 实现位置

### 数据 / 环境层（纯 CPU，无 GPU）
| 文件 | 功能 | 关键函数 |
|------|------|---------|
| `env_utils.py` | 加载 `AlfredTWEnv`、按类型筛游戏、解析真值目标、构造单环境 | `find_games_by_type(task_type,...)`、`find_pen_shelf_games(...)`、`extract_task_target(traj)`（取 object/parent/count）、`load_tw_config_types(ids)`、`make_single_env(...)` |
| `inspect_pen_locations.py` | 零成本统计目标物真实初始位置分布（不开模型） | `main()` |

### 模型层（GPU）
| 文件 | 功能 | 关键点 |
|------|------|--------|
| `agent_vllm.py` | vLLM 封装 Qwen3-4B，按 prompt 产出 `<think>/<action>` | `act_with_meta()` 返回 `(文本, truncated)`；`truncated` 来自 vLLM `finish_reason=='length'`。`_restore_think()` 补回被 chat template 吃掉的开头 `<think>`。`max_tokens=5120` |

### 策略层
| 文件 | 功能 | 关键点 |
|------|------|--------|
| `strategy.py` | 三套自然语言 playbook | `PEN_SHELF_STRATEGY`（pen→shelf 专用）、`GENERIC_PICK_PLACE_STRATEGY`（通用）、`PICK_TWO_STRATEGY`（两物体）。搜索顺序按上帝视角统计：**开放台面优先，抽屉/柜子/架子最后** |

### 主循环 / 解析
| 文件 | 功能 | 关键点 |
|------|------|--------|
| `run_mini_test.py` | pen→shelf 专用主程序（最早的版本） | `--strategy` 开关、`--repeats N`、逐步注入 `[INVENTORY]/[ALREADY SEARCHED]/[HERE]` |
| `run_generic.py` | **通用主程序**，支持 `--mode {generic,pick_two}` | 逐步注入 `[TARGET]/[INVENTORY]/[PROGRESS]/[ALREADY SEARCHED]`；`placed_ids` 按实例去重计数（修 pick_two 重复放置 bug）；`salvage_action_from_back()` 截断时**从后往前匹配** admissible 动作兜底；统计截断/救回步数 |
| `projection.py`（在 `agent_system/.../alfworld/`，**训练管线共用，不改**） | 解析 `<action>` 并校验合法性 | 要求成对 `<think>...</think>`；无标签时回退取末尾 30 字符（截断时会产生垃圾，故 run_generic 加了兜底） |
| `report.py` | 终端逐步彩色打印 | `print_step / print_game_header / print_final_summary` |

### 记录 / 对比
| 文件 | 功能 | 关键点 |
|------|------|--------|
| `trajectory_logger.py` | 每局落盘三件套 + run 级 summary | `log_step(... truncated, salvaged)` 记录截断标记；轨迹文本里标 `⛔THINKING截断(已救回/未救回)` |
| `compare_ab.py` | 读两个 `summary.json` 出 A/B 对比报告 | `build_report(a,b)` → 成功率/步数/合法率对照表 + 逐局 + 结论，归档到 `output_ab/<tag>_report.txt/.json` |

### 驱动脚本
| 文件 | 功能 |
|------|------|
| `run.sh` | 一键跑 pen→shelf（`inspect` + `run_mini_test`） |
| `run_ab.sh` | pen→shelf 的 A/B（固定布局重复） |
| `run_all_ab.sh` | **总驱动**：pick_and_place + pick_two 两组 A/B。支持 `SINGLE_GPU=1`（单卡顺序）、重跑前**自动备份旧结果**到 `backups/<tag>_<时间戳>/` |

---

## 三、用法

### 0. 准备（每次开终端）
```bash
source /data2/myl/miniconda3/etc/profile.d/conda.sh && conda activate skillRL
cd /data2/myl/CoSkill
export MODEL_PATH=/data2/myl/home_configs/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507
export ALFWORLD_DATA=$HOME/.cache/alfworld
# 跑前确认 0 卡干净，清掉残留 vLLM 进程（否则下次 OOM）
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p; done
```

### 1. 零成本看目标物分布（不开模型）
```bash
python -m mini_test_pen_shelf.inspect_pen_locations
```

### 2. 跑全部 A/B（单卡 0 号，顺序）
```bash
SINGLE_GPU=1 GPU_A=0 GPU_MEM_UTIL=0.55 \
    PP_GAMES=6 PP_STEPS=30 P2_REPEATS=3 P2_STEPS=40 \
    bash mini_test_pen_shelf/run_all_ab.sh
```
- 依次跑 4 段：pickplace 有策略 → pickplace 无策略 → picktwo 有策略 → picktwo 无策略
- 重跑会**自动把旧结果备份**到 `mini_test_pen_shelf/backups/<tag>_<时间戳>/`（设 `BACKUP=0` 改为直接清空）
- 双卡并行：去掉 `SINGLE_GPU=1`，设 `GPU_A=0 GPU_B=1`

### 3. 单独跑某类
```bash
# 通用 pick_and_place，10 局，有策略
CUDA_VISIBLE_DEVICES=0 python -m mini_test_pen_shelf.run_generic \
    --mode generic --num_games 10 --max_steps 30 --strategy \
    --gpu_mem_util 0.55 --outdir mini_test_pen_shelf/output_test

# pen→shelf
CUDA_VISIBLE_DEVICES=0 MAX_STEPS=40 STRATEGY=1 bash mini_test_pen_shelf/run.sh
```

---

## 四、怎么看输出

| 想看什么 | 命令 |
|---------|------|
| 实时进度 | `tail -f mini_test_pen_shelf/output_pickplace_strategy/run.log` |
| 各局结果 | `grep -E "结果:\|🏆\|成功率" mini_test_pen_shelf/output_*/run.log` |
| **A/B 对比报告** | `cat mini_test_pen_shelf/output_ab/pickplace_report.txt` |
| 某局完整轨迹（观察/思考/动作，含截断标记） | `cat mini_test_pen_shelf/output_*/game_01_*_trajectory.txt` |
| 某局每步完整 prompt | `cat mini_test_pen_shelf/output_*/game_01_*_prompts.txt` |
| run 级指标（成功率/步数/合法率/截断率） | `cat mini_test_pen_shelf/output_*/summary.json` |

> ⚠️ 别 `tail` 驱动脚本的 stdout 文件（如 `/tmp/myrun.log`）——每臂输出单独重定向到各自目录的 `run.log`，主 stdout 只有几行 echo。

---

## 五、已知要点（踩过的坑）

- **Thinking 模型推理极长**：~5000+ token/步。`max_tokens` 不够会在 `</think>` 前被截断，
  projection 回退取垃圾 → 非法动作。现 `max_tokens=5120`，截断时 run_generic 会**从后往前匹配**
  admissible 动作兜底，并在终端/轨迹里**标注 `⛔THINKING截断`**。
- **chat template 吃掉开头 `<think>`** → `agent_vllm._restore_think` 补回。
- **NO_HIS 模板无记忆**：手持/已搜过/已放置都必须每步显式注入 prompt。
- **放置动词是 `move <obj> to <recep>`**，不是 `put in/on`；以 admissible 列表原文为准。
- **pen ≠ pencil**，目标物按环境任务串精确匹配，否则 `won` 不触发。
- **pick_two 按实例 id 去重**计数，否则同一个物体放两次会被误判成完成。
- **残留 vLLM EngineCore 漏显存**：kill 后要 `kill -9` `nvidia-smi --query-compute-apps` 里的 pid，否则下次 OOM。
- **单卡不能并行两个 vLLM**（24G 装不下），`run_all_ab.sh` 用 `SINGLE_GPU=1` 顺序跑。
