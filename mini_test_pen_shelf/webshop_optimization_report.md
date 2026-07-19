# WebShop mini-test 优化与问题报告

## 最终约束

- 固定 5 个 WebShop 原生类别，每类 2 个任务，共 10 个任务；baseline/template 使用完全相同的 goal index。
- 环境上限 15 步，prompt 仅展示最近 8 条原始 observation/action 历史。
- 不注入程序计算的已用查询、已访问商品、已选/剩余 options、剩余步数或循环警告。
- template 不包含 goal index、目标 ASIN、标准答案属性等 oracle 信息。

## 最佳公平 A/B 结果

| 指标 | Template | Baseline | 差值 |
|---|---:|---:|---:|
| 满分成功率 | 2/10 (20%) | 0/10 (0%) | +20 pp |
| 平均 task score | 0.280 | 0.065 | +0.215 |
| 购买率 | 30% | 20% | +10 pp |
| 平均步数 | 11.9 | 12.6 | -0.7 |
| 可执行动作率 | 98.3% | 100% | -1.7 pp |

template 满分任务是 garden goal 3（6 步）和 electronics goal 55（3 步）；beauty goal 85
得到 0.8。样本仅有 10 个，结果用于快速诊断，不应作为稳定总体估计。

## 所有已观察问题与处理

| 问题 | 轨迹证据/影响 | 处理 | 结果或限制 |
|---|---|---|---|
| Qwen 动作阶段输出 `<tool_call>` 或裸文本 | 动作语义可能正确，但严格标签率和可执行率很低 | thinking 后预填静态 `<action>` 前缀，仍由模型选择动作，并保留 admissible 校验 | 可执行率由早期约 50% 提升到约 98%-100%；不泄漏答案 |
| 重复相同搜索或 Search→同商品→Back 循环 | task 1/2/4/5/10 大量步数耗在重复查询和重复 ASIN | 静态状态机要求从 8 步原始历史自行识别 USED/VISITED，禁止重复 | 有改善但不能完全消除；4B 模型仍会忽略历史中的旧动作 |
| 商品页没有显示 option 的已选状态 | task 3/6 重复点击相同 size/color，浪费步数 | prompt 明确要求从可见历史判断同一商品已点 option，一次后直接 Buy Now | task 3 满分；task 6 仍重复一次但购买得 0.8 |
| Description/Features 后错误使用 Back to Search | 丢失结果页位置，重新搜索后进入循环 | 明确只用 `click[< prev]` 返回商品页 | 能减少上下文丢失，但模型偶尔仍违反 |
| 过度依赖商品标题语义 | goal 30 的任务文字像家具，真实目标却是地毯；goal 500 的目标是甜味剂而非 milk/cream | 将“请求 option 文本同时出现”设为比合成标题更强的匹配信号 | 可缓解合成数据错位，但目标没被打开时无效 |
| 检索结果较深，15 步内探索空间不足 | 多个失败任务反复翻页/换 query，始终未打开目标 ASIN | 首次 query 保持产品类型 + 最多两个稀有属性，优先检查当前页候选，避免立即翻页 | 仍是主要上限；不能用目标 ASIN 或程序检索提示，否则会变成作弊 |
| 过早购买导致部分分 | beauty goal 85 baseline 只得 0.6 | option 任务要求点完可见的请求值再购买 | template 提升到 0.8，但 raw 页面缺少可靠选中标记 |
| 只追求满分导致 15 步超时得 0 | 多个任务最后仍在查询或浏览 | 静态 deadline 规则要求后段选择最佳可见候选，partial reward 优先于超时 | 购买率提升有限；模型经常没有执行 deadline 规则 |
| 更长 thinking 被误认为一定更好 | 2048-token 复测 template 降为 1/10、0.230；baseline 降为 0/10、0.060 | 默认恢复 640 think、256 action、8192 context | 更快且当前样本实测更优；长推理会放大犹豫和重复分析 |
| prompt 尾部重复静态 guard | 2048-token guard 版严格动作率 88.8%、可执行率 93.3%，低于单一状态机 | 删除重复 guard，只保留一份主状态机 | 避免冲突和注意力稀释 |
| 程序生成进度摘要会抬高结果但不公平 | 可直接告诉模型已访问/已选/剩余项，改变 benchmark 信息边界 | 已完全删除相关模块、字段和 prompt 拼接 | 最终实现仅使用原始 8 步历史 |
| 运行结束出现 NCCL/EngineCore shutdown 警告 | 指标和报告已完整落盘后才出现 | 记录为 vLLM 退出清理告警，不当作任务失败或低分根因 | 不影响本轮分数；后续可单独完善显式 engine 销毁 |

## 最终 template 设计

最终采用单一静态页面状态机：Search、Results、Product with options、Product without
options、Description/Features、Deadline/Loop 六种规则。它要求模型自己从最近 8 步原始
历史维护 USED/VISITED/CLICKED，不由程序生成这些字段。完整原文位于
`webshop_template.py`，每步完整 prompt 和模型原始输出保存在 `*_prompts.txt`。

## 实验选择记录

| 版本 | 成功率 | 均分 | 结论 |
|---|---:|---:|---|
| template v1（无稳定 action prefix） | 0/10 | 0.130 | 动作协议和循环均差 |
| template v2（无 action prefix） | 1/10 | 0.160 | prompt 有帮助，但动作格式仍是瓶颈 |
| template v2 + action prefix + 640 think | **2/10** | **0.280** | 最佳，作为最终默认 |
| template v3 + 640 think | 2/10 | 0.260 | 没有超过 v2 |
| 重复 guard + 2048 think | 1/10 | 0.230 | 退化，已撤销 |

每轮中间产物均保存在 `output_webshop/iterations/`，最终 `baseline/`、`template/` 与
`ab_report.*` 指向最佳公平配置。

## 下一轮 v4 静态 template 改动

依据最新的 2048-token 轨迹，v4 只补充由当前页面和最近 8 步原始历史即可判断的规则：
禁止检索后立即 `Back to Search`；结果页不因合成标题不符而跳过低价候选；`ASIN → < prev>`
在可见历史中表示已拒绝，必须选不同 ASIN；出现任何请求 option 后优先完成可见请求
option 并购买；无 option 时，命中商品类型或两个稀有属性即可购买。没有增加任何
程序生成进度字段或 oracle 信息。
