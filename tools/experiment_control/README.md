# 公共实验执行工具

此目录从 train-alfworld-data 迁入，属于 SkillRL 仓库的 Co-Skill 分支，供 Skill0、SkillRL、CoSkill 两组对照共用。数据仓库继续存放原始数据、配置快照、模型清单与绘图工具。

- comparison/：统一配置解析、真实启动器、初始技能和 Search-R1 服务适配。
- models/sft/：SFT准备、训练、导出与发布队列；v2 使用 LLaMA-Factory。
- models/sft/hpc_20261027/：有明确批次语义的SFT/RL队列及审计配套；使用 --root 指定数据/结果根，不包含SSH、资源申请或转发服务。

运行公共 comparison 命令前进入本目录。Co-Skill 的 examples/grpo_trainer/run_controlled_comparison.sh 默认指向这里；Skill0/SkillRL 需将 COMPARISON_ROOT 设置为本目录绝对路径（不再是数据仓库根）。原模型、任务、预算、奖励、token口径与两个独立对照组保持不变。

网络代理和API凭据通过外部环境变量传入；超算部署、连接、转发、供应商连通性检查仅在仓库外维护。遇错误保存状态并通知，不取消作业。

配置及初始技能在此维护，数据仓库里的同名文件是本次迁移快照；每次结果归档保存实际提交和resolved config。脚本历史参数不等于当前已完成训练；恢复旧批次前检查真实输出与状态，防止重复运行。

CPU验证：从本目录运行 python -m pytest comparison -q；模型工具测试 python -m unittest discover -s models/sft -p test_sft_tools.py。实际仓库配置测试需设置 CONTROLLED_SKILL0、CONTROLLED_SKILLRL、CONTROLLED_COSKILL。
