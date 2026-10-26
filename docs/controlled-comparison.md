# 受控对照启动入口

入口：`examples/grpo_trainer/run_controlled_comparison.sh`。

设置 `COMPARISON_ROOT` 为 train-alfworld-data 仓库目录，传入 group、benchmark、SFT model、skills、train-data、dev-data、prompt-tokens、output-root 和 run-id。默认保存解析配置与固定采样清单，加 `--execute` 才真正运行训练。参数和两组对照说明见：

https://github.com/yilinpotato/train-alfworld-data/blob/main/comparison/README.md

该入口读取统一配置，原有启动脚本默认行为保持不变。受控 ALFWorld reset 使用共享种子，WebShop 使用显式 goal_index；SearchQA/ScienceWorld 使用原生 benchmark env_kwargs。数据校验拒绝 train/dev 混用。指标文件输出到独立的组别/环境/方法/run-id 目录。

ScienceWorld 正式实验需要对应领域 SFT checkpoint。Skill0 的空技能版本须明确标为消融，不伪称完成技能内化。CPU 测试及配置解析不能代替 GPU 完整训练验证。

CoSkill 使用 `--method coskill_norl --group norl_primary` 或 `--method coskill_tree_rl --group all_rl`。no-RL 还需要指向同一 SFT checkpoint 的现有模型 API 服务；新 driver 不自行启动 GPU 推理。新环境 Tree-RL 使用 HierarchicalSkillLib，验证 probe/内化状态机的测试不代表模型参数已经经过 RL 更新。
