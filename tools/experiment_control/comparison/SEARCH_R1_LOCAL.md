# SearchQA 的 Wikipedia 本地检索适配

此变更将三方法的搜索后端切换为 Search-R1 提供的 BM25 + wiki-18，仍然由模型发出 `search[query]`、`answer[text]`。这是 SearchQA 题目上的固定语料检索设置，不是原始给定片段阅读任务，也不能与旧 Tavily 实时联网结果混算。

## 来源和存储

- Search-R1：`https://github.com/PeterGriffinJin/Search-R1`，固定提交 `598e61bd1d36895726d28a8d06b3a15bed19f5d3`。
- 索引：`PeterJinGo/wiki-18-bm25-index`，固定 revision `2c7554f25f425038c4bcb155735a0f831851fd78`。
- 语料：`PeterJinGo/wiki-18-corpus`，固定 revision `69c1c00ffe7c5554c68d8548355cb22e46aabc51`。
- 远端文件清单合计索引约2.30GB，压缩语料约5.12GB；这不是最终磁盘需求。还要计算解压语料、datasets Arrow缓存、环境依赖及下载临时文件，不能据此断言超算空间充足。
- 恢复连接后先核实租户配额，不能拿共享文件系统df总空闲量当个人可用空间。三方法共用一份语料和索引。未核实空间前不下载大文件。

## 接入范围

Skill0、SkillRL、CoSkill的原生benchmark manager使用相同的 `agent_system/local_search.py`。统一启动器覆盖Skill0、SkillRL、CoSkill no-RL与Tree-RL；CoSkill旧独立SearchQA入口也支持同样的参数。

统一入口显式增加以下参数，原有模型、技能、train/dev和预算参数不变：

```text
--search-backend search_r1
--retrieval-endpoint http://127.0.0.1:8001/retrieve
--retrieval-corpus-id wiki18-bm25:2c7554f25f425038c4bcb155735a0f831851fd78:69c1c00ffe7c5554c68d8548355cb22e46aabc51
```

公共启动器保留旧默认tavily以便历史配置复现；本批 `models/sft/hpc_20261027/run_search_rl.py` 默认search_r1，必须传显式地址及语料标识。smoke→正式启动检查从smoke的comparison.json继承检索配置，禁止默默回退付费API。新运行必须使用新run-id，从原SFT开始，旧失败目录不覆盖。

`serve_search_r1.py`检查上游固定提交及检索文件无本地改动，然后调用上游BM25实现。默认监听127.0.0.1:8001，避免碰撞模型推理的8000端口。两个作业共享一个服务时，通过已有可信私有连接转发，或显式绑定受限集群地址；不要暴露公网端口。运行环境须满足上游pyserini/Java及导入依赖，采用独立环境，不覆盖原RL环境。

```bash
python comparison/serve_search_r1.py --repo /path/to/pinned/Search-R1 \
  --index /path/to/shared/bm25 --corpus /path/to/shared/wiki-18.jsonl
```

HTTP请求使用 `queries`、`topk=5`、`return_scores=true`，将返回document内容映射到既有观测字段。不会访问所展示的Wikipedia链接，也不执行Open。保留原提示模板、15动作/8搜索、返回5条、评分、token统计、SFT权重、初始技能、12任务×6rollout、dev32和30更新。记录固定语料与检索后端，原模板中的live web措辞沿用以保持本次其他设置不变，论文必须明确实际证据来自Wikipedia快照。

## 检查与当前状态

接口提交：Skill0 main `aa5426d`、SkillRL main `f102eca`、CoSkill Co-Skill `a2b0b8f`。三份接口测试各3项通过，公共配置测试19项通过；服务包装器及HPC启动脚本通过语法检查。

每个方法的CPU测试通过本机HTTP服务返回固定测试文档，检查请求协议、无Authorization、观测呈现、搜索计数、Open拒绝、回答评分及审计。这证明接口连接正确，不证明真实索引加载、检索质量或GPU训练成功。统一启动器同时测试Tavily和本地检索两种配置。

超算真实测试仍待连接恢复：核实配额→下载并校验固定revision文件→启动真实上游检索器→检查真实查询和空结果→分别运行两步训练及保存→审计后重跑正式实验。不能把mock HTTP或配置测试当成以上步骤已经完成。

## 作业保留与通知

用户2026-10-27最新要求：遇到问题保留作业，不自动取消或释放。通过原私有邮件服务通知；凭据和收件地址不进仓库。无资源时每30分钟检查的调度状态以应用自动化为准，不能把本文当成已启用定时任务的证据。
