# SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

<div align="center">

Bridging the gap between raw experience and policy improvement through automatic skill discovery.

</div>

<p align="center">
<img src="figs/pipeline.png" width="80%" alt="SKILLRL Pipeline Overview">
</p>

## 🔥 News

- **[05/10/2026]** Released the code for SFT data generation under `examples/sft_data_generation/`.
- **[04/03/2026]** Released the SFT dataset on [🤗HF](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data)!
- **[03/02/2026]** Due to an accidental misconfiguration, we lost several hundred GitHub stars. If you previously starred this repo, we'd appreciate a re-star ⭐!
- **[02/23/2026]** We released all the model checkpoints on HuggingFace! Feel free to use them as warm starts for continued RL training.
- **[02/18/2026]** The code of SkillRL was released!
- **[02/10/2026]** SkillRL paper was released on [arXiv](https://arxiv.org/abs/2602.08234)!

## 📖 Overview

SkillRL is a framework that enables LLM agents to learn high-level, reusable behavioral patterns from past experiences. While traditional memory-based methods store redundant and noisy raw trajectories, SKILLRL abstracts these into a hierarchical skill library.

> CoSkill 的论文方法、系统流程、端云协同、轨迹池与技能库说明，请先阅读
> [COSKILL论文方法说明.md](COSKILL论文方法说明.md)。实际启动命令、全部运行参数、GPU/数据
> 要求、输出指标与排错见补充操作手册
> [COSKILL运行与实验指南.md](COSKILL运行与实验指南.md)。

## 🤖 Key Features

- **Experience-based Skill Distillation**: Transforms successful trajectories into strategic patterns and failed ones into concise lessons from failure.

- **Hierarchical SKILLBANK**: Organizes knowledge into General Skills for universal strategic guidance and Task-Specific Skills for category-level heuristics.

- **Recursive Skill Evolution**: A dynamic mechanism where the skill library co-evolves with the agent's policy during RL by analyzing validation failures.

- **Context Efficiency**: Achieves 10-20% token compression compared to raw trajectory storage while enhancing reasoning utility. 

---

## 📥 Model Download

You can directly download the model weights by following the links below.

<table>
  <thead>
    <tr>
      <th align="center">Task</th>
      <th align="center">Model</th>
      <th align="center">Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2">🧭 ALFWorld</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Alfworld-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Alfworld-7B-RL">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center" rowspan="2">🛍️ WebShop</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Webshop-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Webshop-7B-RL">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center" rowspan="2">🔍 Search</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Search-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Search-7B-RL">🤗 HuggingFace</a></td>
    </tr>
  </tbody>
</table>


---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/aiming-lab/SkillRL.git
cd SkillRL

pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .

pip install openai
```

### Environment Setup

**ALFWorld**
```bash
pip install alfworld
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0

# Download PDDL & Game files and pre-trained MaskRCNN detector
alfworld-download -f
```

**WebShop**
```bash
cd agent_system/environments/env_package/webshop
./setup.sh -d all
```

**Search**
```bash
cd agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

**API Setup**
```
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT=""
```

---

## 🏃 Training

### CoSkill Tree-RL（2/4/8 GPU 与 Docker）

CoSkill 的渐进式技能树内化由统一入口启动；始终保持每步 `12×6=72` 条
rollout，并按可见算力自动采用 DP=2/4、TP=PP=1；8 卡节点切为两个独立 4 卡
slot，从而并发两个实验且不改变 PPO 几何。通过
`TREE_RL_ORDER=root|leaf` 选择从根或叶开始，模型 checkpoint 与
`skills_tree_rl_latest.json` 会一起自动恢复。

```bash
# ALFWorld，叶到根
rl=1 TREE_RL_ORDER=leaf \
  bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh

# WebShop，根到叶
rl=1 TREE_RL_ORDER=root \
  bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh
```

冻结 no-RL 路径和 Ray Tree-RL 路径现在都在每个活跃环境步只执行一次完整
`<think>...</think><action>...</action>` 生成。实际 prompt/response token 增量和
累计值写入主 `metrics.jsonl`/`group_metrics.jsonl`，不会再重复计入第二次动作请求。

自包含 Docker 镜像会固化当前 `skillRL` Conda 环境，并内嵌 ALFWorld 文本数据、
WebShop 1000 商品数据与索引。它提供 `alfworld-root`、`alfworld-leaf`、
`webshop-root`、`webshop-leaf` 四个入口；模型可挂载，也可在首次启动时由
ModelScope 自动下载。构建与运行命令见
[docker/coskill/README.md](docker/coskill/README.md)。

### ALFWorld 固定轨迹消融（4×A800，非 Docker）

固定轨迹消融使用独立入口，不改变上述主训练过程。它固定六类任务各一条
bootstrap game 与一条非重叠 eval game；所有臂共用冻结 raw traces。bootstrap
保留 CoSkill 检索/树提示词框架但显式使用空技能库，深度树最多经 20 次同证据云端
深化，仍不合格的臂记录为 `N.A.` 而不会中止其他臂。

```bash
conda activate skillRL
cd /path/to/CoSkill

# 仅检查 4 张 A800、模型、数据与云端 API；不启动 rollout。
CUDA_VISIBLE_DEVICES=0,1,2,3 ABLATION_PREFLIGHT_ONLY=1 \
  bash examples/playbook_evolve/run_alfworld_fixed_trajectory_ablation_4xa800.sh

# 正式运行；AB_ROOT 必须保留，后续同一路径可按 phase 继续。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AB_ROOT=/path/to/outputs/alfworld_ablation_4xa800 \
  bash examples/playbook_evolve/run_alfworld_fixed_trajectory_ablation_4xa800.sh
```

预检默认会发起一次最小云端请求；若只检查本地依赖，加入 `CLOUD_PROBE=0`。完整
产物、恢复和指标说明见 [docker/alfworld-ablation/README.md](docker/alfworld-ablation/README.md)。

### Memory Data Generation
The first step of our training pipeline uses the base model to generate memory data. This data serves as the foundation for the agent's initial experiences. The specific prompt used to guide this generation can be found at: `memory_data/prompt/prompt.txt`.

### Supervised Fine-Tuning (SFT)
Prior to RL, we perform SFT to endow the model with basic task capabilities and instruction-following alignment. We use [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) as our framework for the SFT stage. The SFT data was released on [🤗 HF](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data) now! For more details on data generation, please refer to `examples/sft_data_generation/`.

### RL With SkillBank

#### Template Mode

Template mode uses keyword matching to detect the task category and injects all skills for that category into the prompt.  No embedding model is required.

```bash
# ALFWorld
export MODEL_PATH=YOUR_SFT_CKPT
bash examples/grpo_trainer/run_alfworld_skills.sh

# WebShop
bash examples/grpo_trainer/run_webshop_skills.sh

# Search
bash examples/grpo_trainer/run_search_skills.sh
```

Key config flags added by these scripts:

```
+env.use_skills_only_memory=True
+env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json
+env.skills_only_memory.top_k=6              
+env.skills_only_memory.enable_dynamic_update=True
+env.skills_only_memory.update_threshold=0.4
+env.skills_only_memory.max_new_skills=3
```

#### Embedding Mode

Embedding mode uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) to rank skills by semantic similarity to the task description.  Both general skills and task-specific skills are searched cross-category and only the top-k most relevant are injected.  Skill embeddings are pre-computed once at startup.

```bash
export MODEL_PATH=YOUR_SFT_CKPT

python3 -m verl.trainer.main_ppo \
    ... \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=Qwen/Qwen3-Embedding-0.6B \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.task_specific_top_k=5
```

---

## ⚙️ Skill Memory Configuration

All parameters live under `env.skills_only_memory.*` (Hydra / OmegaConf).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skills_json_path` | str | — | **Required.** Path to the skills JSON. |
| `retrieval_mode` | str | `"template"` | `"template"` or `"embedding"`. |
| `embedding_model_path` | str | `"Qwen/Qwen3-Embedding-0.6B"` | Local path or HF model ID.  Only used when `retrieval_mode=embedding`. |
| `top_k` | int | `6` | Number of general skills injected per episode. |
| `task_specific_top_k` | int | `None` | Max task-specific skills per episode.  `None` = all (template) / same as `top_k` (embedding). |
| `enable_dynamic_update` | bool | `False` | Evolve the skill bank during training using validation failures. |
| `update_threshold` | float | `0.4` | Min success rate below which skills are updated. |
| `max_new_skills` | int | `3` | Maximum new skills added per update cycle. |
---

## 📋 Skill Bank Format

Skills are stored in a JSON file with three top-level keys:

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Systematic Exploration",
      "principle": "Search every plausible surface exactly once …",
      "when_to_apply": "Anytime the goal object count is not yet met …"
    }
  ],
  "task_specific_skills": {
    "pick_and_place": [
      {
        "skill_id": "pnp_001",
        "title": "Direct Path Planning",
        "principle": "Navigate directly to the target receptacle …",
        "when_to_apply": "After picking up the object …"
      }
    ],
    "clean": [ … ],
    "heat":  [ … ]
  },
  "common_mistakes": [
    {
      "mistake_id": "err_001",
      "description": "Repeating the same action after it fails.",
      "why_it_happens": "Agent does not track action history.",
      "how_to_avoid": "Check the admissible actions list and try an alternative."
    }
  ]
}
```

### Generating a New Skill Bank

Use the provided generation scripts (requires Azure API access):

```bash
# ALFWorld
python skill_generation/alfworld.py \
    --memory_path memory_data/alfworld/generated_memories_alfworld_total.json \
    --output_path memory_data/alfworld/claude_style_skills.json

# WebShop
python skill_generation/webshop.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json

# Search
python skill_generation/search.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json
```

---

## 📚 Citation
If you find our work helpful, please consider citing:

```bibtex
@article{xia2026skillrl,
  title={SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and Liu, Jiaqi and Zeng, Kaide and Wang, Yu and Han, Siwei and Zhou, Yiyang and Zhao, Xujiang and Chen, Haifeng and others},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}
```

## 🙏 Acknowledgement
We would like to express our gratitude to the open-source community and the following projects for making this work possible: 
[verl-agent](https://github.com/langfengQ/verl-agent), [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [Qwen](https://github.com/QwenLM/Qwen), etc.
