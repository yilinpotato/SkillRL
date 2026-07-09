"""prod_prompt.py — 让 mini_test 的 prompt 与主管线 rollout 逐字段一致。

主管线（`agent_system/environments/env_manager.py::AlfWorldEnvironmentManager.build_text_obs`）
给小模型的 prompt 由三件事决定：
  1. 模板：init/首步用 ``ALFWORLD_TEMPLATE_NO_HIS``；之后开了检索就用
     ``ALFWORLD_TEMPLATE_WITH_MEMORY``，没开就用 ``ALFWORLD_TEMPLATE``。
  2. history：``SimpleMemory`` 存「step 前 raw obs + 动作」，每步取最近
     ``history_length`` 条（主管线默认 2）。
  3. 注入物：playbook（``seed_playbooks`` 经 ``format_for_prompt`` 放最前）+
     skill bullets（general/task/mistakes）。

本模块用**同一套** SkillsOnlyMemory + SimpleMemory + 模板，做单环境忠实镜像，
让 mini_test 不再用自己那套 ``strategy.py`` 头 + ``[INVENTORY]/[ALREADY SEARCHED]``
每步注入，从而与训练时小模型实际看到的 prompt 一致。

默认：playbook 开、skill bullets 关、history=2 —— 即「playbook 默认加、skills 先不加」。
把 ``with_skills=True`` 即把 bullets 接回来，与主管线 skills-on 路径一致。
"""
import os
from typing import Optional

from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_WITH_MEMORY,
)
from agent_system.memory import SimpleMemory
from agent_system.memory.skills_only_memory import SkillsOnlyMemory


_DEFAULT_SKILLS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "memory_data", "alfworld", "claude_style_skills.json")
)


class ProdObsBuilder:
    """单环境版的 ``build_text_obs``。每局 ``reset(task)``，每步 ``build(...)``，
    ``env.step`` 后 ``record(obs_used, action)``。

    Args:
        skills_json_path: 技能库 JSON（与主管线同一份）。
        history_length:   最近历史步数，主管线默认 2。
        with_skills:      是否注入 general/task/mistakes 三类 bullet 技能。默认 False。
        enable_playbook:  是否注入结构化 playbook。默认 True。
    """

    def __init__(self, skills_json_path: Optional[str] = None,
                 history_length: int = 2, with_skills: bool = False, top_k: int = 6,
                 enable_playbook: bool = True, playbook_examples: bool = True,
                 mem_lib=None):
        self.history_length = history_length
        self.with_skills = with_skills
        self.top_k = top_k
        self.enable_playbook = enable_playbook
        # playbook_examples=False -> use the lean playbook (concrete few-shot
        # examples / object->location lists stripped) to ablate their effect.
        self.playbook_examples = playbook_examples
        # mem_lib injection: the standalone playbook-evolution driver passes a
        # LIVE HierarchicalSkillLib (embedding mode, same args as the training
        # script) so the prompt matches env_manager.build_text_obs byte-for-byte
        # AND cloud-evolved playbooks flow in via that same shared object. When
        # not given, build the default template-mode SkillsOnlyMemory (mini_test's
        # original behaviour: template lookup, no embedding model loaded).
        if mem_lib is not None:
            self.mem_lib = mem_lib
            # honour the injected lib's own enable_playbook setting for the flag
            # that gates prompt-time playbook prepending.
            self.enable_playbook = getattr(mem_lib, 'enable_playbook', enable_playbook)
        else:
            self.mem_lib = SkillsOnlyMemory(
                skills_json_path=skills_json_path or _DEFAULT_SKILLS,
                retrieval_mode="template",
                enable_playbook=enable_playbook,
            )
        self.history = SimpleMemory()
        self.retrieved = None
        self._task = ""

    # 检索是否激活：与主管线一致——有 playbook 或 有 bullets 才走 WITH_MEMORY 分支，
    # 两者都关则退化为无记忆的 ALFWORLD_TEMPLATE（真·baseline）。
    def _retrieval_active(self) -> bool:
        return self.enable_playbook or self.with_skills

    def reset(self, task: str):
        self._task = task
        self.history.reset(1)
        self.retrieved = self.mem_lib.retrieve(task_description=task, top_k=self.top_k)
        # Swap to the lean (no-examples) playbook if requested.
        if self.enable_playbook and not self.playbook_examples:
            from agent_system.memory.seed_playbooks import get_seed_playbook
            lean = get_seed_playbook(self.retrieved.get("task_type"), with_examples=False)
            if lean:
                self.retrieved = dict(self.retrieved)
                self.retrieved["playbook"] = lean
        if not self.with_skills:
            # 关掉 bullets：清空三类技能，仅保留 playbook（format_for_prompt 会跳过空段）。
            self.retrieved = dict(self.retrieved)
            self.retrieved["general_skills"] = []
            self.retrieved["task_specific_skills"] = []
            self.retrieved["mistakes_to_avoid"] = []
            # HierarchicalSkillLib.retrieve() 已在过滤前记录候选 skill ids；若这里
            # 不同步清空，OFF 消融组虽然 prompt 没有 bullets，record_usage() 却仍会
            # 把它们算作已注入，污染生命周期与对照指标。
            self.retrieved["injected_skill_ids"] = []

    def build(self, raw_obs: str, admissible, init: bool) -> str:
        reformatted = "\n ".join(f"'{s}'" for s in admissible if s != "help")

        if init:
            obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=raw_obs, admissible_actions=reformatted)
            # init/NO_HIS 首步只 prepend playbook（与 env_manager 同形）。
            if self._retrieval_active():
                pb = (self.retrieved or {}).get("playbook")
                if self.enable_playbook and isinstance(pb, str) and pb.strip():
                    obs = pb.strip() + "\n\n" + obs
            return obs

        memory_contexts, valid_lens = self.history.fetch(
            self.history_length, obs_key="text_obs", action_key="action")
        step_count = len(self.history[0])

        if self._retrieval_active():
            memory_context = self.mem_lib.format_for_prompt(self.retrieved)
            return ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                task_description=self._task,
                retrieved_memories=memory_context,
                step_count=step_count,
                history_length=valid_lens[0],
                action_history=memory_contexts[0],
                current_step=step_count + 1,
                current_observation=raw_obs,
                admissible_actions=reformatted,
            )
        # 无记忆 baseline（playbook+skills 都关）。
        return ALFWORLD_TEMPLATE.format(
            task_description=self._task,
            step_count=step_count,
            history_length=valid_lens[0],
            action_history=memory_contexts[0],
            current_step=step_count + 1,
            current_observation=raw_obs,
            admissible_actions=reformatted,
        )

    def record(self, pre_obs: str, action: str):
        """存「本步动作所基于的 raw obs + 动作」，与 env_manager.step 的 store 一致。"""
        self.history.store({"text_obs": [pre_obs], "action": [action]})
