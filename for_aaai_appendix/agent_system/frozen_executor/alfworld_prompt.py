""
import os
from typing import Optional

from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE_WITH_MEMORY,
)
from agent_system.memory import SimpleMemory
from agent_system.memory.skills_only_memory import SkillsOnlyMemory

_DEFAULT_SKILLS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "memory_data", "alfworld", "initial_skills.json")
)


class AlfWorldObsBuilder:
    ""

    def __init__(self, skills_json_path: Optional[str] = None,
                 history_length: int = 2, with_skills: bool = False, top_k: int = 6,
                 enable_playbook: bool = True, mem_lib=None):
        self.history_length = history_length
        self.with_skills = with_skills
        self.top_k = top_k
        self.enable_playbook = enable_playbook






        if mem_lib is not None:
            self.mem_lib = mem_lib


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



    def _retrieval_active(self) -> bool:
        return self.enable_playbook or self.with_skills

    def reset(self, task: str):
        self._task = task
        self.history.reset(1)
        self.retrieved = self.mem_lib.retrieve(task_description=task, top_k=self.top_k)
        if not self.with_skills:

            self.retrieved = dict(self.retrieved)
            self.retrieved["general_skills"] = []
            self.retrieved["task_specific_skills"] = []
            self.retrieved["mistakes_to_avoid"] = []



            self.retrieved["injected_skill_ids"] = []

    def build(self, raw_obs: str, admissible, init: bool) -> str:
        reformatted = "\n ".join(f"'{s}'" for s in admissible if s != "help")

        if init:
            obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=raw_obs, admissible_actions=reformatted)

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
        ""
        self.history.store({"text_obs": [pre_obs], "action": [action]})
