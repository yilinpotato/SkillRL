from typing import Any, Dict, List, Optional

from agent_system.environments.prompts.webshop import (
    DEFAULT_WEBSHOP_PROMPT_CHAR_LIMIT,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    WEBSHOP_TEMPLATE_WITH_MEMORY,
    fit_webshop_history_to_char_limit,
)
from agent_system.memory import SimpleMemory
from agent_system.memory.skills_only_memory import SkillsOnlyMemory


def format_webshop_actions(available: Dict[str, Any]) -> List[str]:
    unknown = set(available) - {"has_search_bar", "clickables"}
    if unknown:
        raise ValueError(f"Unknown WebShop available-action fields: {sorted(unknown)}")
    actions = []
    if available.get("has_search_bar"):
        actions.append("search[<your query>]")
    actions.extend(f"click[{text}]" for text in available.get("clickables", []))
    return actions


class WebShopObsBuilder:
    def __init__(
        self,
        *,
        mem_lib=None,
        skills_json_path: Optional[str] = None,
        history_length: int = 8,
        with_skills: bool = True,
        top_k: int = 6,
        enable_skill_tree: bool = True,
        prompt_char_limit: int = DEFAULT_WEBSHOP_PROMPT_CHAR_LIMIT,
        fixed_playbook: Optional[str] = None,
    ):
        self.mem_lib = mem_lib or SkillsOnlyMemory(
            skills_json_path=skills_json_path,
            retrieval_mode="template",
            enable_playbook=enable_skill_tree,
        )
        self.history_length = int(history_length)
        self.with_skills = bool(with_skills)
        self.top_k = int(top_k)
        self.enable_skill_tree = bool(
            getattr(self.mem_lib, "enable_playbook", enable_skill_tree)
        )
        self.prompt_char_limit = int(prompt_char_limit)
        self.fixed_playbook = str(fixed_playbook or "").strip()
        self.history = SimpleMemory()
        self.retrieved = None
        self.task = ""

    def _retrieval_active(self):
        return bool(self.fixed_playbook) or self.enable_skill_tree or self.with_skills

    def _memory_text(self):
        sections = []
        if self.fixed_playbook:
            sections.append(self.fixed_playbook)
        if self.enable_skill_tree or self.with_skills:
            sections.append(self.mem_lib.format_for_prompt(self.retrieved))
        return "\n\n".join(section for section in sections if section)

    def reset(self, task: str):
        self.task = task
        self.history.reset(1)
        self.retrieved = self.mem_lib.retrieve(
            task_description=task,
            top_k=self.top_k,
        )
        if not self.with_skills:
            self.retrieved = dict(self.retrieved)
            self.retrieved["general_skills"] = []
            self.retrieved["task_specific_skills"] = []
            self.retrieved["mistakes_to_avoid"] = []
            self.retrieved["injected_skill_ids"] = []

    @staticmethod
    def _actions_text(available: Dict[str, Any]) -> str:
        return "\n".join(
            f"'{action}'," for action in format_webshop_actions(available)
        )

    def _initial_prompt(
        self,
        observation: str,
        available: Dict[str, Any],
    ) -> str:
        if self._retrieval_active():
            memories = self._memory_text()
            return WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                task_description=self.task,
                retrieved_memories=memories,
                step_count=0,
                history_length=0,
                action_history="None",
                current_step=1,
                current_observation=observation,
                available_actions=self._actions_text(available),
            )
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=self.task,
            current_observation=observation,
            available_actions=self._actions_text(available),
        )

    def build(
        self,
        observation: str,
        available: Dict[str, Any],
        init: bool,
    ) -> str:
        if init or self.history_length <= 0:
            return self._initial_prompt(observation, available)

        step_count = len(self.history[0])
        recent_records = self.history[0][-self.history_length:]
        first_step = step_count - len(recent_records) + 1
        retrieved_memories = (
            self._memory_text() if self._retrieval_active() else None
        )

        def _render(history_text: str, kept_history_length: int) -> str:
            fields = dict(
                task_description=self.task,
                step_count=step_count,
                history_length=kept_history_length,
                action_history=history_text,
                current_step=step_count + 1,
                current_observation=observation,
                available_actions=self._actions_text(available),
            )
            if retrieved_memories is not None:
                fields["retrieved_memories"] = retrieved_memories
                return WEBSHOP_TEMPLATE_WITH_MEMORY.format(**fields)
            return WEBSHOP_TEMPLATE.format(**fields)

        prompt, _kept_steps, _dropped_steps, _static_over_limit = (
            fit_webshop_history_to_char_limit(
                recent_records,
                first_step=first_step,
                render_prompt=_render,
                char_limit=self.prompt_char_limit,
            )
        )
        return prompt

    def record(self, observation: str, action: str):
        self.history.store({"text_obs": [observation], "action": [action]})
