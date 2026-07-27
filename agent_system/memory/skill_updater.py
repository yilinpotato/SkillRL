"""
LLM-based skill updater that generates new skills from failed trajectories.

Backend is selected via the SKILL_UPDATER_BACKEND environment variable:
  "deepseek"  (default) – DeepSeek API (OpenAI-compatible)
      DEEPSEEK_API_KEY      – DeepSeek API key
      DEEPSEEK_API_BASE     – base URL (default: https://api.deepseek.com/v1)
      DEEPSEEK_MODEL        – model name (default: deepseek-chat)
  "azure"                  – Azure OpenAI
      AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
"""
import json
import os
import re
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAI


class SkillUpdater:
    def __init__(
        self,
        max_new_skills_per_update: int = 3,
        max_completion_tokens: int = 2048,
    ):
        backend = os.environ.get("SKILL_UPDATER_BACKEND", "deepseek").lower()

        if backend == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            if not api_key or not endpoint:
                raise EnvironmentError(
                    "SkillUpdater (azure) requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
                )
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
            self.model = "o3"
        else:  # deepseek
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise EnvironmentError("SkillUpdater (deepseek) requires DEEPSEEK_API_KEY.")
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            )
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

        self.max_completion_tokens = max_completion_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Kept for audit-oriented callers (for example fixed-trajectory
        # ablations).  Existing training callers do not consume these fields.
        self.last_prompt = None
        self.last_response = None
        self.last_usage = {
            "prompt": 0, "completion": 0, "total": 0,
            "usage_reported": False,
        }

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
    ) -> List[Dict]:
        """
        Analyse failed trajectories and generate new skills to address the gaps.

        Args:
            failed_trajectories: List of dicts with keys:
                ``task``       – task description string
                ``trajectory`` – list of ``{action, observation}`` step dicts
                ``task_type``  – detected task category string
            current_skills: The current skill bank dict (with keys
                ``general_skills``, ``task_specific_skills``, etc.)

        Returns:
            List of new skill dicts ready to be passed to
            ``SkillsOnlyMemory.add_skills()``.
        """
        self.last_prompt = None
        self.last_response = None
        self.last_usage = {
            "prompt": 0, "completion": 0, "total": 0,
            "usage_reported": False,
        }
        if not failed_trajectories:
            return []

        # Compute the next available dyn_ index BEFORE calling the LLM so we
        # can tell it which IDs to use, avoiding duplicate-ID collisions.
        next_dyn_idx = self._next_dyn_index(current_skills)

        prompt = self._build_analysis_prompt(
            failed_trajectories, current_skills, next_dyn_idx
        )
        self.last_prompt = prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            self.last_response = response.choices[0].message.content
            raw_skills = self._parse_skills_response(self.last_response)

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.last_usage = {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                    "usage_reported": True,
                }

            # Reassign dyn_ IDs on our side to guarantee no collisions,
            # regardless of what the LLM returned.
            reassigned = self._reassign_dyn_ids(raw_skills, next_dyn_idx)

            self.update_history.append({
                'num_failures_analyzed': len(failed_trajectories),
                'num_skills_generated': len(reassigned),
                'skill_ids': [s.get('skill_id') for s in reassigned],
            })

            return reassigned[:self.max_new_skills_per_update]

        except Exception as e:
            print(f"[SkillUpdater] Error calling {self.model}: {e}")
            return []

    def analyze_balanced_trajectories(
        self,
        successful_trajectories: List[Dict],
        failed_trajectories: List[Dict],
        current_skills: Dict,
    ) -> List[Dict]:
        """Generate flat skills from complete balanced success/failure evidence.

        This is used by representation ablations where L0 must receive the
        same trajectories as hierarchical arms.  It intentionally preserves
        the flat SkillRL JSON output schema while removing the legacy
        five-failure/last-five-step evidence cap.
        """
        self.last_prompt = None
        self.last_response = None
        self.last_usage = {
            "prompt": 0, "completion": 0, "total": 0,
            "usage_reported": False,
        }
        if not successful_trajectories and not failed_trajectories:
            return []

        next_dyn_idx = self._next_dyn_index(current_skills)
        prompt = self._build_balanced_analysis_prompt(
            successful_trajectories,
            failed_trajectories,
            current_skills,
            next_dyn_idx,
        )
        self.last_prompt = prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            self.last_response = response.choices[0].message.content
            raw_skills = self._parse_skills_response(self.last_response)

            if hasattr(response, "usage") and response.usage:
                prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.last_usage = {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                    "usage_reported": True,
                }

            reassigned = self._reassign_dyn_ids(raw_skills, next_dyn_idx)
            self.update_history.append({
                "num_successes_analyzed": len(successful_trajectories),
                "num_failures_analyzed": len(failed_trajectories),
                "num_trajectories_analyzed": (
                    len(successful_trajectories) + len(failed_trajectories)
                ),
                "num_skills_generated": len(reassigned),
                "skill_ids": [s.get("skill_id") for s in reassigned],
            })
            return reassigned[:self.max_new_skills_per_update]
        except Exception as e:
            print(f"[SkillUpdater] Error calling {self.model}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _next_dyn_index(self, current_skills: Dict) -> int:
        """
        Scan the current skill bank for existing ``dyn_NNN`` IDs and return
        the next unused integer index (1-based).
        """
        max_idx = 0
        pattern = re.compile(r'^dyn_(\d+)$')

        for skill in current_skills.get('general_skills', []):
            m = pattern.match(skill.get('skill_id', ''))
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        for skills in current_skills.get('task_specific_skills', {}).values():
            for skill in skills:
                m = pattern.match(skill.get('skill_id', ''))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

    def _reassign_dyn_ids(self, skills: List[Dict], start_idx: int) -> List[Dict]:
        """
        Replace whatever skill_id values the LLM returned with guaranteed-unique
        ``dyn_NNN`` IDs starting from ``start_idx``.
        """
        reassigned = []
        for i, skill in enumerate(skills):
            updated = dict(skill)
            updated['skill_id'] = f"dyn_{start_idx + i:03d}"
            reassigned.append(updated)
        return reassigned

    def _build_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        next_dyn_idx: int,
    ) -> str:
        # Format failure examples
        failure_examples = []
        for i, traj in enumerate(failed_trajectories[:5]):
            failure_examples.append(
                f"\nExample {i + 1}:\n"
                f"Task: {traj['task']}\n"
                f"Task Type: {traj['task_type']}\n"
                f"Trajectory (last 5 steps):\n"
                f"{self._format_trajectory(traj['trajectory'][-5:])}\n"
            )

        # Collect all existing skill titles (for deduplication hint to the LLM)
        existing_titles = [s['title'] for s in current_skills.get('general_skills', [])]
        for task_type, skills in current_skills.get('task_specific_skills', {}).items():
            for s in skills:
                existing_titles.append(f"[{task_type}] {s.get('title', '')}")

        # Show the LLM what IDs to use (we'll reassign them anyway, but
        # providing the range avoids confusion in the returned JSON)
        example_ids = ", ".join(
            f'"dyn_{next_dyn_idx + j:03d}"'
            for j in range(self.max_new_skills_per_update)
        )

        return f"""Analyze these failed agent trajectories and suggest NEW skills to add to the skill bank.

FAILED TRAJECTORIES:
{''.join(failure_examples)}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

Generate 1-{self.max_new_skills_per_update} NEW actionable skills that would help avoid these failures.
Each skill must have: skill_id, title (3-5 words), principle (1-2 sentences), when_to_apply.

Use skill_ids: {example_ids}

Return ONLY a JSON array of skills, no other text.
Example format:
[{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location by examining the environment.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
"""

    def _build_balanced_analysis_prompt(
        self,
        successful_trajectories: List[Dict],
        failed_trajectories: List[Dict],
        current_skills: Dict,
        next_dyn_idx: int,
    ) -> str:
        def render(label: str, trajectories: List[Dict]) -> str:
            examples = []
            for i, traj in enumerate(trajectories):
                examples.append(
                    f"\n{label} Example {i + 1} [traj_uid={traj.get('traj_uid', '')}]:\n"
                    f"Task: {traj.get('task', '')}\n"
                    f"Task Type: {traj.get('task_type', '')}\n"
                    "Complete trajectory (observation is the state before its action):\n"
                    f"{self._format_trajectory_full(traj.get('trajectory', []))}\n"
                )
            return "".join(examples) or "\n(none)\n"

        existing_titles = [s["title"] for s in current_skills.get("general_skills", [])]
        for task_type, skills in current_skills.get("task_specific_skills", {}).items():
            for skill in skills:
                existing_titles.append(f"[{task_type}] {skill.get('title', '')}")
        example_ids = ", ".join(
            f'"dyn_{next_dyn_idx + j:03d}"'
            for j in range(self.max_new_skills_per_update)
        )

        return f"""Derive NEW flat actionable skills from the balanced successful and failed agent trajectories below.

Use only behavior supported by the supplied trajectories. Compare successes
against failures to identify reusable decisions and recovery rules. Do not
invent hidden object locations, unrecorded post-terminal states, or environment
facts absent from the evidence. The output must remain a flat skill list: do
not create headings, parent/child nodes, or a hierarchy.

SUCCESSFUL TRAJECTORIES:
{render("Success", successful_trajectories)}

FAILED TRAJECTORIES:
{render("Failure", failed_trajectories)}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

Generate 1-{self.max_new_skills_per_update} NEW actionable skills.
Each skill must have: skill_id, title (3-5 words), principle (1-2 sentences), when_to_apply.

Use skill_ids: {example_ids}

Return ONLY a JSON array of skills, no other text.
Example format:
[{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Verify Object Location First", "principle": "Before attempting to pick up an object, verify its location from the current observation.", "when_to_apply": "When the target location is uncertain"}}]
"""

    def _format_trajectory(self, steps: List[Dict]) -> str:
        lines = []
        for step in steps:
            action = step.get('action', 'unknown')
            obs = step.get('observation', '')[:200]
            lines.append(f"  Action: {action}\n  Observation: {obs}")
        return '\n'.join(lines)

    def _format_trajectory_full(self, steps: List[Dict]) -> str:
        lines = []
        for index, step in enumerate(steps, start=1):
            action = step.get("action", "unknown")
            observation = step.get("observation", "")
            lines.append(
                f"  Step {step.get('step', index)} observation_before_action:\n"
                f"{observation}\n"
                f"  Step {step.get('step', index)} action: {action}"
            )
        return "\n".join(lines)

    def _parse_skills_response(self, response: str) -> List[Dict]:
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                skills = json.loads(response[json_start:json_end])
                return [
                    s for s in skills
                    if all(k in s for k in ['skill_id', 'title', 'principle', 'when_to_apply'])
                ]
        except json.JSONDecodeError as e:
            print(f"[SkillUpdater] JSON parse error: {e}")
        return []

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {'total_updates': 0, 'total_skills_generated': 0}
        return {
            'total_updates': len(self.update_history),
            'total_skills_generated': sum(h['num_skills_generated'] for h in self.update_history),
            'all_skill_ids': [sid for h in self.update_history for sid in h['skill_ids']],
            'large_model_prompt_tokens': self.total_prompt_tokens,
            'large_model_completion_tokens': self.total_completion_tokens,
            'large_model_total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
        }
