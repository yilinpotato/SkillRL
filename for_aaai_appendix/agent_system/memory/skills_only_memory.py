# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""

import json
import os
from typing import Dict, Any, List, Optional
from .base import BaseMemory
from agent_system.task_taxonomy import classify_alfworld_task, classify_webshop_task


GLOBAL_SKILL_TREE_KEY = "__agent_skill_tree__"


class SkillsOnlyMemory(BaseMemory):
    ""





    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        enable_playbook: bool = True,
    ):
        ""
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )

        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"Skills file not found: {skills_json_path}")

        with open(skills_json_path, 'r') as f:
            self.skills = json.load(f)

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k
        self.enable_playbook = enable_playbook





        self.task_playbooks: Dict[str, Any] = {}
        json_playbooks = self.skills.get('skill_trees') or self.skills.get('task_playbooks')
        if isinstance(json_playbooks, dict):
            for k, v in json_playbooks.items():
                if isinstance(v, dict) and (v.get('content') or '').strip():
                    self.task_playbooks[k] = v
                elif isinstance(v, str) and v.strip():
                    self.task_playbooks[k] = v


        self._embedding_model = None
        self._skill_embeddings_cache: Optional[Dict] = None


        self._skill_vec_store: Optional[Dict] = None

        n_general = len(self.skills.get('general_skills', []))
        n_task = sum(len(v) for v in self.skills.get('task_specific_skills', {}).values())
        n_mistakes = len(self.skills.get('common_mistakes', []))
        print(
            f"[SkillsOnlyMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes  "
            f"| retrieval_mode={retrieval_mode} "
            f"| skill_tree={'on' if enable_playbook else 'off'} "
            f"({len(self.task_playbooks)} stored trees: {sorted(self.task_playbooks)})"
        )



        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()





    def _detect_task_type(self, task_description: str) -> str:
        ""
        task_specific = self.skills.get('task_specific_skills', {})
        if 'pick_and_place' in task_specific or 'clean' in task_specific:
            return classify_alfworld_task(task_description)
        if 'apparel' in task_specific or 'electronics' in task_specific:
            return classify_webshop_task(task_description)
        return next(iter(task_specific), 'unknown')





    @staticmethod
    def _playbook_content(pb: Any) -> Optional[str]:
        ""
        if isinstance(pb, dict):
            pb = pb.get('content')
        return pb if (isinstance(pb, str) and pb.strip()) else None

    def _tree_rl_states(self) -> Dict[str, Dict[str, Any]]:
        ""
        states = self.skills.get("skill_tree_rl")
        if not isinstance(states, dict):
            states = {}
            self.skills["skill_tree_rl"] = states
        return states

    @staticmethod
    def _live_playbook_levels(playbook: Dict[str, Any]) -> List[int]:
        ""
        return sorted({
            int(node.get("level", 0) or 0)
            for node in (playbook.get("nodes") or {}).values()
            if not node.get("deprecated", False)
            and not node.get("internalized", False)
            and int(node.get("level", 0) or 0) > 0
        })

    @staticmethod
    def _tree_rl_target_ids(playbook: Dict[str, Any], level: Optional[int]) -> List[str]:
        if level is None:
            return []
        return sorted(
            node_id
            for node_id, node in (playbook.get("nodes") or {}).items()
            if int(node.get("level", 0) or 0) == int(level)
            and not node.get("deprecated", False)
            and not node.get("internalized", False)
        )

    def _reset_tree_rl_stage(
        self,
        task_type: str,
        playbook: Dict[str, Any],
        state: Dict[str, Any],
        *,
        global_step: int,
        order: str,
        reason: str,
    ) -> Dict[str, Any]:
        ""
        levels = self._live_playbook_levels(playbook)
        target_level = (min(levels) if order == "root" else max(levels)) if levels else None
        target_ids = self._tree_rl_target_ids(playbook, target_level)
        state.update({
            "order": order,
            "tree_version": int(playbook.get("version", 0) or 0),
            "phase": "train" if target_ids else "done",
            "target_level": target_level,
            "target_node_ids": target_ids,
            "stage_started_step": int(global_step),
            "train_calls": 0,
            "train_successes": 0,
            "probe_calls": 0,
            "probe_successes": 0,
            "last_transition_reason": reason,
        })
        return state

    def _tree_rl_state_for_playbook(
        self,
        task_type: str,
        playbook: Dict[str, Any],
        *,
        order: str,
        global_step: int,
    ) -> Dict[str, Any]:
        states = self._tree_rl_states()
        state = states.get(task_type)
        if not isinstance(state, dict):
            state = {"attempts": 0, "completed_levels": []}
            states[task_type] = state




        if state.get("order") != order:
            state["attempts"] = 0
            state["completed_levels"] = list(state.get("completed_levels") or [])
            return self._reset_tree_rl_stage(
                task_type, playbook, state, global_step=global_step, order=order,
                reason="initialize" if not state.get("order") else "order_changed",
            )

        target_ids = set(state.get("target_node_ids") or [])
        current_ids = set((playbook.get("nodes") or {}).keys())


        if state.get("phase") != "done" and (not target_ids or not target_ids <= current_ids):
            return self._reset_tree_rl_stage(
                task_type, playbook, state, global_step=global_step, order=order,
                reason="tree_rewritten",
            )
        if state.get("phase") == "done" and self._live_playbook_levels(playbook):
            return self._reset_tree_rl_stage(
                task_type, playbook, state, global_step=global_step, order=order,
                reason="new_live_layer",
            )
        return state

    def get_playbook(self, task_type: str) -> Optional[str]:
        ""
        if not self.enable_playbook:
            return None
        pb = self.task_playbooks.get(task_type)
        content = self._playbook_content(pb)
        if content is None:
            return None
        if isinstance(pb, dict):
            skip = {nid for nid, lc in (pb.get("nodes") or {}).items()
                    if lc.get("deprecated")}
            elide = {nid for nid, lc in (pb.get("nodes") or {}).items()
                     if lc.get("internalized")}



            states = self.skills.get("skill_tree_rl") or {}
            state = states.get(task_type) if isinstance(states, dict) else None
            if isinstance(state, dict) and state.get("phase") == "probe":
                elide.update(state.get("target_node_ids") or [])
            if skip or elide:
                try:
                    from . import playbook_tree as _pt
                    return _pt.to_markdown(
                        _pt.parse(content), skip_ids=skip, elide_ids=elide)
                except Exception:
                    return content
        return content

    def get_playbook_record(self, task_type: str) -> Optional[Dict[str, Any]]:
        ""
        pb = self.task_playbooks.get(task_type)
        if isinstance(pb, dict):
            return pb
        if isinstance(pb, str) and pb.strip():
            return {"content": pb, "level": "outline", "version": 0}
        return None

    def update_playbook(
        self,
        task_type: str,
        content: str,
        level: str = "outline",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ""
        storage_key = task_type or "unknown"
        prev = self.get_playbook_record(storage_key)
        version = (prev.get("version", 0) if prev else 0) + 1
        record = {
            "content": content.strip(),
            "level": level or "outline",
            "version": version,
            "kind": "task_skill_tree",
            "task_type": storage_key,
            "history": [],
        }
        if meta:
            record.update({k: v for k, v in meta.items()
                           if k not in ("content", "level", "version", "history")})




        record["nodes"] = self._build_node_lifecycle(
            content.strip(), (prev or {}).get("nodes", {}), version)
        self.task_playbooks[storage_key] = record
        self.skills.setdefault('skill_trees', {})[storage_key] = record
        depths = sorted({lc.get("level", 0) for lc in record["nodes"].values()})
        print(f"[SkillsOnlyMemory] updated skill_tree[{storage_key}] -> "
              f"v{version} level={record['level']} ({len(record['content'])} chars, "
              f"{len(record['nodes'])} nodes, heading depths={depths})")
        return record

    @staticmethod
    def _build_node_lifecycle(content: str, prev_nodes: Dict[str, Any],
                              version: int) -> Dict[str, Any]:
        ""
        try:
            from . import playbook_tree as _pt
        except Exception:
            return {}
        idx = _pt.node_index(_pt.parse(content))
        nodes: Dict[str, Any] = {}
        for nid, info in idx.items():
            prev = prev_nodes.get(nid)
            if prev is None:
                nodes[nid] = {
                    "title": info["title"], "level": info["level"],
                    "body_hash": info["body_hash"],
                    "created_version": version, "last_changed_version": version,
                    "stable_versions": 0,
                    "call_count": 0, "success_when_used": 0,
                    "deprecated": False, "internalized": False,
                }
            else:
                changed = prev.get("body_hash") != info["body_hash"]
                nodes[nid] = {
                    **prev,
                    "title": info["title"], "level": info["level"],
                    "body_hash": info["body_hash"],
                    "last_changed_version": version if changed else prev.get("last_changed_version", version),
                    "stable_versions": 0 if changed else prev.get("stable_versions", 0) + 1,


                    "deprecated": False if changed else prev.get("deprecated", False),
                    "internalized": False if changed else prev.get("internalized", False),
                }
        return nodes

    def record_playbook_usage(self, task_type: str, success: bool) -> None:
        ""
        if not self.enable_playbook:
            return
        pb = self.task_playbooks.get(task_type)
        if not isinstance(pb, dict):
            return
        states = self.skills.get("skill_tree_rl") or {}
        state = states.get(task_type) if isinstance(states, dict) else None
        probe_target_ids = set()
        if isinstance(state, dict) and state.get("phase") == "probe":
            probe_target_ids = set(state.get("target_node_ids") or [])
            if probe_target_ids:
                state["probe_calls"] = int(state.get("probe_calls", 0) or 0) + 1
                state["probe_successes"] = int(state.get("probe_successes", 0) or 0) + int(bool(success))
        elif isinstance(state, dict) and state.get("phase") == "train":
            target_ids = set(state.get("target_node_ids") or [])
            if target_ids:
                state["train_calls"] = int(state.get("train_calls", 0) or 0) + 1
                state["train_successes"] = int(state.get("train_successes", 0) or 0) + int(bool(success))

        for nid, lc in (pb.get("nodes") or {}).items():
            if (lc.get("deprecated") or lc.get("internalized")
                    or nid in probe_target_ids):
                continue
            lc["call_count"] = lc.get("call_count", 0) + 1
            if success:
                lc["success_when_used"] = lc.get("success_when_used", 0) + 1

    def advance_tree_rl_curriculum(
        self,
        *,
        global_step: int,
        order: str = "root",
        min_rl_updates: int = 5,
        min_train_episodes: int = 24,
        train_success_threshold: float = 0.7,
        min_probe_episodes: int = 24,
        probe_success_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        ""
        if order not in {"root", "leaf"}:
            raise ValueError("tree_rl order must be 'root' or 'leaf'")
        if min_rl_updates < 1 or min_train_episodes < 1 or min_probe_episodes < 1:
            raise ValueError("tree_rl minimum update/episode counts must be positive")
        if not 0.0 <= train_success_threshold <= 1.0:
            raise ValueError("tree_rl train_success_threshold must be in [0, 1]")
        if not 0.0 <= probe_success_threshold <= 1.0:
            raise ValueError("tree_rl probe_success_threshold must be in [0, 1]")

        events: List[Dict[str, Any]] = []
        for task_type, playbook in sorted(self.task_playbooks.items()):
            if not isinstance(playbook, dict) or not (playbook.get("nodes") or {}):
                continue
            state = self._tree_rl_state_for_playbook(
                task_type, playbook, order=order, global_step=global_step)
            phase = state.get("phase")
            target_ids = self._tree_rl_target_ids(playbook, state.get("target_level"))


            if phase != "done" and not target_ids:
                state = self._reset_tree_rl_stage(
                    task_type, playbook, state, global_step=global_step, order=order,
                    reason="target_no_longer_live")
                phase = state.get("phase")

            if phase == "train":
                train_calls = int(state.get("train_calls", 0) or 0)
                train_successes = int(state.get("train_successes", 0) or 0)
                train_rate = train_successes / max(train_calls, 1)
                updates = int(global_step) - int(state.get("stage_started_step", global_step) or global_step) + 1
                if (updates >= min_rl_updates and train_calls >= min_train_episodes
                        and train_rate >= train_success_threshold):
                    state.update({
                        "phase": "probe", "probe_calls": 0, "probe_successes": 0,
                        "last_transition_reason": "trained_then_probe",
                    })
                    events.append({
                        "task_type": task_type, "event": "probe_started",
                        "level": state.get("target_level"), "node_ids": target_ids,
                        "train_calls": train_calls, "train_success_rate": train_rate,
                    })
            elif phase == "probe":
                probe_calls = int(state.get("probe_calls", 0) or 0)
                probe_successes = int(state.get("probe_successes", 0) or 0)
                probe_rate = probe_successes / max(probe_calls, 1)
                if probe_calls >= min_probe_episodes:
                    if probe_rate >= probe_success_threshold:
                        for node_id in target_ids:
                            playbook["nodes"][node_id]["internalized"] = True
                            playbook["nodes"][node_id]["internalized_at_step"] = int(global_step)
                        completed = list(state.get("completed_levels") or [])
                        completed.append(int(state.get("target_level")))
                        state["completed_levels"] = completed
                        events.append({
                            "task_type": task_type, "event": "layer_internalized",
                            "level": state.get("target_level"), "node_ids": target_ids,
                            "probe_calls": probe_calls, "probe_success_rate": probe_rate,
                        })
                        self._reset_tree_rl_stage(
                            task_type, playbook, state, global_step=global_step, order=order,
                            reason="probe_passed",
                        )
                    else:
                        state["attempts"] = int(state.get("attempts", 0) or 0) + 1
                        events.append({
                            "task_type": task_type, "event": "probe_failed_restore_layer",
                            "level": state.get("target_level"), "node_ids": target_ids,
                            "probe_calls": probe_calls, "probe_success_rate": probe_rate,
                        })
                        self._reset_tree_rl_stage(
                            task_type, playbook, state, global_step=global_step, order=order,
                            reason="probe_failed",
                        )
        return events

    def tree_rl_metrics(self) -> Dict[str, Any]:
        ""
        states = self.skills.get("skill_tree_rl") or {}
        all_nodes = [
            node for pb in self.task_playbooks.values() if isinstance(pb, dict)
            for node in (pb.get("nodes") or {}).values()
        ]
        active = [s for s in states.values() if isinstance(s, dict)]
        probing = [s for s in active if s.get("phase") == "probe"]
        training = [s for s in active if s.get("phase") == "train"]
        target_levels = [int(s.get("target_level")) for s in active
                         if s.get("target_level") is not None]
        return {
            "coskill/tree_rl/enabled": int(bool(active)),
            "coskill/tree_rl/n_trees": len(active),
            "coskill/tree_rl/n_training_trees": len(training),
            "coskill/tree_rl/n_probe_trees": len(probing),
            "coskill/tree_rl/n_internalized_nodes": sum(
                int(bool(node.get("internalized", False))) for node in all_nodes),
            "coskill/tree_rl/n_live_nodes": sum(
                int(not node.get("deprecated", False) and not node.get("internalized", False))
                for node in all_nodes),
            "coskill/tree_rl/target_level_min": min(target_levels) if target_levels else 0,
            "coskill/tree_rl/target_level_max": max(target_levels) if target_levels else 0,
        }

    def deprecate_playbook_node(self, task_type: str, node_id: str) -> bool:
        ""
        pb = self.task_playbooks.get(task_type)
        if not isinstance(pb, dict):
            return False
        lc = (pb.get("nodes") or {}).get(node_id)
        if lc is None:
            return False
        lc["deprecated"] = True
        return True





    def _get_embedding_model(self):
        ""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install with: pip install sentence-transformers"
                )





            device = os.environ.get("SKILL_EMBED_DEVICE", "cpu")
            print(f"[SkillsOnlyMemory] Loading embedding model: {self.embedding_model_path} (device={device})")
            self._embedding_model = SentenceTransformer(self.embedding_model_path, device=device)
            print("[SkillsOnlyMemory] Embedding model ready.")
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: Dict[str, Any]) -> str:
        ""
        parts = []
        for field in ('title', 'principle', 'when_to_apply', 'trigger'):
            val = (skill.get(field) or '').strip()
            if val:
                parts.append(val)

        action_flow = skill.get('action_flow')
        if isinstance(action_flow, list) and action_flow:
            parts.append("Steps: " + "; ".join(str(a) for a in action_flow))
        elif isinstance(action_flow, str) and action_flow.strip():
            parts.append("Steps: " + action_flow.strip())

        avoid = skill.get('avoid')
        if isinstance(avoid, list) and avoid:
            parts.append("Avoid: " + "; ".join(str(a) for a in avoid))
        elif isinstance(avoid, str) and avoid.strip():
            parts.append("Avoid: " + avoid.strip())

        return ". ".join(parts)

    def _compute_skill_embeddings(self) -> Dict:
        ""
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        import numpy as np

        general_items = [
            ('general', None, s)
            for s in self.skills.get('general_skills', [])
        ]
        task_items = [
            ('task_specific', task_type, s)
            for task_type, skills in self.skills.get('task_specific_skills', {}).items()
            for s in skills
        ]
        all_items = general_items + task_items

        if self._skill_vec_store is None:
            self._skill_vec_store = {}
        store = self._skill_vec_store


        keys = [self._vec_key(item[2]) for item in all_items]
        to_encode_idx = [i for i, k in enumerate(keys) if k not in store]
        if to_encode_idx:
            model = self._get_embedding_model()
            new_texts = [self._skill_to_text(all_items[i][2]) for i in to_encode_idx]
            new_vecs = model.encode(
                new_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for j, i in enumerate(to_encode_idx):
                store[keys[i]] = new_vecs[j]


        live_keys = set(keys)
        for k in list(store.keys()):
            if k not in live_keys:
                del store[k]

        embeddings = (
            np.stack([store[k] for k in keys], axis=0)
            if keys else np.zeros((0, 0))
        )

        self._skill_embeddings_cache = {
            'items': all_items,
            'embeddings': embeddings,
            'n_general': len(general_items),
        }
        print(
            f"[SkillsOnlyMemory] Embeddings ready for {len(all_items)} skills "
            f"({len(general_items)} general + {len(task_items)} task-specific); "
            f"encoded {len(to_encode_idx)} new/changed, reused {len(all_items) - len(to_encode_idx)}"
        )
        return self._skill_embeddings_cache

    @staticmethod
    def _vec_key(skill: Dict[str, Any]) -> tuple:
        ""
        import hashlib
        text = SkillsOnlyMemory._skill_to_text(skill)
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        return (skill.get('skill_id', ''), h)

    def _embedding_retrieve(
        self,
        task_description: str,
        top_k_general: int,
        top_k_task_specific: int,
    ):
        ""
        import numpy as np

        cache = self._compute_skill_embeddings()
        model = self._get_embedding_model()

        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        sims = cache['embeddings'] @ query_emb

        n_general = cache['n_general']
        general_sims = sims[:n_general]
        task_sims = sims[n_general:]


        general_idx = np.argsort(general_sims)[::-1][:top_k_general]
        general_skills = [cache['items'][int(i)][2] for i in general_idx]


        task_idx = np.argsort(task_sims)[::-1][:top_k_task_specific]
        task_skills = [cache['items'][n_general + int(i)][2] for i in task_idx]

        return general_skills, task_skills





    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        **kwargs,
    ) -> Dict[str, Any]:
        ""
        common_mistakes = self.skills.get('common_mistakes', [])[:5]




        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            general_skills, task_skills = self._embedding_retrieve(
                task_description=task_description,
                top_k_general=top_k,
                top_k_task_specific=ts_top_k,
            )

            task_type = self._detect_task_type(task_description)
            result = {
                'general_skills': general_skills,
                'task_specific_skills': task_skills,
                'mistakes_to_avoid': common_mistakes,
                'task_type': task_type,
                'playbook': self.get_playbook(task_type),
                'task_specific_examples': [],
                'retrieval_mode': 'embedding',
            }





            result['injected_skill_ids'] = [
                skill.get('skill_id')
                for skill in general_skills + task_skills
                if skill.get('skill_id')
            ]
            return result




        task_type = self._detect_task_type(task_description)





        all_general = self.skills.get('general_skills', [])
        dynamic_skills = [s for s in all_general if s.get('skill_id', '').startswith('dyn_')]
        static_skills = [s for s in all_general if not s.get('skill_id', '').startswith('dyn_')]
        n_static = max(0, top_k - len(dynamic_skills))
        general_skills = dynamic_skills + static_skills[:n_static]

        all_task_skills = self.skills.get('task_specific_skills', {}).get(task_type, [])

        if self.task_specific_top_k is not None:
            task_skills = all_task_skills[:self.task_specific_top_k]
        else:
            task_skills = all_task_skills

        result = {
            'general_skills': general_skills,
            'task_specific_skills': task_skills,
            'mistakes_to_avoid': common_mistakes,
            'task_type': task_type,
            'playbook': self.get_playbook(task_type),
            'task_specific_examples': [],
            'retrieval_mode': 'template',
        }



        result['injected_skill_ids'] = [
            skill.get('skill_id')
            for skill in general_skills + task_skills
            if skill.get('skill_id')
        ]
        return result

    @staticmethod
    def _format_skill_lines(skill: Dict[str, Any], include_when: bool = False) -> List[str]:
        ""
        title = skill.get('title', '')
        principle = skill.get('principle', '')
        lines = [f"- **{title}**: {principle}"]

        action_flow = skill.get('action_flow')
        if isinstance(action_flow, list) and action_flow:
            lines.append("  Do: " + " → ".join(str(a) for a in action_flow))
        elif isinstance(action_flow, str) and action_flow.strip():
            lines.append("  Do: " + action_flow.strip())

        avoid = skill.get('avoid')
        if isinstance(avoid, list) and avoid:
            lines.append("  Avoid: " + "; ".join(str(a) for a in avoid))
        elif isinstance(avoid, str) and avoid.strip():
            lines.append("  Avoid: " + avoid.strip())

        if include_when:
            when = skill.get('when_to_apply', '')


            if when and not (isinstance(action_flow, list) and action_flow):
                lines.append(f"  _Apply when: {when}_")
        return lines

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        ""
        sections = []
        task_type = retrieved_memories.get('task_type', 'unknown')
        mode = retrieved_memories.get('retrieval_mode', 'template')



        playbook = retrieved_memories.get('playbook')
        if self.enable_playbook and isinstance(playbook, str) and playbook.strip():
            sections.append(playbook.strip())


        general_skills = retrieved_memories.get('general_skills', [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                lines.extend(self._format_skill_lines(skill))
            sections.append("\n".join(lines))


        task_skills = retrieved_memories.get('task_specific_skills', [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = task_type.replace('_', ' ').title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                lines.extend(self._format_skill_lines(skill, include_when=True))
            sections.append("\n".join(lines))


        mistakes = retrieved_memories.get('mistakes_to_avoid', [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get('description', '')
                fix = mistake.get('how_to_avoid', '')
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."





    def reset(self, batch_size: int):
        pass

    def store(self, record: Dict[str, List[Any]]):
        pass

    def fetch(self, step: int):
        pass

    def __len__(self):
        return (
            len(self.skills.get('general_skills', [])) +
            sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()) +
            len(self.skills.get('common_mistakes', []))
        )

    def __getitem__(self, idx: int):
        return self.skills





    def add_skills(self, new_skills: List[Dict], category: str = 'general') -> int:
        ""
        added = 0
        existing_ids = self._get_all_skill_ids()

        for skill in new_skills:
            skill_id = skill.get('skill_id')
            if skill_id in existing_ids:
                print(f"[SkillsOnlyMemory] Skipping duplicate skill: {skill_id}")
                continue

            if category == 'general':
                self.skills.setdefault('general_skills', []).append(skill)
            else:
                self.skills.setdefault('task_specific_skills', {}).setdefault(category, []).append(skill)
            added += 1
            print(f"[SkillsOnlyMemory] Added skill: {skill_id} - {skill.get('title', 'N/A')}")

        if added > 0:

            self._skill_embeddings_cache = None

        return added

    def remove_skill(self, skill_id: str) -> bool:
        ""
        removed = False

        original_len = len(self.skills.get('general_skills', []))
        self.skills['general_skills'] = [
            s for s in self.skills.get('general_skills', [])
            if s.get('skill_id') != skill_id
        ]
        if len(self.skills.get('general_skills', [])) < original_len:
            removed = True

        for task_type in self.skills.get('task_specific_skills', {}):
            original_len = len(self.skills['task_specific_skills'][task_type])
            self.skills['task_specific_skills'][task_type] = [
                s for s in self.skills['task_specific_skills'][task_type]
                if s.get('skill_id') != skill_id
            ]
            if len(self.skills['task_specific_skills'][task_type]) < original_len:
                removed = True

        if removed:
            self._skill_embeddings_cache = None
            print(f"[SkillsOnlyMemory] Removed skill: {skill_id}")
        return removed

    def save_skills(self, path: str):
        ""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        print(f"[SkillsOnlyMemory] Saved {len(self)} skills to {path}")

    def _get_all_skill_ids(self) -> set:
        ids = set()
        for s in self.skills.get('general_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if s.get('skill_id'):
                    ids.add(s['skill_id'])
        return ids

    def get_skill_count(self) -> Dict[str, int]:
        return {
            'general': len(self.skills.get('general_skills', [])),
            'task_specific': sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()),
            'common_mistakes': len(self.skills.get('common_mistakes', [])),
            'total': len(self),
        }
