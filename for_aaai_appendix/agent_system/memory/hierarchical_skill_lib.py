







""

from typing import Any, Dict, List, Optional, Tuple

from .skills_only_memory import SkillsOnlyMemory


class HierarchicalSkillLib(SkillsOnlyMemory):

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        enable_hierarchy: bool = True,
        stable_cycles_l1: int = 3,
        stable_cycles_l2: int = 5,
        success_l1: float = 0.7,
        demote_threshold: float = 0.3,
        min_calls: int = 20,
        min_task_types_l2: int = 3,
        enable_playbook: bool = True,
    ):
        super().__init__(
            skills_json_path=skills_json_path,
            retrieval_mode=retrieval_mode,
            embedding_model_path=embedding_model_path,
            task_specific_top_k=task_specific_top_k,
            enable_playbook=enable_playbook,
        )
        self.enable_hierarchy = enable_hierarchy
        self.stable_cycles_l1 = stable_cycles_l1
        self.stable_cycles_l2 = stable_cycles_l2
        self.success_l1 = success_l1
        self.demote_threshold = demote_threshold
        self.min_calls = min_calls
        self.min_task_types_l2 = min_task_types_l2
        self.cycle = 0









        if self.enable_hierarchy:
            for s in self._iter_all_skills():
                lc = self._ensure_lifecycle(s)
                lc["protected"] = True
                lc["layer"] = "L2"





    @staticmethod
    def _ensure_lifecycle(skill: Dict) -> Dict:
        lc = skill.get("lifecycle")
        if lc is None:
            lc = {}
            skill["lifecycle"] = lc
        lc.setdefault("layer", "L0")
        lc.setdefault("created_cycle", 0)
        lc.setdefault("last_modified_cycle", 0)
        lc.setdefault("stable_cycles", 0)
        lc.setdefault("call_count", 0)
        lc.setdefault("success_when_used", 0)
        lc.setdefault("success_rate", None)
        lc.setdefault("internalized", False)
        lc.setdefault("task_types_seen", [])
        lc.setdefault("protected", False)
        return lc

    def _iter_all_skills(self):
        for s in self.skills.get("general_skills", []):
            yield s
        for skills in self.skills.get("task_specific_skills", {}).values():
            for s in skills:
                yield s

    def _skill_by_id(self, skill_id: str) -> Optional[Dict]:
        for s in self._iter_all_skills():
            if s.get("skill_id") == skill_id:
                return s
        return None





    def ingest_patches(self, patches: List[Dict]) -> int:
        ""
        added = 0
        existing = self._get_all_skill_ids()
        for p in patches:
            sid = p.get("skill_id")
            if not sid or sid in existing:
                print(f"[HierSkillLib] skip duplicate/empty patch: {sid}")
                continue
            lc = self._ensure_lifecycle(p)
            lc["layer"] = "L0"
            lc["created_cycle"] = self.cycle
            lc["last_modified_cycle"] = self.cycle

            scope = p.get("scope", "general")
            if scope == "task_specific" and p.get("task_type"):
                cat = p["task_type"]
                self.skills.setdefault("task_specific_skills", {}).setdefault(cat, []).append(p)
            else:
                self.skills.setdefault("general_skills", []).append(p)
            existing.add(sid)
            added += 1
            print(f"[HierSkillLib] ingested L0 patch: {sid} - {p.get('title', 'N/A')}")
        if added > 0:
            self._skill_embeddings_cache = None
        return added





    def record_usage(self, skill_ids: List[str], success: bool,
                     task_type: Optional[str] = None) -> None:
        ""
        for sid in set(skill_ids or []):
            s = self._skill_by_id(sid)
            if s is None:
                continue
            lc = self._ensure_lifecycle(s)
            lc["call_count"] += 1
            if success:
                lc["success_when_used"] += 1
            cc = lc["call_count"]
            lc["success_rate"] = (lc["success_when_used"] / cc) if cc else None
            if task_type and task_type not in lc["task_types_seen"]:
                lc["task_types_seen"].append(task_type)





    def advance_lifecycle(self, modified_ids: Optional[List[str]] = None) -> Dict:
        ""
        self.cycle += 1
        modified = set(modified_ids or [])
        events = {"to_l1": [], "to_l2": [], "demoted": [], "deprecated": []}

        for s in self._iter_all_skills():
            lc = self._ensure_lifecycle(s)
            sid = s.get("skill_id")


            if sid in modified:
                lc["stable_cycles"] = 0
                lc["last_modified_cycle"] = self.cycle
            else:
                lc["stable_cycles"] += 1

            sr = lc["success_rate"]
            cc = lc["call_count"]
            layer = lc["layer"]



            if (not lc.get("protected", False)
                    and sr is not None and cc >= self.min_calls and sr < self.demote_threshold):
                if layer == "L2":
                    lc["layer"] = "L1"; events["demoted"].append(sid)
                elif layer == "L1":
                    lc["layer"] = "L0"; events["demoted"].append(sid)
                else:
                    lc["internalized"] = False
                    lc["deprecated"] = True
                    events["deprecated"].append(sid)
                continue

            if lc.get("deprecated"):
                continue


            if (layer == "L0"
                    and lc["stable_cycles"] >= self.stable_cycles_l1
                    and sr is not None and sr >= self.success_l1
                    and cc >= self.min_calls):
                lc["layer"] = "L1"; events["to_l1"].append(sid)

            elif (layer == "L1"
                    and lc["stable_cycles"] >= self.stable_cycles_l2
                    and len(lc["task_types_seen"]) >= self.min_task_types_l2):
                lc["layer"] = "L2"; events["to_l2"].append(sid)

        nonempty = {k: v for k, v in events.items() if v}
        if nonempty:
            print(f"[HierSkillLib] cycle {self.cycle} lifecycle events: {nonempty}")
        return events





    def retrieve(self, task_description: str, top_k: int = 6, **kwargs) -> Dict[str, Any]:
        ""
        if not self.enable_hierarchy:
            return super().retrieve(task_description, top_k=top_k, **kwargs)

        def keep(s: Dict) -> bool:
            lc = s.get("lifecycle") or {}
            return not lc.get("internalized", False) and not lc.get("deprecated", False)


        n_filtered = sum(1 for s in self._iter_all_skills() if not keep(s))
        ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
        result = super().retrieve(
            task_description, top_k=top_k + n_filtered, **kwargs
        )

        gen = [s for s in result.get("general_skills", []) if keep(s)][:top_k]
        task = [s for s in result.get("task_specific_skills", []) if keep(s)][:ts_top_k]
        result["general_skills"] = gen
        result["task_specific_skills"] = task

        result["injected_skill_ids"] = (
            [s.get("skill_id") for s in gen] + [s.get("skill_id") for s in task]
        )
        return result

    def get_cold_skills(self) -> List[Dict]:
        ""
        return [s for s in self._iter_all_skills()
                if (s.get("lifecycle") or {}).get("layer") == "L2"
                and not (s.get("lifecycle") or {}).get("internalized", False)]

    def has_cold_skills(self) -> bool:
        return len(self.get_cold_skills()) > 0

    def mark_internalized(self, skill_ids: List[str]) -> None:
        ""
        for sid in skill_ids:
            s = self._skill_by_id(sid)
            if s is not None:
                self._ensure_lifecycle(s)["internalized"] = True

    def layer_counts(self) -> Dict[str, int]:
        counts = {"L0": 0, "L1": 0, "L2": 0, "internalized": 0, "deprecated": 0}
        for s in self._iter_all_skills():
            lc = s.get("lifecycle") or {}
            counts[lc.get("layer", "L0")] = counts.get(lc.get("layer", "L0"), 0) + 1
            if lc.get("internalized"):
                counts["internalized"] += 1
            if lc.get("deprecated"):
                counts["deprecated"] += 1
        return counts
