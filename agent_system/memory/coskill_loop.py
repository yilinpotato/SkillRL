# Copyright 2025 CoSkill.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
CoSkill Cloud Loop (云端闭环编排) —— 共享编排层。

把"水位线触发后的云端更新"从 ray_trainer 里抽出来，做成一个不依赖 verl/Ray/torch
的独立类，供两个调用方复用、保证行为 100% 一致、不漂移：
  - `verl.trainer.ppo.ray_trainer.RayPPOTrainer`（RL 训练路径）
  - `examples.playbook_evolve.run_playbook_evolve`（无 verl 的冻结模型 driver）

一次 `maybe_update(traces_pool, skill_lib, global_step)` 的内部序列（与原
`_update_skills_coskill` 步骤 2–6 完全一致）：
  should_trigger → export_batch
  → [enable_coskill] contrastive_distill → ingest_patches → advance_lifecycle
  → [enable_playbook_evolve] diagnose_failures → evolve per-task skill trees
       → update_playbook(compat API) → 落盘 cloud_io/
  → save_skills(skill_lib/skills_step{N}.json)
  → 写 coskill_status.json

调用方只需负责把轨迹喂进 `traces_pool`（verl 侧解析 batch，driver 侧直接组 RawTrace）。
"""

import json
import os
import time
from typing import Any, Dict, Optional


class CoSkillCloudLoop:
    """水位线触发的云端更新编排。进程内单实例即可。"""

    def __init__(
        self,
        output_dir: str,
        *,
        enable_coskill: bool = False,
        enable_playbook_evolve: bool = False,
        enable_failure_analysis: bool = True,
        max_new_skills: int = 3,
        playbook_evolve_min_samples: int = 6,
        coskill_debug: bool = False,
        environment_name: str = "generic",
    ):
        self.output_dir = output_dir
        self.enable_coskill = enable_coskill
        self.enable_playbook_evolve = enable_playbook_evolve
        self.enable_failure_analysis = enable_failure_analysis
        self.max_new_skills = max_new_skills
        self.playbook_evolve_min_samples = playbook_evolve_min_samples
        self.coskill_debug = coskill_debug
        self.environment_name = str(environment_name or "generic")

        self.cloud_analyzer = None
        self._analyzer_init_failed = False
        self.last_timing: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # CloudAnalyzer 懒初始化（缺 DEEPSEEK_API_KEY 时不崩，跳过本轮）         #
    # ------------------------------------------------------------------ #

    def _get_analyzer(self):
        if self.cloud_analyzer is None and not self._analyzer_init_failed:
            from .cloud_analyzer import CloudAnalyzer
            try:
                self.cloud_analyzer = CloudAnalyzer(
                    max_new_skills_per_update=self.max_new_skills,
                    output_dir=self.output_dir,
                    environment_name=self.environment_name,
                )
            except Exception as e:
                # e.g. missing DEEPSEEK_API_KEY. The compressed batch is already
                # persisted to cloud_io/ and can be distilled later.
                print(f"[CoSkill] CloudAnalyzer init failed ({e}); "
                      f"skipping distillation, batch saved to cloud_io/")
                self._analyzer_init_failed = True
                self.cloud_analyzer = None
        return self.cloud_analyzer

    # ------------------------------------------------------------------ #
    # 主入口                                                               #
    # ------------------------------------------------------------------ #

    def maybe_update(
        self,
        traces_pool,
        skill_lib,
        global_step: int,
        force_reason: Optional[str] = None,
    ) -> bool:
        """水位线触发则跑一轮云端更新，返回是否真的更新了。

        ``force_reason`` 仅供可复现实验在确定的 episode 边界强制更新；生产
        trainer 不传该参数，仍完全遵循容量/表现双水位线。这样固定任务 A/B 的
        两臂会在相同样本数上调用云端，而不受轨迹文本长度差异干扰。
        """
        if force_reason:
            fire, reason = True, force_reason
        else:
            fire, reason = traces_pool.should_trigger()
        if self.coskill_debug:
            print(f"[CoSkill][dbg] step={global_step} "
                  f"pool={traces_pool.stats()} trigger=({fire},{reason})")
        if not fire:
            return False

        if skill_lib is None:
            print("[CoSkill] No skill_lib provided, skipping update")
            return False

        if not (self.enable_coskill or self.enable_playbook_evolve):
            if self.coskill_debug:
                print(f"[CoSkill][dbg] update skipped ({reason}): "
                      "skill bullets and skill tree evolution are both disabled")
            return False

        # Export compressed batch.
        _t0 = time.time()
        compressed = traces_pool.export_batch(trigger_reason=reason)
        export_seconds = time.time() - _t0
        print(f"[CoSkill] watermark fired ({reason}); "
              f"succ={compressed['stats']['n_success']} fail={compressed['stats']['n_failure']}")

        analyzer = self._get_analyzer()
        if analyzer is None:
            return False

        # Skill distillation (bullets) — only when enable_coskill.
        current_skills = getattr(skill_lib, 'skills', {})
        patches = []
        distill_seconds = 0.0
        if self.enable_coskill:
            _t1 = time.time()
            patches = analyzer.contrastive_distill(compressed, current_skills)
            distill_seconds = time.time() - _t1
        self.last_timing = {
            'export_seconds': export_seconds,
            'distill_seconds': distill_seconds,
            'last_trigger_reason': reason,
        }

        # Debug: dump structured patches.
        if self.coskill_debug and patches:
            try:
                dbg_dir = os.path.join(self.output_dir, 'cloud_io')
                os.makedirs(dbg_dir, exist_ok=True)
                dbg_path = os.path.join(dbg_dir, f'patches_step{global_step}_debug.json')
                with open(dbg_path, 'w') as f:
                    json.dump([
                        {k: p.get(k) for k in ('skill_id', 'title', 'scope', 'task_type',
                                               'trigger', 'action_flow', 'avoid')}
                        for p in patches
                    ], f, ensure_ascii=False, indent=2)
                print(f"[CoSkill][dbg] dumped {len(patches)} patches → {dbg_path}")
            except Exception as e:
                print(f"[CoSkill][dbg] patch dump failed: {e}")

        # Ingest patches + advance lifecycle.
        added_ids = []
        if patches and hasattr(skill_lib, 'ingest_patches'):
            skill_lib.ingest_patches(patches)
            added_ids = [p.get('skill_id') for p in patches]
        if self.enable_coskill and hasattr(skill_lib, 'advance_lifecycle'):
            skill_lib.advance_lifecycle(modified_ids=added_ids)
            if hasattr(skill_lib, 'layer_counts'):
                print(f"[CoSkill] layer counts: {skill_lib.layer_counts()}")

        # Skill-tree evolution.
        if self.enable_playbook_evolve and hasattr(skill_lib, 'update_playbook'):
            _tp = time.time()
            self._evolve_playbooks(analyzer, skill_lib, compressed, global_step)
            self.last_timing['playbook_seconds'] = time.time() - _tp

        # Persist evolved skill lib.
        save_dir = os.path.join(self.output_dir, 'skill_lib')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'skills_step{global_step}.json')
        if hasattr(skill_lib, 'save_skills'):
            skill_lib.save_skills(save_path)
            print(f"[CoSkill] saved evolved skill lib to {save_path}")

        # Health snapshot.
        self._write_status(traces_pool, skill_lib, reason, global_step)
        return True

    # ------------------------------------------------------------------ #
    # Skill-tree 进化：按 task_type 诊断 + 生成/细化                         #
    # ------------------------------------------------------------------ #

    def _evolve_playbooks(self, analyzer, skill_lib, compressed, global_step) -> None:
        """Diagnose failures (batched) then evolve one skill tree per task type.

        The large model authors/refines a separate tree for each task type that
        has enough fresh evidence in the exported batch. ``update_playbook``
        remains the compatibility API, but now stores the tree under that
        task_type, so the next rollout injects only the matching tree.
        """
        min_samples = self.playbook_evolve_min_samples

        # 1) One batched failure diagnosis for the whole compressed batch.
        diagnoses = {}
        if self.enable_failure_analysis:
            diagnoses = analyzer.diagnose_failures(compressed) or {}

        succ_all = compressed.get('success_samples', []) or []
        fail_all = compressed.get('failure_samples', []) or []

        by_type = {}
        for tr in succ_all + fail_all:
            tt = tr.get('task_type') or 'unknown'
            by_type.setdefault(tt, {'success': [], 'failure': []})
            by_type[tt]['success' if tr.get('outcome') == 'success' else 'failure'].append(tr)

        for task_type in sorted(by_type):
            succ = by_type[task_type]['success']
            fail = by_type[task_type]['failure']
            if len(succ) + len(fail) < min_samples:
                print(f"[CoSkill] skill_tree[{task_type}] skipped: "
                      f"{len(succ)+len(fail)} < min_samples={min_samples}")
                continue

            task_diags = []
            for d in (diagnoses or {}).get(task_type, []) or []:
                task_diags.append(d)

            current = skill_lib.get_playbook_record(task_type)
            current_content = (current or {}).get('content') if current else None
            result = analyzer.evolve_playbook(
                task_type=task_type,
                current_playbook=current_content,
                success_traces=succ,
                failure_traces=fail,
                diagnoses=task_diags,
                history=[],
            )
            if not result:
                continue
            tree_text = result.get('skill_tree') or result.get('playbook') or ''
            if result.get('action') == 'keep' or not tree_text:
                print(f"[CoSkill] skill_tree[{task_type}] kept "
                      f"(v{(current or {}).get('version', 0)})")
                continue

            round_failures = [{'failure_type': d.get('failure_type', 'other'),
                               'root_cause': d.get('root_cause', ''),
                               'skill_tree_gap': d.get('skill_tree_gap', d.get('playbook_gap', '')),
                               'task_type': d.get('task_type', task_type)}
                              for d in task_diags[:12]]
            rec = skill_lib.update_playbook(
                task_type=task_type,
                content=tree_text,
                level=result.get('level', 'outline'),
                meta={'critique': result.get('critique', ''),
                      'changelog': result.get('changelog', ''),
                      'round_failures': round_failures,
                      'task_scope': task_type,
                      'updated_at': f'step_{global_step}'},
            )
            # Dump each installed version for inspection.
            if getattr(analyzer, 'playbook_io_dir', None):
                safe_tt = str(task_type).replace("/", "_").replace(" ", "_")
                try:
                    path = os.path.join(analyzer.playbook_io_dir,
                                        f'skill_tree_{safe_tt}_v{rec["version"]}.json')
                    with open(path, 'w') as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[CoSkill] skill tree dump failed: {e}")
                self._dump_playbook_debug(analyzer, safe_tt, rec, result, task_diags,
                                          prev_content=current_content)

    def _dump_playbook_debug(self, analyzer, task_type, rec, result, diagnoses,
                             prev_content=None) -> None:
        """人类可读的 skill tree debug：本轮 action/critique/共识折叠、失败诊断，
        以及**逐节点生命周期表**（新增/稳定几版/改写/累积调用与成功率/是否被剪枝），
        便于直接核对"层次化技能树按需生长"这个机制是否真的按预期在工作。
        """
        try:
            from . import playbook_tree as _pt
        except Exception:
            return
        nodes = rec.get('nodes') or {}
        version = rec.get('version')
        content = rec.get('content', '')
        try:
            depth = _pt.max_depth(_pt.parse(content))
        except Exception:
            depth = '?'

        lines = [
            f"=== Skill Tree Debug — scope={task_type}  v{version} "
            f"(level={rec.get('level')}, tree_depth={depth}, {len(nodes)} nodes) ===",
            f"action: {result.get('action')}",
            f"critique: {result.get('critique', '')}",
            f"changelog: {result.get('changelog', '')}",
            f"consensus prefix (folded, not re-taught this round): "
            f"{' → '.join(result.get('consensus_prefix') or []) or '(none)'}",
            f"n_success this round: {result.get('n_success', '?')}  "
            f"n_failure this round: {result.get('n_failure', '?')}",
            "",
            "--- Failure diagnoses feeding this round ---",
        ]
        if diagnoses:
            for d in diagnoses:
                lines.append(f"- [{d.get('failure_type', '?')}] {d.get('root_cause', '')} "
                             f"| evidence: {d.get('evidence', '')} "
                             f"| fix: {d.get('corrective_rule', '')} "
                             f"| skill_tree_gap: {d.get('skill_tree_gap', d.get('playbook_gap', ''))} "
                             f"| patch_location: {d.get('patch_location', '(unspecified)')}")
        else:
            lines.append("(none)")

        lines.append("")
        lines.append(f"--- Node tree (per-node lifecycle, this version=v{version}) ---")
        for nid, lc in sorted(nodes.items()):
            if lc.get('created_version') == version:
                tag = 'NEW'
            elif lc.get('last_changed_version') == version:
                tag = 'MODIFIED'
            else:
                tag = 'STABLE'
            if lc.get('deprecated'):
                tag = 'DEPRECATED(pruned)'
            elif lc.get('internalized'):
                tag = 'INTERNALIZED(pruned)'
            cc = lc.get('call_count', 0)
            sw = lc.get('success_when_used', 0)
            rate = f"{100.0 * sw / cc:.1f}%" if cc else "n/a"
            lines.append(
                f"[{tag:<20}] {nid:<40} level={lc.get('level')} "
                f"stable_versions={lc.get('stable_versions', 0)} "
                f"calls={cc} success={sw} rate={rate}"
            )

        lines.append("")
        lines.append(f"--- Line diff vs previous version (v{version - 1} -> v{version}) ---")
        if prev_content is None:
            lines.append("(none — this is the first version, nothing to diff against)")
        elif prev_content.strip() == content.strip():
            lines.append("(unchanged)")
        else:
            import difflib
            diff = list(difflib.unified_diff(
                prev_content.splitlines(), content.splitlines(),
                fromfile=f"v{version - 1}", tofile=f"v{version}", lineterm="",
            ))
            lines.extend(diff if diff else ["(unchanged)"])

        lines.append("")
        lines.append("--- Rendered content actually shown to the agent (post-prune) ---")
        skip = {nid for nid, lc in nodes.items()
                if lc.get('deprecated') or lc.get('internalized')}
        try:
            rendered = _pt.to_markdown(_pt.parse(content), skip_ids=skip) if skip else content
        except Exception:
            rendered = content
        lines.append(rendered)

        try:
            path = os.path.join(analyzer.playbook_io_dir,
                                f'skill_tree_{task_type}_v{version}_debug.txt')
            with open(path, 'w') as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            print(f"[CoSkill] skill tree debug dump failed ({task_type}): {e}")

    # ------------------------------------------------------------------ #
    # 观测：健康快照 + metrics                                             #
    # ------------------------------------------------------------------ #

    def _write_status(self, traces_pool, skill_lib, last_reason, global_step) -> None:
        """Write OUTPUT_DIR/coskill_status.json (overwritten each update)."""
        status = {
            'global_step': global_step,
            'last_trigger_reason': last_reason,
            'timing': self.last_timing,
        }
        if traces_pool is not None:
            status['pool'] = traces_pool.stats()
        if hasattr(skill_lib, 'layer_counts'):
            status['skill_lib'] = skill_lib.layer_counts()
        if self.cloud_analyzer is not None:
            status['cloud'] = self.cloud_analyzer.get_update_summary()
        try:
            path = os.path.join(self.output_dir, 'coskill_status.json')
            with open(path, 'w') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CoSkill] status write failed: {e}")

    def metrics(self, traces_pool=None, skill_lib=None) -> Dict[str, Any]:
        """CoSkill observability metrics (pool / skill-lib / cloud / skill tree /
        timing). Only includes blocks whose backing object exists."""
        m: Dict[str, Any] = {}
        if traces_pool is not None:
            s = traces_pool.stats()
            m['coskill/pool/total_added'] = s.get('total_added', 0)
            m['coskill/pool/pending_added'] = s.get('pending_added', 0)
            m['coskill/pool/pending_tokens'] = s.get('pending_tokens', 0)
            m['coskill/pool/total_dropped_loops'] = s.get('total_dropped_loops', 0)
            m['coskill/pool/n_task_types'] = len(s.get('task_types', []))
        if skill_lib is not None and hasattr(skill_lib, 'layer_counts'):
            for k, v in skill_lib.layer_counts().items():
                m[f'coskill/skilllib/{k}'] = v
        if self.cloud_analyzer is not None:
            summary = self.cloud_analyzer.get_update_summary()
            m['coskill/cloud/total_updates'] = summary.get('total_updates', 0)
            m['coskill/cloud/total_patches'] = summary.get('total_patches', 0)
            m['coskill/cloud/large_model_prompt_tokens'] = summary.get(
                'large_model_prompt_tokens', 0)
            m['coskill/cloud/large_model_completion_tokens'] = summary.get(
                'large_model_completion_tokens', 0)
            m['coskill/cloud/large_model_total_tokens'] = summary.get('large_model_total_tokens', 0)
            m['coskill/skill_tree/diagnose_calls'] = summary.get('diagnose_calls', 0)
            m['coskill/skill_tree/evolve_calls'] = summary.get('evolve_calls', 0)
            m['coskill/skill_tree/updates'] = summary.get(
                'skill_tree_updates', summary.get('playbook_updates', 0))
        if skill_lib is not None and hasattr(skill_lib, 'task_playbooks'):
            m['coskill/skill_tree/n_trees'] = len(getattr(skill_lib, 'task_playbooks', {}) or {})
        if self.last_timing:
            m['coskill/timing/export_seconds'] = self.last_timing.get('export_seconds', 0.0)
            m['coskill/timing/distill_seconds'] = self.last_timing.get('distill_seconds', 0.0)
            m['coskill/timing/skill_tree_seconds'] = self.last_timing.get('playbook_seconds', 0.0)
            reason = self.last_timing.get('last_trigger_reason')
            m['coskill/last_trigger_reason'] = (
                3 if str(reason or '').startswith('episode_interval_')
                else 2 if reason in ('performance_watermark', 'success_stagnation', 'success_decline')
                else 1 if reason == 'capacity_watermark' else 0
            )
        return m
