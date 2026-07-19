"""Standalone, fixed-task WebShop mini test with ALFWorld-style artifacts.

For every run this driver writes:
  * one JSON + readable trajectory + complete prompts file per task;
  * resolved_tasks.json with the exact fixed WebShop goals;
  * summary.json, summary.txt, and a self-contained report.html.

The default manifest contains two tasks from each native WebShop product
category (fashion/garden/beauty/electronics/grocery).  ``baseline`` and
``template`` therefore evaluate exactly the same ten goal indices.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent_system.environments.env_package.webshop.projection import webshop_projection
from mini_test_pen_shelf.webshop_template import MAX_SCORE_WEBSHOP_PLAYBOOK
from mini_test_pen_shelf.webshop_utils import (
    WebShopObsBuilder,
    _load_webshop_env_class,
    extract_webshop_task,
    format_webshop_actions,
    format_webshop_observation,
    webshop_trace_observation,
)


NATIVE_CATEGORIES = ("fashion", "garden", "beauty", "electronics", "grocery")
DEFAULT_MANIFEST = Path(__file__).with_name("webshop_tasks_2_per_category.json")
DEFAULT_SKILLS = Path(__file__).parents[1] / "memory_data/webshop/claude_style_skills.json"


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _safe_name(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_")
    return value[:60] or "unknown"


def _one_line(text: Any, limit: Optional[int] = None) -> str:
    value = " ".join(str(text or "").split())
    if limit and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _think(raw: str) -> str:
    match = re.search(r"<think>(.*?)</think>", raw or "", re.I | re.S)
    return match.group(1).strip() if match else ""


def find_webshop_data_dir(explicit: Optional[str] = None) -> Path:
    """Find the small 1000-product data and its required Lucene index."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path(os.environ["WEBSHOP_DATA_DIR"]).expanduser()
        if os.environ.get("WEBSHOP_DATA_DIR") else None,
        root / "agent_system/environments/env_package/webshop/webshop/data",
        root.parent / "Skill0/agent_system/environments/env_package/webshop/webshop/data",
        Path("/data2/myl/Skill0/agent_system/environments/env_package/webshop/webshop/data"),
    ]
    required = ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json")
    tried = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        tried.append(str(candidate))
        if (all((candidate / name).is_file() for name in required)
                and (candidate.parent / "search_engine/indexes").is_dir()):
            return candidate
    raise FileNotFoundError(
        "没有找到完整 WebShop 资源（3 个 JSON + search_engine/indexes）。\n"
        "请设置 WEBSHOP_DATA_DIR。已检查:\n  - " + "\n  - ".join(tried)
    )


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    tasks = manifest.get("tasks") or []
    if not tasks:
        raise ValueError(f"Manifest has no tasks: {path}")
    counts = Counter(str(item.get("category")) for item in tasks)
    expected = int(manifest.get("tasks_per_category", 2))
    missing = [category for category in NATIVE_CATEGORIES if counts[category] != expected]
    if missing:
        raise ValueError(
            f"Manifest must contain {expected} tasks for every native category; "
            f"bad categories={missing}, counts={dict(counts)}")
    indices = [int(item["goal_index"]) for item in tasks]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Manifest contains duplicate goal indices: {indices}")
    return manifest


def resolve_tasks(server, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Join the compact manifest with authoritative shuffled WebShop goals."""
    resolved = []
    for order, expected in enumerate(manifest["tasks"], 1):
        goal_index = int(expected["goal_index"])
        if goal_index < 0 or goal_index >= len(server.goals):
            raise IndexError(f"goal_index {goal_index} outside [0, {len(server.goals)})")
        goal = server.goals[goal_index]
        for key in ("category", "query", "asin"):
            actual = str(goal.get(key, "")).lower()
            wanted = str(expected.get(key, "")).lower()
            if wanted and actual != wanted:
                raise ValueError(
                    f"Manifest drift at goal_index={goal_index}: {key} expected "
                    f"{expected.get(key)!r}, dataset has {goal.get(key)!r}")
        resolved.append({
            "task_idx": order,
            "goal_index": goal_index,
            "category": goal["category"],
            "query": goal["query"],
            "asin": goal["asin"],
            "product_name": goal.get("name"),
            "product_category": goal.get("product_category"),
            "instruction": goal["instruction_text"],
            "required_attributes": list(goal.get("attributes") or []),
            "required_options": dict(goal.get("goal_options") or {}),
            "price_upper": goal.get("price_upper"),
        })
    return resolved


def _is_executable(action: str, available: Dict[str, Any]) -> bool:
    match = re.fullmatch(r"\s*(search|click)\[(.*)]\s*", action or "", re.I | re.S)
    if not match:
        return False
    kind, argument = match.group(1).lower(), match.group(2).strip().lower()
    if kind == "search":
        return bool(argument and available.get("has_search_bar"))
    return argument in {str(value).lower() for value in available.get("clickables", [])}


def _salvage_executable_action(
    raw: str,
    projected: str,
    available: Dict[str, Any],
) -> Tuple[str, str]:
    """Recover only actions that are provably executable in the current page.

    This mirrors the ALFWorld mini test's admissible-action salvage.  It never
    invents an unavailable click and does not use target ASIN/goal metadata.
    """
    if _is_executable(projected, available):
        return projected, "direct"

    # Prefer an explicit bare/bracketed action near the end of malformed output.
    bracketed = re.findall(r"(?:<action>\s*)?((?:search|click)\[[^\]\n]+])",
                           raw or "", flags=re.I)
    for candidate in reversed(bracketed):
        candidate = candidate.strip().lower()
        if _is_executable(candidate, available):
            return candidate, "salvaged_explicit"

    # Forced thoughts often conclude "click/open/select B0..." but run out of
    # action tokens before emitting tags. Recover that mentioned current link.
    clickables = [str(value).lower() for value in available.get("clickables", [])]
    lowered = str(raw or "").lower()
    ranked = []
    for clickable in clickables:
        if clickable in {"search", "back to search", "back to results", "next >", "< prev"}:
            continue
        positions = [match.start() for match in re.finditer(re.escape(clickable), lowered)]
        for position in positions:
            prefix = lowered[max(0, position - 80):position]
            if re.search(r"\b(?:click|open|select|choose|pick|candidate|best|first)\b", prefix):
                ranked.append((position, clickable))
    if ranked:
        _, clickable = max(ranked)
        candidate = f"click[{clickable}]"
        if _is_executable(candidate, available):
            return candidate, "salvaged_mentioned_clickable"
    return projected, "malformed"


class WebShopEpisodeLogger:
    def __init__(self, outdir: Path, task: Dict[str, Any], variant: str):
        self.outdir = outdir
        self.task = task
        self.variant = variant
        self.steps: List[Dict[str, Any]] = []

    def add(self, **row: Any) -> None:
        self.steps.append(row)

    def flush(self, result: Dict[str, Any]) -> Dict[str, str]:
        status = "WIN" if result["won"] else ("BOUGHT" if result["purchased"] else "FAIL")
        base = (
            f"task_{self.task['task_idx']:02d}_{_safe_name(self.task['category'])}_"
            f"{status}_score{result['task_score']:.3f}_{result['used_steps']}steps"
        )
        json_path = self.outdir / f"{base}.json"
        trajectory_path = self.outdir / f"{base}_trajectory.txt"
        prompts_path = self.outdir / f"{base}_prompts.txt"
        payload = {
            "variant": self.variant,
            **self.task,
            **{key: value for key, value in result.items() if key != "files"},
            "steps": self.steps,
        }
        _json_dump(payload, json_path)

        lines = [
            "=" * 92,
            f"WebShop Task #{self.task['task_idx']:02d} | {self.task['category']} | "
            f"{status} | score={result['task_score']:.4f} | {result['used_steps']} steps",
            f"variant={self.variant}  goal_index={self.task['goal_index']}  "
            f"target_asin={self.task['asin']}  purchased_asin={result.get('purchased_asin') or '-'}",
            f"instruction: {self.task['instruction']}",
            f"required options: {self.task['required_options']}",
            f"required attributes: {self.task['required_attributes']}",
            "=" * 92,
        ]
        for step in self.steps:
            flags = [
                "TAG_OK" if step["strict_valid_action"] else "TAG_BAD",
                "EXEC_OK" if step["executable_action"] else "EXEC_BAD",
                "FORCED" if step["forced"] else "NATURAL",
            ]
            lines.extend([
                "",
                f"┌─ Step {step['step']:02d} [{' | '.join(flags)}]",
                f"│ page before : {_one_line(step['observation_before'], 1800)}",
                f"│ think       : {_one_line(step['think'], 2400)}",
                f"│ action      : {step['action']}",
                f"│ page after  : {_one_line(step['observation_after'], 1800)}",
                f"│ done={step['done']} score={step['task_score']:.4f} won={step['won']}",
                "└" + "─" * 72,
            ])
        trajectory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        prompt_lines = [
            "#" * 92,
            f"# WebShop Task #{self.task['task_idx']:02d} complete prompts",
            f"# variant={self.variant} goal_index={self.task['goal_index']} "
            f"score={result['task_score']:.4f}",
            f"# instruction: {self.task['instruction']}",
            "#" * 92,
        ]
        for step in self.steps:
            prompt_lines.extend([
                "", "/" * 92,
                f"// Step {step['step']} | action={step['action']!r} | "
                f"strict={step['strict_valid_action']} executable={step['executable_action']}",
                "/" * 92,
                step["prompt"],
                "\n// RAW MODEL OUTPUT\n" + step["raw_model_output"],
            ])
        prompts_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
        return {
            "json": json_path.name,
            "trajectory": trajectory_path.name,
            "prompts": prompts_path.name,
        }


def _aggregate(results: Sequence[Dict[str, Any]], variant: str, args,
               elapsed: float, token_usage: Dict[str, int]) -> Dict[str, Any]:
    total = len(results)
    steps = sum(int(item["used_steps"]) for item in results)
    strict = sum(int(item["strict_valid_actions"]) for item in results)
    executable = sum(int(item["executable_actions"]) for item in results)

    def stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(rows)
        return {
            "count": count,
            "wins": sum(int(row["won"]) for row in rows),
            "success_rate": round(sum(int(row["won"]) for row in rows) / max(count, 1), 4),
            "purchases": sum(int(row["purchased"]) for row in rows),
            "purchase_rate": round(sum(int(row["purchased"]) for row in rows) / max(count, 1), 4),
            "mean_task_score": round(
                sum(float(row["task_score"]) for row in rows) / max(count, 1), 4),
            "mean_steps": round(
                sum(int(row["used_steps"]) for row in rows) / max(count, 1), 2),
        }

    by_category = {}
    for category in NATIVE_CATEGORIES:
        by_category[category] = stats([row for row in results if row["category"] == category])
    overall = stats(results)
    return {
        "benchmark": "WebShop",
        "variant": variant,
        "template_enabled": variant == "template",
        "template_name": "MAX_SCORE_WEBSHOP_PLAYBOOK" if variant == "template" else None,
        "model_path": args.model_path or os.environ.get("MODEL_PATH"),
        "dataset_seed": args.seed,
        "manifest": str(Path(args.manifest).resolve()),
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "temperature": args.temperature,
        "think_budget": args.think_budget,
        "action_budget": args.action_budget,
        "force_action_prefix": True,
        "elapsed_seconds": round(elapsed, 2),
        "num_tasks": total,
        **overall,
        "strict_valid_action_rate": round(strict / max(steps, 1), 4),
        "executable_action_rate": round(executable / max(steps, 1), 4),
        "forced_thinking_steps": sum(int(row["forced_steps"]) for row in results),
        "salvaged_actions": sum(int(row["salvaged_actions"]) for row in results),
        "token_usage": token_usage,
        "by_category": by_category,
        "per_task": list(results),
    }


def write_reports(summary: Dict[str, Any], outdir: Path) -> None:
    lines = [
        "=" * 92,
        f"WebShop Mini Test | variant={summary['variant']} | tasks={summary['num_tasks']}",
        "=" * 92,
        f"成功 {summary['wins']}/{summary['num_tasks']} ({summary['success_rate']*100:.1f}%)  |  "
        f"平均得分 {summary['mean_task_score']:.4f}  |  "
        f"购买率 {summary['purchase_rate']*100:.1f}%  |  平均步数 {summary['mean_steps']}",
        f"动作：strict tags {summary['strict_valid_action_rate']*100:.1f}%  |  "
        f"环境可执行 {summary['executable_action_rate']*100:.1f}%  |  "
        f"thinking forced {summary['forced_thinking_steps']} steps  |  "
        f"salvaged {summary['salvaged_actions']} actions",
        "",
        f"{'类别':<16}{'成功':>10}{'平均分':>12}{'购买率':>12}{'平均步数':>12}  可视化",
        "-" * 92,
    ]
    for category in NATIVE_CATEGORIES:
        row = summary["by_category"][category]
        bar = "█" * int(round(row["mean_task_score"] * 20))
        lines.append(
            f"{category:<16}{row['wins']}/{row['count']:>7}"
            f"{row['mean_task_score']:>12.4f}{row['purchase_rate']*100:>11.1f}%"
            f"{row['mean_steps']:>12.2f}  {bar}"
        )
    lines.extend(["", "逐任务：", "-" * 92])
    for row in summary["per_task"]:
        mark = "✅" if row["won"] else ("🟡" if row["purchased"] else "❌")
        lines.append(
            f"{mark} #{row['task_idx']:02d} {row['category']:<12} "
            f"goal={row['goal_index']:<4} score={row['task_score']:.4f} "
            f"steps={row['used_steps']:<2} query={row['query']}"
        )
        lines.append(f"   task: {_one_line(row['instruction'], 160)}")
        lines.append(f"   trajectory: {row['files']['trajectory']}")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cards = []
    for category in NATIVE_CATEGORIES:
        row = summary["by_category"][category]
        cards.append(f"""
        <section class="card">
          <h3>{html.escape(category)}</h3>
          <div class="big">{row['mean_task_score']:.3f}</div>
          <div class="bar"><span style="width:{row['mean_task_score']*100:.1f}%"></span></div>
          <p>{row['wins']}/{row['count']} exact wins · {row['purchase_rate']*100:.0f}% purchased</p>
        </section>""")
    task_rows = []
    for row in summary["per_task"]:
        state = "win" if row["won"] else ("partial" if row["purchased"] else "fail")
        task_rows.append(f"""
        <tr class="{state}">
          <td>#{row['task_idx']:02d}</td><td>{html.escape(row['category'])}</td>
          <td>{row['goal_index']}</td><td><strong>{row['task_score']:.4f}</strong></td>
          <td>{row['used_steps']}</td><td>{html.escape(row['query'])}</td>
          <td><a href="{html.escape(row['files']['trajectory'])}">轨迹</a> ·
              <a href="{html.escape(row['files']['prompts'])}">prompts</a> ·
              <a href="{html.escape(row['files']['json'])}">JSON</a></td>
        </tr>""")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>WebShop Mini Test</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;background:#f5f7fb;color:#172033}}
main{{max-width:1180px;margin:auto;padding:30px}} h1{{margin-bottom:6px}} .sub{{color:#637083}}
.hero,.card,table{{background:white;border:1px solid #e3e8f0;border-radius:14px;box-shadow:0 4px 18px #1f29370d}}
.hero{{padding:22px;margin:22px 0;display:flex;gap:40px;flex-wrap:wrap}} .metric .n{{font-size:30px;font-weight:750}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:20px 0}}
.card{{padding:18px}} .card h3{{margin:0;color:#526075}} .big{{font-size:34px;font-weight:760;margin:10px 0}}
.bar{{height:9px;background:#edf1f6;border-radius:6px;overflow:hidden}} .bar span{{display:block;height:100%;background:linear-gradient(90deg,#4f7cff,#25b887)}}
table{{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;margin-top:20px}}
th,td{{padding:11px 12px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:top}} th{{background:#f0f3f8}}
tr.win td:first-child{{border-left:5px solid #25b887}} tr.partial td:first-child{{border-left:5px solid #f3ad39}} tr.fail td:first-child{{border-left:5px solid #e45b65}}
a{{color:#315fd6;text-decoration:none}} code{{background:#edf1f6;padding:2px 5px;border-radius:5px}}
</style></head><body><main>
<h1>WebShop Mini Test</h1><div class="sub">variant=<code>{html.escape(summary['variant'])}</code> · 固定 5 类 × 2 任务 · exact success 优先，task score 次之</div>
<section class="hero">
 <div class="metric"><div class="n">{summary['success_rate']*100:.1f}%</div><div>exact success</div></div>
 <div class="metric"><div class="n">{summary['mean_task_score']:.4f}</div><div>mean task score</div></div>
 <div class="metric"><div class="n">{summary['purchase_rate']*100:.1f}%</div><div>purchase rate</div></div>
 <div class="metric"><div class="n">{summary['mean_steps']:.1f}</div><div>mean steps</div></div>
</section>
<div class="grid">{''.join(cards)}</div>
<table><thead><tr><th>#</th><th>类别</th><th>Goal</th><th>得分</th><th>步数</th><th>商品查询</th><th>详情</th></tr></thead>
<tbody>{''.join(task_rows)}</tbody></table>
</main></body></html>"""
    (outdir / "report.html").write_text(document, encoding="utf-8")


def run(args) -> Dict[str, Any]:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = find_webshop_data_dir(args.webshop_data_dir)
    manifest = load_manifest(Path(args.manifest))
    WebAgentTextEnv = _load_webshop_env_class(str(data_dir))
    file_path = str(data_dir / "items_shuffle_1000.json")
    attr_path = str(data_dir / "items_ins_v2_1000.json")

    # One expensive product/search server is shared by all ten independent sessions.
    first_env = WebAgentTextEnv(
        observation_mode="text", num_products=None, human_goals=False,
        file_path=file_path, attr_path=attr_path, seed=args.seed,
        session_prefix=f"mini-{args.variant}-00-",
    )
    resolved = resolve_tasks(first_env.server, manifest)
    if args.task_limit:
        resolved = resolved[: args.task_limit]
    _json_dump({
        "manifest": manifest,
        "webshop_data_dir": str(data_dir),
        "num_server_goals": len(first_env.server.goals),
        "resolved_tasks": resolved,
    }, outdir / "resolved_tasks.json")
    print(f"[WebShop] data={data_dir}")
    print(f"[WebShop] variant={args.variant} fixed_tasks={len(resolved)} "
          f"indices={[row['goal_index'] for row in resolved]}")
    if args.validate_only:
        print(f"[WebShop] manifest 校验完成: {outdir / 'resolved_tasks.json'}")
        return {}

    envs = [first_env]
    for index in range(1, len(resolved)):
        envs.append(WebAgentTextEnv(
            observation_mode="text", num_products=None, human_goals=False,
            file_path=file_path, attr_path=attr_path, seed=args.seed,
            server=first_env.server, session_prefix=f"mini-{args.variant}-{index:02d}-",
        ))

    observations, available, tasks = [], [], []
    for env, target in zip(envs, resolved):
        observation, _ = env.reset(session=int(target["goal_index"]))
        task = extract_webshop_task(observation)
        if task != target["instruction"]:
            raise RuntimeError(
                f"Goal reset mismatch at {target['goal_index']}: {task!r} != {target['instruction']!r}")
        observations.append(format_webshop_observation(observation, task))
        available.append(env.get_available_actions())
        tasks.append(task)

    fixed_playbook = MAX_SCORE_WEBSHOP_PLAYBOOK if args.variant == "template" else None
    builders, strategy_categories = [], []
    for task in tasks:
        builder = WebShopObsBuilder(
            skills_json_path=args.skills_json,
            history_length=args.history_length,
            with_skills=False,
            enable_skill_tree=False,
            prompt_char_limit=args.prompt_char_limit,
            fixed_playbook=fixed_playbook,
        )
        builder.reset(task)
        builders.append(builder)
        strategy_categories.append((builder.retrieved or {}).get("task_type", "unknown"))

    from mini_test_pen_shelf.agent_vllm import VLLMAgent
    agent = VLLMAgent(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_tokens=args.think_budget + args.action_budget,
        think_budget=args.think_budget,
        action_budget=args.action_budget,
        temperature=args.temperature,
        seed=args.seed,
        no_wait=args.nowait,
        max_num_seqs=len(resolved),
        enforce_eager=True,
        force_action_prefix=True,
    )

    loggers = [WebShopEpisodeLogger(outdir, target, args.variant) for target in resolved]
    done = [False] * len(envs)
    won = [False] * len(envs)
    purchased = [False] * len(envs)
    scores = [0.0] * len(envs)
    used = [0] * len(envs)
    strict_counts = [0] * len(envs)
    executable_counts = [0] * len(envs)
    forced_counts = [0] * len(envs)
    salvage_counts = [0] * len(envs)
    last_actions: List[Optional[str]] = [None] * len(envs)
    repeat_counts = [0] * len(envs)
    stalled = [False] * len(envs)
    purchased_asins: List[Optional[str]] = [None] * len(envs)
    reward_details: List[Optional[Dict[str, Any]]] = [None] * len(envs)
    started = time.time()

    for step in range(1, args.max_steps + 1):
        active = [index for index, value in enumerate(done) if not value]
        if not active:
            break
        prompts = [
            builders[index].build(observations[index], available[index], init=(step == 1))
            for index in active
        ]
        generated = agent.act_batch_with_meta(prompts)
        raw_outputs = [item[0] for item in generated]
        forced = [bool(item[1]) for item in generated]
        actions, strict_flags, details = webshop_projection(
            list(raw_outputs), return_details=True)

        for pos, index in enumerate(active):
            before = observations[index]
            before_available = available[index]
            action, execution_source = _salvage_executable_action(
                raw_outputs[pos], actions[pos], before_available)
            executable = _is_executable(action, before_available)
            # WebAgentTextEnv automatically resets itself after Buy Now, so
            # preserve the just-finished session id before calling step().
            purchase_session_id = envs[index].session
            raw_observation, raw_score, env_done, _ = envs[index].step(action)
            after = format_webshop_observation(raw_observation, tasks[index])
            next_available = envs[index].get_available_actions()
            raw_score = float(raw_score or 0.0)
            slot_won = bool(env_done and abs(raw_score - 1.0) < 1e-8)
            strict = bool(strict_flags[pos])

            used[index] = step
            strict_counts[index] += int(strict)
            executable_counts[index] += int(executable)
            forced_counts[index] += int(forced[pos])
            salvage_counts[index] += int(execution_source.startswith("salvaged_"))
            scores[index] = raw_score if env_done else scores[index]
            won[index] = slot_won
            purchased[index] = bool(env_done)
            if env_done:
                finished_session = envs[index].server.user_sessions.get(
                    purchase_session_id, {})
                purchased_asins[index] = finished_session.get("asin")
                reward_details[index] = finished_session.get("verbose_info")
            repeat_counts[index] = repeat_counts[index] + 1 if action == last_actions[index] else 0
            last_actions[index] = action
            builders[index].record(before, action)
            loggers[index].add(
                step=step,
                prompt=prompts[pos],
                raw_model_output=raw_outputs[pos],
                think=_think(raw_outputs[pos]),
                action=action,
                strict_valid_action=strict,
                relaxed_valid_action=bool(details[pos]["valid_action"]),
                executable_action=executable,
                execution_source=execution_source,
                forced=bool(forced[pos]),
                observation_before=webshop_trace_observation(before),
                available_actions_before=format_webshop_actions(before_available),
                observation_after=webshop_trace_observation(after),
                done=bool(env_done),
                task_score=raw_score,
                won=slot_won,
            )
            observations[index] = after
            available[index] = next_available
            if env_done or repeat_counts[index] >= args.repeat_stop_threshold:
                done[index] = True
                stalled[index] = bool(not env_done and repeat_counts[index] >= args.repeat_stop_threshold)

        print(
            f"[WebShop] step={step:02d}/{args.max_steps} active={len(active)} "
            f"done={sum(done)}/{len(done)} exact_wins={sum(won)} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )

    results = []
    for index, (env, target, logger) in enumerate(zip(envs, resolved, loggers)):
        result = {
            **target,
            "strategy_category": strategy_categories[index],
            "won": bool(won[index]),
            "purchased": bool(purchased[index]),
            "task_score": round(float(scores[index]), 6),
            "used_steps": int(used[index] or args.max_steps),
            "strict_valid_actions": strict_counts[index],
            "executable_actions": executable_counts[index],
            "forced_steps": forced_counts[index],
            "salvaged_actions": salvage_counts[index],
            "stopped_for_repetition": stalled[index],
            "purchased_asin": purchased_asins[index],
            "reward_details": reward_details[index],
        }
        result["files"] = logger.flush(result)
        results.append(result)

    elapsed = time.time() - started
    summary = _aggregate(results, args.variant, args, elapsed, agent.get_token_usage())
    _json_dump(summary, outdir / "summary.json")
    write_reports(summary, outdir)
    print("\n" + (outdir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[WebShop] HTML 可视化: {outdir / 'report.html'}")
    agent.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "template"), required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--webshop_data_dir", default=None)
    parser.add_argument("--skills_json", default=str(DEFAULT_SKILLS))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument("--prompt_char_limit", type=int, default=24000)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--think_budget", type=int, default=640)
    parser.add_argument("--action_budget", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--gpu_mem_util", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nowait", action="store_true")
    parser.add_argument("--repeat_stop_threshold", type=int, default=3)
    parser.add_argument("--task_limit", type=int, default=0,
                        help="仅调试时取 manifest 前 N 个；正式 A/B 保持 0")
    parser.add_argument("--validate_only", action="store_true",
                        help="只加载环境、校验固定任务并写 resolved_tasks.json，不加载模型")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
