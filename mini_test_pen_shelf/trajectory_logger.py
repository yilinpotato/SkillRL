"""trajectory_logger.py — 把每局完整轨迹和每步 prompt 落盘到 output/ 文件夹

每局产出两个文件（写进 run_one_game 传入的 outdir）：
  1. game_<idx>_<status>_trajectory.txt
       完整原始轨迹：每一步的 观察(observation) / 思考(think) / 动作(action) /
       是否合法 / 环境反馈，最多 40 步，人类可读。
  2. game_<idx>_<status>_prompts.txt
       逐步检查发给 4B 小模型的**完整 prompt 原文**（含策略 playbook、
       注入的 INVENTORY/SEARCHED/HERE、observation、admissible actions）。

另外整局结束写一份 game_<idx>_<status>.json，便于程序化读取。
"""
import os
import json


class TrajectoryLogger:
    def __init__(self, outdir, game_idx, task, target, gt):
        self.outdir = outdir
        self.game_idx = game_idx
        self.task = task
        self.target = target
        self.gt = gt
        self.steps = []          # 每步结构化记录
        os.makedirs(outdir, exist_ok=True)

    def log_step(self, step, prompt, raw, think, action, valid, obs,
                 holding, searched, found_here, won, reward,
                 truncated=False, salvaged=False):
        self.steps.append({
            "step": step,
            "prompt": prompt,
            "raw_model_output": raw,
            "think": think,
            "action": action,
            "valid": bool(valid),
            "truncated": bool(truncated),    # thinking 用满 max_tokens 被截断
            "salvaged": bool(salvaged),      # 截断后靠从后往前匹配救回了动作
            "observation": obs,
            "holding": holding,
            "searched": sorted(searched) if searched else [],
            "found_here": found_here,
            "won": bool(won),
            "reward": float(reward) if reward is not None else None,
        })

    def _fname(self, suffix, won, used_steps):
        status = "WIN" if won else "FAIL"
        base = f"game_{self.game_idx:02d}_{status}_{used_steps}steps"
        return os.path.join(self.outdir, base + suffix)

    def flush(self, won, used_steps):
        self._write_trajectory(won, used_steps)
        self._write_prompts(won, used_steps)
        self._write_json(won, used_steps)

    # ---- 文件 1：完整轨迹（观察/思考/动作）----
    def _write_trajectory(self, won, used_steps):
        path = self._fname("_trajectory.txt", won, used_steps)
        lines = []
        bar = "=" * 78
        lines.append(bar)
        lines.append(f" 游戏 #{self.game_idx}  ::  pen → shelf   "
                     f"[{'WIN' if won else 'FAIL'} / {used_steps} 步]")
        lines.append(bar)
        lines.append(f" 任务 (task)      : {self.task}")
        lines.append(f" 目标物体 (target): {self.target}")
        locs = self.gt.get("pen_locations") or []
        lines.append(f" pen 真值初始位置 : {', '.join(locs) if locs else '(未解析)'}")
        lines.append(bar)
        for s in self.steps:
            lines.append("")
            flag = "✅合法" if s["valid"] else "❌非法"
            head = (f"┌── Step {s['step']:>2}  [{flag}]  "
                    f"holding={s['holding'] or '空手'}")
            if s.get("truncated"):
                head += "  ⛔THINKING截断" + ("(已救回)" if s.get("salvaged") else "(未救回)")
            lines.append(head)
            lines.append(f"│ 🧠 think  : {self._oneline(s['think'])}")
            lines.append(f"│ ▶️  action : {s['action']}")
            lines.append(f"│ 👁️  obs    : {self._oneline(s['observation'])}")
            if s["found_here"]:
                lines.append(f"│ 🖊️  眼前可见目标: {s['found_here']}")
            if s["won"]:
                lines.append("│ 🏆 任务完成！")
            lines.append("└" + "─" * 60)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ---- 文件 2：逐步 prompt 检查 ----
    def _write_prompts(self, won, used_steps):
        path = self._fname("_prompts.txt", won, used_steps)
        lines = []
        bar = "#" * 78
        lines.append(bar)
        lines.append(f"# 游戏 #{self.game_idx} 发给 Qwen3-4B 的逐步 prompt 原文")
        lines.append(f"# 任务: {self.task}  |  目标物: {self.target}  |  "
                     f"{'WIN' if won else 'FAIL'} / {used_steps} 步")
        lines.append(bar)
        for s in self.steps:
            lines.append("")
            lines.append("/" * 78)
            lines.append(f"// ====== Step {s['step']} 的完整 PROMPT (发给小模型的输入) ======")
            lines.append(f"// 此步动作: {s['action']}  | 合法: {s['valid']}")
            lines.append("/" * 78)
            lines.append(s["prompt"])
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ---- JSON（程序化）----
    def _write_json(self, won, used_steps):
        path = self._fname(".json", won, used_steps)
        payload = {
            "game_idx": self.game_idx,
            "task": self.task,
            "target": self.target,
            "ground_truth_pen_locations": self.gt.get("pen_locations") or [],
            "won": bool(won),
            "used_steps": used_steps,
            "steps": self.steps,
        }
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _oneline(text):
        if not text:
            return "(空)"
        return " ".join(str(text).split())
