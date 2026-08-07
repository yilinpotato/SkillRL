# CoSkill Appendix

## Scope and notation

This appendix contains the exact repository-owned prompt templates used by the
online edge executor and cloud analyzer. Runtime values—task descriptions,
observations, admissible actions, trajectory evidence, retrieved skills, and
the current skill tree—are represented by placeholders. Qwen's tokenizer-owned
chat template is applied after these strings are rendered and is not duplicated
here.

The retrieved-memory placeholder is rendered at runtime in this order:

1. learned skill tree;
2. general principles;
3. task-relevant skills;
4. mistakes to avoid.

## A. Online experiment prompts

### 1. ALFWorld: initial step

~~~~text
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
~~~~

### 2. ALFWorld: history without retrieved skills

~~~~text
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
~~~~

### 3. ALFWorld: history with retrieved skills

~~~~text
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
~~~~

### 4. WebShop: initial step


~~~~text
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
First reason about the current situation inside exactly one <think>...</think> block. Then return exactly one action block in this form: <action>click[...]</action> or <action>search[...]</action>. Do not output a bare action or any text after </action>.
~~~~

### 5. WebShop: history without retrieved skills


~~~~text
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
First reason about the current situation inside exactly one <think>...</think> block. Then return exactly one action block in this form: <action>click[...]</action> or <action>search[...]</action>. Do not output a bare action or any text after </action>.
~~~~

### 6. WebShop: history with retrieved skills


~~~~text
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
First reason about the current situation inside exactly one <think>...</think> block. Then return exactly one action block in this form: <action>click[...]</action> or <action>search[...]</action>. Do not output a bare action or any text after </action>.
~~~~

### 7. Cloud: contrastive skill distillation


~~~~text
You are an expert at distilling sequential-agent experience into reusable skills.
The current environment is {{ self.environment_name }}.
ENVIRONMENT CONTRACT: {{ self._domain_context() }}

You are given COMPRESSED trajectories for task_type="{{ task_type }}". Each step shows the action
taken and the OBSERVATION DELTA (only what changed in the environment: +added / -removed lines).

SUCCESSFUL TRAJECTORIES (what to do):
{{ succ_txt }}

FAILED TRAJECTORIES (what went wrong):
{{ fail_txt }}

DECISION FORKS (where successful and failed runs diverged after a shared prefix):
{{ fork_txt }}

TASK — Contrastive Analysis:
1. Compare success vs failure: identify what successful runs did right and the exact step where
   failed runs went off track (use the decision forks).
2. Abstract the (success - failure) difference into 1-{{ self.max_new_skills_per_update }} NEW, generalizable skills.
3. Avoid duplicating these existing skills: {{ existing_titles }}

WRITING STYLE — keep skills SHORT and SIMPLE so a small 4B model can follow them.
Match the concise format of the existing hand-written seed skills.

Return ONLY a JSON array. Each skill MUST have EXACTLY these fields (no extra fields):
  - "skill_id":      one of {{ example_ids }}
  - "title":         3-5 word title
  - "scope":         "general" or "task_specific"
  - "task_type":     one of [{{ allowed_types }}] when task_specific, or "" if general. Never use "ALL".
  - "principle":     ONE or TWO plain sentences stating the rule. Keep it under 30 words. No JSON, no lists.
  - "when_to_apply": ONE short sentence naming the situation that triggers this skill. Under 20 words.

Example:
[{{ seed_example }}]

Return ONLY the JSON array, no other text.
~~~~

### 8. Cloud: failure diagnosis


~~~~text
Role: You are an expert failure-analysis agent for sequential decision-making agent
tasks. The current environment is {{ self.environment_name }}. Reason from the trajectories and the
environment contract below; do not import assumptions from a different benchmark.

ENVIRONMENT CONTRACT:
{{ domain_context }}

Goal: For each FAILED trajectory, diagnose WHY it failed. Use successful rollouts of the same
task_type as observed references of successful behaviour, and the environment's success criterion
as the correctness anchor. They are not oracle demonstrations or ground-truth action plans. Each
step shows the action taken and the OBSERVATION DELTA
(+added / -removed lines).

{{ chr(10).join(sections) }}

DECISION FORKS (where successful and failed runs diverged after a shared prefix):
{{ forks }}

For EACH failed trajectory, identify the single causal failure reason, how it could be avoided, and
WHERE a corrective patch belongs in the agent's skill tree. The skill tree is a markdown TREE whose
heading depth is its hierarchy (a deeper heading refines its parent). So point the patch at a location
by naming the section heading, and say whether the fix belongs directly in it or as a deeper
refinement nested under it — not just "somewhere in the skill tree".
Return ONLY a JSON array. One object per failed trajectory, EXACTLY these fields:
  - "traj_ref":      the [ref=...] tag of the failed trajectory
  - "task_type":     its task_type
  - "failure_type":  one of "wrong_target" | "wrong_order" | "inefficient_exploration" | "loop" |
                     "premature_stop" | "invalid_action" | "misread_state" | "gave_up" | "other"
  - "root_cause":    ONE sentence, the causal reason it failed.
  - "evidence":      the step / observation that proves it (quote briefly).
  - "corrective_rule": ONE short imperative rule that would have prevented it.
  - "skill_tree_gap": what is missing/weak in the skill tree that let this error through.
  - "patch_location": WHERE the fix belongs — name the target markdown heading, and whether it should
                     be a new/deeper subsection ('##'/'###') under it. Use "new top-level section" if
                     no fitting heading exists yet.
  - "confidence":    a number 0.0-1.0.
Return ONLY the JSON array, no other text.
~~~~

### 9. Cloud: skill-tree evolution


~~~~text
You are the SKILL TREE author and editor for one small language-model agent.
The current environment is {{ self.environment_name }}.
ENVIRONMENT CONTRACT: {{ self._domain_context() }}

This agent keeps ONE skill tree PER goal family / task type. The tree you are editing here is ONLY for
the current goal family "{{ task_type }}" and will be read at the TOP of the agent's prompt when a future
task is detected as the same family. Do not mix in rules for unrelated goal families.

Although the tree is task-family-specific, keep the content GENERAL within that family: infer the
goal, sub-goals, state phases, decision bottlenecks, and failure modes from the trajectories. Do not
hard-code benchmark labels, exact sampled entity IDs, layouts, product IDs, or dataset-specific
category names as if they were the method. The skill tree should teach the agent how to assess its own
situation and what to do next — the goal, the decision flow, and the concrete actions — using only
what the prompt already contains, with no externally injected state hints.

CURRENT SKILL TREE (exactly what the agent was shown this round):
"""
{{ cur }}
"""

{{ consensus_section }}

NEW SUCCESSFUL TRAJECTORIES (what worked):
{{ succ_txt }}

NEW FAILED TRAJECTORIES (what went wrong):
{{ fail_txt }}

FAILURE DIAGNOSES (root cause + which skill-tree part was missing/weak):
{{ diag_txt }}
{{ depth_constraint }}
{{ depth_execution_override }}
{{ repair_section }}
{{ full_evidence_section }}
{{ progressive_section }}

Do these steps IN ORDER:

1. INDUCE THE REGULARITY (reason, don't just patch). Look ACROSS the trajectories and diagnoses and
   ask: what general regularity explains success vs failure? Combine two sources of evidence:
   (a) DATA INDUCTION — what do the successful runs consistently do that the failed ones do not, and
   at which decision point do they diverge? Prefer a pattern seen MULTIPLE times over a one-off.
   (b) VERIFIED ENVIRONMENT CONTRACT — use only the explicit environment contract above for mechanics,
   and use general reasoning only to organize rules already supported by trajectories. Do not use
   unstated common sense to invent a sub-goal order, object location, state transition, or precondition.
   Use verified evidence to GENERALIZE beyond the few sampled trajectories so the rule transfers.
   State every regularity as a GENERAL principle grounded in a stated reason — never an
   instance-specific fix tied to one concrete entity or location.

   EVIDENCE SAFETY: keep a transition CONDITIONAL when the evidence shows that its validity depends
   on the current observation or carried-object state. Do not promote an action order seen in one
   trajectory into a universal prerequisite. When successful traces disagree about an order, write an
   observation-conditioned decision rule instead; when the evidence does not establish a transition,
   tell the agent to inspect its current feedback/admissible action rather than inventing a mechanic.

2. USAGE CRITIQUE (skip if there is no current skill tree). Judge how the agent USED the current
   skill tree: did it follow it, misread it, or ignore a section? IMPORTANT — also check whether the
   skill tree ITSELF caused the failure: is any wording ambiguous, over-specific, contradictory, or
   misleading enough to push the agent into the wrong action? If a failure traces back to your own
   text, FIX THE TEXT (that is a higher-priority edit than adding new rules).

3. DECIDE THE ACTION:
   - No current skill tree -> action="rewrite": author the FIRST version from the induced regularities.
     Start SHALLOW (see step 4) — do NOT pre-emptively add depth the evidence has not shown to need.
   - Current skill tree works and shows NO avoidable failures -> action="keep".
   - Otherwise -> action="refine": change ONLY the section(s) the diagnoses point at (use each
     diagnosis's patch_location), including fixing your own misleading wording.

4. HIERARCHICAL MARKDOWN — this is the core format. Write the skill tree as a TREE, using markdown
   heading depth for the branches: a heading nested one level deeper than its parent (a '##' under a
   '#', a '###' under a '##', and so on) is a REFINEMENT of that parent — it elaborates, clarifies, or
   details what the parent says. YOU decide what every node contains and HOW MANY levels there are;
   there is no fixed meaning for any level and no fixed number of levels. Let the content decide the
   shape — go exactly as deep as the material needs and no deeper.
   Organize this task-family tree with clear categories or bullet-like branches when helpful (for
   example by goal phase, state-assessment phase, decision bottleneck, or recurring mistake), then
   break each branch down step by step only as far as evidence justifies.
   DEEPEN BY JUDGEMENT, NOT BY RULE. Keep every branch as SHALLOW as it can be while still working.
   Add a child heading under a section ONLY when the evidence — a recurring failure, a diagnosis's
   patch_location, or a misread — shows the agent did NOT grasp that parent at its current depth.
   Well-understood sections stay shallow; different branches may sit at different depths at once.
   Depth follows demonstrated need, branch by branch — never deepen the whole document uniformly.
   When a diagnosis names a patch_location, put the fix under exactly that heading (or create the new
   heading it asks for).

5. LAYOUT & CONSTRAINTS: lead with the goal, then order sections in the natural flow the agent acts
   (assess state -> choose the right sub-goal -> act -> recognize completion and stop). One idea per
   line; most decision-critical rule first within each section; no duplication or contradiction across
   sections. Keep it SHORT — a small model pays a thinking tax per line; spend depth only where it
   earns its keep.

Return ONLY one JSON object, EXACTLY these fields:
  - "action":    "keep" | "refine" | "rewrite"
  - "level":     the deepest heading depth present, as a number (1 = only '#', 2 = a '##' exists,
                 3 = a '###' exists, ...)
  - "skill_tree": the FULL new skill tree MARKDOWN (empty string if action="keep")
  - "critique":  1-3 sentences: how the agent used the skill tree AND whether the skill tree's own
                 wording misled it (or "" if there was no current skill tree)
  - "changelog": 1 sentence naming which section(s) you changed or deepened, and why
{{ grounding_schema }}
Return ONLY the JSON object, no other text.
~~~~


Example skilltree: Cool and Place

```text

# Goal: Cool an object and move it to a specified target location.

## Navigation

- Before moving, check that the intended destination is reachable from your current location. If you are not sure, use 'look' to see observable locations.
- After executing a 'go to' action, read the observation carefully. Verify that you arrived at the exact location you intended (e.g., "You arrive at countertop 2"). If the observation says a different location, you did not reach it — do not assume success. Instead, consider that the target may not be adjacent or you may have misidentified it. Use 'look' to reassess adjacent locations, then try a different route (e.g., go to a known adjacent location first).
- Keep track of which instances you have already visited (e.g., countertop 1, countertop 2) to avoid revisiting them accidentally.

## Phase 1: Find and take the target object

### Surface search (priority order, not strict)

- Preferred search order: Start by searching the surfaces that most often hold objects: first check all countertops (every instance you saw in the initial look), then all diningtables. After those are exhausted, check sidetables, shelves, sinkbasins, stoveburners, and the inside of the fridge.
- Flexibility: You do not need to enforce this order rigidly. If you pass by a sinkbasin while moving between countertops, you may check it. Always prioritize the high-yield surfaces (countertops, diningtables) early, but do not skip a surface entirely just because you are not following the exact sequence.
- Use the 'go to' action to move to each surface instance. After arriving, read the observation to see what is on that surface. Do not use 'look' or 'examine' — the arrival observation already shows all items.
- For the fridge: if target found inside, take it and go to Phase 2. If not, close the fridge before moving on.
- Handling arrival at an unintended location: If you arrive at a location that is not the next one you intended, treat that location as visited (search it if it is a surface type you have not yet searched), then move to the next logical surface in the priority order.

### Exhaustive container search

- Only begin container search after you have checked every surface type (countertops, diningtables, sidetables, shelves, sinkbasins, stoveburners, and the inside of the fridge) and the target was not found.
- Open each cabinet one by one: Move to the cabinet, then use 'open cabinet X'. After opening, read the observation to see what is inside. If target present, take it, then close the cabinet before moving on. If not, close the cabinet and move to the next unopened cabinet.
- Then check each drawer (same process: move, open, look inside, close).
- Then check the microwave (open, look inside).
- Crucial: After moving to a container, you must open it to see its contents. The arrival observation does not show what is inside a closed container. If you skip opening, you will miss the target.

### Object verification

- Before taking, read the exact name. After taking, verify again. If wrong, put it back on the exact surface you took it from and resume searching from that location.

## Phase 2: Cool the object using the fridge

- As soon as you are carrying the correct target object, go to a fridge, open it (if closed), and use 'cool <object> with fridge'.

## Phase 3: Move the cooled object to the target location

- Go directly to the target location (e.g., cabinet, diningtable, garbagecan). If it is a closed container, open it first.
- Use 'move <object> to <target>'.

## Completion check

- The object is now cooled and placed at the target. Task complete.
```

Example skilltree: Examine an Object Under a Light

```text
# Goal: Examine or look at a target object using a light source (e.g., desklamp)

## Critical rules (read first)

- After you take the correct target object, stop immediately. No further actions of any kind. The task completes the moment the lamp is on and the object is in your inventory. Do not examine, move, look, or travel after taking.
- You must visit every single receptacle from your initial mental list. Do not skip any, even if you think the target is unlikely there. If you find the target before visiting all, stop. Otherwise, persist until every receptacle has been checked.

## Phase 1: Find and turn on the desklamp

- As soon as you see the initial room description, mentally list every furniture item (each numbered entity). You will need this list later to search every location.
- Start by moving to a surface receptacle (e.g., desk, dresser, sidetable) to see its contents. Check for a desklamp.
- If you do not see a desklamp, move to another surface receptacle from your list and check its contents. Continue until you find a desklamp.
- Once you are at the same receptacle as the desklamp, use it (`use desklamp <number>`). The observation will confirm the lamp is on.
- If you encounter the target object before finding the lamp, note its location but do not take it yet. Continue searching for the lamp first.

## Phase 2: Locate the target object (systematic search after lamp is on)

- You must visit every single receptacle from your initial list – no exceptions. Use `look` to refresh the list if unsure.
- For each receptacle in your list:
  1. Go to that receptacle (`go to <receptacle>`).
  2. Read the arrival observation. If the target object's exact name appears on the surface (or inside if open), take it immediately (`take <object> from <receptacle>`) and stop – the task is done.
  3. If the receptacle is a container (e.g., drawer, cabinet) and it is closed, open it (`open <receptacle>`). Then check its contents. If the target object is inside, take it and stop.
- Do not move away from a receptacle that contains the target object without taking it. If you see it, take it at that moment.
- After visiting all receptacles and still not finding the target, re-read the initial description to confirm you didn't miss any. Persist until you have visited every receptacle.

### Target object search (detailed steps)

- Use only the actions: `go`, `look`, `open`, `close`, `take`, `inventory`, and `use`. Do not use `examine` – the arrival observation already shows what is on the receptacle. Repeating `look` also gives the same information if you need to refresh.
- After turning on the lamp, systematically go to each receptacle from your initial list. If a receptacle is a closed container, open it before checking its contents. Do not skip any receptacle type (e.g., drawers, cabinets, shelves, coffeetables, etc.).
- If the container is open, simply read the observation. Do not waste actions examining or looking repeatedly at the same receptacle.

### Object verification

- Before taking any object, verify that its exact name matches the target object named in the task. For example, if the task says "look at box", only take an object whose name is "box" – do not take "statue" or "newspaper".
- If you see multiple objects, focus only on the target name. Ignore and do not pick up other objects, even if they appear interesting.
- After taking the correct object, the task is complete. Do not perform any further actions.

## Phase 3: Task completion (stop immediately)

- With the desklamp still on and the target object in your inventory, the task is complete. No further action is required – no examine, move, look, or travel.
- Key rule: Once you take the target object, stop immediately. Any subsequent action (e.g., moving the object, examining it again, going elsewhere) may undo the success or waste steps. Do not perform any additional actions.
```
