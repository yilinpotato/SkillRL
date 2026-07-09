# Copyright 2025 CoSkill.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Seed natural-language PLAYBOOKS for ALFWorld task types (CoSkill Phase 1).

Background
----------
The flat ``general/task_specific`` skills are short bullet "principles".  A
mini-test (``mini_test_pen_shelf/``) showed that a *structured, phase-based
playbook* — ``GOAL / FIND / DELIVER`` conditional flow — lifts a 4B model far
more than bullets do (pen->shelf success 20%->100%, pick_two 30%->80%,
pick_and_place steps 12.3->6.7).

Neither the main training pipeline nor the aligned mini_test injects per-step
state tags (``[INVENTORY]`` / ``[HERE]`` / ``[TARGET]`` / ``[ALREADY SEARCHED]``).
So the playbook itself teaches the model how to judge its own phase from what the
prompt already contains — the task line, the current observation ("you see ..." /
"You pick up ..."), and the recent action history — rather than relying on any
injected hint.

Keys are the outputs of ``SkillsOnlyMemory._detect_task_type`` so the lookup is
the same in both ``template`` and ``embedding`` retrieval modes.  Task types not
listed here simply fall back to the bullet skills (no playbook injected).

Runtime source of truth is the skill-bank JSON's optional ``task_playbooks``
section (so the cloud analyzer can evolve playbooks in Phase 2).  These Python
seeds are the human-authored default used only when the JSON has none.
"""

from typing import Dict


PICK_AND_PLACE_PLAYBOOK = """\
## STRATEGY PLAYBOOK — pick_and_place (put ONE object onto a receptacle)

GOAL: put the TARGET object named in the task onto the TARGET receptacle. Re-read the
task line for the exact object and receptacle; only that exact object counts.
CRITICAL: the object is EXACTLY the word in the task line. Do NOT swap it for a different
object you happen to see, and do NOT decide the task is a typo. If the named object is not
here, it is simply elsewhere — keep searching; never substitute another object.

You get NO status reminders — work out your situation yourself each step:
- HOLDING IT? You hold the target if your recent actions show a "take <target> ..."
  that you have not yet followed with a "move/put". A successful pickup also says
  "You pick up the <target>".
- SEE IT HERE? If the current observation's "you see ..." names the target at your
  spot, it is right here — take it now, don't wander off.
Then: holding the target -> DELIVER; otherwise -> FIND.

FIND (not holding it yet):
- Guess where this kind of object is kept and go straight there. Kitchen things
  (knife/fork/cup/bowl/bread/egg/pot...) -> countertop, diningtable, sink, fridge,
  cabinet. Bathroom things (soap/toiletpaper/spraybottle...) -> countertop, toilet,
  sink, cabinet. Room/office things (book/pen/laptop/cd/remote/vase/keychain...) ->
  desk, sidetable, coffeetable, dresser, sofa, shelf, bed.
- Check OPEN surfaces first; open a drawer/cabinet only if it is closed AND open
  surfaces are exhausted. Don't go back to a spot your recent actions already checked;
  don't open one closed container after another while surfaces remain.
- When the observation shows the target here -> "take <object> <id> from <recep> <id>"
  (take ONLY the named object).

DELIVER (holding it):
- "go to <receptacle> <id>", then "move <object> <id> to <receptacle> <id>" (copy the
  exact admissible string; some worlds phrase it "put ... in/on"). This FINISHES the task.
- Once holding it, only go-to-receptacle and move/put are valid — never search again.

Only choose actions that appear verbatim in the admissible list. If your preferred spot
is not listed, pick any unchecked listed spot — never stall. If the observation is
"Nothing happens", your last action was illegal; pick a different listed one.
Think briefly (about 2 short lines), then act."""


PICK_TWO_PLAYBOOK = """\
## STRATEGY PLAYBOOK — pick_two_obj_and_place (put TWO of the same object onto one receptacle)

GOAL: put TWO separate instances of the TARGET object (two DIFFERENT numbered ids of the
SAME object the task names, e.g. <obj> 1 AND <obj> 2) onto the TARGET receptacle, one at a
time. Re-read the task line for the exact object and receptacle.
CRITICAL: the object is EXACTLY the word in the task line. Do NOT swap it for a different
object you happen to see (a spoon is not a peppershaker), and do NOT decide the task is a
typo. If the named object is not here, it is simply elsewhere — keep searching for it.

You get NO status reminders — work out your situation yourself each step:
- HOLDING ONE? You hold an instance if your recent actions show a "take <target> ..."
  not yet followed by a "move/put" (the pickup says "You pick up the <target>").
- HOW MANY DONE? Count the target instances already on/in the receptacle from what you
  have placed so far. An instance you SEE INSIDE the target receptacle is ALREADY DONE —
  never take it back out; re-placing it does NOT count and only wastes steps.
Then: holding an instance -> DELIVER; otherwise -> FIND.

FIND (not holding one):
- Go find a DIFFERENT instance (a different number id) somewhere ELSE. Guess by kind
  (kitchen -> countertop/sink/cabinet/diningtable; bathroom -> countertop/toilet/sink;
  room/office -> desk/sidetable/shelf/dresser). The two instances often sit near each
  other, so re-check around where you found the first. Open surfaces first; don't go back
  to a spot your recent actions already checked.
- When the observation shows a not-yet-delivered instance -> "take <object> <id> from
  <recep> <id>".

DELIVER (holding one):
- "go to <receptacle> <id>", then "move <object> <id> to <receptacle> <id>" (copy the
  exact admissible string). After it is placed your hands are EMPTY again — go back to
  FIND and search for the NEXT not-yet-delivered instance. Repeat until TWO are placed.
- You can carry only ONE object at a time: never take a second instance before you have
  delivered the one you are holding.

STOP only when TWO different ids are on/in the receptacle. Use only actions in the
admissible list; if your preferred spot is missing, pick any unsearched listed one —
never stall. "Nothing happens" = your last action was illegal.
Think briefly (about 2 short lines), then act."""


# ---------------------------------------------------------------------------
# LEAN variants: same structure, but with the concrete "few-shot" examples
# stripped — the object->location enumerations and the parenthetical "e.g." /
# "(a spoon is not a peppershaker)" instances. Used to ablate whether the
# concrete examples help or hurt (toggle via get_seed_playbook(with_examples=False)
# / ProdObsBuilder(playbook_examples=False)). Structure, phase-detection and the
# CRITICAL anti-substitution rule are kept identical.
# ---------------------------------------------------------------------------

PICK_AND_PLACE_PLAYBOOK_LEAN = """\
## STRATEGY PLAYBOOK — pick_and_place (put ONE object onto a receptacle)

GOAL: put the TARGET object named in the task onto the TARGET receptacle. Re-read the
task line for the exact object and receptacle; only that exact object counts.
CRITICAL: the object is EXACTLY the word in the task line. Do NOT swap it for a different
object you happen to see, and do NOT decide the task is a typo. If the named object is not
here, it is simply elsewhere — keep searching; never substitute another object.

You get NO status reminders — work out your situation yourself each step:
- HOLDING IT? You hold the target if your recent actions show a "take <target> ..."
  that you have not yet followed with a "move/put". A successful pickup also says
  "You pick up the <target>".
- SEE IT HERE? If the current observation's "you see ..." names the target at your
  spot, it is right here — take it now, don't wander off.
Then: holding the target -> DELIVER; otherwise -> FIND.

FIND (not holding it yet):
- Think about where this kind of object is usually kept and go straight there.
- Check OPEN surfaces first; open a drawer/cabinet only if it is closed AND open
  surfaces are exhausted. Don't go back to a spot your recent actions already checked;
  don't open one closed container after another while surfaces remain.
- When the observation shows the target here -> "take <object> <id> from <recep> <id>"
  (take ONLY the named object).

DELIVER (holding it):
- "go to <receptacle> <id>", then "move <object> <id> to <receptacle> <id>" (copy the
  exact admissible string; some worlds phrase it "put ... in/on"). This FINISHES the task.
- Once holding it, only go-to-receptacle and move/put are valid — never search again.

Only choose actions that appear verbatim in the admissible list. If your preferred spot
is not listed, pick any unchecked listed spot — never stall. If the observation is
"Nothing happens", your last action was illegal; pick a different listed one.
Think briefly (about 2 short lines), then act."""


PICK_TWO_PLAYBOOK_LEAN = """\
## STRATEGY PLAYBOOK — pick_two_obj_and_place (put TWO of the same object onto one receptacle)

GOAL: put TWO separate instances of the TARGET object (two DIFFERENT numbered ids of the
SAME object the task names) onto the TARGET receptacle, one at a time. Re-read the task
line for the exact object and receptacle.
CRITICAL: the object is EXACTLY the word in the task line. Do NOT swap it for a different
object you happen to see, and do NOT decide the task is a typo. If the named object is not
here, it is simply elsewhere — keep searching for it.

You get NO status reminders — work out your situation yourself each step:
- HOLDING ONE? You hold an instance if your recent actions show a "take <target> ..."
  not yet followed by a "move/put" (the pickup says "You pick up the <target>").
- HOW MANY DONE? Count the target instances already on/in the receptacle from what you
  have placed so far. An instance you SEE INSIDE the target receptacle is ALREADY DONE —
  never take it back out; re-placing it does NOT count and only wastes steps.
Then: holding an instance -> DELIVER; otherwise -> FIND.

FIND (not holding one):
- Go find a DIFFERENT instance (a different number id) somewhere ELSE. Think about where
  this kind of object is usually kept. The two instances often sit near each other, so
  re-check around where you found the first. Open surfaces first; don't go back to a spot
  your recent actions already checked.
- When the observation shows a not-yet-delivered instance -> "take <object> <id> from
  <recep> <id>".

DELIVER (holding one):
- "go to <receptacle> <id>", then "move <object> <id> to <receptacle> <id>" (copy the
  exact admissible string). After it is placed your hands are EMPTY again — go back to
  FIND and search for the NEXT not-yet-delivered instance. Repeat until TWO are placed.
- You can carry only ONE object at a time: never take a second instance before you have
  delivered the one you are holding.

STOP only when TWO different ids are on/in the receptacle. Use only actions in the
admissible list; if your preferred spot is missing, pick any unsearched listed one —
never stall. "Nothing happens" = your last action was illegal.
Think briefly (about 2 short lines), then act."""


# Keyed by SkillsOnlyMemory._detect_task_type outputs.
SEED_PLAYBOOKS: Dict[str, str] = {
    "pick_and_place": PICK_AND_PLACE_PLAYBOOK,
    "pick_two_obj_and_place": PICK_TWO_PLAYBOOK,
}

# No-examples ("lean") variants, same keys.
SEED_PLAYBOOKS_LEAN: Dict[str, str] = {
    "pick_and_place": PICK_AND_PLACE_PLAYBOOK_LEAN,
    "pick_two_obj_and_place": PICK_TWO_PLAYBOOK_LEAN,
}


def get_seed_playbook(task_type: str, with_examples: bool = True):
    """Return the seed playbook for ``task_type``; ``with_examples=False`` returns
    the lean variant (concrete few-shot examples stripped). None if no playbook."""
    src = SEED_PLAYBOOKS if with_examples else SEED_PLAYBOOKS_LEAN
    return src.get(task_type)
