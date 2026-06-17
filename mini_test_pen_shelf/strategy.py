"""strategy.py — pen→shelf 任务的「自然语言策略模板」

目标：用最简单、对小模型最友好的自然语言，把 put some pen on shelf 这一类任务
拆成确定性的 5 步法，让 Qwen3-4B 在 40 步内以最大概率完成。

设计依据（来自 inspect_pen_locations.py 的真值统计）：
    pen / pencil 初始位置出现频率（按游戏数）:
        drawer      23.5%
        desk        20.6%
        sidetable   17.6%
        shelf       17.6%
        dresser     11.8%
        garbagecan   8.8%
所以「搜索顺序」按这个先验排：先 drawer，再 desk，再 sidetable，再 dresser，
最后 garbagecan / shelf。这样命中 pen 的期望步数最短。

把策略写进 prompt 的开头，模型每一步都能看到，且只需照着「当前处于哪个阶段」选动作。
"""

# 注意：用 {{ }} 转义花括号，方便和 ALFWORLD_TEMPLATE_NO_HIS 一起 .format()
PEN_SHELF_STRATEGY = """\
## TASK PLAYBOOK: put a pen on a shelf

You must put the TARGET object named in the task ("Your task is to: ...") onto a
shelf. The target is usually "pen" — sometimes "pencil". They are DIFFERENT
objects: if the task says pen, only a PEN counts (a pencil will NOT finish the
task), and vice-versa. Pick up ONLY the object whose name matches the task.

### THINK BRIEFLY
Keep your <think> to AT MOST 3 short sentences. Do not restate these rules,
do not argue with yourself, do not re-derive the plan every turn. Just:
(1) say in one line whether you are holding the target object yet, (2) name the
next action. Then stop thinking and output the action. Long reasoning wastes steps.

### FIRST, ANSWER ONE QUESTION: am I holding the target object?
Read the [INVENTORY] line in the observation — it tells you exactly what you
hold. If it says you hold the target object, you are DELIVERING. Otherwise you
are still SEARCHING.

### IF NOT HOLDING IT  — find the target object
Search receptacles in this priority order, skipping ones you already opened:
    drawer  ->  desk  ->  sidetable  ->  dresser  ->  garbagecan  ->  shelf
  - Move to the next unchecked spot with the admissible action  "go to <recep> <id>".
  - If the observation says it is closed, use  "open <recep> <id>".
  - If you see the TARGET object here (the exact word from the task), grab it:
    "take <target> <id> from <recep> <id>"  (copy the exact names shown).
    Do NOT grab a different object (e.g. do not take a pencil if you need a pen).
  - If the target is not here, this spot is empty: move to the NEXT spot. Never
    re-open a spot you already searched.

### IF HOLDING IT  — deliver to a shelf (only TWO actions left, in order)
  1. If you are NOT at a shelf yet:  "go to shelf 1"  (or any shelf id listed).
  2. If you ARE at a shelf:  >>> "move <target> <id> to shelf <id>" <<<
     This FINISHES the task. Do nothing else after it.

  !!! The action that places the object is spelled "move <obj> to shelf <id>"
      (some worlds also accept "put <obj> in/on shelf <id>"). ALWAYS copy the
      EXACT string from the admissible actions list — do not guess the wording.
  !!! Once you hold the target, the ONLY two verbs you may use are "go to shelf"
      and "move ... to shelf". Never "open", never search again.
      Reaching the shelf is NOT the goal — moving the object onto it is.

### HARD RULES
  - Output exactly ONE action, copied verbatim from the admissible actions list.
  - The words SEARCH / DELIVER / playbook are NOT actions — never output them.
  - If the observation is "Nothing happens", your last action was illegal:
    pick a different action that literally appears in the admissible list.

---
"""


def build_strategy_prompt(base_template, obs_text, admissible_actions):
    """把策略说明拼到标准 NO_HIS 模板前面。

    base_template: ALFWORLD_TEMPLATE_NO_HIS
    obs_text:      当前 observation 文本
    admissible_actions: 已经格式化好的合法动作字符串
    """
    body = base_template.format(
        current_observation=obs_text,
        admissible_actions=admissible_actions,
    )
    return PEN_SHELF_STRATEGY + body


# ============================================================================
# 通用 pick_and_place 策略：把任意 object 搬到任意 receptacle
# ============================================================================
# 搜索优先级依据上帝视角统计（pick_and_place 前30局物体真实位置）：
#   台面大件家具 countertop/sidetable/toilet/coffeetable/diningtable/tvstand/
#   bed/dresser/desk/armchair/sofa 占绝大多数；drawer/shelf/cabinet 反而很少。
#   早期版本把抽屉/架子排在前面，导致模型在抽屉里空转撞超时——这里纠正过来。
GENERIC_PICK_PLACE_STRATEGY = """\
## TASK PLAYBOOK: pick up an object and put it on/in a receptacle

You must put the TARGET object onto the TARGET receptacle named in the task
("Your task is to: put a <OBJECT> in/on the <RECEPTACLE>"). Read the task line
and the [TARGET] hint to know exactly which object and which receptacle.

### THINK BRIEFLY
Keep your <think> to AT MOST 3 short sentences: (1) say whether you are holding
the target object yet, (2) name the next action. Then output the action.

### FIRST: am I holding the target object?
Read the [INVENTORY] line. If it says you hold the target object -> DELIVER.
Otherwise -> SEARCH.

### SEARCH — find the target object  (ORDER MATTERS)
Small objects almost always sit on OPEN FURNITURE SURFACES, NOT inside drawers.
Visit spots in THIS priority, skipping ones already searched:
  1) OPEN SURFACES FIRST — these hold the object ~85% of the time:
     countertop, sidetable, coffeetable, diningtable, desk, dresser,
     tvstand, bed, sofa, armchair, toilet, sinkbasin.
  2) ONLY IF none of the above has it, then try closed containers LAST:
     drawer, cabinet, fridge, shelf, garbagecan.
  - Move with  "go to <recep> <id>".  Surfaces are visible on arrival — no need
    to open them. Only "open <recep> <id>" for a closed drawer/cabinet/fridge.
  - When you SEE the target object here, grab it:
    "take <object> <id> from <recep> <id>"  (copy the exact names shown).
    Take ONLY the target object, not a similar-looking different one.
  - If the target is not here, go to the NEXT unsearched spot (stay in group 1
    before touching group 2). Never re-visit a spot already searched.
  - DO NOT open drawer after drawer while open surfaces remain unchecked.

### DELIVER — put it on/in the target receptacle (TWO actions, in order)
  1. If NOT at the target receptacle yet:  "go to <receptacle> <id>".
  2. If AT the target receptacle:  >>> "move <object> <id> to <receptacle> <id>" <<<
     This FINISHES the task. Do nothing after it.
  !!! The placing action is usually "move <obj> to <recep> <id>" (some worlds
      use "put <obj> in/on <recep> <id>"). ALWAYS copy the EXACT string from the
      admissible list. Once holding the object, only "go to <receptacle>" and
      "move/put ..." are allowed — never search again.

### HARD RULES
  - Output exactly ONE action, copied verbatim from the admissible list.
  - SEARCH/DELIVER are NOT actions. If "Nothing happens", your last action was
    illegal — pick a different one that literally appears in the list.

---
"""


# ============================================================================
# pick_two 策略：把【两个】同类 object 搬到同一个 receptacle
# ============================================================================
PICK_TWO_STRATEGY = """\
## TASK PLAYBOOK: put TWO objects of the same kind onto a receptacle

The task asks for TWO of the same object (e.g. "find two soapbottle and put
them in cart"). Deliver TWO SEPARATE instances (different number ids, e.g.
soapbottle 1 AND soapbottle 2) to the receptacle, one at a time. You can only
carry ONE at a time.

### THINK BRIEFLY
Keep your <think> to AT MOST 3 short sentences: (1) read [PROGRESS] — how many
placed and whether holding one, (2) name the next action.

### USE THE [PROGRESS] LINE — it is your memory
It tells you how many you placed, WHICH instances are already in the receptacle,
and whether your hands are full. Decide:
  - placed 2/2                  -> DONE, stop.
  - holding one                 -> go to receptacle, then move/put it.
  - hands empty and placed < 2  -> find a DIFFERENT instance you have NOT placed.

### CRITICAL — do NOT loop on one object
  - NEVER "take" an object that is ALREADY in the receptacle (listed in
    [PROGRESS]). Taking it back and re-placing it does NOT count as two — you
    must deliver a SECOND, different instance (a different number id).
  - After placing the first one, GO SEARCH for the other instance. The two
    instances often sit in the SAME spot or nearby — re-check where you found
    the first.

### SEARCH — find the next, not-yet-placed instance  (ORDER MATTERS)
Open surfaces first (countertop, sidetable, toilet, diningtable, coffeetable,
tvstand, bed, dresser, sofa, armchair, sinkbasin), closed containers
(drawer, cabinet, fridge, shelf) LAST. Take an instance whose id you have not
placed: "take <object> <id> from <recep> <id>".

### DELIVER — put the carried object on the receptacle
  1. If NOT at the receptacle:  "go to <receptacle> <id>".
  2. If AT the receptacle:  "move <object> <id> to <receptacle> <id>"
     (copy the exact admissible string). Then, if placed < 2, SEARCH for the
     remaining instance.

### HARD RULES
  - Output exactly ONE action, copied verbatim from the admissible list.
  - Deliver BOTH distinct instances before stopping. Do not stop after one.
  - If "Nothing happens", pick a different action that is actually in the list.

---
"""


def build_generic_prompt(base_template, obs_text, admissible_actions):
    body = base_template.format(current_observation=obs_text,
                                admissible_actions=admissible_actions)
    return GENERIC_PICK_PLACE_STRATEGY + body


def build_pick_two_prompt(base_template, obs_text, admissible_actions):
    body = base_template.format(current_observation=obs_text,
                                admissible_actions=admissible_actions)
    return PICK_TWO_STRATEGY + body
