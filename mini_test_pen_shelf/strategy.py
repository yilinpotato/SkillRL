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


# ============================================================================
# 共享前缀：极简思考格式提示（一行示范即可，不堆规则）
# 注意：之前用长 header + 两条 few-shot，反而让模型花大量 token 去复述/消化格式
# 规则（元-思考），思考更长、更易撞预算。这里精简到最短。
# ============================================================================
STRUCTURED_THINK_HEADER = """\
Think in at most 2 short lines, then act. Example:
<think>Holding nothing; pen is likely on a desk; go to desk 1.</think>
<action>go to desk 1</action>
---
"""

# 注意：用 {{ }} 转义花括号，方便和 ALFWORLD_TEMPLATE_NO_HIS 一起 .format()
PEN_SHELF_STRATEGY = """\
## GOAL: put the TARGET object (the exact one named in the task, pen OR pencil —
they differ; only the named one counts) onto a shelf.

First check [INVENTORY]: holding the target -> go to IF HOLDING. Else -> IF NOT HOLDING.

### IF NOT HOLDING IT — find the target
  - Pens/pencils are usually on desk, sidetable, dresser, drawer. Check open
    surfaces first; open a drawer only if it is closed and surfaces are exhausted.
  - See the target here -> "take <target> <id> from <recep> <id>". Take only the
    named object. Skip spots already in [ALREADY SEARCHED].

### IF HOLDING IT — deliver to a shelf (two actions)
  1. Not at a shelf yet -> "go to shelf <id>".
  2. At a shelf -> "move <target> <id> to shelf <id>" (copy the exact admissible
     string; some worlds use "put ... in/on shelf"). This FINISHES the task.
  Once holding it, only go-to-shelf and move/put are allowed — never search again.

If obs is "Nothing happens", your last action was illegal — pick a different one
that appears verbatim in the admissible list.
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
    return STRUCTURED_THINK_HEADER + PEN_SHELF_STRATEGY + body


# ============================================================================
# 通用 pick_and_place 策略：把任意 object 搬到任意 receptacle（精简版）
# 搜索优先级依据上帝视角统计：物体多在开放台面（countertop/sidetable/桌子/
# 床/dresser/sofa...），drawer/shelf/cabinet 少；让模型按物体类别自己推断位置。
# ============================================================================
GENERIC_PICK_PLACE_STRATEGY = """\
## GOAL: put the TARGET object onto the TARGET receptacle named in the task.

First check [INVENTORY]: holding the target -> go to IF HOLDING. Else -> IF NOT HOLDING.

### IF NOT HOLDING IT — find the target (guess where its kind is kept, go there)
  - Kitchen things (knife/fork/cup/bowl/bread/egg/pot...) -> countertop,
    diningtable, sink, fridge, cabinet.
  - Bathroom things (soap/toiletpaper/spraybottle...) -> countertop, toilet,
    sink, cabinet.
  - Room/office things (book/pen/laptop/cd/remote/vase/keychain...) -> desk,
    sidetable, coffeetable, dresser, sofa, shelf, bed.
  Prefer OPEN SURFACES first; open a drawer/cabinet only if it is closed and
  surfaces are exhausted. See the target -> "take <object> <id> from <recep> <id>"
  (only the named object). Skip [ALREADY SEARCHED] spots; don't open one closed
  container after another while surfaces remain.

### IF HOLDING IT — deliver
  "go to <receptacle> <id>", then "move <object> <id> to <receptacle> <id>"
  (copy the exact admissible string). This FINISHES the task.

Only choose actions that appear verbatim in the admissible list. If your
preferred spot isn't listed, pick any unsearched listed spot — never stall. If
obs is "Nothing happens", your last action was illegal; pick a different one.
---
"""


# ============================================================================
# pick_two 策略：把【两个】同类 object 搬到同一个 receptacle（精简版）
# ============================================================================
PICK_TWO_STRATEGY = """\
## GOAL: put TWO separate instances of the TARGET object (different ids, e.g.
soapbottle 1 AND soapbottle 2) onto the TARGET receptacle, one at a time.

Read [PROGRESS]: it says how many placed and which ids are already there.
  - placed 2/2 -> DONE, stop.
  - holding one -> "go to <receptacle> <id>", then "move <object> <id> to <receptacle> <id>".
  - hands empty, placed < 2 -> find a DIFFERENT instance you have NOT placed.

NEVER take back an instance already in the receptacle ([PROGRESS] lists them) —
that does not count. After placing one, search for the OTHER instance; the two
usually sit in the same area, so re-check near where you found the first.

SEARCH: guess by object kind (kitchen->countertop/sink/cabinet; bathroom->
countertop/toilet/sink; room/office->desk/sidetable/shelf), open surfaces first,
closed containers last. "take <object> <id> from <recep> <id>".

Only actions in the admissible list. If preferred spot missing, pick any
unsearched listed one — never stall. "Nothing happens" = illegal last action.
---
"""


def build_generic_prompt(base_template, obs_text, admissible_actions):
    body = base_template.format(current_observation=obs_text,
                                admissible_actions=admissible_actions)
    return STRUCTURED_THINK_HEADER + GENERIC_PICK_PLACE_STRATEGY + body


def build_pick_two_prompt(base_template, obs_text, admissible_actions):
    body = base_template.format(current_observation=obs_text,
                                admissible_actions=admissible_actions)
    return STRUCTURED_THINK_HEADER + PICK_TWO_STRATEGY + body
