# Representative Skill Tree Examples

This appendix presents four high-support skill trees learned during an
ALFWorld frozen-executor run. The snapshot was taken at update step 7,128; the
available run summary covered 7,056 episodes with 5,834 successes (82.68%).
The run was still in progress when the artifacts were collected.

These examples were selected using a fixed rule: every root node had at least
300 recorded uses and a `success_when_used / call_count` ratio of at least
0.85. The examples are ordered by their task-family episode success rate.
These training-time statistics explain the selection and must not be
interpreted as independent held-out test results.

The learned procedural content below is reproduced without manual correction.
The stored `#`, `##`, and `###` heading levels are retained. Only JSON escaping
and the heat tree's enclosing code fence were removed, blank lines were added,
and bold markers were omitted for a cleaner plain-text presentation.

## Summary

| Task family | Tree version | Stored depth | Episodes | Wins | Episode success rate | Root-node evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `look_at_obj_in_light` | 30 | 3 | 613 | 570 | 92.99% | 330/361 successful uses (91.41%) |
| `pick_and_place` | 66 | 4 | 1,565 | 1,398 | 89.33% | 1,407/1,573 successful uses (89.45%) |
| `heat` | 41 | 3 | 911 | 803 | 88.14% | 799–909 uses per root; 88.78%–92.49% |
| `cool` | 61 | 3 | 1,060 | 918 | 86.60% | 918/1,053 successful uses (87.18%) |

---

Example 1: Examine an Object Under a Light (`look_at_obj_in_light`)

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

---

Example 2: Pick and Place (`pick_and_place`)

# pick_and_place skill tree

## Goal

Pick up a specified object and place it into or onto a specified receptacle.

## Sub-Goal Flow

1. Extract the full list of all receptacles from the initial observation.
2. Systematically visit every receptacle in that exact order, starting with the target receptacle.
3. When the target object is found, take it and go to the target receptacle to place it.
4. Stop immediately after the move action.

## Initial Observation & List Extraction

- From the welcome observation, note every furniture item (e.g., bed 1, drawer 2, cabinet 3). Include all surfaces (countertop, sidetable, sofa, armchair) and containers (drawer, cabinet, safe).
- If the observation is truncated (ends abruptly), use `look` once to get the full list. If the look result is empty, keep the original truncated list as your mental list. After that, do not `look` again.
- Create a mental list in the exact order they appear. This list determines the search order.

## Systematic Search (mandatory)

### Step 1: Go to the target receptacle first

- Navigate to the receptacle named in the task (e.g., if task says "put on armchair", go to armchair 1).
- If the receptacle is closed, `open` it immediately. Check if the target object is there.

### Step 2: Visit every other receptacle in list order

- If the object is not at the target receptacle, go to the next receptacle in your list (the one that appears after the target in the original observation).
- For each receptacle:
  - If it is closed, `open` it immediately upon arrival.
  - Read the observation. If the target object is present, `take <object> from <receptacle>` and go to Placement.
  - If not present, immediately move to the next receptacle in the list. Do not `examine`, `close`, or `look`.
- Strictly follow the list order – do not skip any receptacle, and do not revisit a receptacle.
- Continue until every single receptacle on the list has been visited exactly once. Do not stop early.

### Loop Avoidance

- If you ever find yourself at a receptacle you have already visited (check your mental list), you have looped. Break the loop by moving to the next unvisited receptacle in the list. Do not go back to a previous one.
- If the observation ever says "nothing" or repeats the same description, do not repeat actions; move on to the next receptacle.

## Object Selection Filter

- Before taking any object, verify that the base noun matches the target object exactly.
  - Example: target "soapbottle" is not "soapbar". Target "pillow" is not "cushion".
- Do not take an object if the name does not match. Continue searching.
- If you are holding an object that does not match, you may have picked it by mistake. Check inventory (`inventory` action) and then continue search, but avoid picking objects outside the systematic flow.

## Object Acquisition

- Action: `take <object> from <receptacle>` only after verification passes.
- Use the exact object name as it appears in the observation.

## Placement

- Go to the target receptacle. If it is closed, open it. Then `move <object> to <receptacle>`.
- Stop immediately after the move action. Do not perform any further actions.

## Critical Rules Summary

- Do not use object-specific guesses – the object can be anywhere. The systematic list guarantees you find it.
- Include all surface types and container types – do not skip any.
- No `look` or `inventory` during search (except the single allowed `look` on truncation).
- Never revisit a receptacle – if you catch yourself, go to the next unvisited one.

---

Example 3: Heat and Place (`heat`)

# heat

## 1. Interpret the task

- Identify the object to heat and the target placement location.
- The heating appliance is the microwave.
- The object must be heated before placing; do not place raw.

## 2. Locate the object

### 2.1 Immediate check: fridge

- Open the fridge. If the target object is inside, take it and go directly to Section 3 (Heat).
- If not, immediately start the exhaustive search below. Do not go anywhere else first.

### 2.2 Systematic Exhaustive Search (mandatory, follow exactly in order)

You must try every receptacle type in the fixed order below. For each type, try numbers sequentially (e.g., countertop 1, countertop 2, ...) until you receive an error that the location does not exist. At each, look and take if found. Do not skip any type; do not return to a previously checked type.

1. Countertops: Check all countertop numbers until error.
2. Dining tables: Check all diningtable numbers until error. (Do not skip even if none were visible initially.)
3. Cabinets: For each cabinet number, open it, look inside, take if found. Continue until error.
4. Drawers: For each drawer number, open it, look inside, take if found. Continue until error.
5. Other openable containers: Check sinkbasin 1, garbagecan 1, microwave 1, coffeemachine 1, toaster 1, etc. – try each type incrementally until error.
6. Repeat look: If still not found, perform a `look`. If new receptacles become visible, repeat the entire sequence from step 1. If another room is accessible, go there and repeat.

Critical rules:

- If you ever find yourself at the target placement location (e.g., fridge, diningtable) without the object, you have made a mistake. Immediately stop and resume the exhaustive search from the next unchecked receptacle type.
- Do not drop or place the object before heating.
- Once you pick up the object, go directly to the microwave (Section 3).

## 3. Heat the object

- Go to the microwave.
- Use `heat <object> with microwave` while holding it. Optionally open microwave first if direct heat fails.
- After heating, the object is ready. Do not return it to the microwave.

## 4. Place the heated object

- Go to the target receptacle specified in the task.
- If it is openable (cabinet, drawer, fridge, garbagecan), open it first.
- Use `move <object> to <receptacle>`.
- Do not take the object back after placing.

## 5. Completion check and termination

- After placing, verify: check inventory (must be empty) and examine the receptacle (must contain the object).
- If both conditions hold, task complete. Stop immediately. Do not loop.

---

Example 4: Cool and Place (`cool`)

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
