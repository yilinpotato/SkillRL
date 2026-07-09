# Copyright 2025 CoSkill.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Playbook Tree (下行渐进式技能树) —— 把 markdown playbook 当**树**做增删改查。

markdown 的标题层级（'#' 的数量）天然就是一棵树：一个比父标题深一级的标题
（'##' 在 '#' 下）就是对父节点的**细化叶子**。本模块把 playbook markdown 解析成
可寻址的节点树，为每个节点算一个稳定 id（按标题路径 slug），从而支持：
  - 寻址 / 查询：按标题路径或 id 找节点
  - 增 / 改：在指定标题下插入或替换子节点（对接失败诊断的 patch_location）
  - 删 / 剪枝：渲染时跳过被弃用/内化的节点子树，收缩注入 context
  - diff：跨版本按 id 匹配，判断哪些节点新增/改写/消失（供逐节点生命周期继承）

纯字符串/正则，无第三方依赖。渲染 `to_markdown(skip_ids=...)` 逐字节还原（跳过被剪
节点），故"解析→渲染"对未剪枝的 playbook 是可逆的。
"""

import re
from typing import Dict, List, Optional, Set, Tuple

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-z]+", "-", text.strip().lower()).strip("-")
    return s or "section"


class PBNode:
    """一个 playbook 节点 = 一个标题 + 它自己的正文行 + 子节点。root 的 level=0。"""

    __slots__ = ("level", "title", "body", "children")

    def __init__(self, level: int, title: str,
                 body: Optional[List[str]] = None,
                 children: Optional[List["PBNode"]] = None):
        self.level = level
        self.title = title
        self.body = body if body is not None else []
        self.children = children if children is not None else []


def parse(md: str) -> PBNode:
    """把 markdown 解析成节点树。首个标题前的行归 root.body（如开头的 goal 行）。"""
    root = PBNode(0, "")
    stack: List[PBNode] = [root]
    for line in (md or "").splitlines():
        m = _HEADING.match(line)
        if m:
            lvl = len(m.group(1))
            node = PBNode(lvl, m.group(2).strip())
            while len(stack) > 1 and stack[-1].level >= lvl:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].body.append(line)
    return root


def _walk(node: PBNode, path: List[str], seen: Set[str]):
    """深度优先。产出 (node, path_titles, node_id)。同父下重名标题自动加后缀去重。"""
    for child in node.children:
        p = path + [child.title]
        base = "/".join(_slug(t) for t in p)
        nid = base
        k = 2
        while nid in seen:
            nid = f"{base}-{k}"
            k += 1
        seen.add(nid)
        yield child, p, nid
        yield from _walk(child, p, seen)


def node_index(root: PBNode) -> Dict[str, Dict]:
    """返回 {node_id: {title, path, level, body_hash}}。body_hash 用于判定节点内容是否变化。"""
    import hashlib
    idx: Dict[str, Dict] = {}
    for node, path, nid in _walk(root, [], set()):
        h = hashlib.md5(("\n".join(node.body)).encode("utf-8")).hexdigest()[:12]
        idx[nid] = {"title": node.title, "path": path, "level": node.level, "body_hash": h}
    return idx


def find(root: PBNode, path_titles: List[str]) -> Optional[PBNode]:
    """按标题路径寻址节点。"""
    node = root
    for t in path_titles:
        nxt = next((c for c in node.children if c.title == t), None)
        if nxt is None:
            return None
        node = nxt
    return node


def to_markdown(root: PBNode, skip_ids: Optional[Set[str]] = None) -> str:
    """渲染回 markdown；skip_ids 中的节点连同其子树整体跳过（剪枝/内化用）。"""
    skip = skip_ids or set()
    out: List[str] = []
    out.extend(root.body)

    def emit(node: PBNode, path: List[str], seen: Set[str]):
        for child in node.children:
            p = path + [child.title]
            base = "/".join(_slug(t) for t in p)
            nid = base
            k = 2
            while nid in seen:
                nid = f"{base}-{k}"
                k += 1
            seen.add(nid)
            if nid in skip:
                continue  # 跳过该节点及其整棵子树
            out.append("#" * child.level + " " + child.title)
            out.extend(child.body)
            emit(child, p, seen)

    emit(root, [], set())
    # 去掉首尾多余空行，保持整洁
    return "\n".join(out).strip()


def diff(old_index: Dict[str, Dict], new_index: Dict[str, Dict]) -> Dict[str, List[str]]:
    """按 node_id 比较两版：新增 / 消失 / 改写(同 id 但 body_hash 变) / 不变。"""
    old_ids, new_ids = set(old_index), set(new_index)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    modified, unchanged = [], []
    for nid in sorted(new_ids & old_ids):
        if old_index[nid].get("body_hash") != new_index[nid].get("body_hash"):
            modified.append(nid)
        else:
            unchanged.append(nid)
    return {"added": added, "removed": removed, "modified": modified, "unchanged": unchanged}


def max_depth(root: PBNode) -> int:
    """最深标题深度（1 = 只有 '#'，2 = 出现 '##' …；无标题返回 0）。"""
    best = 0
    for node, _, _ in _walk(root, [], set()):
        best = max(best, node.level)
    return best
