







""

import re
from typing import Dict, List, Optional, Set, Tuple

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-z]+", "-", text.strip().lower()).strip("-")
    return s or "section"


class PBNode:
    ""

    __slots__ = ("level", "title", "body", "children")

    def __init__(self, level: int, title: str,
                 body: Optional[List[str]] = None,
                 children: Optional[List["PBNode"]] = None):
        self.level = level
        self.title = title
        self.body = body if body is not None else []
        self.children = children if children is not None else []


def parse(md: str) -> PBNode:
    ""
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
    ""
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
    ""
    import hashlib
    idx: Dict[str, Dict] = {}
    for node, path, nid in _walk(root, [], set()):
        h = hashlib.md5(("\n".join(node.body)).encode("utf-8")).hexdigest()[:12]
        idx[nid] = {"title": node.title, "path": path, "level": node.level, "body_hash": h}
    return idx


def find(root: PBNode, path_titles: List[str]) -> Optional[PBNode]:
    ""
    node = root
    for t in path_titles:
        nxt = next((c for c in node.children if c.title == t), None)
        if nxt is None:
            return None
        node = nxt
    return node


def to_markdown(
    root: PBNode,
    skip_ids: Optional[Set[str]] = None,
    elide_ids: Optional[Set[str]] = None,
) -> str:
    ""
    skip = skip_ids or set()
    elide = elide_ids or set()
    out: List[str] = []
    out.extend(root.body)

    def emit(node: PBNode, path: List[str], seen: Set[str], hidden_levels: int = 0):
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
                continue
            if nid in elide:



                emit(child, p, seen, hidden_levels + 1)
                continue
            out.append("#" * max(1, child.level - hidden_levels) + " " + child.title)
            out.extend(child.body)
            emit(child, p, seen, hidden_levels)

    emit(root, [], set())

    return "\n".join(out).strip()


def diff(old_index: Dict[str, Dict], new_index: Dict[str, Dict]) -> Dict[str, List[str]]:
    ""
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
    ""
    best = 0
    for node, _, _ in _walk(root, [], set()):
        best = max(best, node.level)
    return best
