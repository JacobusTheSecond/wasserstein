from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models import CandidateEvaluation, CompactInterval


@dataclass
class _TreapNode:
    interval: CompactInterval
    priority: int
    left: Optional["_TreapNode"] = None
    right: Optional["_TreapNode"] = None
    subtree_max_red_end: int = 0
    subtree_total_cost: float = 0.0
    subtree_total_size: int = 0


def _treap_priority(interval: CompactInterval) -> int:
    x = ((interval.red_start + 1) << 32) ^ (interval.blue_start + 1) ^ 0x9E3779B97F4A7C15
    x ^= x >> 30
    x *= 0xBF58476D1CE4E5B9
    x &= (1 << 64) - 1
    x ^= x >> 27
    x *= 0x94D049BB133111EB
    x &= (1 << 64) - 1
    x ^= x >> 31
    return int(x)


def _treap_update(node: Optional[_TreapNode]) -> Optional[_TreapNode]:
    if node is None:
        return None
    node.subtree_max_red_end = node.interval.red_end
    node.subtree_total_cost = node.interval.cost
    node.subtree_total_size = node.interval.size
    if node.left is not None:
        node.subtree_max_red_end = max(node.subtree_max_red_end, node.left.subtree_max_red_end)
        node.subtree_total_cost += node.left.subtree_total_cost
        node.subtree_total_size += node.left.subtree_total_size
    if node.right is not None:
        node.subtree_max_red_end = max(node.subtree_max_red_end, node.right.subtree_max_red_end)
        node.subtree_total_cost += node.right.subtree_total_cost
        node.subtree_total_size += node.right.subtree_total_size
    return node


def _treap_merge(left: Optional[_TreapNode], right: Optional[_TreapNode]) -> Optional[_TreapNode]:
    if left is None:
        return right
    if right is None:
        return left
    if left.priority <= right.priority:
        left.right = _treap_merge(left.right, right)
        return _treap_update(left)
    right.left = _treap_merge(left, right.left)
    return _treap_update(right)


def _treap_split(root: Optional[_TreapNode], key: int):
    if root is None:
        return None, None
    if root.interval.red_start < key:
        a, b = _treap_split(root.right, key)
        root.right = a
        return _treap_update(root), _treap_update(b)
    a, b = _treap_split(root.left, key)
    root.left = b
    return _treap_update(a), _treap_update(root)


def _treap_rightmost(root: Optional[_TreapNode]) -> Optional[_TreapNode]:
    if root is None:
        return None
    while root.right is not None:
        root = root.right
    return root


def _treap_collect_inorder(root: Optional[_TreapNode], out: list[CompactInterval]) -> None:
    if root is None:
        return
    _treap_collect_inorder(root.left, out)
    out.append(root.interval)
    _treap_collect_inorder(root.right, out)


class DynamicIntervalSet:
    """Dynamic set of disjoint matched intervals with O(log n + t) range updates."""

    def __init__(self):
        self.root: Optional[_TreapNode] = None

    @property
    def total_cost(self) -> float:
        return 0.0 if self.root is None else float(self.root.subtree_total_cost)

    @property
    def total_size(self) -> int:
        return 0 if self.root is None else int(self.root.subtree_total_size)

    def extract_subsumed_intervals(self, merged_interval: CompactInterval):
        left_tree, mid_right = _treap_split(self.root, merged_interval.red_start)
        middle_tree, right_tree = _treap_split(mid_right, merged_interval.red_end + 1)

        predecessor = _treap_rightmost(left_tree)
        if predecessor is not None and predecessor.interval.red_end >= merged_interval.red_start:
            self.root = _treap_merge(left_tree, _treap_merge(middle_tree, right_tree))
            raise ValueError(
                "Candidate interval partially overlaps an existing interval. "
                "The greedy invariant of disjoint intervals has been violated."
            )

        if middle_tree is not None and middle_tree.subtree_max_red_end > merged_interval.red_end:
            self.root = _treap_merge(left_tree, _treap_merge(middle_tree, right_tree))
            raise ValueError(
                "Candidate interval partially overlaps an existing interval. "
                "The greedy invariant of disjoint intervals has been violated."
            )

        subsumed: list[CompactInterval] = []
        _treap_collect_inorder(middle_tree, subsumed)
        for current in subsumed:
            contained = (
                merged_interval.red_start <= current.red_start <= current.red_end <= merged_interval.red_end
                and merged_interval.blue_start <= current.blue_start <= current.blue_end <= merged_interval.blue_end
            )
            if not contained:
                self.root = _treap_merge(left_tree, _treap_merge(middle_tree, right_tree))
                raise ValueError(
                    "Candidate interval partially overlaps an existing interval. "
                    "The greedy invariant of disjoint intervals has been violated."
                )

        removed_cost = 0.0 if middle_tree is None else float(middle_tree.subtree_total_cost)
        removed_size = 0 if middle_tree is None else int(middle_tree.subtree_total_size)
        self.root = _treap_merge(left_tree, _treap_merge(middle_tree, right_tree))
        return subsumed, removed_cost, removed_size

    def apply_candidate(self, delta: CandidateEvaluation) -> None:
        left_tree, mid_right = _treap_split(self.root, delta.merged_interval.red_start)
        _middle_tree, right_tree = _treap_split(mid_right, delta.merged_interval.red_end + 1)
        new_node = _TreapNode(interval=delta.merged_interval, priority=_treap_priority(delta.merged_interval))
        _treap_update(new_node)
        self.root = _treap_merge(left_tree, _treap_merge(new_node, right_tree))
