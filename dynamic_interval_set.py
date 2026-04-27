from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional
from typing import List, Optional, Sequence, Tuple

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

def _collect_intervals_inorder(node: Optional[_IntervalNode], out: List[CompactInterval]) -> None:
    if node is None:
        return
    _collect_intervals_inorder(node.left, out)
    out.append(node.interval)
    _collect_intervals_inorder(node.right, out)

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

class _IntervalNode:
    __slots__ = (
        "interval",
        "prio",
        "left",
        "right",
        "subtree_cost_sum",
        "subtree_pair_sum",
        "subtree_max_red_end",
    )

    def __init__(self, interval: "CompactInterval"):
        self.interval = interval
        self.prio = random.random()
        self.left: Optional["_IntervalNode"] = None
        self.right: Optional["_IntervalNode"] = None

        self.subtree_cost_sum = float(interval.cost)
        self.subtree_pair_sum = int(interval.size)
        self.subtree_max_red_end = int(interval.red_end)


def _subtree_cost(node: Optional[_IntervalNode]) -> float:
    return 0.0 if node is None else node.subtree_cost_sum


def _subtree_pairs(node: Optional[_IntervalNode]) -> int:
    return 0 if node is None else node.subtree_pair_sum


def _subtree_max_red_end(node: Optional[_IntervalNode]) -> int:
    return -10**18 if node is None else node.subtree_max_red_end


def _pull_interval(node: Optional[_IntervalNode]) -> None:
    if node is None:
        return
    node.subtree_cost_sum = (
        float(node.interval.cost)
        + _subtree_cost(node.left)
        + _subtree_cost(node.right)
    )
    node.subtree_pair_sum = (
        int(node.interval.size)
        + _subtree_pairs(node.left)
        + _subtree_pairs(node.right)
    )
    node.subtree_max_red_end = max(
        int(node.interval.red_end),
        _subtree_max_red_end(node.left),
        _subtree_max_red_end(node.right),
    )


def _merge_interval_treaps(
    a: Optional[_IntervalNode],
    b: Optional[_IntervalNode],
) -> Optional[_IntervalNode]:
    if a is None:
        return b
    if b is None:
        return a
    if a.prio < b.prio:
        a.right = _merge_interval_treaps(a.right, b)
        _pull_interval(a)
        return a
    else:
        b.left = _merge_interval_treaps(a, b.left)
        _pull_interval(b)
        return b


def _split_interval_treap(
    root: Optional[_IntervalNode],
    key: int,
) -> Tuple[Optional[_IntervalNode], Optional[_IntervalNode]]:
    """
    Split by red_start:
      left  = intervals with red_start < key
      right = intervals with red_start >= key
    """
    if root is None:
        return None, None

    if root.interval.red_start < key:
        a, b = _split_interval_treap(root.right, key)
        root.right = a
        _pull_interval(root)
        return root, b
    else:
        a, b = _split_interval_treap(root.left, key)
        root.left = b
        _pull_interval(root)
        return a, root


def _rightmost(node: Optional[_IntervalNode]) -> Optional[_IntervalNode]:
    if node is None:
        return None
    while node.right is not None:
        node = node.right
    return node


def _inorder_collect(
    node: Optional[_IntervalNode],
    out: List["CompactInterval"],
) -> None:
    if node is None:
        return
    _inorder_collect(node.left, out)
    out.append(node.interval)
    _inorder_collect(node.right, out)

class DynamicIntervalSetSummary:
    """
    Maintains the current disjoint interval solution using a treap with subtree
    aggregates. Candidate evaluation never scans the whole solution.

    Expected time:
      - range_summary: O(log m)
      - apply_interval: O(log m)
      - current cost/size: O(1)
    """

    def __init__(self):
        self.root: Optional[_IntervalNode] = None
        self.total_cost: float = 0.0
        self.total_size: int = 0

    def _restore(
        self,
        a: Optional[_IntervalNode],
        b: Optional[_IntervalNode],
        c: Optional[_IntervalNode],
    ) -> None:
        self.root = _merge_interval_treaps(a, _merge_interval_treaps(b, c))

    def range_summary(
        self,
        red_start: int,
        red_end: int,
    ) -> Tuple[float, int]:
        """
        Return aggregate (swallowed_cost, swallowed_pairs) for all current
        intervals whose red_start lies in [red_start, red_end].

        This is the logarithmic replacement for scanning the current solution.
        """
        a, bc = _split_interval_treap(self.root, red_start)
        b, c = _split_interval_treap(bc, red_end + 1)

        # Defensive check against partial overlap from the left.
        pred = _rightmost(a)
        if pred is not None and pred.interval.red_end >= red_start:
            self._restore(a, b, c)
            raise ValueError(
                "Candidate interval partially overlaps an existing interval on the left."
            )

        # Defensive check against partial overlap inside the middle block.
        if b is not None and _subtree_max_red_end(b) > red_end:
            self._restore(a, b, c)
            raise ValueError(
                "Candidate interval partially overlaps an existing interval in the middle block."
            )

        swallowed_cost = _subtree_cost(b)
        swallowed_pairs = _subtree_pairs(b)

        self._restore(a, b, c)
        return swallowed_cost, swallowed_pairs

    def apply_interval(
        self,
        merged_interval: "CompactInterval",
    ) -> Tuple[float, int]:
        """
        Replace all current intervals with red_start in
        [merged.red_start, merged.red_end] by merged_interval.

        Returns:
            (swallowed_cost, swallowed_pairs)
        """
        a, bc = _split_interval_treap(self.root, merged_interval.red_start)
        b, c = _split_interval_treap(bc, merged_interval.red_end + 1)

        pred = _rightmost(a)
        if pred is not None and pred.interval.red_end >= merged_interval.red_start:
            self._restore(a, b, c)
            raise ValueError(
                "Chosen interval partially overlaps an existing interval on the left."
            )
        if b is not None and _subtree_max_red_end(b) > merged_interval.red_end:
            self._restore(a, b, c)
            raise ValueError(
                "Chosen interval partially overlaps an existing interval in the middle block."
            )

        swallowed_cost = _subtree_cost(b)
        swallowed_pairs = _subtree_pairs(b)

        new_node = _IntervalNode(merged_interval)
        self.root = _merge_interval_treaps(a, _merge_interval_treaps(new_node, c))

        self.total_cost = self.total_cost - swallowed_cost + float(merged_interval.cost)
        self.total_size = self.total_size - swallowed_pairs + int(merged_interval.size)

        return swallowed_cost, swallowed_pairs

    def apply_interval_with_extract(
        self,
        merged_interval: CompactInterval,
    ) -> Tuple[List[CompactInterval], float, int]:
        """
        Replace all current intervals with red_start in
        [merged_interval.red_start, merged_interval.red_end] by merged_interval.

        Returns
        -------
        subsumed_intervals : list[CompactInterval]
            The swallowed intervals, in sorted order.
        swallowed_cost : float
        swallowed_pairs : int

        Expected time:
            O(log m + t), where t is the number of swallowed intervals.
        Across a whole replay, sum(t) = O(n).
        """
        a, bc = _split_interval_treap(self.root, merged_interval.red_start)
        b, c = _split_interval_treap(bc, merged_interval.red_end + 1)

        pred = _rightmost(a)
        if pred is not None and pred.interval.red_end >= merged_interval.red_start:
            self._restore(a, b, c)
            raise ValueError(
                "Chosen interval partially overlaps an existing interval on the left."
            )
        if b is not None and _subtree_max_red_end(b) > merged_interval.red_end:
            self._restore(a, b, c)
            raise ValueError(
                "Chosen interval partially overlaps an existing interval in the middle block."
            )

        subsumed_intervals: List[CompactInterval] = []
        _collect_intervals_inorder(b, subsumed_intervals)

        swallowed_cost = _subtree_cost(b)
        swallowed_pairs = _subtree_pairs(b)

        new_node = _IntervalNode(merged_interval)
        self.root = _merge_interval_treaps(a, _merge_interval_treaps(new_node, c))

        self.total_cost = self.total_cost - swallowed_cost + float(merged_interval.cost)
        self.total_size = self.total_size - swallowed_pairs + int(merged_interval.size)

        return subsumed_intervals, swallowed_cost, swallowed_pairs

    def to_sorted_intervals(self) -> List["CompactInterval"]:
        out: List["CompactInterval"] = []
        _inorder_collect(self.root, out)
        return out

