from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fft_utils import combine_shift_arrays, real_convolution, shift_array_value
from models import ColoredPoint, CompactInterval
from mp_types import Interval, Point
from utils import validate_instance, validate_sorted


def points_in_interval(points: Sequence[Point], interval: Interval) -> list[Point]:
    left, right = interval
    if left > right:
        raise ValueError("interval must satisfy left <= right.")
    validate_sorted(points, "points")
    i = bisect_left(points, left)
    j = bisect_right(points, right)
    return list(points[i:j])


def interval_is_balanced(R: Sequence[Point], B: Sequence[Point], interval: Interval) -> bool:
    return len(points_in_interval(R, interval)) == len(points_in_interval(B, interval))


def interval_matching_query(R: Sequence[Point], B: Sequence[Point], interval: Interval) -> CompactInterval:
    validate_instance(R, B)
    left, right = interval
    red_start = bisect_left(R, left)
    red_after_end = bisect_right(R, right)
    blue_start = bisect_left(B, left)
    blue_after_end = bisect_right(B, right)

    red_count = red_after_end - red_start
    blue_count = blue_after_end - blue_start
    if red_count != blue_count:
        raise ValueError(
            "The interval is not balanced: "
            f"found {red_count} red points and {blue_count} blue points."
        )
    if red_count == 0:
        raise ValueError("The interval contains no points and cannot define a matching.")

    red_slice = R[red_start:red_after_end]
    blue_slice = B[blue_start:blue_after_end]
    cost = float(sum((r - b) ** 2 for r, b in zip(red_slice, blue_slice)))
    return CompactInterval(
        red_start=red_start,
        red_end=red_after_end - 1,
        blue_start=blue_start,
        blue_end=blue_after_end - 1,
        cost=cost,
    )


def build_colored_sequence(R: Sequence[Point], B: Sequence[Point]) -> list[ColoredPoint]:
    raw: list[ColoredPoint] = []
    uid = 0
    for x in R:
        raw.append(ColoredPoint(position=float(x), color="R", uid=uid))
        uid += 1
    for x in B:
        raw.append(ColoredPoint(position=float(x), color="B", uid=uid))
        uid += 1
    raw.sort(key=lambda p: (p.position, p.color, p.uid))
    return [ColoredPoint(position=p.position, color=p.color, uid=i) for i, p in enumerate(raw)]


@dataclass
class _RangeTreeNode:
    l: int
    r: int
    mid: int
    left: Optional["_RangeTreeNode"] = None
    right: Optional["_RangeTreeNode"] = None
    red_start: Optional[int] = None
    red_end: Optional[int] = None
    blue_start: Optional[int] = None
    blue_end: Optional[int] = None
    total_shift_min: int = 0
    total_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cross_lr_shift_min: int = 0
    cross_lr_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cross_rl_shift_min: int = 0
    cross_rl_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


class BalancedIntervalSquaredCostDataStructure:
    """Range-tree data structure for balanced interval matching with squared cost."""

    def __init__(self, R: Sequence[Point], B: Sequence[Point]):
        validate_instance(R, B)
        self.R = np.asarray(list(R), dtype=np.float64)
        self.B = np.asarray(list(B), dtype=np.float64)
        self.n = len(R)
        self.all_points = build_colored_sequence(R, B)
        self.merged_positions = np.asarray([p.position for p in self.all_points], dtype=np.float64)

        m = len(self.all_points)
        self.prefix_red_in_merged = np.zeros(m + 1, dtype=np.int64)
        self.prefix_blue_in_merged = np.zeros(m + 1, dtype=np.int64)
        for i, point in enumerate(self.all_points):
            self.prefix_red_in_merged[i + 1] = self.prefix_red_in_merged[i] + int(point.color == "R")
            self.prefix_blue_in_merged[i + 1] = self.prefix_blue_in_merged[i] + int(point.color == "B")

        self.prefix_r2 = np.zeros(self.n + 1, dtype=np.float64)
        self.prefix_b2 = np.zeros(self.n + 1, dtype=np.float64)
        self.prefix_r2[1:] = np.cumsum(self.R * self.R)
        self.prefix_b2[1:] = np.cumsum(self.B * self.B)
        self.root = self._build_node(0, m - 1)

    def _range_color_indices(self, l: int, r: int, color: str):
        if color == "R":
            prefix = self.prefix_red_in_merged
        elif color == "B":
            prefix = self.prefix_blue_in_merged
        else:
            raise ValueError("color must be 'R' or 'B'.")
        start = int(prefix[l])
        after_end = int(prefix[r + 1])
        if after_end == start:
            return None, None
        return start, after_end - 1

    def _build_cross_left_red_right_blue(self, left: _RangeTreeNode, right: _RangeTreeNode):
        if left.red_start is None or right.blue_start is None:
            return 0, np.zeros(0, dtype=np.float64)
        u, v = left.red_start, left.red_end
        s, t = right.blue_start, right.blue_end
        assert v is not None and t is not None
        values = real_convolution(self.R[u : v + 1][::-1], self.B[s : t + 1])
        return s - v, values

    def _build_cross_right_red_left_blue(self, left: _RangeTreeNode, right: _RangeTreeNode):
        if right.red_start is None or left.blue_start is None:
            return 0, np.zeros(0, dtype=np.float64)
        u, v = right.red_start, right.red_end
        s, t = left.blue_start, left.blue_end
        assert v is not None and t is not None
        values = real_convolution(self.R[u : v + 1][::-1], self.B[s : t + 1])
        return s - v, values

    def _build_node(self, l: int, r: int) -> _RangeTreeNode:
        node = _RangeTreeNode(l=l, r=r, mid=(l + r) // 2)
        node.red_start, node.red_end = self._range_color_indices(l, r, "R")
        node.blue_start, node.blue_end = self._range_color_indices(l, r, "B")
        if l == r:
            return node
        node.left = self._build_node(l, node.mid)
        node.right = self._build_node(node.mid + 1, r)
        assert node.left is not None and node.right is not None
        node.cross_lr_shift_min, node.cross_lr_values = self._build_cross_left_red_right_blue(node.left, node.right)
        node.cross_rl_shift_min, node.cross_rl_values = self._build_cross_right_red_left_blue(node.left, node.right)
        node.total_shift_min, node.total_values = combine_shift_arrays(
            [
                (node.left.total_shift_min, node.left.total_values),
                (node.cross_lr_shift_min, node.cross_lr_values),
                (node.cross_rl_shift_min, node.cross_rl_values),
                (node.right.total_shift_min, node.right.total_values),
            ]
        )
        return node

    def _query_product_sum(self, node: _RangeTreeNode, query_l: int, query_r: int, shift: int) -> float:
        if query_l <= node.l and node.r <= query_r:
            return shift_array_value(node.total_shift_min, node.total_values, shift)
        if node.left is None or node.right is None:
            return 0.0
        if query_r <= node.mid:
            return self._query_product_sum(node.left, query_l, query_r, shift)
        if query_l > node.mid:
            return self._query_product_sum(node.right, query_l, query_r, shift)
        return (
            shift_array_value(node.cross_lr_shift_min, node.cross_lr_values, shift)
            + shift_array_value(node.cross_rl_shift_min, node.cross_rl_values, shift)
            + self._query_product_sum(node.left, query_l, query_r, shift)
            + self._query_product_sum(node.right, query_l, query_r, shift)
        )

    def match_interval_by_indices(self, merged_left: int, merged_right: int) -> CompactInterval:
        red_count = int(self.prefix_red_in_merged[merged_right + 1] - self.prefix_red_in_merged[merged_left])
        blue_count = int(self.prefix_blue_in_merged[merged_right + 1] - self.prefix_blue_in_merged[merged_left])
        if red_count != blue_count:
            raise ValueError(
                "Queried merged interval is not balanced: "
                f"found {red_count} red points and {blue_count} blue points."
            )
        if red_count == 0:
            raise ValueError("The interval contains no points and cannot define a matching.")
        red_start = int(self.prefix_red_in_merged[merged_left])
        blue_start = int(self.prefix_blue_in_merged[merged_left])
        red_end = red_start + red_count - 1
        blue_end = blue_start + blue_count - 1
        sum_r2 = float(self.prefix_r2[red_end + 1] - self.prefix_r2[red_start])
        sum_b2 = float(self.prefix_b2[blue_end + 1] - self.prefix_b2[blue_start])
        prod_sum = self._query_product_sum(self.root, merged_left, merged_right, blue_start - red_start)
        return CompactInterval(
            red_start=red_start,
            red_end=red_end,
            blue_start=blue_start,
            blue_end=blue_end,
            cost=float(sum_r2 + sum_b2 - 2.0 * prod_sum),
        )

    def match_interval_by_coordinates(self, left: float, right: float) -> CompactInterval:
        merged_left = bisect_left(self.merged_positions, left)
        merged_after_right = bisect_right(self.merged_positions, right)
        if merged_left >= merged_after_right:
            raise ValueError("The interval contains no points and cannot define a matching.")
        return self.match_interval_by_indices(merged_left, merged_after_right - 1)

@dataclass
class _LazyRangeTreeNode:
    l: int
    r: int
    mid: int
    left: Optional["_LazyRangeTreeNode"] = None
    right: Optional["_LazyRangeTreeNode"] = None
    red_start: Optional[int] = None
    red_end: Optional[int] = None
    blue_start: Optional[int] = None
    blue_end: Optional[int] = None

    # Lazily computed tables.
    total_shift_min: int = 0
    total_values: Optional[np.ndarray] = None

    cross_lr_shift_min: int = 0
    cross_lr_values: Optional[np.ndarray] = None

    cross_rl_shift_min: int = 0
    cross_rl_values: Optional[np.ndarray] = None


class LazyBalancedIntervalSquaredCostDataStructure:
    """Range-tree data structure for balanced interval matching with squared cost.

    This version is lazy:
    - the tree skeleton is built eagerly,
    - FFT tables are computed only when first needed.
    """

    def __init__(self, R: Sequence[Point], B: Sequence[Point]):
        validate_instance(R, B)
        self.R = np.asarray(list(R), dtype=np.float64)
        self.B = np.asarray(list(B), dtype=np.float64)
        self.n = len(R)
        self.all_points = build_colored_sequence(R, B)
        self.merged_positions = np.asarray([p.position for p in self.all_points], dtype=np.float64)

        m = len(self.all_points)
        self.prefix_red_in_merged = np.zeros(m + 1, dtype=np.int64)
        self.prefix_blue_in_merged = np.zeros(m + 1, dtype=np.int64)
        for i, point in enumerate(self.all_points):
            self.prefix_red_in_merged[i + 1] = self.prefix_red_in_merged[i] + int(point.color == "R")
            self.prefix_blue_in_merged[i + 1] = self.prefix_blue_in_merged[i] + int(point.color == "B")

        self.prefix_r2 = np.zeros(self.n + 1, dtype=np.float64)
        self.prefix_b2 = np.zeros(self.n + 1, dtype=np.float64)
        self.prefix_r2[1:] = np.cumsum(self.R * self.R)
        self.prefix_b2[1:] = np.cumsum(self.B * self.B)

        # Only build the tree skeleton here.
        self.root = self._build_node(0, m - 1)

    def _range_color_indices(self, l: int, r: int, color: str):
        if color == "R":
            prefix = self.prefix_red_in_merged
        elif color == "B":
            prefix = self.prefix_blue_in_merged
        else:
            raise ValueError("color must be 'R' or 'B'.")
        start = int(prefix[l])
        after_end = int(prefix[r + 1])
        if after_end == start:
            return None, None
        return start, after_end - 1

    def _build_cross_left_red_right_blue(self, left: _LazyRangeTreeNode, right: _LazyRangeTreeNode):
        if left.red_start is None or right.blue_start is None:
            return 0, np.zeros(0, dtype=np.float64)
        u, v = left.red_start, left.red_end
        s, t = right.blue_start, right.blue_end
        assert v is not None and t is not None
        values = real_convolution(self.R[u : v + 1][::-1], self.B[s : t + 1])
        return s - v, values

    def _build_cross_right_red_left_blue(self, left: _LazyRangeTreeNode, right: _LazyRangeTreeNode):
        if right.red_start is None or left.blue_start is None:
            return 0, np.zeros(0, dtype=np.float64)
        u, v = right.red_start, right.red_end
        s, t = left.blue_start, left.blue_end
        assert v is not None and t is not None
        values = real_convolution(self.R[u : v + 1][::-1], self.B[s : t + 1])
        return s - v, values

    def _build_node(self, l: int, r: int) -> _LazyRangeTreeNode:
        node = _LazyRangeTreeNode(l=l, r=r, mid=(l + r) // 2)
        node.red_start, node.red_end = self._range_color_indices(l, r, "R")
        node.blue_start, node.blue_end = self._range_color_indices(l, r, "B")

        if l == r:
            return node

        node.left = self._build_node(l, node.mid)
        node.right = self._build_node(node.mid + 1, r)
        return node

    def _ensure_cross_lr(self, node: _LazyRangeTreeNode) -> None:
        if node.cross_lr_values is not None:
            return
        if node.left is None or node.right is None:
            node.cross_lr_shift_min = 0
            node.cross_lr_values = np.zeros(0, dtype=np.float64)
            return
        node.cross_lr_shift_min, node.cross_lr_values = self._build_cross_left_red_right_blue(node.left, node.right)

    def _ensure_cross_rl(self, node: _LazyRangeTreeNode) -> None:
        if node.cross_rl_values is not None:
            return
        if node.left is None or node.right is None:
            node.cross_rl_shift_min = 0
            node.cross_rl_values = np.zeros(0, dtype=np.float64)
            return
        node.cross_rl_shift_min, node.cross_rl_values = self._build_cross_right_red_left_blue(node.left, node.right)

    def _ensure_total(self, node: _LazyRangeTreeNode) -> None:
        if node.total_values is not None:
            return

        if node.left is None or node.right is None:
            # Leaf: no red-blue pair can be fully contained in a single merged position.
            node.total_shift_min = 0
            node.total_values = np.zeros(0, dtype=np.float64)
            return

        self._ensure_total(node.left)
        self._ensure_total(node.right)
        self._ensure_cross_lr(node)
        self._ensure_cross_rl(node)

        node.total_shift_min, node.total_values = combine_shift_arrays(
            [
                (node.left.total_shift_min, node.left.total_values),
                (node.cross_lr_shift_min, node.cross_lr_values),
                (node.cross_rl_shift_min, node.cross_rl_values),
                (node.right.total_shift_min, node.right.total_values),
            ]
        )

    def _query_product_sum(self, node: _LazyRangeTreeNode, query_l: int, query_r: int, shift: int) -> float:
        if query_l <= node.l and node.r <= query_r:
            self._ensure_total(node)
            assert node.total_values is not None
            return shift_array_value(node.total_shift_min, node.total_values, shift)

        if node.left is None or node.right is None:
            return 0.0

        if query_r <= node.mid:
            return self._query_product_sum(node.left, query_l, query_r, shift)

        if query_l > node.mid:
            return self._query_product_sum(node.right, query_l, query_r, shift)

        self._ensure_cross_lr(node)
        self._ensure_cross_rl(node)
        assert node.cross_lr_values is not None
        assert node.cross_rl_values is not None

        return (
            shift_array_value(node.cross_lr_shift_min, node.cross_lr_values, shift)
            + shift_array_value(node.cross_rl_shift_min, node.cross_rl_values, shift)
            + self._query_product_sum(node.left, query_l, query_r, shift)
            + self._query_product_sum(node.right, query_l, query_r, shift)
        )

    def match_interval_by_indices(self, merged_left: int, merged_right: int) -> CompactInterval:
        red_count = int(self.prefix_red_in_merged[merged_right + 1] - self.prefix_red_in_merged[merged_left])
        blue_count = int(self.prefix_blue_in_merged[merged_right + 1] - self.prefix_blue_in_merged[merged_left])
        if red_count != blue_count:
            raise ValueError(
                "Queried merged interval is not balanced: "
                f"found {red_count} red points and {blue_count} blue points."
            )
        if red_count == 0:
            raise ValueError("The interval contains no points and cannot define a matching.")

        red_start = int(self.prefix_red_in_merged[merged_left])
        blue_start = int(self.prefix_blue_in_merged[merged_left])
        red_end = red_start + red_count - 1
        blue_end = blue_start + blue_count - 1

        sum_r2 = float(self.prefix_r2[red_end + 1] - self.prefix_r2[red_start])
        sum_b2 = float(self.prefix_b2[blue_end + 1] - self.prefix_b2[blue_start])
        prod_sum = self._query_product_sum(self.root, merged_left, merged_right, blue_start - red_start)

        return CompactInterval(
            red_start=red_start,
            red_end=red_end,
            blue_start=blue_start,
            blue_end=blue_end,
            cost=float(sum_r2 + sum_b2 - 2.0 * prod_sum),
        )

    def match_interval_by_coordinates(self, left: float, right: float) -> CompactInterval:
        merged_left = bisect_left(self.merged_positions, left)
        merged_after_right = bisect_right(self.merged_positions, right)
        if merged_left >= merged_after_right:
            raise ValueError("The interval contains no points and cannot define a matching.")
        return self.match_interval_by_indices(merged_left, merged_after_right - 1)
