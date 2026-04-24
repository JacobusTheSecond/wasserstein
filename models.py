from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from mp_types import Interval, Pair, Point


@dataclass(frozen=True)
class CompactInterval:
    """A balanced interval stored only by red/blue index ranges and its cost."""

    red_start: int
    red_end: int
    blue_start: int
    blue_end: int
    cost: float

    @property
    def size(self) -> int:
        return self.red_end - self.red_start + 1

    def key(self) -> Tuple[int, int, int, int]:
        return (self.red_start, self.red_end, self.blue_start, self.blue_end)

    def left_coordinate(self, R: Sequence[Point], B: Sequence[Point]) -> float:
        return float(min(R[self.red_start], B[self.blue_start]))

    def right_coordinate(self, R: Sequence[Point], B: Sequence[Point]) -> float:
        return float(max(R[self.red_end], B[self.blue_end]))

    def red_points(self, R: Sequence[Point]) -> List[Point]:
        return list(R[self.red_start : self.red_end + 1])

    def blue_points(self, B: Sequence[Point]) -> List[Point]:
        return list(B[self.blue_start : self.blue_end + 1])

    def pairs(self, R: Sequence[Point], B: Sequence[Point]) -> List[Pair]:
        return list(zip(self.red_points(R), self.blue_points(B)))


@dataclass(frozen=True)
class ColoredPoint:
    """A point in the merged order R ∪ B with its color and uid."""

    position: Point
    color: str
    uid: int = -1


@dataclass(frozen=True)
class CandidateEvaluation:
    """A candidate interval together with the delta it induces on the solution."""

    left_endpoint: ColoredPoint
    right_endpoint: ColoredPoint
    merged_interval: CompactInterval
    subsumed_intervals: List[CompactInterval]
    delta_cost: float

    @property
    def interval(self) -> Interval:
        return (self.left_endpoint.position, self.right_endpoint.position)

    @property
    def added_pairs(self) -> int:
        previous_pairs = sum(interval.size for interval in self.subsumed_intervals)
        return self.merged_interval.size - previous_pairs


@dataclass(frozen=True)
class AlgorithmStats:
    algorithm_name: str
    query_count: int
    candidate_count: int
    heap_pushes: int = 0
    heap_pops: int = 0
    initialization_time_seconds: float = 0.0
    query_data_structure_time_seconds: float = 0.0
    candidate_processing_time_seconds: float = 0.0
    selection_time_seconds: float = 0.0
    solution_update_time_seconds: float = 0.0
    total_time_seconds: float = 0.0


@dataclass(frozen=True)
class GreedyProfileResult:
    """Compact output of a greedy interval-growing algorithm."""

    costs: List[float]
    deltas: List[CandidateEvaluation]
    algorithm_name: str
    stats: Optional[AlgorithmStats] = None

    @property
    def n(self) -> int:
        return len(self.costs) - 1

    def cost_at(self, k: int) -> float:
        if not (0 <= k <= self.n):
            raise ValueError(f"k must lie in [0, {self.n}].")
        return float(self.costs[k])

    def step_at(self, k: int) -> Optional[CandidateEvaluation]:
        if not (0 <= k <= self.n):
            raise ValueError(f"k must lie in [0, {self.n}].")
        return None if k == 0 else self.deltas[k - 1]

    def solution_at(self, k: int) -> List[CompactInterval]:
        from utils import apply_delta_to_solution

        if not (0 <= k <= self.n):
            raise ValueError(f"k must lie in [0, {self.n}].")
        solution: List[CompactInterval] = []
        for i in range(k):
            solution = apply_delta_to_solution(solution, self.deltas[i])
        return solution
