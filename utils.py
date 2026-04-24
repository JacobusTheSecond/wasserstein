from __future__ import annotations

from typing import Sequence, Tuple

from models import CandidateEvaluation, CompactInterval
from mp_types import Point


def validate_sorted(points: Sequence[Point], name: str) -> None:
    if any(points[i] > points[i + 1] for i in range(len(points) - 1)):
        raise ValueError(f"{name} must be sorted in nondecreasing order.")


def validate_instance(R: Sequence[Point], B: Sequence[Point]) -> None:
    validate_sorted(R, "R")
    validate_sorted(B, "B")
    if len(R) != len(B):
        raise ValueError("The instance must contain equally many red and blue points.")


def interval_sort_key(interval: CompactInterval) -> Tuple[int, int]:
    return (interval.red_start, interval.blue_start)


def validate_disjoint_sorted_intervals(intervals: Sequence[CompactInterval]) -> None:
    for i in range(len(intervals) - 1):
        left = intervals[i]
        right = intervals[i + 1]
        if left.red_end >= right.red_start:
            raise ValueError("Red index ranges must be disjoint and sorted.")
        if left.blue_end >= right.blue_start:
            raise ValueError("Blue index ranges must be disjoint and sorted.")


def candidate_key(candidate: CandidateEvaluation) -> Tuple[float, float, float, str, str]:
    return (
        candidate.delta_cost,
        candidate.left_endpoint.position,
        candidate.right_endpoint.position,
        candidate.left_endpoint.color,
        candidate.right_endpoint.color,
    )


def current_total_cost(solution: Sequence[CompactInterval]) -> float:
    return float(sum(interval.cost for interval in solution))


def current_total_size(solution: Sequence[CompactInterval]) -> int:
    return int(sum(interval.size for interval in solution))


def apply_delta_to_solution(
    solution: Sequence[CompactInterval], delta: CandidateEvaluation
) -> list[CompactInterval]:
    removed_keys = {interval.key() for interval in delta.subsumed_intervals}
    updated = [interval for interval in solution if interval.key() not in removed_keys]
    updated.append(delta.merged_interval)
    updated.sort(key=interval_sort_key)
    validate_disjoint_sorted_intervals(updated)
    return updated


def undo_delta_from_solution(
    solution: Sequence[CompactInterval], delta: CandidateEvaluation
) -> list[CompactInterval]:
    merged_key = delta.merged_interval.key()
    updated = [interval for interval in solution if interval.key() != merged_key]
    updated.extend(delta.subsumed_intervals)
    updated.sort(key=interval_sort_key)
    validate_disjoint_sorted_intervals(updated)
    return updated
