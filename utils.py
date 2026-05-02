from __future__ import annotations

from typing import Sequence, Tuple
from typing import List, Optional, Sequence, Tuple

from models import CandidateEvaluation, CompactInterval, GreedyProfileResult, LightGreedyProfileResult, \
    LightCandidateEvaluation
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

def recover_all_solutions(
    deltas: Sequence[LightCandidateEvaluation],
) -> List[List["CompactInterval"]]:
    """
    Expensive on purpose. Only call this if you truly need every intermediate
    solution materialized.
    """
    interval_set = DynamicIntervalSetSummary()
    out: List[List["CompactInterval"]] = [interval_set.to_sorted_intervals()]
    for i, delta in enumerate(deltas, start=1):
        swallowed_cost, swallowed_pairs = interval_set.apply_interval(delta.merged_interval)
        if swallowed_pairs != delta.swallowed_pairs:
            raise ValueError(
                f"Replay mismatch at step {i}: swallowed_pairs changed "
                f"({swallowed_pairs} vs stored {delta.swallowed_pairs})."
            )
        out.append(interval_set.to_sorted_intervals())
    return out

def _find_subsumed_intervals_by_scan(
    current_solution: Sequence[CompactInterval],
    merged_interval: CompactInterval,
) -> List[CompactInterval]:
    """
    Recover exactly which current intervals are swallowed by merged_interval.

    This is only for offline conversion / debugging, so a scan is fine.
    """
    subsumed: List[CompactInterval] = []

    for cur in current_solution:
        red_disjoint = cur.red_end < merged_interval.red_start or merged_interval.red_end < cur.red_start
        blue_disjoint = cur.blue_end < merged_interval.blue_start or merged_interval.blue_end < cur.blue_start

        contained = (
            merged_interval.red_start <= cur.red_start <= cur.red_end <= merged_interval.red_end
            and merged_interval.blue_start <= cur.blue_start <= cur.blue_end <= merged_interval.blue_end
        )

        if red_disjoint and blue_disjoint:
            continue
        if contained:
            subsumed.append(cur)
        else:
            raise ValueError(
                "While converting light profile to full profile, found a partial overlap. "
                "The light profile appears inconsistent."
            )

    return subsumed


def _apply_full_delta_to_solution(
    current_solution: Sequence[CompactInterval],
    merged_interval: CompactInterval,
) -> List[CompactInterval]:
    """
    Apply one accepted merged interval to the current reconstructed solution.
    """
    updated = [
        cur
        for cur in current_solution
        if not (
            merged_interval.red_start <= cur.red_start
            and cur.red_end <= merged_interval.red_end
            and merged_interval.blue_start <= cur.blue_start
            and cur.blue_end <= merged_interval.blue_end
        )
    ]
    updated.append(merged_interval)
    updated.sort(key=lambda x: (x.red_start, x.blue_start))
    return updated

def light_to_full_greedy_profile(
    light_profile: LightGreedyProfileResult,
) -> GreedyProfileResult:
    interval_set = DynamicIntervalSetSummary()
    full_deltas: List[CandidateEvaluation] = []

    for step_index, light_delta in enumerate(light_profile.deltas, start=1):
        subsumed, swallowed_cost, swallowed_pairs = interval_set.apply_interval_with_extract(
            light_delta.merged_interval
        )

        if abs(swallowed_cost - light_delta.swallowed_cost) > 1e-5:
            raise ValueError(
                f"Step {step_index}: swallowed cost mismatch while reconstructing full profile "
                f"({swallowed_cost} vs stored {light_delta.swallowed_cost})."
            )
        if swallowed_pairs != light_delta.swallowed_pairs:
            raise ValueError(
                f"Step {step_index}: swallowed pair-count mismatch while reconstructing full profile "
                f"({swallowed_pairs} vs stored {light_delta.swallowed_pairs})."
            )

        full_delta = CandidateEvaluation(
            left_endpoint=light_delta.left_endpoint,
            right_endpoint=light_delta.right_endpoint,
            merged_interval=light_delta.merged_interval,
            subsumed_intervals=subsumed,
            delta_cost=light_delta.delta_cost,
        )
        full_deltas.append(full_delta)

    return GreedyProfileResult(
        costs=list(light_profile.costs),
        deltas=full_deltas,
        algorithm_name=light_profile.algorithm_name + "_expanded",
        stats=light_profile.stats,
    )


from typing import List, Sequence

from dynamic_interval_set import DynamicIntervalSetSummary
from models import CompactInterval, LightCandidateEvaluation


def recover_solution_at_k(
    deltas: Sequence[LightCandidateEvaluation],
    k: int,
) -> List[CompactInterval]:
    """
    Reconstruct the solution after k accepted light deltas by replaying them.
    """
    if not (0 <= k <= len(deltas)):
        raise ValueError(f"k must lie in [0, {len(deltas)}].")

    interval_set = DynamicIntervalSetSummary()
    for i in range(k):
        delta = deltas[i]
        swallowed_cost, swallowed_pairs = interval_set.apply_interval(delta.merged_interval)

        # Optional consistency checks
        if abs(swallowed_cost - delta.swallowed_cost) > 1e-9:
            raise ValueError(
                f"Replay mismatch at step {i + 1}: swallowed cost changed "
                f"({swallowed_cost} vs stored {delta.swallowed_cost})."
            )
        if swallowed_pairs != delta.swallowed_pairs:
            raise ValueError(
                f"Replay mismatch at step {i + 1}: swallowed pair-count changed "
                f"({swallowed_pairs} vs stored {delta.swallowed_pairs})."
            )

    return interval_set.to_sorted_intervals()

