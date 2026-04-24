from __future__ import annotations

from typing import Sequence

from dynamic_interval_set import DynamicIntervalSet
from models import CandidateEvaluation, ColoredPoint, CompactInterval
from queries import BalancedIntervalSquaredCostDataStructure, build_colored_sequence, interval_matching_query
from mp_types import Point
from utils import current_total_cost, current_total_size, validate_disjoint_sorted_intervals, validate_instance


def collect_unmatched_points(points: Sequence[Point], color: str, solution: Sequence[CompactInterval]) -> list[ColoredPoint]:
    validate_disjoint_sorted_intervals(solution)
    unmatched: list[ColoredPoint] = []
    interval_index = 0
    for point_index, x in enumerate(points):
        while interval_index < len(solution):
            current = solution[interval_index]
            current_end = current.red_end if color == "R" else current.blue_end
            if current_end < point_index:
                interval_index += 1
            else:
                break
        covered = False
        if interval_index < len(solution):
            current = solution[interval_index]
            current_start = current.red_start if color == "R" else current.blue_start
            current_end = current.red_end if color == "R" else current.blue_end
            covered = current_start <= point_index <= current_end
        if not covered:
            unmatched.append(ColoredPoint(position=float(x), color=color))
    return unmatched


def unmatched_union(R: Sequence[Point], B: Sequence[Point], solution: Sequence[CompactInterval]) -> list[ColoredPoint]:
    validate_instance(R, B)
    unmatched_red = collect_unmatched_points(R, "R", solution)
    unmatched_blue = collect_unmatched_points(B, "B", solution)
    merged = unmatched_red + unmatched_blue
    merged.sort(key=lambda p: (p.position, p.color, p.uid))
    return merged


def find_subsumed_intervals(merged_interval: CompactInterval, solution: Sequence[CompactInterval]) -> list[CompactInterval]:
    subsumed: list[CompactInterval] = []
    for current in solution:
        red_disjoint = current.red_end < merged_interval.red_start or merged_interval.red_end < current.red_start
        blue_disjoint = current.blue_end < merged_interval.blue_start or merged_interval.blue_end < current.blue_start
        contained = (
            merged_interval.red_start <= current.red_start <= current.red_end <= merged_interval.red_end
            and merged_interval.blue_start <= current.blue_start <= current.blue_end <= merged_interval.blue_end
        )
        if red_disjoint and blue_disjoint:
            continue
        if contained:
            subsumed.append(current)
        else:
            raise ValueError(
                "Candidate interval partially overlaps an existing interval. "
                "The greedy invariant of disjoint intervals has been violated."
            )
    return subsumed


def finalize_candidate(
    solution: Sequence[CompactInterval],
    left_endpoint: ColoredPoint,
    right_endpoint: ColoredPoint,
    merged_interval: CompactInterval,
) -> CandidateEvaluation:
    subsumed = find_subsumed_intervals(merged_interval, solution)
    delta_cost = merged_interval.cost - current_total_cost(subsumed)
    previous_size = current_total_size(subsumed)
    if merged_interval.size != previous_size + 1:
        raise ValueError(
            "Unexpected candidate size: the merged interval should increase the "
            "number of matched pairs by exactly one."
        )
    return CandidateEvaluation(
        left_endpoint=left_endpoint,
        right_endpoint=right_endpoint,
        merged_interval=merged_interval,
        subsumed_intervals=list(subsumed),
        delta_cost=float(delta_cost),
    )


def finalize_candidate_dynamic(
    interval_set: DynamicIntervalSet,
    left_endpoint: ColoredPoint,
    right_endpoint: ColoredPoint,
    merged_interval: CompactInterval,
) -> CandidateEvaluation:
    subsumed, removed_cost, removed_size = interval_set.extract_subsumed_intervals(merged_interval)
    if merged_interval.size != removed_size + 1:
        raise ValueError(
            "Unexpected candidate size: the merged interval should increase the "
            "number of matched pairs by exactly one."
        )
    return CandidateEvaluation(
        left_endpoint=left_endpoint,
        right_endpoint=right_endpoint,
        merged_interval=merged_interval,
        subsumed_intervals=subsumed,
        delta_cost=float(merged_interval.cost - removed_cost),
    )


def evaluate_candidate_slow(
    R: Sequence[Point], B: Sequence[Point], solution: Sequence[CompactInterval], left_endpoint: ColoredPoint, right_endpoint: ColoredPoint
) -> CandidateEvaluation:
    if left_endpoint.position > right_endpoint.position:
        raise ValueError("Endpoints must be ordered from left to right.")
    if left_endpoint.color == right_endpoint.color:
        raise ValueError("A candidate must be formed by opposite-color endpoints.")
    merged_interval = interval_matching_query(R, B, (left_endpoint.position, right_endpoint.position))
    return finalize_candidate(solution, left_endpoint, right_endpoint, merged_interval)


def evaluate_candidate_fast(
    data_structure: BalancedIntervalSquaredCostDataStructure,
    solution: Sequence[CompactInterval],
    left_endpoint: ColoredPoint,
    right_endpoint: ColoredPoint,
) -> CandidateEvaluation:
    if left_endpoint.position > right_endpoint.position:
        raise ValueError("Endpoints must be ordered from left to right.")
    if left_endpoint.color == right_endpoint.color:
        raise ValueError("A candidate must be formed by opposite-color endpoints.")
    merged_interval = data_structure.match_interval_by_coordinates(left_endpoint.position, right_endpoint.position)
    return finalize_candidate(solution, left_endpoint, right_endpoint, merged_interval)


def evaluate_candidate_fast_dynamic(
    data_structure: BalancedIntervalSquaredCostDataStructure,
    interval_set: DynamicIntervalSet,
    left_endpoint: ColoredPoint,
    right_endpoint: ColoredPoint,
) -> CandidateEvaluation:
    if left_endpoint.position > right_endpoint.position:
        raise ValueError("Endpoints must be ordered from left to right.")
    if left_endpoint.color == right_endpoint.color:
        raise ValueError("A candidate must be formed by opposite-color endpoints.")
    merged_interval = data_structure.match_interval_by_coordinates(left_endpoint.position, right_endpoint.position)
    return finalize_candidate_dynamic(interval_set, left_endpoint, right_endpoint, merged_interval)


def enumerate_candidates(R: Sequence[Point], B: Sequence[Point], solution: Sequence[CompactInterval]) -> list[CandidateEvaluation]:
    candidates: list[CandidateEvaluation] = []
    unmatched = unmatched_union(R, B, solution)
    for i in range(len(unmatched) - 1):
        left_point = unmatched[i]
        right_point = unmatched[i + 1]
        if left_point.color != right_point.color:
            candidates.append(evaluate_candidate_slow(R, B, solution, left_point, right_point))
    return candidates
