from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Optional, Sequence

from candidates import (
    enumerate_candidates,
    evaluate_candidate_fast_dynamic,
    evaluate_candidate_slow,
)
from dynamic_interval_set import DynamicIntervalSet
from models import AlgorithmStats, CandidateEvaluation, GreedyProfileResult
from queries import BalancedIntervalSquaredCostDataStructure, build_colored_sequence
from mp_types import Point
from utils import apply_delta_to_solution, candidate_key, current_total_cost, current_total_size, validate_instance


def greedy_interval_baseline(R: Sequence[Point], B: Sequence[Point]) -> GreedyProfileResult:
    validate_instance(R, B)
    total_start = perf_counter()
    solution = []
    costs = [0.0]
    deltas = []
    query_count = 0
    candidate_count = 0
    candidate_processing_time = 0.0
    selection_time = 0.0
    solution_update_time = 0.0

    n = len(R)
    for k in range(1, n + 1):
        t0 = perf_counter()
        candidates = enumerate_candidates(R, B, solution)
        candidate_processing_time += perf_counter() - t0
        query_count += len(candidates)
        candidate_count += len(candidates)
        if not candidates:
            raise RuntimeError("No valid candidate exists, but the matching is not complete.")
        t0 = perf_counter()
        chosen = min(candidates, key=candidate_key)
        selection_time += perf_counter() - t0
        t0 = perf_counter()
        solution = apply_delta_to_solution(solution, chosen)
        solution_update_time += perf_counter() - t0
        if current_total_size(solution) != k:
            raise RuntimeError(f"Internal error at step {k}: expected {k} matched pairs, but found {current_total_size(solution)}.")
        costs.append(current_total_cost(solution))
        deltas.append(chosen)

    total_time = perf_counter() - total_start
    return GreedyProfileResult(
        costs=costs,
        deltas=deltas,
        algorithm_name="algorithm_1_full_rescan",
        stats=AlgorithmStats(
            algorithm_name="algorithm_1_full_rescan",
            query_count=query_count,
            candidate_count=candidate_count,
            candidate_processing_time_seconds=candidate_processing_time,
            selection_time_seconds=selection_time,
            solution_update_time_seconds=solution_update_time,
            total_time_seconds=total_time,
        ),
    )


def _candidate_is_valid(candidate: CandidateEvaluation, alive, prev_idx, next_idx) -> bool:
    left_id = candidate.left_endpoint.uid
    right_id = candidate.right_endpoint.uid
    return (
        0 <= left_id < len(alive)
        and 0 <= right_id < len(alive)
        and alive[left_id]
        and alive[right_id]
        and next_idx[left_id] == right_id
        and prev_idx[right_id] == left_id
    )


def _push_candidate_slow(heap, serial_counter, all_points, solution, left_id, right_id, R, B) -> bool:
    if left_id < 0 or right_id < 0 or left_id >= len(all_points) or right_id >= len(all_points):
        return False
    left_point = all_points[left_id]
    right_point = all_points[right_id]
    if left_point.color == right_point.color:
        return False
    candidate = evaluate_candidate_slow(R, B, solution, left_point, right_point)
    heappush(heap, (candidate_key(candidate), next(serial_counter), candidate))
    return True


def _push_candidate_fast_dynamic(heap, serial_counter, all_points, interval_set, left_id, right_id, ds) -> bool:
    if left_id < 0 or right_id < 0 or left_id >= len(all_points) or right_id >= len(all_points):
        return False
    left_point = all_points[left_id]
    right_point = all_points[right_id]
    if left_point.color == right_point.color:
        return False
    candidate = evaluate_candidate_fast_dynamic(ds, interval_set, left_point, right_point)
    heappush(heap, (candidate_key(candidate), next(serial_counter), candidate))
    return True


def greedy_interval_priority_queue(R: Sequence[Point], B: Sequence[Point]) -> GreedyProfileResult:
    validate_instance(R, B)
    total_start = perf_counter()
    n = len(R)
    all_points = build_colored_sequence(R, B)
    m = len(all_points)
    prev_idx = [-1] + [i for i in range(m - 1)]
    next_idx = [i + 1 for i in range(m - 1)] + [-1]
    alive = [True] * m
    solution = []
    costs = [0.0]
    deltas = []
    heap = []
    serial_counter = count()

    query_count = candidate_count = heap_pushes = heap_pops = 0
    candidate_processing_time = selection_time = solution_update_time = 0.0

    for left_id in range(m - 1):
        t0 = perf_counter()
        pushed = _push_candidate_slow(heap, serial_counter, all_points, solution, left_id, left_id + 1, R, B)
        candidate_processing_time += perf_counter() - t0
        if pushed:
            query_count += 1
            candidate_count += 1
            heap_pushes += 1

    for k in range(1, n + 1):
        t0 = perf_counter()
        chosen: Optional[CandidateEvaluation] = None
        while heap:
            _, _, candidate = heappop(heap)
            heap_pops += 1
            if _candidate_is_valid(candidate, alive, prev_idx, next_idx):
                chosen = candidate
                break
        selection_time += perf_counter() - t0
        if chosen is None:
            raise RuntimeError("Priority queue became empty before the matching was complete.")

        t0 = perf_counter()
        solution = apply_delta_to_solution(solution, chosen)
        left_id = chosen.left_endpoint.uid
        right_id = chosen.right_endpoint.uid
        left_neighbor = prev_idx[left_id]
        right_neighbor = next_idx[right_id]
        alive[left_id] = alive[right_id] = False
        if left_neighbor != -1:
            next_idx[left_neighbor] = right_neighbor
        if right_neighbor != -1:
            prev_idx[right_neighbor] = left_neighbor
        prev_idx[left_id] = next_idx[left_id] = -1
        prev_idx[right_id] = next_idx[right_id] = -1
        solution_update_time += perf_counter() - t0

        t0 = perf_counter()
        pushed = _push_candidate_slow(heap, serial_counter, all_points, solution, left_neighbor, right_neighbor, R, B)
        candidate_processing_time += perf_counter() - t0
        if pushed:
            query_count += 1
            candidate_count += 1
            heap_pushes += 1

        if current_total_size(solution) != k:
            raise RuntimeError(f"Internal error at step {k}: expected {k} matched pairs, but found {current_total_size(solution)}.")
        costs.append(current_total_cost(solution))
        deltas.append(chosen)

    total_time = perf_counter() - total_start
    return GreedyProfileResult(
        costs=costs,
        deltas=deltas,
        algorithm_name="algorithm_2_heap_slow_queries",
        stats=AlgorithmStats(
            algorithm_name="algorithm_2_heap_slow_queries",
            query_count=query_count,
            candidate_count=candidate_count,
            heap_pushes=heap_pushes,
            heap_pops=heap_pops,
            candidate_processing_time_seconds=candidate_processing_time,
            selection_time_seconds=selection_time,
            solution_update_time_seconds=solution_update_time,
            total_time_seconds=total_time,
        ),
    )


def greedy_interval_priority_queue_range_tree(R: Sequence[Point], B: Sequence[Point]) -> GreedyProfileResult:
    validate_instance(R, B)
    total_start = perf_counter()
    t0 = perf_counter()
    ds = BalancedIntervalSquaredCostDataStructure(R, B)
    interval_set = DynamicIntervalSet()
    ds_time = perf_counter() - t0

    n = len(R)
    all_points = ds.all_points
    m = len(all_points)
    prev_idx = [-1] + [i for i in range(m - 1)]
    next_idx = [i + 1 for i in range(m - 1)] + [-1]
    alive = [True] * m
    costs = [0.0]
    deltas = []
    heap = []
    serial_counter = count()

    query_count = candidate_count = heap_pushes = heap_pops = 0
    candidate_processing_time = selection_time = solution_update_time = 0.0

    for left_id in range(m - 1):
        t0 = perf_counter()
        pushed = _push_candidate_fast_dynamic(heap, serial_counter, all_points, interval_set, left_id, left_id + 1, ds)
        candidate_processing_time += perf_counter() - t0
        if pushed:
            query_count += 1
            candidate_count += 1
            heap_pushes += 1

    for k in range(1, n + 1):
        t0 = perf_counter()
        chosen: Optional[CandidateEvaluation] = None
        while heap:
            _, _, candidate = heappop(heap)
            heap_pops += 1
            if not _candidate_is_valid(candidate, alive, prev_idx, next_idx):
                continue
            refreshed = evaluate_candidate_fast_dynamic(ds, interval_set, candidate.left_endpoint, candidate.right_endpoint)
            if candidate_key(refreshed) != candidate_key(candidate):
                heappush(heap, (candidate_key(refreshed), next(serial_counter), refreshed))
                heap_pushes += 1
                query_count += 1
                candidate_count += 1
                continue
            chosen = refreshed
            break
        selection_time += perf_counter() - t0
        if chosen is None:
            raise RuntimeError("Priority queue became empty before the matching was complete.")

        t0 = perf_counter()
        interval_set.apply_candidate(chosen)
        left_id = chosen.left_endpoint.uid
        right_id = chosen.right_endpoint.uid
        left_neighbor = prev_idx[left_id]
        right_neighbor = next_idx[right_id]
        alive[left_id] = alive[right_id] = False
        if left_neighbor != -1:
            next_idx[left_neighbor] = right_neighbor
        if right_neighbor != -1:
            prev_idx[right_neighbor] = left_neighbor
        prev_idx[left_id] = next_idx[left_id] = -1
        prev_idx[right_id] = next_idx[right_id] = -1
        solution_update_time += perf_counter() - t0

        t0 = perf_counter()
        pushed = _push_candidate_fast_dynamic(heap, serial_counter, all_points, interval_set, left_neighbor, right_neighbor, ds)
        candidate_processing_time += perf_counter() - t0
        if pushed:
            query_count += 1
            candidate_count += 1
            heap_pushes += 1

        if interval_set.total_size != k:
            raise RuntimeError(f"Internal error at step {k}: expected {k} matched pairs, but found {interval_set.total_size}.")
        costs.append(interval_set.total_cost)
        deltas.append(chosen)

    total_time = perf_counter() - total_start
    return GreedyProfileResult(
        costs=costs,
        deltas=deltas,
        algorithm_name="algorithm_3_heap_range_tree_queries",
        stats=AlgorithmStats(
            algorithm_name="algorithm_3_heap_range_tree_queries",
            query_count=query_count,
            candidate_count=candidate_count,
            heap_pushes=heap_pushes,
            heap_pops=heap_pops,
            query_data_structure_time_seconds=ds_time,
            candidate_processing_time_seconds=candidate_processing_time,
            selection_time_seconds=selection_time,
            solution_update_time_seconds=solution_update_time,
            total_time_seconds=total_time,
        ),
    )
