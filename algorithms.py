from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Optional, Sequence, Tuple, List

from candidates import (
    enumerate_candidates,
    evaluate_candidate_fast_dynamic,
    evaluate_candidate_slow,
)
from dynamic_interval_set import DynamicIntervalSet, DynamicIntervalSetSummary
from models import AlgorithmStats, CandidateEvaluation, GreedyProfileResult, LightCandidateEvaluation, \
    LightGreedyProfileResult
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

from heapq import heappop, heappush

# ---------------------------------------------------------------------------
# Assumed existing types / classes from your code base:
#
# - Point = float
# - CompactInterval
# - ColoredPoint
# - AlgorithmStats
# - BalancedIntervalSquaredCostDataStructure
# - validate_instance(R, B)
#
# The code below only depends on those.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cheap dynamic interval set with subtree aggregates
# Keyed by interval.red_start
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Candidate helpers for algorithm 4
# ---------------------------------------------------------------------------


def _light_candidate_key(
    candidate: LightCandidateEvaluation,
) -> Tuple[float, float, float, str, str]:
    return (
        candidate.delta_cost,
        candidate.left_endpoint.position,
        candidate.right_endpoint.position,
        candidate.left_endpoint.color,
        candidate.right_endpoint.color,
    )


def _same_light_priority(
    a: LightCandidateEvaluation,
    b: LightCandidateEvaluation,
    *,
    atol: float = 1e-12,
) -> bool:
    return (
        abs(a.delta_cost - b.delta_cost) <= atol
        and a.left_endpoint.uid == b.left_endpoint.uid
        and a.right_endpoint.uid == b.right_endpoint.uid
        and a.swallowed_pairs == b.swallowed_pairs
    )


def _candidate_is_valid(
    candidate: LightCandidateEvaluation,
    alive: Sequence[bool],
    prev_idx: Sequence[int],
    next_idx: Sequence[int],
) -> bool:
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


def evaluate_candidate_algo4(
    query_ds: "BalancedIntervalSquaredCostDataStructure",
    interval_set: DynamicIntervalSetSummary,
    left_endpoint: "ColoredPoint",
    right_endpoint: "ColoredPoint",
) -> LightCandidateEvaluation:
    """
    Build a candidate using:
      - O(log n) balanced-interval query for merged_interval,
      - O(log m) solution-summary query for swallowed aggregates.

    No swallowed interval list is materialized.
    """
    if left_endpoint.position > right_endpoint.position:
        raise ValueError("Endpoints must be ordered from left to right.")
    if left_endpoint.color == right_endpoint.color:
        raise ValueError("Candidate endpoints must have opposite colors.")

    # Because uid is the merged-order index in the same sorted sequence used by the query DS,
    # we can query by indices directly.
    merged_interval = query_ds.match_interval_by_indices(
        left_endpoint.uid,
        right_endpoint.uid,
    )

    swallowed_cost, swallowed_pairs = interval_set.range_summary(
        merged_interval.red_start,
        merged_interval.red_end,
    )

    if merged_interval.size != swallowed_pairs + 1:
        raise ValueError(
            "Unexpected candidate size: merged interval should add exactly one new pair."
        )

    return LightCandidateEvaluation(
        left_endpoint=left_endpoint,
        right_endpoint=right_endpoint,
        merged_interval=merged_interval,
        delta_cost=float(merged_interval.cost - swallowed_cost),
        swallowed_cost=float(swallowed_cost),
        swallowed_pairs=int(swallowed_pairs),
    )


def refresh_candidate_algo4(
    candidate: LightCandidateEvaluation,
    interval_set: DynamicIntervalSetSummary,
) -> LightCandidateEvaluation:
    """
    Recompute the swallowed aggregate against the current interval set.
    The merged interval itself does not change.
    """
    merged_interval = candidate.merged_interval

    swallowed_cost, swallowed_pairs = interval_set.range_summary(
        merged_interval.red_start,
        merged_interval.red_end,
    )

    if merged_interval.size != swallowed_pairs + 1:
        raise ValueError(
            "Unexpected refreshed candidate size: merged interval should add exactly one new pair."
        )

    return LightCandidateEvaluation(
        left_endpoint=candidate.left_endpoint,
        right_endpoint=candidate.right_endpoint,
        merged_interval=merged_interval,
        delta_cost=float(merged_interval.cost - swallowed_cost),
        swallowed_cost=float(swallowed_cost),
        swallowed_pairs=int(swallowed_pairs),
    )


def _push_candidate_algo4(
    heap: List[Tuple[Tuple[float, float, float, str, str], int, LightCandidateEvaluation]],
    serial_counter,
    all_points: Sequence["ColoredPoint"],
    interval_set: DynamicIntervalSetSummary,
    left_id: int,
    right_id: int,
    query_ds: "BalancedIntervalSquaredCostDataStructure",
) -> bool:
    if left_id < 0 or right_id < 0:
        return False
    if left_id >= len(all_points) or right_id >= len(all_points):
        return False

    left_point = all_points[left_id]
    right_point = all_points[right_id]
    if left_point.color == right_point.color:
        return False

    candidate = evaluate_candidate_algo4(query_ds, interval_set, left_point, right_point)
    heappush(heap, (_light_candidate_key(candidate), next(serial_counter), candidate))
    return True

# ---------------------------------------------------------------------------
# Entire fourth algorithm
# ---------------------------------------------------------------------------


def greedy_interval_priority_queue_summary(
    R: Sequence[float],
    B: Sequence[float],
) -> LightGreedyProfileResult:
    """
    Algorithm 4:
      - heap over neighboring unmatched opposite-color points,
      - balanced-interval range-tree query for merged interval,
      - logarithmic candidate delta via DynamicIntervalSetSummary,
      - cheap history (accepted deltas only; no swallowed interval lists).

    This fixes the hidden linear-time blowup from scanning the whole current
    solution during candidate finalization.

    Note:
      The stale-candidate heap refresh issue is still handled lazily here.
      This code fixes the *delta-evaluation* bottleneck, not the separate
      amortization question for refreshes.
    """
    validate_instance(R, B)

    total_start = perf_counter()

    # Query data structure
    init_start = perf_counter()
    query_ds = BalancedIntervalSquaredCostDataStructure(R, B)
    all_points = query_ds.all_points
    m_points = len(all_points)

    prev_idx = [-1] + [i for i in range(m_points - 1)]
    next_idx = [i + 1 for i in range(m_points - 1)] + [-1]
    alive = [True] * m_points

    interval_set = DynamicIntervalSetSummary()
    costs: List[float] = [0.0]
    deltas: List[LightCandidateEvaluation] = []

    heap: List[Tuple[Tuple[float, float, float, str, str], int, LightCandidateEvaluation]] = []
    serial_counter = count()

    init_time = perf_counter() - init_start

    # Stats
    query_count = 0
    candidate_count = 0
    heap_pushes = 0
    heap_pops = 0
    candidate_processing_time = 0.0
    selection_time = 0.0
    solution_update_time = 0.0

    # Initial neighboring pairs
    for left_id in range(m_points - 1):
        right_id = left_id + 1
        t0 = perf_counter()
        pushed = _push_candidate_algo4(
            heap,
            serial_counter,
            all_points,
            interval_set,
            left_id,
            right_id,
            query_ds,
        )
        candidate_processing_time += perf_counter() - t0

        if pushed:
            query_count += 1
            candidate_count += 1
            heap_pushes += 1

    n = len(R)
    for k in range(1, n + 1):
        # Pop best valid candidate, refreshing lazily if needed.
        t0 = perf_counter()
        chosen: Optional[LightCandidateEvaluation] = None

        while heap:
            _key, _serial, candidate = heappop(heap)
            heap_pops += 1

            if not _candidate_is_valid(candidate, alive, prev_idx, next_idx):
                continue

            refreshed = refresh_candidate_algo4(candidate, interval_set)

            if _same_light_priority(candidate, refreshed):
                chosen = refreshed
                break

            heappush(heap, (_light_candidate_key(refreshed), next(serial_counter), refreshed))
            heap_pushes += 1

        selection_time += perf_counter() - t0

        if chosen is None:
            raise RuntimeError("Priority queue became empty before the matching was complete.")

        # Apply chosen interval to the current solution summary structure.
        t0 = perf_counter()
        swallowed_cost, swallowed_pairs = interval_set.apply_interval(chosen.merged_interval)

        # Defensive consistency check against the stored cheap history.
        if (
            abs(swallowed_cost - chosen.swallowed_cost) > 1e-9
            or swallowed_pairs != chosen.swallowed_pairs
        ):
            raise RuntimeError(
                "Chosen candidate summary changed between refresh and application."
            )

        left_id = chosen.left_endpoint.uid
        right_id = chosen.right_endpoint.uid
        left_neighbor = prev_idx[left_id]
        right_neighbor = next_idx[right_id]

        alive[left_id] = False
        alive[right_id] = False

        if left_neighbor != -1:
            next_idx[left_neighbor] = right_neighbor
        if right_neighbor != -1:
            prev_idx[right_neighbor] = left_neighbor

        prev_idx[left_id] = -1
        next_idx[left_id] = -1
        prev_idx[right_id] = -1
        next_idx[right_id] = -1

        solution_update_time += perf_counter() - t0

        # Only one new neighboring pair can appear across the gap.
        t0 = perf_counter()
        pushed = _push_candidate_algo4(
            heap,
            serial_counter,
            all_points,
            interval_set,
            left_neighbor,
            right_neighbor,
            query_ds,
        )
        candidate_processing_time += perf_counter() - t0

        if pushed:
            query_count += 1
            candidate_count += 1
            heap_pushes += 1

        if interval_set.total_size != k:
            raise RuntimeError(
                f"Internal error at step {k}: expected {k} matched pairs, "
                f"but found {interval_set.total_size}."
            )

        costs.append(float(interval_set.total_cost))
        deltas.append(chosen)

    total_time = perf_counter() - total_start

    stats = AlgorithmStats(
        algorithm_name="algorithm_4_heap_range_tree_summary",
        query_count=query_count,
        candidate_count=candidate_count,
        heap_pushes=heap_pushes,
        heap_pops=heap_pops,
        initialization_time_seconds=float(init_time),
        query_data_structure_time_seconds=float(init_time),  # if you split DS build separately, change this
        candidate_processing_time_seconds=float(candidate_processing_time),
        selection_time_seconds=float(selection_time),
        solution_update_time_seconds=float(solution_update_time),
        total_time_seconds=float(total_time),
    )

    return LightGreedyProfileResult(
        costs=costs,
        deltas=deltas,
        algorithm_name="algorithm_4_heap_range_tree_summary",
        stats=stats,
    )
