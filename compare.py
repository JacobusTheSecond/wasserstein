from __future__ import annotations

from typing import Any, Dict, Sequence, List

from algorithms import (
    greedy_interval_baseline,
    greedy_interval_priority_queue,
    greedy_interval_priority_queue_range_tree,
)
from models import GreedyProfileResult
from mp_types import Point


def profiles_match(reference: GreedyProfileResult, other: GreedyProfileResult, *, atol: float = 1e-3) -> bool:
    if reference.n != other.n:
        return False
    return all(abs(a - b) <= atol for a, b in zip(reference.costs, other.costs))


def assert_profiles_match(reference: GreedyProfileResult, other: GreedyProfileResult, *, atol: float = 1e-3) -> None:
    if reference.n != other.n:
        raise AssertionError(
            f"Profile lengths differ: reference has n={reference.n}, other has n={other.n}."
        )
    for k, (a, b) in enumerate(zip(reference.costs, other.costs)):
        if abs(a - b) > atol:
            raise AssertionError(f"Profiles disagree at k={k}: reference cost = {a}, other cost = {b}.")


def compare_all_algorithms(R: Sequence[Point], B: Sequence[Point], names: List[str], atol: float = 1e-3) -> Dict[str, GreedyProfileResult]:
    result = {}
    for name in names:
        if name == "naive":
            result[name] = greedy_interval_baseline(R,B)
        elif name == "noFFT":
            result[name] = greedy_interval_priority_queue(R,B)
        elif name == "FFT":
            result[name] = greedy_interval_priority_queue_range_tree(R,B)
        assert_profiles_match(result[names[0]],result[name])

    return result


def stats_table(profiles: Dict[str, GreedyProfileResult]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for name, profile in profiles.items():
        stats = profile.stats
        if stats is None:
            continue
        rows.append(
            {
                "algorithm": name,
                "query_count": stats.query_count,
                "candidate_count": stats.candidate_count,
                "heap_pushes": stats.heap_pushes,
                "heap_pops": stats.heap_pops,
                "initialization_time_seconds": stats.initialization_time_seconds,
                "query_data_structure_time_seconds": stats.query_data_structure_time_seconds,
                "candidate_processing_time_seconds": stats.candidate_processing_time_seconds,
                "selection_time_seconds": stats.selection_time_seconds,
                "solution_update_time_seconds": stats.solution_update_time_seconds,
                "total_time_seconds": stats.total_time_seconds,
            }
        )
    return rows
