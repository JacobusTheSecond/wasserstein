from __future__ import annotations

from typing import Any, Dict, Sequence

from algorithms import (
    greedy_interval_baseline,
    greedy_interval_priority_queue,
    greedy_interval_priority_queue_range_tree,
)
from models import GreedyProfileResult
from mp_types import Point


def profiles_match(reference: GreedyProfileResult, other: GreedyProfileResult, *, atol: float = 1e-5) -> bool:
    if reference.n != other.n:
        return False
    return all(abs(a - b) <= atol for a, b in zip(reference.costs, other.costs))


def assert_profiles_match(reference: GreedyProfileResult, other: GreedyProfileResult, *, atol: float = 1e-5) -> None:
    if reference.n != other.n:
        raise AssertionError(
            f"Profile lengths differ: reference has n={reference.n}, other has n={other.n}."
        )
    for k, (a, b) in enumerate(zip(reference.costs, other.costs)):
        if abs(a - b) > atol:
            raise AssertionError(f"Profiles disagree at k={k}: reference cost = {a}, other cost = {b}.")


def compare_all_algorithms(R: Sequence[Point], B: Sequence[Point], *, atol: float = 1e-5) -> Dict[str, GreedyProfileResult]:
    #algo1 = greedy_interval_baseline(R, B)
    algo2 = greedy_interval_priority_queue(R, B)
    algo3 = greedy_interval_priority_queue_range_tree(R, B)
    #assert_profiles_match(algo1, algo2, atol=atol)
    assert_profiles_match(algo2, algo3, atol=atol)
    return { #"naive": algo1,
        "noFFT": algo2,
        "FFT": algo3}


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
