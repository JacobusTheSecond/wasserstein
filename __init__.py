from .algorithms import (
    greedy_interval_baseline,
    greedy_interval_priority_queue,
    greedy_interval_priority_queue_range_tree,
)
from .compare import assert_profiles_match, compare_all_algorithms, profiles_match, stats_table
from .generators import generate_biased_instance, generate_instance
from .models import AlgorithmStats, CandidateEvaluation, ColoredPoint, CompactInterval, GreedyProfileResult
from .queries import (
    BalancedIntervalSquaredCostDataStructure,
    build_colored_sequence,
    interval_is_balanced,
    interval_matching_query,
    points_in_interval,
)
from .visualization import (
    GreedyProfileStepper,
    describe_solution_step,
    make_stepper,
    plot_instance,
    plot_solution,
    plot_solution_at_k,
)

__all__ = [
    "AlgorithmStats",
    "BalancedIntervalSquaredCostDataStructure",
    "CandidateEvaluation",
    "ColoredPoint",
    "CompactInterval",
    "GreedyProfileResult",
    "GreedyProfileStepper",
    "assert_profiles_match",
    "build_colored_sequence",
    "compare_all_algorithms",
    "describe_solution_step",
    "generate_biased_instance",
    "generate_instance",
    "greedy_interval_baseline",
    "greedy_interval_priority_queue",
    "greedy_interval_priority_queue_range_tree",
    "interval_is_balanced",
    "interval_matching_query",
    "make_stepper",
    "plot_instance",
    "plot_solution",
    "plot_solution_at_k",
    "points_in_interval",
    "profiles_match",
    "stats_table",
]
