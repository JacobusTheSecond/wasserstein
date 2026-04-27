from __future__ import annotations

from pathlib import Path
from pprint import pprint
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

from IO import (
    equalize_cardinality,
    load_3d_points,
    principal_axis_union,
    project_onto_axis,
)
from generators import generate_biased_instance, generate_instance
from compare import compare_all_algorithms, stats_table
from utils import recover_solution_at_k
from visualization import describe_solution_step, make_stepper, plot_solution_at_k

# Change this import only if your slimFFT entry function has a different name.
from algorithms import greedy_interval_priority_queue_summary as run_slim_fft

MODE = "benchmark"   # "real_data" or "benchmark"

#real data configs
DATA_DIR = Path(__file__).parent / "data"
RED_FILE = "dragonStandRight_0.ply"
BLUE_FILE = "dragonStandRight_24.ply"

TARGET_N = 100_000
REAL_DATA_SEED = 7

TRANSLATE_RED = True
TRANSLATE_BLUE = False
RED_TRANSLATION = np.array([0.03, 0.00, 0.00], dtype=np.float64)
BLUE_TRANSLATION = np.array([0.00, 0.00, 0.00], dtype=np.float64)

SHOW_PROJECTION_PLOT = True
SHOW_SOLUTION_PLOT = True
OPEN_STEPPER_REAL = False
K_TO_PLOT_REAL = None   # None means use k = n

#benchmark configs
USE_BIASED_INSTANCE = True
BENCHMARK_SEED = 0x1337BAD1

LOW = 0.0
HIGH = 100.0
LOW_COLOR = "R"
LOW_BIAS_STRENGTH = 10.0
HIGH_BIAS_STRENGTH = 10.0

INITIAL_ALGORITHMS = ["naive", "noFFT", "FFT", "slimFFT","slimFFT_c"]

N_START = 10
N_MULTIPLIER = 1.5
TIME_LIMIT_SECONDS = 5.0

REPEAT_UNTIL_N = 2000
REPEAT_COUNT_SMALL = 7
REPEAT_COUNT_LARGE = 3

def make_synthetic_instance(n: int, seed: int):
    if USE_BIASED_INSTANCE:
        return generate_biased_instance(
            n=n,
            low=LOW,
            high=HIGH,
            seed=seed,
            low_color=LOW_COLOR,
            low_bias_strength=LOW_BIAS_STRENGTH,
            high_bias_strength=HIGH_BIAS_STRENGTH,
        )
    return generate_instance(
        n=n,
        low=LOW,
        high=HIGH,
        seed=seed,
    )


def run_real_data() -> None:
    red_path = DATA_DIR / RED_FILE
    blue_path = DATA_DIR / BLUE_FILE

    if not red_path.exists():
        raise FileNotFoundError(f"Could not find {red_path}")
    if not blue_path.exists():
        raise FileNotFoundError(f"Could not find {blue_path}")

    red_points_3d = load_3d_points(red_path, target_n=TARGET_N, seed=REAL_DATA_SEED)
    blue_points_3d = load_3d_points(blue_path, target_n=TARGET_N, seed=REAL_DATA_SEED + 1)

    red_points_3d, blue_points_3d = equalize_cardinality(
        red_points_3d,
        blue_points_3d,
        target_n=TARGET_N,
        seed=REAL_DATA_SEED,
    )

    if TRANSLATE_RED:
        red_points_3d = np.ascontiguousarray(red_points_3d + RED_TRANSLATION, dtype=np.float64)

    if TRANSLATE_BLUE:
        blue_points_3d = np.ascontiguousarray(blue_points_3d + BLUE_TRANSLATION, dtype=np.float64)

    mean, axis = principal_axis_union(red_points_3d, blue_points_3d)

    R = np.sort(project_onto_axis(red_points_3d, mean, axis)).tolist()
    B = np.sort(project_onto_axis(blue_points_3d, mean, axis)).tolist()

    print(f"Loaded {len(R)} red points and {len(B)} blue points.")
    print("PCA mean =", mean)
    print("PCA axis =", axis)

    # IMPORTANT: no compare call here, only slimFFT.
    profile = run_slim_fft(R, B)

    if profile.stats is not None:
        print("\nslimFFT stats:")
        pprint(profile.stats)

    k = profile.n if K_TO_PLOT_REAL is None else max(0, min(K_TO_PLOT_REAL, profile.n))

    print("\nSelected solution summary:")
    print(describe_solution_step(profile, k))

    if SHOW_PROJECTION_PLOT:
        plt.figure(figsize=(9, 3))
        plt.scatter(R, np.full(len(R), 0.05), s=8, c="red", label="R")
        plt.scatter(B, np.full(len(B), -0.05), s=8, c="blue", label="B")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.yticks([])
        plt.xlabel("Projection onto first PCA axis")
        plt.title("Projected 1D point sets")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if SHOW_SOLUTION_PLOT:
        fig, ax = plt.subplots(figsize=(11, 3.0))
        plot_solution_at_k(R, B, profile, k, ax=ax)
        plt.show()

    if OPEN_STEPPER_REAL:
        stepper = make_stepper(R, B, profile)
        stepper.interactive_plot()


def run_benchmark() -> None:
    active_algorithms = list(INITIAL_ALGORITHMS)

    n_values_by_algo = {name: [] for name in INITIAL_ALGORITHMS}
    times_by_algo = {name: [] for name in INITIAL_ALGORITHMS}

    n = N_START

    print("Running multiplicative benchmark...")
    while active_algorithms:
        print(f"\nN = {n}")
        print(f"Active algorithms: {active_algorithms}")

        repeat_count = REPEAT_COUNT_SMALL if n <= REPEAT_UNTIL_N else REPEAT_COUNT_LARGE
        print(f"Repetitions: {repeat_count}")

        per_algo_times = {name: [] for name in active_algorithms}
        last_profiles = None

        for rep in range(repeat_count):
            rep_seed = BENCHMARK_SEED + rep
            R, B = make_synthetic_instance(n, rep_seed)

            profiles = compare_all_algorithms(R, B, active_algorithms)
            last_profiles = profiles

            for name in active_algorithms:
                profile = profiles[name]
                if profile.stats is None:
                    continue
                per_algo_times[name].append(profile.stats.total_time_seconds)

        print("\nStats from last repetition:")
        if last_profiles is not None:
            for row in stats_table(last_profiles):
                pprint(row)

        print("\nAggregated timings:")
        next_active = []
        for name in active_algorithms:
            if not per_algo_times[name]:
                continue

            t_med = median(per_algo_times[name])
            t_min = min(per_algo_times[name])
            t_max = max(per_algo_times[name])

            print(
                f"{name}: median={t_med:.6f}s, "
                f"min={t_min:.6f}s, max={t_max:.6f}s"
            )

            n_values_by_algo[name].append(n)
            times_by_algo[name].append(t_med)

            if t_med <= TIME_LIMIT_SECONDS:
                next_active.append(name)
            else:
                print(
                    f"Dropping {name}: median time {t_med:.3f} seconds "
                    f"(> {TIME_LIMIT_SECONDS})"
                )

        active_algorithms = next_active

        next_n = max(n + 1, int(round(n * N_MULTIPLIER)))
        if next_n == n:
            next_n = n + 1
        n = next_n

    plt.figure(figsize=(8, 5))
    for name in INITIAL_ALGORITHMS:
        if n_values_by_algo[name]:
            plt.loglog(
                n_values_by_algo[name],
                times_by_algo[name],
                marker="o",
                label=name,
            )

    plt.xlabel("N")
    plt.ylabel("running time (seconds)")
    plt.title("Running time comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if MODE == "real_data":
        run_real_data()
    elif MODE == "benchmark":
        run_benchmark()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")