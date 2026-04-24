from __future__ import annotations

from pprint import pprint

import matplotlib.pyplot as plt

from generators import generate_biased_instance, generate_instance
from compare import compare_all_algorithms, stats_table
from visualization import (
    describe_solution_step,
    make_stepper,
    plot_solution_at_k,
)


# ---------------------------------------------------------------------------
# Edit these values directly.
# ---------------------------------------------------------------------------
USE_BIASED_INSTANCE = True
N = 500
LOW = 0.0
HIGH = 100.0
SEED = 7

# Only used when USE_BIASED_INSTANCE = True
LOW_COLOR = "R"          # "R" or "B"
LOW_BIAS_STRENGTH = 10.0
HIGH_BIAS_STRENGTH = 10.0

SHOW_PLOTS = True
OPEN_STEPPER = True
K_TO_PLOT = 9 * N // 10          # will be clipped to [0, n]

# Benchmark mode
RUN_BENCHMARK = True
N_START = 10
N_MULTIPLIER = 1.5
TIME_LIMIT_SECONDS = 30.0
INITIAL_ALGORITHMS = ["naive", "noFFT", "FFT"]


def make_instance(n: int):
    if USE_BIASED_INSTANCE:
        return generate_biased_instance(
            n=n,
            low=LOW,
            high=HIGH,
            seed=SEED,
            low_color=LOW_COLOR,
            low_bias_strength=LOW_BIAS_STRENGTH,
            high_bias_strength=HIGH_BIAS_STRENGTH,
        )
    return generate_instance(
        n=n,
        low=LOW,
        high=HIGH,
        seed=SEED,
    )


if __name__ == "__main__":
    if RUN_BENCHMARK:
        active_algorithms = list(INITIAL_ALGORITHMS)

        n_values_by_algo = {name: [] for name in INITIAL_ALGORITHMS}
        times_by_algo = {name: [] for name in INITIAL_ALGORITHMS}

        n = N_START

        print("Running multiplicative benchmark...")
        while active_algorithms:
            print(f"\nN = {n}")
            print(f"Active algorithms: {active_algorithms}")

            R, B = make_instance(n)
            profiles = compare_all_algorithms(R, B, active_algorithms)

            print("\nStats:")
            for row in stats_table(profiles):
                pprint(row)

            next_active = []
            for name in active_algorithms:
                profile = profiles[name]
                if profile.stats is None:
                    continue

                t = profile.stats.total_time_seconds
                n_values_by_algo[name].append(n)
                times_by_algo[name].append(t)

                if t <= TIME_LIMIT_SECONDS:
                    next_active.append(name)
                else:
                    print(f"Dropping {name}: took {t:.3f} seconds (> {TIME_LIMIT_SECONDS})")

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

    else:
        R, B = make_instance(N)

        print("R =")
        pprint(R)
        print("\nB =")
        pprint(B)

        profiles = compare_all_algorithms(R, B, INITIAL_ALGORITHMS)

        print("\nComparison stats:")
        for row in stats_table(profiles):
            pprint(row)

        algo3 = profiles["FFT"]
        k = max(0, min(K_TO_PLOT, algo3.n))

        print("\nSelected solution summary:")
        print(describe_solution_step(algo3, k))

        if SHOW_PLOTS:
            fig, ax = plt.subplots(figsize=(11, 3.0))
            plot_solution_at_k(R, B, algo3, k, ax=ax)
            plt.show()

        if OPEN_STEPPER:
            stepper = make_stepper(R, B, algo3)
            stepper.interactive_plot()