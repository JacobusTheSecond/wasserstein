from __future__ import annotations

from pprint import pprint

import matplotlib.pyplot as plt

from generators import generate_biased_instance, generate_instance
from compare import compare_all_algorithms, stats_table
from visualization import (
    describe_solution_step,
    make_stepper,
    plot_instance,
    plot_solution_at_k,
)


# ---------------------------------------------------------------------------
# Edit these values directly.
# ---------------------------------------------------------------------------
USE_BIASED_INSTANCE = True
N = 5000
LOW = 0.0
HIGH = 100.0
SEED = 7

# Only used when USE_BIASED_INSTANCE = True
LOW_COLOR = "R"          # "R" or "B"
LOW_BIAS_STRENGTH = 10.0
HIGH_BIAS_STRENGTH = 10.0

SHOW_PLOTS = True
OPEN_STEPPER = True
K_TO_PLOT = 9*N//10            # will be clipped to [0, n]


# ---------------------------------------------------------------------------
# Main demo/comparison script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if USE_BIASED_INSTANCE:
        R, B = generate_biased_instance(
            n=N,
            low=LOW,
            high=HIGH,
            seed=SEED,
            low_color=LOW_COLOR,
            low_bias_strength=LOW_BIAS_STRENGTH,
            high_bias_strength=HIGH_BIAS_STRENGTH,
        )
    else:
        R, B = generate_instance(
            n=N,
            low=LOW,
            high=HIGH,
            seed=SEED,
        )

    print("R =")
    pprint(R)
    print("\nB =")
    pprint(B)

    profiles = compare_all_algorithms(R, B)

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