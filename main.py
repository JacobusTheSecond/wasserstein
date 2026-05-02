from __future__ import annotations

from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

from compare import compare_all_algorithms
from competition import partial_ot_1d

OUT_DIR = Path("runtime_grid_plots")
OUT_DIR.mkdir(exist_ok=True)

ALGORITHMS = [
    "naive",
    # "FFT_py",
    # "slimFFT_py",
    "FFT_c",
]
BENCHMARK_COMPETITOR = True
COMPETITOR_NAME = "Theirs"

N_START = 50
N_MULTIPLIER = 1.5
TIME_LIMIT_SECONDS = 60.0

REPEAT_UNTIL_N = 2000
REPEAT_COUNT_SMALL = 5
REPEAT_COUNT_LARGE = 3

BASE_MEAN = 0.0
BASE_STD = 1.0
MASTER_SEED = 1337

SAVE_INDIVIDUAL_PDFS = True
SAVE_COMBINED_FIGURE = True

# Visual style
LINEWIDTH = 1.7
MARKERSIZE = 2.5
MARKEREDGEWIDTH = 0.7

STYLE = {
    "naive": dict(color="#222222", marker="o"),   # black / dark gray
    "FFT": dict(color="#009E73", marker="s"),
    "slimFFT": dict(color="#CC79A7", marker="^"),
    "slimFFT_c": dict(color="#D55E00", marker="D"),
    "Ours": dict(color="#D55E00", marker="D"),    # vermillion
    "competitor": dict(color="#009E73", marker="x"),
    "Theirs": dict(color="#009E73", marker="x"),  # bluish green
}

def sample_normal(n: int, mean: float, std: float, seed: int) -> tuple[list[float], list[float]]:
    rng_r = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 10_000)
    R = np.sort(rng_r.normal(mean, std, size=n)).astype(np.float64)
    B = np.sort(rng_b.normal(mean, std, size=n)).astype(np.float64)
    return R.tolist(), B.tolist()


def make_same_mean_different_std(blue_std: float) -> Callable[[int, int], tuple[list[float], list[float]]]:
    def maker(n: int, seed: int):
        rng_r = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 10_000)
        R = np.sort(rng_r.normal(BASE_MEAN, BASE_STD, size=n)).astype(np.float64)
        B = np.sort(rng_b.normal(BASE_MEAN, blue_std, size=n)).astype(np.float64)
        return R.tolist(), B.tolist()

    return maker


def make_same_std_different_mean(blue_mean: float) -> Callable[[int, int], tuple[list[float], list[float]]]:
    def maker(n: int, seed: int):
        rng_r = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 10_000)
        R = np.sort(rng_r.normal(BASE_MEAN, BASE_STD, size=n)).astype(np.float64)
        B = np.sort(rng_b.normal(blue_mean, BASE_STD, size=n)).astype(np.float64)
        return R.tolist(), B.tolist()

    return maker


def make_unimodal_vs_bimodal(mean_left: float, mean_right: float) -> Callable[[int, int], tuple[list[float], list[float]]]:
    def maker(n: int, seed: int):
        rng_r = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 10_000)

        R = np.sort(rng_r.normal(BASE_MEAN, BASE_STD, size=n)).astype(np.float64)

        n_left = n // 2
        n_right = n - n_left
        B_left = rng_b.normal(mean_left, BASE_STD, size=n_left)
        B_right = rng_b.normal(mean_right, BASE_STD, size=n_right)
        B = np.sort(np.concatenate([B_left, B_right])).astype(np.float64)

        return R.tolist(), B.tolist()

    return maker

def time_competitor(R: list[float], B: list[float]) -> float:
    R_np = np.asarray(R, dtype=np.float64)
    B_np = np.asarray(B, dtype=np.float64)

    t0 = perf_counter()
    partial_ot_1d(R_np, B_np, max_iter=min(len(R_np), len(B_np)), p=2)
    return perf_counter() - t0


def benchmark_family(
    instance_maker: Callable[[int, int], tuple[list[float], list[float]]],
    *,
    seed: int,
) -> dict[str, tuple[list[int], list[float]]]:
    active_algorithms = list(ALGORITHMS)
    competitor_active = BENCHMARK_COMPETITOR

    names = list(ALGORITHMS)
    if competitor_active:
        names.append(COMPETITOR_NAME)

    n_values_by_algo = {name: [] for name in names}
    times_by_algo = {name: [] for name in names}

    # Warm up competitor once so JIT compile time does not pollute the first point.
    if competitor_active:
        warm_x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        warm_y = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        partial_ot_1d(warm_x, warm_y, max_iter=3, p=2)

    n = N_START
    while active_algorithms or competitor_active:
        repeat_count = REPEAT_COUNT_SMALL if n <= REPEAT_UNTIL_N else REPEAT_COUNT_LARGE
        per_algo_times = {name: [] for name in names}

        for rep in range(repeat_count):
            rep_seed = seed + rep
            R, B = instance_maker(n, rep_seed)

            if active_algorithms:
                profiles = compare_all_algorithms(R, B, active_algorithms)
                for name in active_algorithms:
                    profile = profiles[name]
                    if profile.stats is not None:
                        per_algo_times[name].append(profile.stats.total_time_seconds)

            if competitor_active:
                per_algo_times[COMPETITOR_NAME].append(time_competitor(R, B))

        next_active = []
        for name in active_algorithms:
            if not per_algo_times[name]:
                continue

            t_med = median(per_algo_times[name])
            n_values_by_algo[name].append(n)
            times_by_algo[name].append(t_med)

            if t_med <= TIME_LIMIT_SECONDS:
                next_active.append(name)

        active_algorithms = next_active

        if competitor_active and per_algo_times[COMPETITOR_NAME]:
            t_med = median(per_algo_times[COMPETITOR_NAME])
            n_values_by_algo[COMPETITOR_NAME].append(n)
            times_by_algo[COMPETITOR_NAME].append(t_med)

            if t_med > TIME_LIMIT_SECONDS:
                competitor_active = False

        next_n = max(n + 1, int(round(n * N_MULTIPLIER)))
        if next_n == n:
            next_n = n + 1
        n = next_n

    return {
        name: (n_values_by_algo[name], times_by_algo[name])
        for name in names
    }

panels = []

# Row 1
for col, blue_std in enumerate([1.0, 1.01, 1.1, 2.0]):
    panels.append(
        {
            "row": 0,
            "col": col,
            "title": rf"$R \sim \mathcal{{N}}(0,1)$" + "\n" + rf"$B \sim \mathcal{{N}}(0,{blue_std}^2)$",
            "stem": f"row1_col{col+1}_std_{blue_std}".replace(".", "p"),
            "maker": make_same_mean_different_std(blue_std),
        }
    )

# Row 2
for col, blue_mean in enumerate([0.0, 0.01, 0.1, 1.0]):
    panels.append(
        {
            "row": 1,
            "col": col,
            "title": rf"$R \sim \mathcal{{N}}(0,1)$" + "\n" + rf"$B \sim \mathcal{{N}}({blue_mean},1)$",
            "stem": f"row2_col{col+1}_mean_{blue_mean}".replace(".", "p"),
            "maker": make_same_std_different_mean(blue_mean),
        }
    )

# Row 3
for col, (m1, m2) in enumerate([(0.0, 0.0), (-0.1, 0.1), (-0.5, 0.5), (-1.0, 1.0)]):
    panels.append(
        {
            "row": 2,
            "col": col,
            "title": (
                rf"$R \sim \mathcal{{N}}(0,1)$"
                + "\n"
                + rf"$B \sim \frac{{1}}{{2}}\mathcal{{N}}({m1},1)+\frac{{1}}{{2}}\mathcal{{N}}({m2},1)$"
            ),
            "stem": f"row3_col{col+1}_mix_{m1}_{m2}".replace(".", "p").replace("-", "m"),
            "maker": make_unimodal_vs_bimodal(m1, m2),
        }
    )

all_results = []
for i, panel in enumerate(panels, start=1):
    print(f"Benchmarking panel {i}/12: {panel['stem']}")
    curves = benchmark_family(panel["maker"], seed=MASTER_SEED + 1000 * i)
    all_results.append((panel, curves))

all_positive_times = []
all_ns = []

for _, curves in all_results:
    for _, (ns, times) in curves.items():
        all_positive_times.extend([t for t in times if t > 0])
        all_ns.extend(ns)

if not all_positive_times:
    raise RuntimeError("No timing data collected.")
if not all_ns:
    raise RuntimeError("No n values collected.")

global_y_min = min(all_positive_times) * 0.85
global_y_max = max(max(all_positive_times), TIME_LIMIT_SECONDS) * 1.05

global_x_min = min(all_ns) * 0.95
global_x_max = max(all_ns) * 1.05

def powers_of_ten_in_range(lo: float, hi: float) -> list[float]:
    if lo <= 0 or hi <= 0:
        return []
    k_min = int(np.floor(np.log10(lo)))
    k_max = int(np.ceil(np.log10(hi)))
    vals = []
    for k in range(k_min, k_max + 1):
        v = 10.0 ** k
        if lo <= v <= hi:
            vals.append(v)
    return vals


def format_power_tick(v: float) -> str:
    k = int(round(np.log10(v)))
    return rf"$10^{{{k}}}$"

def style_log_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)

    # Minor ticks completely off
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Force major ticks at powers of 10
    xticks = powers_of_ten_in_range(global_x_min, global_x_max)
    yticks = powers_of_ten_in_range(global_y_min, global_y_max)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([format_power_tick(v) for v in xticks], fontsize=7)
    ax.set_yticklabels([format_power_tick(v) for v in yticks], fontsize=7)

    # Grid only at powers of 10
    ax.grid(False)
    for xg in xticks:
        ax.axvline(xg, color="0.5", alpha=0.20, linewidth=0.4, zorder=0)
    for yg in yticks:
        ax.axhline(yg, color="0.5", alpha=0.20, linewidth=0.4, zorder=0)

    # Square panel
    ax.set_box_aspect(1)

    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.set_xlabel(r"$n$", fontsize=8)
    ax.set_ylabel("time (s)", fontsize=8)


def draw_panel(ax, title: str, curves: dict[str, tuple[list[int], list[float]]]) -> None:
    style_log_axes(ax)

    for name, (ns, ts) in curves.items():
        if not ns:
            continue
        style = STYLE.get(name, {})
        ax.plot(
            ns,
            ts,
            linestyle="-",
            linewidth=LINEWIDTH,
            marker=style.get("marker", "o"),
            markersize=MARKERSIZE,
            markerfacecolor="none",
            markeredgewidth=MARKEREDGEWIDTH,
            color=style.get("color", None),
            alpha=0.95,
            label=name,
        )

    ax.set_title(title, fontsize=9)

if SAVE_INDIVIDUAL_PDFS:
    for panel, curves in all_results:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        draw_panel(ax, panel["title"], curves)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{panel['stem']}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{panel['stem']}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

if SAVE_COMBINED_FIGURE:
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 11.5))

    for panel, curves in all_results:
        ax = axes[panel["row"], panel["col"]]
        draw_panel(ax, panel["title"], curves)

    # One shared legend gathered from all panels
    seen = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h

    fig.legend(
        seen.values(),
        seen.keys(),
        loc="upper center",
        ncol=max(1, len(seen)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.995),
    )

    row_labels = [
        "Row 1: variance perturbation",
        "Row 2: mean perturbation",
        "Row 3: bimodal perturbation",
    ]
    for r, label in enumerate(row_labels):
        axes[r, 0].annotate(
            label,
            xy=(-0.58, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
    fig.savefig(OUT_DIR / "runtime_grid_3x4.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "runtime_grid_3x4.png", dpi=240, bbox_inches="tight")
    plt.show()

print(f"\nDone. Output written to: {OUT_DIR.resolve()}")