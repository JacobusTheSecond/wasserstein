from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable
import shutil
import sys

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

BASE_MEAN = 0.0
BASE_STD = 1.0
MASTER_SEED = 1337

SAVE_INDIVIDUAL_PDFS = True
SAVE_COMBINED_FIGURE = True

# Progress display
VERBOSE = True
LIVE_PROGRESS = True

# Correctness check against PAWL
COMPARE_TO_PAWL = True
PAWL_COMPARE_ALGORITHM = "FFT_c"
PAWL_ATOL = 1e-4
PAWL_RTOL = 1e-4

# Quantile band
LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90

# Visual style
LINEWIDTH = 1.7
MARKERSIZE = 2.5
MARKEREDGEWIDTH = 0.7
BAND_ALPHA = 0.16

STYLE = {
    "naive": dict(color="#222222", marker="o"),
    "FFT_py": dict(color="#0072B2", marker="s"),
    "slimFFT_py": dict(color="#CC79A7", marker="^"),
    "FFT_c": dict(color="#D55E00", marker="D"),
    "Theirs": dict(color="#009E73", marker="x"),
}

_PROGRESS_LAST_LEN = 0


def _terminal_width() -> int:
    return shutil.get_terminal_size((160, 20)).columns


def _fit_progress_message(msg: str) -> str:
    width = max(20, _terminal_width() - 1)
    if len(msg) <= width:
        return msg

    sep = " || "
    if sep in msg:
        prefix, suffix = msg.split(sep, 1)
        suffix = sep + suffix
        if len(suffix) >= width:
            return "..." + suffix[-(width - 3):]
        keep = width - len(suffix) - 3
        if keep <= 0:
            return suffix[-width:]
        return prefix[:keep] + "..." + suffix

    if width <= 3:
        return msg[:width]
    return msg[: width - 3] + "..."


def progress_update(msg: str) -> None:
    global _PROGRESS_LAST_LEN
    if not VERBOSE or not LIVE_PROGRESS:
        return
    line = _fit_progress_message(msg)
    pad = max(0, _PROGRESS_LAST_LEN - len(line))
    sys.stdout.write("\r" + line + (" " * pad))
    sys.stdout.flush()
    _PROGRESS_LAST_LEN = len(line)


def progress_end(msg: str | None = None) -> None:
    global _PROGRESS_LAST_LEN
    if not VERBOSE or not LIVE_PROGRESS:
        return
    if msg is None:
        sys.stdout.write("\r" + (" " * _PROGRESS_LAST_LEN) + "\r")
        sys.stdout.flush()
    else:
        line = _fit_progress_message(msg)
        pad = max(0, _PROGRESS_LAST_LEN - len(line))
        sys.stdout.write("\r" + line + (" " * pad) + "\n")
        sys.stdout.flush()
    _PROGRESS_LAST_LEN = 0


def vprint(*args, **kwargs) -> None:
    if not VERBOSE:
        return
    if LIVE_PROGRESS and _PROGRESS_LAST_LEN > 0:
        progress_end()
    print(*args, **kwargs)


def target_repeat_count(seconds: float) -> int:
    if seconds < 0.1:
        return 20
    if seconds < 1.0:
        return 10
    if seconds < 5.0:
        return 5
    return 3


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


def time_competitor_and_profile(R: list[float], B: list[float]) -> tuple[float, np.ndarray]:
    R_np = np.asarray(R, dtype=np.float64)
    B_np = np.asarray(B, dtype=np.float64)

    t0 = perf_counter()
    _, _, marginal_costs = partial_ot_1d(R_np, B_np, max_iter=min(len(R_np), len(B_np)), p=2)
    elapsed = perf_counter() - t0

    marginal_costs = np.asarray(marginal_costs, dtype=np.float64)
    profile_costs = np.concatenate(([0.0], np.cumsum(marginal_costs)))
    return elapsed, profile_costs


def assert_profile_matches_pawl(
    ours: list[float] | np.ndarray,
    pawl: list[float] | np.ndarray,
    *,
    atol: float = PAWL_ATOL,
    rtol: float = PAWL_RTOL,
    context: str = "",
) -> None:
    ours_arr = np.asarray(ours, dtype=np.float64)
    pawl_arr = np.asarray(pawl, dtype=np.float64)

    if ours_arr.shape != pawl_arr.shape:
        raise AssertionError(
            f"Profile length mismatch{': ' + context if context else ''}: "
            f"{ours_arr.shape} vs {pawl_arr.shape}"
        )

    if not np.allclose(ours_arr, pawl_arr, atol=atol, rtol=rtol):
        diff = np.abs(ours_arr - pawl_arr)
        k = int(np.argmax(diff))
        raise AssertionError(
            f"Profile mismatch against PAWL{': ' + context if context else ''} "
            f"at k={k}: ours={ours_arr[k]:.12g}, pawl={pawl_arr[k]:.12g}, "
            f"abs diff={diff[k]:.12g}"
        )


def benchmark_family(
    instance_maker: Callable[[int, int], tuple[list[float], list[float]]],
    *,
    seed: int,
    panel_label: str = "",
) -> dict[str, dict[str, list[float] | list[int]]]:
    active_algorithms = list(ALGORITHMS)
    competitor_active = BENCHMARK_COMPETITOR

    names = list(ALGORITHMS)
    if competitor_active:
        names.append(COMPETITOR_NAME)

    n_values_by_algo = {name: [] for name in names}
    median_times_by_algo = {name: [] for name in names}
    q10_times_by_algo = {name: [] for name in names}
    q90_times_by_algo = {name: [] for name in names}

    if competitor_active:
        progress_update(f"{panel_label} | warming up {COMPETITOR_NAME}")
        warm_x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        warm_y = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        partial_ot_1d(warm_x, warm_y, max_iter=3, p=2)

    n = N_START
    while active_algorithms or competitor_active:
        per_algo_times = {name: [] for name in names}
        target_repeats: dict[str, int | None] = {name: None for name in names}

        rep = 0
        while True:
            names_to_run = [
                name
                for name in active_algorithms
                if target_repeats[name] is None or len(per_algo_times[name]) < target_repeats[name]
            ]

            competitor_needs_more = (
                competitor_active
                and (
                    target_repeats[COMPETITOR_NAME] is None
                    or len(per_algo_times[COMPETITOR_NAME]) < target_repeats[COMPETITOR_NAME]
                )
            )

            fft_c_needs_more = (
                PAWL_COMPARE_ALGORITHM in target_repeats
                and (
                    target_repeats[PAWL_COMPARE_ALGORITHM] is None
                    or len(per_algo_times[PAWL_COMPARE_ALGORITHM]) < target_repeats[PAWL_COMPARE_ALGORITHM]
                )
            )

            need_competitor_for_own_curve = competitor_needs_more

            need_competitor_for_check = (
                competitor_active
                and COMPARE_TO_PAWL
                and competitor_needs_more
                and fft_c_needs_more
                and PAWL_COMPARE_ALGORITHM in names_to_run
            )

            run_competitor_now = need_competitor_for_own_curve or need_competitor_for_check

            if not names_to_run and not run_competitor_now:
                break

            rep_seed = seed + rep
            rep += 1
            R, B = instance_maker(n, rep_seed)

            profiles = None
            latest_chunks: list[str] = []

            if names_to_run:
                profiles = compare_all_algorithms(R, B, names_to_run)
                for name in names_to_run:
                    profile = profiles[name]
                    if profile.stats is not None:
                        t = profile.stats.total_time_seconds
                        per_algo_times[name].append(t)
                        latest_chunks.append(f"{name}={t:.3f}s")
                        if target_repeats[name] is None:
                            target_repeats[name] = target_repeat_count(t)

            pawl_profile = None
            if run_competitor_now:
                t_comp, pawl_profile = time_competitor_and_profile(R, B)
                if need_competitor_for_own_curve:
                    per_algo_times[COMPETITOR_NAME].append(t_comp)
                    latest_chunks.append(f"{COMPETITOR_NAME}={t_comp:.3f}s")
                    if target_repeats[COMPETITOR_NAME] is None:
                        target_repeats[COMPETITOR_NAME] = target_repeat_count(t_comp)

            if need_competitor_for_check and profiles is not None and pawl_profile is not None:
                if PAWL_COMPARE_ALGORITHM in profiles:
                    ours_profile = profiles[PAWL_COMPARE_ALGORITHM].costs
                    assert_profile_matches_pawl(
                        ours_profile,
                        pawl_profile,
                        context=f"{panel_label}, n={n}, repetition={rep}",
                    )
                    latest_chunks.append("profile same: yes")

            progress_chunks = []
            for name in names:
                target = target_repeats[name]
                done = len(per_algo_times[name])
                if target is None:
                    progress_chunks.append(f"{name}:?")
                else:
                    progress_chunks.append(f"{name}:{min(done, target)}/{target}")

            status = f"{panel_label} | n={n} | rep={rep} | " + " | ".join(progress_chunks)
            if latest_chunks:
                status += " || " + ", ".join(latest_chunks)
            progress_update(status)

        for name in active_algorithms:
            if not per_algo_times[name]:
                continue

            arr = np.asarray(per_algo_times[name], dtype=np.float64)
            t_med = float(np.quantile(arr, 0.50))
            t_q10 = float(np.quantile(arr, LOWER_QUANTILE))
            t_q90 = float(np.quantile(arr, UPPER_QUANTILE))

            n_values_by_algo[name].append(n)
            median_times_by_algo[name].append(t_med)
            q10_times_by_algo[name].append(t_q10)
            q90_times_by_algo[name].append(t_q90)

        next_active = []
        for name in active_algorithms:
            if not per_algo_times[name]:
                continue
            arr = np.asarray(per_algo_times[name], dtype=np.float64)
            t_med = float(np.quantile(arr, 0.50))
            if t_med <= TIME_LIMIT_SECONDS:
                next_active.append(name)
        active_algorithms = next_active

        if competitor_active and per_algo_times[COMPETITOR_NAME]:
            arr = np.asarray(per_algo_times[COMPETITOR_NAME], dtype=np.float64)
            t_med = float(np.quantile(arr, 0.50))
            t_q10 = float(np.quantile(arr, LOWER_QUANTILE))
            t_q90 = float(np.quantile(arr, UPPER_QUANTILE))

            n_values_by_algo[COMPETITOR_NAME].append(n)
            median_times_by_algo[COMPETITOR_NAME].append(t_med)
            q10_times_by_algo[COMPETITOR_NAME].append(t_q10)
            q90_times_by_algo[COMPETITOR_NAME].append(t_q90)

            if t_med > TIME_LIMIT_SECONDS:
                competitor_active = False

        next_n = max(n + 1, int(round(n * N_MULTIPLIER)))
        if next_n == n:
            next_n = n + 1
        n = next_n

    progress_end(f"{panel_label} | done")

    return {
        name: {
            "n": n_values_by_algo[name],
            "median": median_times_by_algo[name],
            "q10": q10_times_by_algo[name],
            "q90": q90_times_by_algo[name],
        }
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
    panel_label = f"panel {i}/12 ({panel['stem']})"
    curves = benchmark_family(panel["maker"], seed=MASTER_SEED + 1000 * i, panel_label=panel_label)
    all_results.append((panel, curves))

all_positive_times = []
all_ns = []

for _, curves in all_results:
    for _, curve in curves.items():
        ns = curve["n"]
        q10s = curve["q10"]
        q90s = curve["q90"]
        all_ns.extend(ns)
        all_positive_times.extend([t for t in q10s if t > 0])
        all_positive_times.extend([t for t in q90s if t > 0])

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

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    xticks = powers_of_ten_in_range(global_x_min, global_x_max)
    yticks = powers_of_ten_in_range(global_y_min, global_y_max)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([format_power_tick(v) for v in xticks], fontsize=7)
    ax.set_yticklabels([format_power_tick(v) for v in yticks], fontsize=7)

    ax.grid(False)
    for xg in xticks:
        ax.axvline(xg, color="0.5", alpha=0.20, linewidth=0.4, zorder=0)
    for yg in yticks:
        ax.axhline(yg, color="0.5", alpha=0.20, linewidth=0.4, zorder=0)

    ax.set_box_aspect(1)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.set_xlabel(r"$n$", fontsize=8)
    ax.set_ylabel("time (s)", fontsize=8)


def draw_panel(ax, title: str, curves: dict[str, dict[str, list[float] | list[int]]]) -> None:
    style_log_axes(ax)

    for name, curve in curves.items():
        ns = np.asarray(curve["n"], dtype=np.float64)
        meds = np.asarray(curve["median"], dtype=np.float64)
        q10s = np.asarray(curve["q10"], dtype=np.float64)
        q90s = np.asarray(curve["q90"], dtype=np.float64)

        if ns.size == 0:
            continue

        style = STYLE.get(name, {})
        color = style.get("color", None)

        ax.fill_between(
            ns,
            q10s,
            q90s,
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=1,
        )

        ax.plot(
            ns,
            meds,
            linestyle="-",
            linewidth=LINEWIDTH,
            marker=style.get("marker", "o"),
            markersize=MARKERSIZE,
            markerfacecolor="none",
            markeredgewidth=MARKEREDGEWIDTH,
            color=color,
            alpha=0.95,
            label=name,
            zorder=2,
        )

    ax.set_title(title, fontsize=9)


if SAVE_INDIVIDUAL_PDFS:
    vprint("Saving individual panels...")
    for panel, curves in all_results:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        draw_panel(ax, panel["title"], curves)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{panel['stem']}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{panel['stem']}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        vprint(f"  saved {panel['stem']}.pdf/.png")

if SAVE_COMBINED_FIGURE:
    vprint("Saving combined figure...")
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 11.5))

    for panel, curves in all_results:
        ax = axes[panel["row"], panel["col"]]
        draw_panel(ax, panel["title"], curves)

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
    vprint("  saved runtime_grid_3x4.pdf/.png")

print(f"\nDone. Output written to: {OUT_DIR.resolve()}")