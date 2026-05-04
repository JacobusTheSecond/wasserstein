from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from models import CompactInterval, GreedyProfileResult
from mp_types import Point
from utils import apply_delta_to_solution, undo_delta_from_solution, validate_disjoint_sorted_intervals, \
    validate_instance, recover_solution_at_k


class GreedyProfileStepper:
    """Helper object for stepping through the k-solutions."""

    def __init__(self, R: Sequence[Point], B: Sequence[Point], profile: GreedyProfileResult):
        validate_instance(R, B)
        self.R = list(R)
        self.B = list(B)
        self.profile = profile
        self.k = 0
        self.current_intervals: list[CompactInterval] = []

    def goto(self, k: int) -> int:
        if not (0 <= k <= self.profile.n):
            raise ValueError(f"k must lie in [0, {self.profile.n}].")
        if k < self.k:
            self.current_intervals = []
            self.k = 0
        while self.k < k:
            delta = self.profile.step_at(self.k + 1)
            assert delta is not None
            self.current_intervals = apply_delta_to_solution(self.current_intervals, delta)
            self.k += 1
        return self.k

    def next(self) -> int:
        if self.k < self.profile.n:
            delta = self.profile.step_at(self.k + 1)
            assert delta is not None
            self.current_intervals = apply_delta_to_solution(self.current_intervals, delta)
            self.k += 1
        return self.k

    def prev(self) -> int:
        if self.k > 0:
            delta = self.profile.step_at(self.k)
            assert delta is not None
            self.current_intervals = undo_delta_from_solution(self.current_intervals, delta)
            self.k -= 1
        return self.k

    def current_solution(self) -> list[CompactInterval]:
        return list(self.current_intervals)

    def current_cost(self) -> float:
        return self.profile.cost_at(self.k)

    def current_step(self):
        return self.profile.step_at(self.k)

    def plot(self, ax=None, show_pairs: bool = True) -> None:
        plot_solution(
            self.R,
            self.B,
            self.current_intervals,
            ax=ax,
            title=describe_solution_step(self.profile, self.k),
            show_pairs=show_pairs,
        )

    def interactive_plot(self, show_pairs: bool = True):
        fig = plt.figure(figsize=(11, 4.0))
        ax = fig.add_axes([0.06, 0.24, 0.88, 0.70])
        ax_prev = fig.add_axes([0.34, 0.06, 0.12, 0.10])
        ax_next = fig.add_axes([0.54, 0.06, 0.12, 0.10])

        button_prev = Button(ax_prev, "Previous")
        button_next = Button(ax_next, "Next")

        def redraw() -> None:
            ax.clear()
            self.plot(ax=ax, show_pairs=show_pairs)
            fig.canvas.draw_idle()

        def on_prev(_event) -> None:
            self.prev()
            redraw()

        def on_next(_event) -> None:
            self.next()
            redraw()

        def on_key(event) -> None:
            if event.key in {"left", "p"}:
                self.prev()
                redraw()
            elif event.key in {"right", "n"}:
                self.next()
                redraw()
            elif event.key == "home":
                self.goto(0)
                redraw()
            elif event.key == "end":
                self.goto(self.profile.n)
                redraw()

        button_prev.on_clicked(on_prev)
        button_next.on_clicked(on_next)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()
        return fig


def plot_instance(
    R: Sequence[Point],
    B: Sequence[Point],
    *,
    ax=None,
    title: str = "Red/blue points on the line",
    show_legend: bool = True,
    point_size: int = 50,
) -> None:
    validate_instance(R, B)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    if len(R) > 0:
        ax.scatter(R, [0.08] * len(R), c="red", s=point_size, label="R", zorder=3)
    if len(B) > 0:
        ax.scatter(B, [-0.08] * len(B), c="blue", s=point_size, label="B", zorder=3)

    all_points = list(R) + list(B)
    if all_points:
        xmin = min(all_points)
        xmax = max(all_points)
        pad = max(1.0, 0.03 * max(1.0, xmax - xmin))
        ax.set_xlim(xmin - pad, xmax + pad)
    else:
        ax.set_xlim(-1.0, 1.0)

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=1)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("position on the real line")
    ax.set_title(title)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(loc="upper right", frameon=False)


def plot_solution(
    R: Sequence[Point],
    B: Sequence[Point],
    intervals: Sequence[CompactInterval],
    *,
    ax=None,
    title: str = "Current interval solution",
    show_pairs: bool = True,
) -> None:
    validate_instance(R, B)
    validate_disjoint_sorted_intervals(intervals)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.0))

    plot_instance(R, B, ax=ax, title=title)
    for idx, interval in enumerate(intervals):
        left = interval.left_coordinate(R, B)
        right = interval.right_coordinate(R, B)
        ax.plot([left, right], [0.0, 0.0], linewidth=4, alpha=0.35, zorder=2)
        ax.text(0.5 * (left + right), 0.18, f"I{idx + 1}", ha="center", va="bottom", fontsize=9)
        if show_pairs:
            for red, blue in interval.pairs(R, B):
                ax.plot([red, blue], [0.08, -0.08], linewidth=1.5, alpha=0.6, zorder=2)


def describe_solution_step(profile: GreedyProfileResult, k: int) -> str:
    if not (0 <= k <= profile.n):
        raise ValueError(f"k must lie in [0, {profile.n}].")
    lines = [profile.algorithm_name, f"k = {k}", f"cost = {profile.cost_at(k):.6f}"]
    if k == 0:
        lines.append("no interval chosen yet")
        return " | ".join(lines)
    delta = profile.step_at(k)
    assert delta is not None
    lines.append(
        "chosen endpoints = "
        f"({delta.left_endpoint.position:.6f}, {delta.left_endpoint.color}) -- "
        f"({delta.right_endpoint.position:.6f}, {delta.right_endpoint.color})"
    )
    lines.append(f"delta cost = {delta.delta_cost:.6f}")
    lines.append(f"new interval size = {delta.merged_interval.size}")
    return " | ".join(lines)

