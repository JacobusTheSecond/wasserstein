from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_generated_instance(
    R,
    B,
    *,
    bins: int = 30,
    title: str = "Synthetic instance distribution",
    show: bool = True,
):
    """
    Plot bucketed histograms of the generated red/blue point sets.
    """
    R = np.asarray(R, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    plt.figure(figsize=(9, 4.5))
    plt.hist(R, bins=bins, alpha=0.6, label="R", color="red")
    plt.hist(B, bins=bins, alpha=0.6, label="B", color="blue")
    plt.xlabel("Position")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def generate_instance(
    n: int,
    low: float = 0.0,
    high: float = 100.0,
    seed: Optional[int] = None,
    distribution: str = "uniform",
    *,
    plot: bool = False,
    bins: int = 30,
):
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if high < low:
        raise ValueError("high must be at least low.")

    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        red = rng.uniform(low, high, size=n)
        blue = rng.uniform(low, high, size=n)
    elif distribution == "normal":
        center = 0.5 * (low + high)
        scale = max((high - low) / 6.0, 1e-12)
        red = np.clip(rng.normal(center, scale, size=n), low, high)
        blue = np.clip(rng.normal(center, scale, size=n), low, high)
    else:
        raise ValueError("distribution must be either 'uniform' or 'normal'.")

    R = sorted(float(x) for x in red)
    B = sorted(float(x) for x in blue)

    if plot:
        plot_generated_instance(
            R,
            B,
            bins=bins,
            title=f"Synthetic instance ({distribution})",
        )

    return R, B


def generate_biased_instance(
    n: int,
    low: float = 0.0,
    high: float = 100.0,
    seed: Optional[int] = None,
    low_bias_strength: float = 4.0,
    high_bias_strength: float = 4.0,
    low_color: str = "R",
    *,
    plot: bool = False,
    bins: int = 30,
):
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if high < low:
        raise ValueError("high must be at least low.")
    if low_bias_strength <= 0 or high_bias_strength <= 0:
        raise ValueError("bias strengths must be positive.")
    if low_color not in {"R", "B"}:
        raise ValueError("low_color must be 'R' or 'B'.")

    rng = np.random.default_rng(seed)

    low_samples = rng.beta(1.0, low_bias_strength, size=n)
    high_samples = rng.beta(high_bias_strength, 1.0, size=n)

    low_points = low + (high - low) * low_samples
    high_points = low + (high - low) * high_samples

    if low_color == "R":
        red, blue = low_points, high_points
    else:
        red, blue = high_points, low_points

    R = sorted(float(x) for x in red)
    B = sorted(float(x) for x in blue)

    if plot:
        plot_generated_instance(
            R,
            B,
            bins=bins,
            title=(
                f"Biased synthetic instance "
                f"(low_color={low_color}, low_bias={low_bias_strength}, high_bias={high_bias_strength})"
            ),
        )

    return R, B

def generate_bimodal_vs_unimodal_instance(
    n: int,
    mean: float = 0.0,
    red_std: float = 1.0,
    blue_std: float = 1.0,
    mode_separation: float = 4.0,
    seed: Optional[int] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
    *,
    exact_sample_mean: bool = True,
    plot: bool = False,
    bins: int = 30,
):
    """
    Red:   n samples from one normal distribution N(mean, red_std^2)
    Blue:  n samples from a balanced mixture of two normals
           N(mean - mode_separation/2, blue_std^2)
           N(mean + mode_separation/2, blue_std^2)

    If exact_sample_mean=True, both realized samples are shifted so that their
    sample mean is exactly `mean`. This makes the red and blue sample means
    exactly equal.

    If low/high are provided, values are clipped into [low, high].
    Note: clipping can slightly disturb exact mean equality.
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if red_std <= 0 or blue_std <= 0:
        raise ValueError("red_std and blue_std must be positive.")
    if mode_separation < 0:
        raise ValueError("mode_separation must be nonnegative.")
    if (low is None) != (high is None):
        raise ValueError("Either specify both low and high, or neither.")
    if low is not None and high is not None and high < low:
        raise ValueError("high must be at least low.")

    rng = np.random.default_rng(seed)

    # Red: unimodal
    red = rng.normal(mean, red_std, size=n)

    # Blue: bimodal, split counts as evenly as possible
    n_left = n // 2
    n_right = n - n_left
    mu_left = mean - 0.5 * mode_separation
    mu_right = mean + 0.5 * mode_separation

    blue_left = rng.normal(mu_left, blue_std, size=n_left)
    blue_right = rng.normal(mu_right, blue_std, size=n_right)
    blue = np.concatenate([blue_left, blue_right])

    # Force the realized sample means to match exactly if requested
    if exact_sample_mean and n > 0:
        red = red + (mean - float(np.mean(red)))
        blue = blue + (mean - float(np.mean(blue)))

    if low is not None and high is not None:
        red = np.clip(red, low, high)
        blue = np.clip(blue, low, high)

    R = sorted(float(x) for x in red)
    B = sorted(float(x) for x in blue)

    if plot:
        plot_generated_instance(
            R,
            B,
            bins=bins,
            title=(
                f"Bimodal vs unimodal "
                f"(mean={mean}, red_std={red_std}, blue_std={blue_std}, "
                f"sep={mode_separation})"
            ),
        )

    return R, B

def generate_instance_with_means(
    n: int,
    red_mean: float,
    blue_mean: float,
    std: float = 1.0,
    low: Optional[float] = None,
    high: Optional[float] = None,
    seed: Optional[int] = None,
    *,
    plot: bool = False,
    bins: int = 30,
):
    """
    Generate two 1D point sets from normal distributions with specified means.

    If low and high are provided, samples are clipped into [low, high].
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if std <= 0:
        raise ValueError("std must be positive.")
    if (low is None) != (high is None):
        raise ValueError("Either specify both low and high, or neither.")
    if low is not None and high is not None and high < low:
        raise ValueError("high must be at least low.")

    rng = np.random.default_rng(seed)

    red = rng.normal(red_mean, std, size=n)
    blue = rng.normal(blue_mean, std, size=n)

    if low is not None and high is not None:
        red = np.clip(red, low, high)
        blue = np.clip(blue, low, high)

    R = sorted(float(x) for x in red)
    B = sorted(float(x) for x in blue)

    if plot:
        plot_generated_instance(
            R,
            B,
            bins=bins,
            title=(
                f"Synthetic instance with means "
                f"(red_mean={red_mean}, blue_mean={blue_mean}, std={std})"
            ),
        )

    return R, B