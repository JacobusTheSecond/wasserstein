from __future__ import annotations

from typing import Optional

import numpy as np


def generate_instance(
    n: int,
    low: float = 0.0,
    high: float = 100.0,
    seed: Optional[int] = None,
    distribution: str = "uniform",
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

    return sorted(float(x) for x in red), sorted(float(x) for x in blue)


def generate_biased_instance(
    n: int,
    low: float = 0.0,
    high: float = 100.0,
    seed: Optional[int] = None,
    low_bias_strength: float = 4.0,
    high_bias_strength: float = 4.0,
    low_color: str = "R",
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

    return sorted(float(x) for x in red), sorted(float(x) for x in blue)
