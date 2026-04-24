from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def real_convolution(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, dtype=np.float64)
    out_len = len(a) + len(b) - 1
    fft_len = 1 << (out_len - 1).bit_length()
    fa = np.fft.rfft(np.asarray(a, dtype=np.float64), fft_len)
    fb = np.fft.rfft(np.asarray(b, dtype=np.float64), fft_len)
    conv = np.fft.irfft(fa * fb, fft_len)[:out_len]
    conv[np.abs(conv) < 1e-12] = 0.0
    return conv.astype(np.float64, copy=False)


def combine_shift_arrays(parts: Sequence[Tuple[int, np.ndarray]]) -> Tuple[int, np.ndarray]:
    nonempty = [(mn, arr) for mn, arr in parts if len(arr) > 0]
    if not nonempty:
        return 0, np.zeros(0, dtype=np.float64)
    shift_min = min(mn for mn, _ in nonempty)
    shift_max = max(mn + len(arr) - 1 for mn, arr in nonempty)
    out = np.zeros(shift_max - shift_min + 1, dtype=np.float64)
    for mn, arr in nonempty:
        start = mn - shift_min
        out[start : start + len(arr)] += arr
    return shift_min, out


def shift_array_value(shift_min: int, values: np.ndarray, d: int) -> float:
    if len(values) == 0:
        return 0.0
    idx = d - shift_min
    if 0 <= idx < len(values):
        return float(values[idx])
    return 0.0
