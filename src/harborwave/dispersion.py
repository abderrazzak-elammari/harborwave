"""Linear wave theory: dispersion relation and wave celerities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

G = 9.81


def wavenumber(omega: ArrayLike, depth: ArrayLike, tol: float = 1e-12) -> NDArray[np.float64]:
    """Solve omega^2 = g k tanh(k h) for k, elementwise."""
    omega = np.asarray(omega, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if np.any(depth <= 0):
        raise ValueError("depth must be positive")

    x = omega**2 * depth / G
    # Guo (2002) closed-form approximation, accurate to ~0.1%, then Newton.
    kh = x / (1.0 - np.exp(-(x**1.25))) ** 0.4
    for _ in range(20):
        t = np.tanh(kh)
        f = kh * t - x
        df = t + kh * (1.0 - t**2)
        step = f / df
        kh = kh - step
        if np.all(np.abs(step) < tol * np.maximum(kh, 1.0)):
            break
    return kh / depth


def phase_speed(omega: ArrayLike, depth: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(omega, float) / wavenumber(omega, depth)


def group_speed(omega: ArrayLike, depth: ArrayLike) -> NDArray[np.float64]:
    omega = np.asarray(omega, float)
    depth = np.asarray(depth, float)
    k = wavenumber(omega, depth)
    kh = k * depth
    # 2kh / sinh(2kh) underflows cleanly to 0 in deep water; guard the overflow.
    with np.errstate(over="ignore"):
        ratio = np.where(kh < 50.0, 2.0 * kh / np.sinh(2.0 * np.minimum(kh, 50.0)), 0.0)
    return 0.5 * (omega / k) * (1.0 + ratio)


def wavelength(period: ArrayLike, depth: ArrayLike) -> NDArray[np.float64]:
    omega = 2.0 * np.pi / np.asarray(period, float)
    return 2.0 * np.pi / wavenumber(omega, depth)
