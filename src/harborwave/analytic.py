"""Closed-form solutions used for verification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import h1vp, hankel1, jv, jvp

from . import dispersion


def maccamy_fuchs(
    p: NDArray[np.float64],
    k: float,
    radius: float,
    amplitude: float = 1.0,
    n_terms: int | None = None,
) -> NDArray[np.complex128]:
    """Diffraction of the plane wave exp(ikx) by a vertical circular cylinder.

    MacCamy & Fuchs (1954). Valid in constant depth, where the mild-slope equation
    reduces to Helmholtz.
    """
    r = np.linalg.norm(p, axis=1)
    theta = np.arctan2(p[:, 1], p[:, 0])
    ka = k * radius
    if n_terms is None:
        n_terms = int(ka + 10 * ka ** (1 / 3) + 15)

    eta = np.zeros(len(p), dtype=complex)
    for m in range(n_terms):
        eps = 1.0 if m == 0 else 2.0
        coef = jvp(m, ka) / h1vp(m, ka)
        eta += eps * 1j**m * (jv(m, k * r) - coef * hankel1(m, k * r)) * np.cos(m * theta)
    return amplitude * eta


def shoaling_amplitude(
    omega: float, depth: NDArray[np.float64], depth_ref: float, amplitude: float = 1.0
) -> NDArray[np.float64]:
    """Amplitude of a wave crossing a gentle slope, from energy flux conservation."""
    cg = dispersion.group_speed(omega, depth)
    cg_ref = dispersion.group_speed(omega, depth_ref)
    return amplitude * np.sqrt(cg_ref / cg)


def helmholtz_period(
    basin_area: float, entrance_width: float, depth: float, entrance_length: float = 0.0
) -> float:
    """Period of the pumping (Helmholtz) mode of a basin connected to the sea by a channel.

    Rough estimate from the lumped-mass model; the effective channel length includes an
    end correction of about half the entrance width on each side.
    """
    g = dispersion.G
    length = entrance_length + entrance_width
    return 2 * np.pi * np.sqrt(basin_area * length / (g * depth * entrance_width))
