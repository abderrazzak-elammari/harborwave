"""Incident wave fields. Time dependence is exp(-i omega t)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Points: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class PlaneWave:
    """Monochromatic plane wave travelling in direction `angle` (radians from +x)."""

    k: float
    angle: float = 0.0
    amplitude: float = 1.0

    @property
    def direction(self) -> NDArray[np.float64]:
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    def __call__(self, p: Points) -> NDArray[np.complex128]:
        return self.amplitude * np.exp(1j * self.k * (p @ self.direction))

    def gradient(self, p: Points) -> NDArray[np.complex128]:
        return 1j * self.k * self(p)[:, None] * self.direction[None, :]


@dataclass(frozen=True)
class ReflectedPlaneWave:
    """Plane wave plus its mirror image on the straight coast y = 0.

    This is the field that would exist without the harbour, so the Sommerfeld condition
    on the offshore boundary only has to let the harbour-scattered part out.
    """

    k: float
    angle: float = np.pi / 2
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.angle < np.pi:
            raise ValueError("incident wave must travel towards the coast (0 < angle < pi)")

    @property
    def parts(self) -> tuple[PlaneWave, PlaneWave]:
        return (
            PlaneWave(self.k, self.angle, self.amplitude),
            PlaneWave(self.k, -self.angle, self.amplitude),
        )

    def __call__(self, p: Points) -> NDArray[np.complex128]:
        a, b = self.parts
        return a(p) + b(p)

    def gradient(self, p: Points) -> NDArray[np.complex128]:
        a, b = self.parts
        return a.gradient(p) + b.gradient(p)
