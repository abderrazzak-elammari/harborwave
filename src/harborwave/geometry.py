"""Idealised harbour on a straight coast, and the signed distance functions behind it.

Layout (y points inland, the coast is y = 0):

                      y = W   +------------------+
                              |      basin       |
                              |                  |
       coast  ----------------+===+          +===+----------------  y = 0
                                  |          |
                              breakwaters    entrance
                                    .  sea  .
                                       .  .
                             offshore semicircle, radius R
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import dispersion
from .mesh import Mesh, arc_points, from_distance, polyline_points

Points: TypeAlias = NDArray[np.float64]


def disk(p: Points, radius: float) -> NDArray[np.float64]:
    return np.linalg.norm(p, axis=1) - radius


def rect(p: Points, x0: float, y0: float, x1: float, y1: float) -> NDArray[np.float64]:
    dx = np.maximum(x0 - p[:, 0], p[:, 0] - x1)
    dy = np.maximum(y0 - p[:, 1], p[:, 1] - y1)
    outside = np.linalg.norm(np.maximum(np.column_stack([dx, dy]), 0.0), axis=1)
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside


@dataclass(frozen=True)
class Harbor:
    radius: float = 500.0  # offshore boundary
    basin_length: float = 300.0  # along the coast
    basin_width: float = 200.0  # inland
    entrance: float = 60.0
    breakwater_width: float = 12.0
    depth_sea: float = 10.0
    depth_basin: float = 6.0

    @property
    def half_length(self) -> float:
        return 0.5 * self.basin_length

    @property
    def breakwaters(self) -> list[tuple[float, float, float, float]]:
        w = 0.5 * self.breakwater_width
        e = 0.5 * self.entrance
        return [
            (-self.half_length, -w, -e, w),
            (e, -w, self.half_length, w),
        ]

    def distance(self, p: Points) -> NDArray[np.float64]:
        sea = np.maximum(disk(p, self.radius), p[:, 1])
        basin = rect(p, -self.half_length, 0.0, self.half_length, self.basin_width)
        d = np.minimum(sea, basin)
        for box in self.breakwaters:
            d = np.maximum(d, -rect(p, *box))
        return d

    def outline(self, h: float) -> Points:
        """Boundary nodes at spacing h, walking the domain counter-clockwise."""
        R, L, W = self.radius, self.half_length, self.basin_width
        w, e = 0.5 * self.breakwater_width, 0.5 * self.entrance
        offshore = arc_points(R, np.pi, 2 * np.pi, h)
        land = polyline_points(
            [
                (R, 0.0),
                (L, 0.0),
                (L, -w),
                (e, -w),
                (e, w),
                (L, w),
                (L, W),
                (-L, W),
                (-L, w),
                (-e, w),
                (-e, -w),
                (-L, -w),
                (-L, 0.0),
                (-R, 0.0),
            ],
            h,
        )
        # The arc ends at (R, 0) and the land outline starts there: drop the duplicate.
        return np.vstack([offshore[:-1], land[:-1]])

    def tag(self, p: Points, tol: float = 1e-6) -> NDArray[np.str_]:
        tags = np.full(len(p), "quay", dtype="<U12")
        # Edge midpoints sit on chords, slightly inside the circle.
        on_arc = np.abs(np.linalg.norm(p, axis=1) - self.radius) < 0.02 * self.radius
        tags[on_arc] = "offshore"
        on_coast = (np.abs(p[:, 1]) < tol) & (np.abs(p[:, 0]) > self.half_length)
        tags[on_coast] = "coast"
        for box in self.breakwaters:
            tags[np.abs(rect(p, *box)) < tol] = "breakwater"
        return tags

    def depth(self, p: Points) -> NDArray[np.float64]:
        """Constant depth at sea, linear ramp over the first third of the basin."""
        ramp = np.clip(p[:, 1] / (self.basin_width / 3.0), 0.0, 1.0)
        return self.depth_sea + (self.depth_basin - self.depth_sea) * ramp

    def mesh(self, period_min: float, nodes_per_wavelength: int = 12) -> Mesh:
        """Mesh fine enough for the shortest period of interest."""
        depth_min = min(self.depth_sea, self.depth_basin)
        h = float(dispersion.wavelength(period_min, depth_min)) / nodes_per_wavelength
        if h > 0.75 * self.breakwater_width:
            raise ValueError(
                f"mesh size {h:.1f} m cannot resolve breakwaters {self.breakwater_width} m wide"
            )
        bbox = (-self.radius, -self.radius, self.radius, self.basin_width)
        return from_distance(self.distance, self.outline(h), bbox, h, self.tag)

    def basin_nodes(self, mesh: Mesh) -> NDArray[np.int64]:
        p = mesh.nodes
        inside = (np.abs(p[:, 0]) < self.half_length) & (p[:, 1] > 0.5 * self.breakwater_width)
        return np.flatnonzero(inside)
