"""Matplotlib helpers. Imported lazily so the core package does not need matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh


def triangulation(mesh: Mesh) -> mtri.Triangulation:
    return mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cells)


def field(
    mesh: Mesh,
    values: NDArray,
    ax: plt.Axes | None = None,
    title: str = "",
    label: str = "",
    vmax: float | None = None,
    cmap: str = "viridis",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    tri = triangulation(mesh)
    art = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    plt.colorbar(art, ax=ax, label=label, shrink=0.8)
    return ax


def surface(mesh: Mesh, eta: NDArray[np.complex128], phase: float = 0.0, ax=None, title: str = ""):
    """Instantaneous free surface Re(eta e^{-i phase})."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    z = np.real(eta * np.exp(-1j * phase))
    lim = np.abs(eta).max()
    tri = triangulation(mesh)
    art = ax.tripcolor(tri, z, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(art, ax=ax, label="elevation / A", shrink=0.8)
    return ax


def wireframe(mesh: Mesh, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    kwargs.setdefault("lw", 0.3)
    kwargs.setdefault("color", "0.3")
    ax.triplot(triangulation(mesh), **kwargs)
    ax.set_aspect("equal")
    return ax
