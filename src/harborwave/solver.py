"""Mild-slope equation (Berkhoff) with absorbing and partially reflecting boundaries.

    div(c cg grad eta) + k^2 c cg eta = 0

Boundary conditions:
    open     d eta/dn = (ik - kappa/2) (eta - eta_inc) + d eta_inc/dn
    wall     d eta/dn = i k alpha eta,   alpha = (1 - Kr) / (1 + Kr)

kappa is the curvature of the open boundary (1/R on a circle, 0 on a straight line).
Keeping it is the first-order Bayliss-Turkel correction; it matters for kR < 10.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from . import dispersion, fem
from .mesh import Mesh


def reflection_to_alpha(kr: float) -> float:
    if not 0.0 <= kr <= 1.0:
        raise ValueError("reflection coefficient must be in [0, 1]")
    return (1.0 - kr) / (1.0 + kr)


@dataclass
class MildSlope:
    mesh: Mesh
    depth: NDArray[np.float64]  # nodal
    open: Mapping[str, float] = field(default_factory=lambda: {"offshore": 0.0})  # tag -> kappa
    reflection: Mapping[str, float] = field(default_factory=dict)  # tag -> Kr

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, float)
        if self.depth.shape != (self.mesh.n_nodes,):
            raise ValueError("depth must be given at every node")
        tags = set(self.mesh.edge_tags)
        unknown = (set(self.open) | set(self.reflection)) - tags
        if unknown:
            raise ValueError(f"no boundary tagged {sorted(unknown)}")
        if set(self.open) & set(self.reflection):
            raise ValueError("a boundary cannot be both open and reflecting")
        for kr in self.reflection.values():
            reflection_to_alpha(kr)

    def coefficients(self, omega: float) -> tuple[NDArray, NDArray]:
        """Nodal wavenumber and c*cg."""
        k = dispersion.wavenumber(omega, self.depth)
        ccg = (omega / k) * dispersion.group_speed(omega, self.depth)
        return k, ccg

    def _open_beta(self, tag: str, k: NDArray) -> NDArray[np.complex128]:
        edges = self.mesh.boundary(tag)
        return 1j * fem.edge_average(self.mesh, edges, k) - 0.5 * self.open[tag]

    def assemble(self, omega: float) -> sp.csc_matrix:
        mesh = self.mesh
        k, ccg = self.coefficients(omega)

        a = fem.stiffness(mesh, fem.cell_average(mesh, ccg)) - fem.mass(
            mesh, fem.cell_average(mesh, k**2 * ccg)
        )
        a = a.astype(complex)

        for tag, kr in self.reflection.items():
            alpha = reflection_to_alpha(kr)
            if alpha == 0.0:
                continue
            edges = mesh.boundary(tag)
            beta = 1j * alpha * fem.edge_average(mesh, edges, k)
            a = a - fem.boundary_mass(mesh, edges, beta * fem.edge_average(mesh, edges, ccg))

        for tag in self.open:
            edges = mesh.boundary(tag)
            beta = self._open_beta(tag, k)
            a = a - fem.boundary_mass(mesh, edges, beta * fem.edge_average(mesh, edges, ccg))
        return a.tocsc()

    def load(self, omega: float, incident, tags: Iterable[str]) -> NDArray[np.complex128]:
        mesh = self.mesh
        k, ccg = self.coefficients(omega)
        rhs = np.zeros(mesh.n_nodes, dtype=complex)
        for tag in tags:
            edges = mesh.boundary(tag)
            n = len(edges)
            pts = mesh.nodes[edges].reshape(-1, 2)
            eta_inc = incident(pts).reshape(n, 2)
            grad = incident.gradient(pts).reshape(n, 2, 2)
            dn = np.einsum("eij,ej->ei", grad, mesh.edge_normals(edges))
            beta = self._open_beta(tag, k)
            p = fem.edge_average(mesh, edges, ccg)
            rhs += fem.boundary_load(mesh, edges, p[:, None] * (dn - beta[:, None] * eta_inc))
        return rhs

    def solve(
        self, omega: float, incident, incident_tags: Iterable[str] | None = None
    ) -> NDArray[np.complex128]:
        """Nodal complex free-surface elevation.

        The incident field enters through `incident_tags` (all open boundaries by default).
        """
        tags = list(self.open) if incident_tags is None else list(incident_tags)
        missing = set(tags) - set(self.open)
        if missing:
            raise ValueError(f"incident boundaries must be open: {sorted(missing)}")
        return spla.spsolve(self.assemble(omega), self.load(omega, incident, tags))

    def solve_period(self, period: float, incident, incident_tags=None) -> NDArray[np.complex128]:
        return self.solve(2.0 * np.pi / period, incident, incident_tags)


def agitation(eta: NDArray[np.complex128], amplitude: float = 1.0) -> NDArray[np.float64]:
    """Local wave height over incident wave height."""
    return np.abs(eta) / amplitude
