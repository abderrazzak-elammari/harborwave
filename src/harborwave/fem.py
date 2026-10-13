"""P1 assembly with cellwise and edgewise coefficients, complex-valued"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .mesh import Mesh, signed_areas


def gradients(mesh: Mesh) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cell areas and the (n_cells, 3, 2) gradients of the P1 basis functions"""
    p = mesh.nodes[mesh.cells]
    area = signed_areas(mesh.nodes, mesh.cells)
    if np.any(area <= 0):
        raise ValueError("mesh has degenerate or clockwise cells")
    grad = np.empty((mesh.n_cells, 3, 2))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        grad[:, i, 0] = p[:, j, 1] - p[:, k, 1]
        grad[:, i, 1] = p[:, k, 0] - p[:, j, 0]
    grad /= (2.0 * area)[:, None, None]
    return area, grad


def _assemble(n: int, conn: NDArray[np.int64], local: NDArray) -> sp.csr_matrix:
    m = conn.shape[1]
    rows = np.repeat(conn, m, axis=1).ravel()
    cols = np.tile(conn, (1, m)).ravel()
    return sp.coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def stiffness(mesh: Mesh, coef: NDArray) -> sp.csr_matrix:
    """int coef grad(u).grad(v), coef constant per cell."""
    area, grad = gradients(mesh)
    local = np.einsum("eid,ejd->eij", grad, grad) * (coef * area)[:, None, None]
    return _assemble(mesh.n_nodes, mesh.cells, local)


def mass(mesh: Mesh, coef: NDArray) -> sp.csr_matrix:
    """int coef u v, coef constant per cell."""
    area = signed_areas(mesh.nodes, mesh.cells)
    ref = (np.ones((3, 3)) + np.eye(3)) / 12.0
    local = (coef * area)[:, None, None] * ref[None]
    return _assemble(mesh.n_nodes, mesh.cells, local)


def edge_lengths(mesh: Mesh, edges: NDArray[np.int64]) -> NDArray[np.float64]:
    return np.linalg.norm(mesh.nodes[edges[:, 1]] - mesh.nodes[edges[:, 0]], axis=1)


def boundary_mass(mesh: Mesh, edges: NDArray[np.int64], coef: NDArray) -> sp.csr_matrix:
    """int_edges coef u v, coef constant per edge."""
    length = edge_lengths(mesh, edges)
    ref = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    local = (coef * length)[:, None, None] * ref[None]
    return _assemble(mesh.n_nodes, edges, local)


def boundary_load(mesh: Mesh, edges: NDArray[np.int64], values: NDArray) -> NDArray:
    """int_edges g v with g given at the two nodes of each edge, shape (n_edges, 2)."""
    length = edge_lengths(mesh, edges)
    ref = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    local = length[:, None] * (values @ ref)
    b = np.zeros(mesh.n_nodes, dtype=np.result_type(values, float))
    np.add.at(b, edges.ravel(), local.ravel())
    return b


def cell_average(mesh: Mesh, nodal: NDArray) -> NDArray:
    return nodal[mesh.cells].mean(axis=1)


def edge_average(mesh: Mesh, edges: NDArray[np.int64], nodal: NDArray) -> NDArray:
    return nodal[edges].mean(axis=1)


def l2_norm(mesh: Mesh, u: NDArray) -> float:
    m = mass(mesh, np.ones(mesh.n_cells))
    return float(np.sqrt(np.real(np.conj(u) @ (m @ u))))


def l2_error(mesh: Mesh, u: NDArray, exact: NDArray) -> float:
    return l2_norm(mesh, u - exact) / l2_norm(mesh, exact)
