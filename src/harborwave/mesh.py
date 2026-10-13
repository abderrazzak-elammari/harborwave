"""Triangular meshes: structured generators and a distance-function mesher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay

Points: TypeAlias = NDArray[np.float64]
Distance: TypeAlias = Callable[[Points], NDArray[np.float64]]
Tagger: TypeAlias = Callable[[Points], NDArray[np.str_]]


@dataclass(frozen=True)
class Mesh:
    nodes: NDArray[np.float64]  # (n_nodes, 2)
    cells: NDArray[np.int64]  # (n_cells, 3), counter-clockwise
    edges: NDArray[np.int64]  # boundary edges, oriented as in their cell
    edge_tags: NDArray[np.str_]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def boundary(self, tag: str) -> NDArray[np.int64]:
        return self.edges[self.edge_tags == tag]

    def boundary_nodes(self, tag: str) -> NDArray[np.int64]:
        return np.unique(self.boundary(tag))

    def edge_normals(self, edges: NDArray[np.int64]) -> NDArray[np.float64]:
        """Outward unit normals of oriented boundary edges."""
        d = self.nodes[edges[:, 1]] - self.nodes[edges[:, 0]]
        n = np.column_stack([d[:, 1], -d[:, 0]])
        return n / np.linalg.norm(n, axis=1)[:, None]

    def h_max(self) -> float:
        p = self.nodes[self.cells]
        lengths = [np.linalg.norm(p[:, (i + 1) % 3] - p[:, i], axis=1) for i in range(3)]
        return float(np.max(lengths))

    def min_angle(self) -> float:
        """Smallest angle in the mesh, in degrees."""
        p = self.nodes[self.cells]
        angles = []
        for i in range(3):
            a = p[:, (i + 1) % 3] - p[:, i]
            b = p[:, (i + 2) % 3] - p[:, i]
            cos = np.einsum("ij,ij->i", a, b) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            )
            angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return float(np.min(angles))


def signed_areas(nodes: Points, cells: NDArray[np.int64]) -> NDArray[np.float64]:

    p = nodes[cells]
    twice_area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 2, 0] - p[:, 0, 0]
    ) * (p[:, 1, 1] - p[:, 0, 1])
    return np.asarray(0.5 * twice_area, dtype=np.float64)


def _orient(nodes: Points, cells: NDArray[np.int64]) -> NDArray[np.int64]:
    cells = cells.copy()
    flip = signed_areas(nodes, cells) < 0
    cells[flip] = cells[flip][:, [0, 2, 1]]
    return cells


def _boundary_edges(cells: NDArray[np.int64]) -> NDArray[np.int64]:
    """Edges that belong to a single cell, keeping the cell orientation."""
    edges = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return edges[first[counts == 1]]


def build(nodes: Points, cells: NDArray[np.int64], tag: Tagger) -> Mesh:
    cells = _orient(nodes, cells.astype(np.int64))
    edges = _boundary_edges(cells)
    mid = 0.5 * (nodes[edges[:, 0]] + nodes[edges[:, 1]])
    return Mesh(np.asarray(nodes, float), cells, edges, np.asarray(tag(mid)))


def rectangle(length: float, width: float, nx: int, ny: int) -> Mesh:
    """Structured mesh of [0, length] x [0, width]. Tags: inlet, outlet, wall."""
    x, y = np.meshgrid(np.linspace(0, length, nx + 1), np.linspace(0, width, ny + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()])

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = ii.ravel(), jj.ravel()
    n00 = j * (nx + 1) + i
    n10, n01, n11 = n00 + 1, n00 + nx + 1, n00 + nx + 2
    flip = (i + j) % 2 == 0
    a = np.where(flip[:, None], np.column_stack([n00, n10, n11]), np.column_stack([n00, n10, n01]))
    b = np.where(flip[:, None], np.column_stack([n00, n11, n01]), np.column_stack([n10, n11, n01]))

    def tag(p: Points) -> NDArray[np.str_]:
        out = np.full(len(p), "wall", dtype="<U8")
        out[np.isclose(p[:, 0], 0.0)] = "inlet"
        out[np.isclose(p[:, 0], length)] = "outlet"
        return out

    return build(nodes, np.vstack([a, b]), tag)


def annulus(r_inner: float, r_outer: float, n_radial: int, n_angular: int) -> Mesh:
    """Structured polar mesh between two concentric circles. Tags: inner, outer."""
    r = np.linspace(r_inner, r_outer, n_radial + 1)
    theta = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    nodes = np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()])

    ii, jj = np.meshgrid(np.arange(n_radial), np.arange(n_angular), indexing="ij")
    i, j = ii.ravel(), jj.ravel()
    jn = (j + 1) % n_angular
    n00 = i * n_angular + j
    n01 = i * n_angular + jn
    n10 = (i + 1) * n_angular + j
    n11 = (i + 1) * n_angular + jn
    flip = (i + j) % 2 == 0
    a = np.where(flip[:, None], np.column_stack([n00, n10, n11]), np.column_stack([n00, n10, n01]))
    b = np.where(flip[:, None], np.column_stack([n00, n11, n01]), np.column_stack([n10, n11, n01]))

    def tag(p: Points) -> NDArray[np.str_]:
        rad = np.linalg.norm(p, axis=1)
        return np.where(rad < 0.5 * (r_inner + r_outer), "inner", "outer").astype("<U8")

    return build(nodes, np.vstack([a, b]), tag)


def from_distance(
    dist: Distance,
    boundary_points: Points,
    bbox: tuple[float, float, float, float],
    h: float,
    tag: Tagger,
    n_smooth: int = 6,
) -> Mesh:
    """Unstructured mesh of (dist < 0) with prescribed boundary points.

    Interior nodes start on a hexagonal lattice, are triangulated with Delaunay and
    relaxed by a few Laplacian smoothing passes. Boundary nodes do not move. Good
    enough for smooth domains with features larger than a few h, not a general mesher.
    """
    x0, y0, x1, y1 = bbox
    x = np.arange(x0, x1 + h, h)
    y = np.arange(y0, y1 + h, h * np.sqrt(3) / 2)
    xx, yy = np.meshgrid(x, y)
    xx[1::2] += 0.5 * h
    interior = np.column_stack([xx.ravel(), yy.ravel()])
    interior = interior[dist(interior) < -0.6 * h]

    fixed = np.asarray(boundary_points, float)
    n_fixed = len(fixed)
    nodes = np.vstack([fixed, interior])

    def triangulate(pts: Points) -> NDArray[np.int64]:
        cells = Delaunay(pts).simplices
        centroid = pts[cells].mean(axis=1)
        return cells[dist(centroid) < -1e-6 * h]

    cells = triangulate(nodes)
    for _ in range(n_smooth):
        adj = sp.coo_matrix(
            (np.ones(3 * len(cells)), (cells[:, [0, 1, 2]].ravel(), cells[:, [1, 2, 0]].ravel())),
            shape=(len(nodes), len(nodes)),
        )
        adj = ((adj + adj.T) > 0).astype(float)
        degree = np.asarray(adj.sum(axis=1)).ravel()
        used = degree > 0
        target = nodes.copy()
        target[used] = (adj @ nodes)[used] / degree[used, None]
        target[:n_fixed] = fixed
        nodes = target
        cells = triangulate(nodes)

    keep = np.zeros(len(nodes), dtype=bool)
    keep[cells] = True
    renumber = np.cumsum(keep) - 1
    return build(nodes[keep], renumber[cells], tag)


def polyline_points(vertices: ArrayLike, h: float, closed: bool = False) -> Points:
    """Sample a polyline at spacing close to h, keeping every vertex."""
    vertices = np.asarray(vertices, float)
    if closed:
        vertices = np.vstack([vertices, vertices[:1]])
    out = []
    for a, b in zip(vertices[:-1], vertices[1:], strict=True):
        n = max(1, round(np.linalg.norm(b - a) / h))
        t = np.linspace(0, 1, n, endpoint=False)
        out.append(a + t[:, None] * (b - a))
    if not closed:
        out.append(vertices[-1:])
    return np.vstack(out)


def arc_points(radius: float, start: float, end: float, h: float, center=(0.0, 0.0)) -> Points:
    n = max(2, round(abs(end - start) * radius / h))
    theta = np.linspace(start, end, n + 1)
    return np.column_stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)])
