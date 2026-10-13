import numpy as np
import pytest

from harborwave import Harbor, fem
from harborwave.geometry import disk, rect
from harborwave.mesh import annulus, from_distance, polyline_points, rectangle, signed_areas


def test_rectangle_counts_and_tags():
    m = rectangle(2.0, 1.0, 8, 4)
    assert m.n_nodes == 9 * 5
    assert m.n_cells == 64
    assert np.all(signed_areas(m.nodes, m.cells) > 0)
    assert fem.edge_lengths(m, m.boundary("inlet")).sum() == pytest.approx(1.0)
    assert fem.edge_lengths(m, m.boundary("outlet")).sum() == pytest.approx(1.0)
    assert fem.edge_lengths(m, m.boundary("wall")).sum() == pytest.approx(4.0)


def test_rectangle_normals_point_outward():
    m = rectangle(2.0, 1.0, 4, 2)
    n_in = m.edge_normals(m.boundary("inlet"))
    n_out = m.edge_normals(m.boundary("outlet"))
    assert np.allclose(n_in, [-1.0, 0.0])
    assert np.allclose(n_out, [1.0, 0.0])


def test_annulus_area_and_perimeters():
    m = annulus(1.0, 3.0, 40, 200)
    area = signed_areas(m.nodes, m.cells)
    assert np.all(area > 0)
    # Polygonal approximation of the circles: slightly less than pi (R^2 - r^2).
    assert area.sum() == pytest.approx(np.pi * 8.0, rel=2e-3)
    assert fem.edge_lengths(m, m.boundary("inner")).sum() == pytest.approx(2 * np.pi, rel=2e-3)
    assert fem.edge_lengths(m, m.boundary("outer")).sum() == pytest.approx(6 * np.pi, rel=2e-3)
    n = m.edge_normals(m.boundary("inner"))
    mid = m.nodes[m.boundary("inner")].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", n, mid) < 0)


def test_signed_distance_rect():
    p = np.array([[0.5, 0.5], [2.0, 0.5], [0.5, -1.0], [2.0, 2.0]])
    d = rect(p, 0.0, 0.0, 1.0, 1.0)
    assert d == pytest.approx([-0.5, 1.0, 1.0, np.sqrt(2.0)])


def test_from_distance_disk():
    h = 0.1
    theta = np.linspace(0, 2 * np.pi, int(2 * np.pi / h), endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    m = from_distance(
        lambda p: disk(p, 1.0),
        ring,
        (-1.0, -1.0, 1.0, 1.0),
        h,
        lambda p: np.full(len(p), "rim"),
    )
    assert np.all(signed_areas(m.nodes, m.cells) > 0)
    assert signed_areas(m.nodes, m.cells).sum() == pytest.approx(np.pi, rel=5e-3)
    assert m.min_angle() > 20.0
    assert set(m.edge_tags) == {"rim"}
    assert fem.edge_lengths(m, m.boundary("rim")).sum() == pytest.approx(2 * np.pi, rel=5e-3)


def test_polyline_keeps_vertices():
    pts = polyline_points([(0, 0), (1, 0), (1, 1)], 0.3)
    assert np.allclose(pts[0], [0, 0])
    assert np.allclose(pts[-1], [1, 1])
    assert any(np.allclose(p, [1, 0]) for p in pts)


def test_harbor_mesh_boundaries():
    hb = Harbor()
    m = hb.mesh(period_min=8.0, nodes_per_wavelength=8)
    assert np.all(signed_areas(m.nodes, m.cells) > 0)
    assert m.min_angle() > 15.0
    length = {t: fem.edge_lengths(m, m.boundary(t)).sum() for t in set(m.edge_tags)}
    w, e, half = 0.5 * hb.breakwater_width, 0.5 * hb.entrance, hb.half_length
    assert length["offshore"] == pytest.approx(np.pi * hb.radius, rel=5e-3)
    assert length["coast"] == pytest.approx(2 * (hb.radius - half))
    assert length["breakwater"] == pytest.approx(2 * (3 * w + 2 * (half - e)))
    assert length["quay"] == pytest.approx(2 * (hb.basin_width - w) + 2 * half)
