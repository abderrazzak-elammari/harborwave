"""Comparison with closed-form solutions."""

import numpy as np
import pytest

from harborwave import MildSlope, PlaneWave, analytic, dispersion, fem
from harborwave.mesh import annulus, rectangle


def omega_for(k: float, depth: float) -> float:
    return float(np.sqrt(dispersion.G * k * np.tanh(k * depth)))


def rate(h, err):
    return np.polyfit(np.log(h), np.log(err), 1)[0]


def cylinder_model(n: int, radius=1.0, outer=8.0, depth=1.0, kappa=None):
    m = annulus(radius, outer, n, int(2 * np.pi * outer / ((outer - radius) / n)))
    kappa = 1.0 / outer if kappa is None else kappa
    model = MildSlope(
        m, np.full(m.n_nodes, depth), open={"outer": kappa}, reflection={"inner": 1.0}
    )
    return m, model


def test_plane_wave_in_flat_channel_converges_at_order_two():
    depth, k = 2.0, 1.0
    omega = omega_for(k, depth)
    wave = PlaneWave(k)
    hs, errs = [], []
    for n in [20, 40, 80]:
        m = rectangle(2 * np.pi, np.pi, n, n // 2)
        model = MildSlope(
            m,
            np.full(m.n_nodes, depth),
            open={"inlet": 0.0, "outlet": 0.0},
            reflection={"wall": 1.0},
        )
        eta = model.solve(omega, wave)
        hs.append(m.h_max())
        errs.append(fem.l2_error(m, eta, wave(m.nodes)))
    assert errs[-1] < 2e-3
    assert rate(hs, errs) == pytest.approx(2.0, abs=0.2)


def test_cylinder_diffraction_matches_maccamy_fuchs():
    radius, depth, k = 1.0, 1.0, 2.0
    omega = omega_for(k, depth)
    hs, errs = [], []
    for n in [28, 56, 112]:
        m, model = cylinder_model(n)
        eta = model.solve(omega, PlaneWave(k))
        hs.append(m.h_max())
        errs.append(fem.l2_error(m, eta, analytic.maccamy_fuchs(m.nodes, k, radius)))
    assert errs[-1] < 1e-2
    assert rate(hs, errs) == pytest.approx(2.0, abs=0.2)


def test_curvature_correction_reduces_boundary_reflection():
    radius, outer, depth, k = 1.0, 6.0, 1.0, 0.7
    omega = omega_for(k, depth)
    errs = {}
    for kappa in (0.0, 1.0 / outer):
        m, model = cylinder_model(60, outer=outer, kappa=kappa)
        exact = analytic.maccamy_fuchs(m.nodes, k, radius)
        errs[kappa] = fem.l2_error(m, model.solve(omega, PlaneWave(k)), exact)
    assert errs[1.0 / outer] < 0.2 * errs[0.0]


def test_shoaling_follows_energy_flux_conservation():
    """Wave going up a 1:37 slope; amplitude must grow like sqrt(cg0 / cg)."""
    length, width = 600.0, 20.0
    nx, ny = 300, 10
    omega = 2 * np.pi / 8.0
    m = rectangle(length, width, nx, ny)
    x = m.nodes[:, 0]
    depth = np.interp(x, [0.0, 100.0, 400.0, length], [10.0, 10.0, 2.0, 2.0])
    model = MildSlope(m, depth, open={"inlet": 0.0, "outlet": 0.0}, reflection={"wall": 1.0})
    k0 = float(dispersion.wavenumber(omega, 10.0))
    eta = model.solve(omega, PlaneWave(k0), incident_tags=["inlet"])

    amplitude = np.abs(eta).reshape(ny + 1, nx + 1)
    # Across the channel the wave is uniform up to the asymmetry of the split quads.
    assert np.abs(amplitude - amplitude[0]).max() < 0.02
    profile = amplitude.mean(axis=0)
    x_line = x.reshape(ny + 1, nx + 1)[0]
    expected = analytic.shoaling_amplitude(omega, np.interp(x_line, x, depth), 10.0)
    assert np.abs(profile / expected - 1.0).max() < 0.02


def test_fully_absorbing_wall_is_the_sommerfeld_condition():
    depth, k = 2.0, 1.0
    omega = omega_for(k, depth)
    m = rectangle(10.0, 2.0, 50, 10)
    as_open = MildSlope(m, np.full(m.n_nodes, depth), open={"inlet": 0.0, "outlet": 0.0})
    as_wall = MildSlope(
        m, np.full(m.n_nodes, depth), open={"inlet": 0.0}, reflection={"outlet": 0.0}
    )
    diff = as_open.assemble(omega) - as_wall.assemble(omega)
    assert abs(diff).max() < 1e-12


def test_boundary_validation():
    m = rectangle(1.0, 1.0, 2, 2)
    ones = np.ones(m.n_nodes)
    with pytest.raises(ValueError):
        MildSlope(m, ones, open={"outlet": 0.0}, reflection={"wall": 1.5})
    with pytest.raises(ValueError):
        MildSlope(m, ones, open={"outlet": 0.0}, reflection={"beach": 0.5})
    with pytest.raises(ValueError):
        MildSlope(m, ones, open={"outlet": 0.0}, reflection={"outlet": 0.5})
    with pytest.raises(ValueError):
        MildSlope(m, ones[:-1], open={"outlet": 0.0})
