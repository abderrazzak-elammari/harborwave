import numpy as np
import pytest

from harborwave import dispersion
from harborwave.dispersion import G


def test_dispersion_relation_satisfied():
    omega = np.linspace(0.1, 3.0, 50)
    depth = np.logspace(-1, 3, 50)
    k = dispersion.wavenumber(omega, depth)
    residual = G * k * np.tanh(k * depth) - omega**2
    assert np.abs(residual).max() < 1e-10 * omega.max() ** 2


def test_deep_water_limit():
    omega = 1.0
    k = dispersion.wavenumber(omega, 1000.0)
    assert k == pytest.approx(omega**2 / G, rel=1e-10)


def test_shallow_water_limit():
    omega = 0.01
    depth = 1.0
    k = dispersion.wavenumber(omega, depth)
    assert k == pytest.approx(omega / np.sqrt(G * depth), rel=1e-4)


def test_group_speed_limits():
    assert dispersion.group_speed(1.0, 1000.0) == pytest.approx(0.5 * G / 1.0, rel=1e-8)
    assert dispersion.group_speed(0.01, 1.0) == pytest.approx(np.sqrt(G), rel=1e-4)


def test_group_speed_is_derivative_of_omega():
    depth = 5.0
    omega = 1.2
    k = dispersion.wavenumber(omega, depth)
    dk = 1e-6

    def w(kk):
        return np.sqrt(G * kk * np.tanh(kk * depth))

    numerical = (w(k + dk) - w(k - dk)) / (2 * dk)
    assert dispersion.group_speed(omega, depth) == pytest.approx(numerical, rel=1e-6)


def test_wavelength_of_ten_second_wave():
    # Textbook value: T = 10 s in 10 m of water gives L ~ 92.3 m.
    assert dispersion.wavelength(10.0, 10.0) == pytest.approx(92.3, abs=0.2)


def test_negative_depth_rejected():
    with pytest.raises(ValueError):
        dispersion.wavenumber(1.0, -1.0)
