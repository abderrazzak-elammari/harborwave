"""Diffraction by a vertical cylinder: finite elements against MacCamy & Fuchs.

python examples/cylinder_verification.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harborwave import MildSlope, PlaneWave, analytic, dispersion, fem, plotting
from harborwave.mesh import annulus

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"

radius, outer, depth = 1.0, 8.0, 1.0
k = 2.0
omega = float(np.sqrt(dispersion.G * k * np.tanh(k * depth)))


def solve(n_radial):
    m = annulus(radius, outer, n_radial, int(2 * np.pi * outer / ((outer - radius) / n_radial)))
    model = MildSlope(
        m, np.full(m.n_nodes, depth), open={"outer": 1.0 / outer}, reflection={"inner": 1.0}
    )
    return m, model.solve(omega, PlaneWave(k))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"ka = {k * radius:.2f}, kR = {k * outer:.1f}")
    print("| n_radial | DOFs | h | rel. L2 error |")
    print("|---:|---:|---:|---:|")
    hs, errs = [], []
    for n in [14, 28, 56, 112]:
        m, eta = solve(n)
        err = fem.l2_error(m, eta, analytic.maccamy_fuchs(m.nodes, k, radius))
        hs.append(m.h_max())
        errs.append(err)
        print(f"| {n} | {m.n_nodes:,} | {m.h_max():.3f} | {err:.2e} |")
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    print(f"observed order: {slope:.2f}")

    m, eta = solve(56)
    exact = analytic.maccamy_fuchs(m.nodes, k, radius)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plotting.field(m, np.abs(eta), ax=axes[0], title="Finite elements", label="|eta| / A", vmax=2.0)
    plotting.field(
        m, np.abs(exact), ax=axes[1], title="MacCamy & Fuchs", label="|eta| / A", vmax=2.0
    )
    plotting.field(
        m,
        np.abs(eta - exact),
        ax=axes[2],
        title="Pointwise error",
        label="|error| / A",
        cmap="magma",
    )
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    fig.tight_layout()
    fig.savefig(FIG / "cylinder_verification.png", dpi=130)

    theta = np.linspace(0, 2 * np.pi, 400)
    ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    on_wall = m.boundary_nodes("inner")
    angle_fe = np.arctan2(m.nodes[on_wall, 1], m.nodes[on_wall, 0]) % (2 * np.pi)
    order = np.argsort(angle_fe)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(
        np.degrees(theta), np.abs(analytic.maccamy_fuchs(ring, k, radius)), "k-", label="analytic"
    )
    ax.plot(
        np.degrees(angle_fe[order][::6]),
        np.abs(eta[on_wall][order][::6]),
        "o",
        ms=3,
        label="P1, h = 0.18",
)
    ax.set_xlabel("angle from the lee side [deg]")
    ax.set_ylabel("run-up |eta| / A")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cylinder_runup.png", dpi=130)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.loglog(hs, errs, "o-", label="P1")
    ax.loglog(hs, errs[0] * (np.array(hs) / hs[0]) ** 2, "k--", label="slope 2")
    ax.set_xlabel("h")
    ax.set_ylabel("relative L2 error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cylinder_convergence.png", dpi=130)


if __name__ == "__main__":
    main()
