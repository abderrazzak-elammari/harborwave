"""Agitation map in the idealised harbour, with vertical quays and with absorbing quays.

python examples/harbor_agitation.py --period 10 --direction 90
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harborwave import Harbor, MildSlope, ReflectedPlaneWave, agitation, dispersion, plotting

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--period", type=float, default=10.0, help="wave period [s]")
    p.add_argument(
        "--direction", type=float, default=90.0, help="direction of travel [deg from +x]"
    )
    p.add_argument(
        "--period-min", type=float, default=7.0, help="shortest period the mesh resolves"
    )
    p.add_argument("--nodes-per-wavelength", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    harbor = Harbor()
    mesh = harbor.mesh(args.period_min, args.nodes_per_wavelength)
    depth = harbor.depth(mesh.nodes)
    omega = 2 * np.pi / args.period
    k_sea = float(dispersion.wavenumber(omega, harbor.depth_sea))
    wave = ReflectedPlaneWave(k_sea, np.radians(args.direction))
    basin = harbor.basin_nodes(mesh)
    print(f"mesh: {mesh.n_nodes:,} nodes, h = {mesh.h_max():.1f} m")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, kr_quay in zip(axes, (1.0, 0.4), strict=True):
        model = MildSlope(
            mesh,
            depth,
            open={"offshore": 1.0 / harbor.radius},
            reflection={"coast": 1.0, "breakwater": 1.0, "quay": kr_quay},
        )
        eta = model.solve(omega, wave)
        kag = agitation(eta)
        print(
            f"quay Kr = {kr_quay}: basin agitation mean {kag[basin].mean():.2f}, "
            f"max {kag[basin].max():.2f}"
        )
        plotting.field(
            mesh,
            kag,
            ax=ax,
            title=f"T = {args.period:.0f} s, quay reflection Kr = {kr_quay}",
            label="agitation |eta| / A",
            vmax=2.5,
        )
        ax.set_xlim(-320, 320)
        ax.set_ylim(-300, 210)
    fig.tight_layout()
    fig.savefig(FIG / "harbor_agitation.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotting.wireframe(mesh, ax=ax)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-120, 210)
    ax.set_title("Mesh around the entrance")
    fig.tight_layout()
    fig.savefig(FIG / "harbor_mesh.png", dpi=130)


if __name__ == "__main__":
    main()
