"""Response of the basin to the wave period: where are the resonances?

python examples/harbor_resonance.py
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harborwave import (
    Harbor,
    MildSlope,
    ReflectedPlaneWave,
    agitation,
    analytic,
    dispersion,
    plotting,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "output"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t-min", type=float, default=7.0)
    p.add_argument("--t-max", type=float, default=60.0)
    p.add_argument("--n-periods", type=int, default=150)
    p.add_argument("--nodes-per-wavelength", type=int, default=12)
    p.add_argument("--quay-reflection", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    harbor = Harbor()
    mesh = harbor.mesh(args.t_min, args.nodes_per_wavelength)
    depth = harbor.depth(mesh.nodes)
    basin = harbor.basin_nodes(mesh)
    model = MildSlope(
        mesh,
        depth,
        open={"offshore": 1.0 / harbor.radius},
        reflection={"coast": 1.0, "breakwater": 1.0, "quay": args.quay_reflection},
    )
    print(f"mesh: {mesh.n_nodes:,} nodes, h = {mesh.h_max():.1f} m")

    # Log spacing: the long-period end is where the modes are, and they are narrow.
    periods = np.geomspace(args.t_min, args.t_max, args.n_periods)
    mean_resp, max_resp = [], []
    tic = time.perf_counter()
    for period in periods:
        omega = 2 * np.pi / period
        k_sea = float(dispersion.wavenumber(omega, harbor.depth_sea))
        eta = model.solve(omega, ReflectedPlaneWave(k_sea, np.pi / 2))
        kag = agitation(eta)[basin]
        mean_resp.append(kag.mean())
        max_resp.append(kag.max())
    elapsed = time.perf_counter() - tic
    print(f"{len(periods)} solves in {elapsed:.0f} s ({elapsed / len(periods):.2f} s each)")

    mean_resp = np.array(mean_resp)
    max_resp = np.array(max_resp)
    peaks = [
        i
        for i in range(1, len(periods) - 1)
        if mean_resp[i] > mean_resp[i - 1]
        and mean_resp[i] > mean_resp[i + 1]
        and mean_resp[i] > 2.0
    ]
    t_helmholtz = analytic.helmholtz_period(
        harbor.basin_length * harbor.basin_width,
        harbor.entrance,
        harbor.depth_basin,
        harbor.breakwater_width,
    )
    print(f"resonant periods: {', '.join(f'{periods[i]:.1f} s' for i in peaks)}")
    print(
        f"lumped Helmholtz estimate: {t_helmholtz:.0f} s "
        "(outside the sweep: at that period the wavelength exceeds the domain)"
    )
    json.dump(
        {
            "periods": periods.tolist(),
            "mean_agitation": mean_resp.tolist(),
            "max_agitation": max_resp.tolist(),
            "peaks_s": [float(periods[i]) for i in peaks],
            "helmholtz_estimate_s": t_helmholtz,
        },
        open(OUT / "resonance.json", "w"),
        indent=2,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(periods, mean_resp, label="basin mean")
    ax.semilogx(periods, max_resp, alpha=0.5, label="basin max")
    for i in peaks:
        ax.annotate(
            f"{periods[i]:.0f} s",
            (periods[i], mean_resp[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("wave period [s]")
    ax.set_ylabel("agitation |eta| / A")
    ax.set_xticks([7, 10, 15, 20, 30, 40, 60])
    ax.set_xticklabels([7, 10, 15, 20, 30, 40, 60])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "harbor_resonance.png", dpi=130)

    # The two strongest modes, side by side.
    strongest = sorted(peaks, key=lambda i: -mean_resp[i])[:2]
    if strongest:
        fig, axes = plt.subplots(
            1, len(strongest), figsize=(6.5 * len(strongest), 5.5), squeeze=False
        )
        for ax, i in zip(axes[0], strongest, strict=True):
            omega = 2 * np.pi / periods[i]
            k_sea = float(dispersion.wavenumber(omega, harbor.depth_sea))
            eta = model.solve(omega, ReflectedPlaneWave(k_sea, np.pi / 2))
            plotting.field(
                mesh,
                agitation(eta),
                ax=ax,
                title=f"T = {periods[i]:.1f} s",
                label="agitation |eta| / A",
            )
            ax.set_xlim(-200, 200)
            ax.set_ylim(-120, 210)
        fig.tight_layout()
        fig.savefig(FIG / "harbor_modes.png", dpi=130)


if __name__ == "__main__":
    main()
