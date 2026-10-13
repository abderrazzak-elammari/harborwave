"""Interactive agitation map: pick a period, a direction and quay reflection.

streamlit run app/streamlit_app.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from harborwave import Harbor, MildSlope, ReflectedPlaneWave, agitation, dispersion, plotting


@st.cache_resource(show_spinner="Meshing the harbour...")
def build_mesh(period_min: float, nodes_per_wavelength: int):
    harbor = Harbor()
    mesh = harbor.mesh(period_min, nodes_per_wavelength)
    return harbor, mesh, harbor.depth(mesh.nodes)


def main():
    st.set_page_config(page_title="Harbour agitation", layout="wide")
    st.title("Harbour wave agitation")
    st.caption(
        "Mild-slope equation, P1 finite elements. "
        "Agitation is the local wave height over the incident wave height."
    )

    with st.sidebar:
        st.header("Incident wave")
        period = st.slider("Period [s]", 7.0, 60.0, 12.0, step=0.5)
        direction = st.slider("Direction of travel [deg from +x]", 30, 150, 90, step=5)
        st.header("Structures")
        kr_quay = st.slider("Quay reflection coefficient", 0.0, 1.0, 1.0, step=0.05)
        kr_breakwater = st.slider("Breakwater reflection coefficient", 0.0, 1.0, 1.0, step=0.05)
        st.header("Mesh")
        nodes_per_wavelength = st.select_slider(
            "Nodes per wavelength at T = 7 s", [8, 10, 12, 15], 10
        )

    harbor, mesh, depth = build_mesh(7.0, nodes_per_wavelength)
    model = MildSlope(
        mesh,
        depth,
        open={"offshore": 1.0 / harbor.radius},
        reflection={"coast": 1.0, "breakwater": kr_breakwater, "quay": kr_quay},
    )
    omega = 2 * np.pi / period
    k_sea = float(dispersion.wavenumber(omega, harbor.depth_sea))

    tic = time.perf_counter()
    eta = model.solve(omega, ReflectedPlaneWave(k_sea, np.radians(direction)))
    elapsed = time.perf_counter() - tic
    kag = agitation(eta)
    basin = harbor.basin_nodes(mesh)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Degrees of freedom", f"{mesh.n_nodes:,}")
    c2.metric("Solve time", f"{elapsed:.2f} s")
    c3.metric("Basin mean agitation", f"{kag[basin].mean():.2f}")
    c4.metric("Basin max agitation", f"{kag[basin].max():.2f}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 6))
        plotting.field(mesh, kag, ax=ax, title="Agitation", label="|eta| / A", vmax=3.0)
        ax.set_xlim(-320, 320)
        ax.set_ylim(-300, 210)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        phase = st.slider("Phase of the snapshot [deg]", 0, 360, 0, step=15)
        fig, ax = plt.subplots(figsize=(7, 6))
        plotting.surface(mesh, eta, phase=np.radians(phase), ax=ax, title="Free surface")
        ax.set_xlim(-320, 320)
        ax.set_ylim(-300, 210)
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
