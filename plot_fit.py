"""
Plots for fit_quadratic.py (Al-Ag-Cu eutectic).

plot_fit   plotly: ternary view of the hull phases, fit points and equilibrium
           triangle (left), and the data and fitted quadratics relative to the
           common tangent plane in 3D (right)
plot_cuts  matplotlib: 1D cuts through every c_ref along the eigenvectors of A,
           G - plane from the TDB, the fitted quadratic and the capped quadratic
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tdb_sampling import PHASE_DESCRIPTION

COLORS = {"LIQUID": "#9467bd", "FCC_A1": "#1f77b4", "HCP_A3": "#2ca02c", "THETA": "#d62728"}


def color_of(name):
    for phase, c in COLORS.items():
        if name == phase or name.startswith(phase + "_"):
            return c
    return "gray"


def tern(x):  # (x_Ag, x_Cu) -> ternary a, b, c = Cu, Al, Ag
    x = np.atleast_2d(x)
    return dict(a=x[:, 1], b=1 - x[:, 0] - x[:, 1], c=x[:, 0])


def plot_fit(system, equilibrium, near_eq, quadratic, check, query, T,
             potential, intercept, xdists):
    def plane(x):
        return x @ potential + intercept

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "ternary"}, {"type": "scene"}]],
        subplot_titles=["Hull phases, fit points, equilibrium",
                        "G - tangent plane: data (points) and quadratic fits (surfaces)"])
    hover = "x<sub>Ag</sub>=%{c:.4f}<br>x<sub>Cu</sub>=%{a:.4f}<br>x<sub>Al</sub>=%{b:.4f}"

    # --- Ternary: hull vertices of the eutectic phases, fit points, equilibrium ---
    hull_sys = system.get_equilibrium_system()
    others = [ph.xdata for name, ph in hull_sys.phases.items() if color_of(name) == "gray"]
    if others:
        fig.add_trace(go.Scatterternary(
            **tern(np.vstack(others)), mode="markers", name="other phases on hull",
            marker=dict(color="gray", size=3, opacity=0.2),
            hovertemplate=hover + "<extra>other phase on hull</extra>"), row=1, col=1)
    for name, phase in hull_sys.phases.items():
        if color_of(name) == "gray":
            continue
        fig.add_trace(go.Scatterternary(
            **tern(phase.xdata), mode="markers", name=f"{name} on hull", legendgroup=name,
            marker=dict(color=color_of(name), size=3, opacity=0.25),
            hovertemplate=hover + f"<extra>{name} on hull</extra>"), row=1, col=1)
    for name, phase in near_eq.phases.items():
        fig.add_trace(go.Scatterternary(
            **tern(phase.xdata), mode="markers", name=f"{name} fit points", legendgroup=name,
            marker=dict(color=color_of(name), size=4),
            hovertemplate=hover + f"<extra>{name} fit point</extra>"), row=1, col=1)

    eq_comps = np.array([comp for _, comp, _ in equilibrium])
    fig.add_trace(go.Scatterternary(
        **tern(np.vstack([eq_comps, eq_comps[:1]])), mode="lines", name="equilibrium triangle",
        line=dict(color="crimson", width=2.5), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatterternary(
        **tern(query), mode="markers", name="eutectic liquid composition",
        marker=dict(symbol="star", size=16, color="crimson", line=dict(color="black", width=1)),
        hovertemplate=hover + "<extra>eutectic</extra>"), row=1, col=1)
    for name, comp, G in equilibrium:
        fig.add_trace(go.Scatterternary(
            **tern(comp), mode="markers", name=f"eq: {name}", legendgroup=name,
            marker=dict(symbol="diamond", size=13, color=color_of(name),
                        line=dict(color="black", width=1)),
            hovertemplate=hover + f"<br>G={G:.1f} J/mol<extra>eq: {name}</extra>"), row=1, col=1)
    fig.add_trace(go.Scatterternary(
        **tern(np.array([comp for _, comp, _ in check])), mode="markers",
        name="eq: fitted quadratics",
        marker=dict(symbol="circle-open", size=9, color="black", line=dict(width=2)),
        hovertemplate=hover + "<extra>eq: fitted quadratics</extra>"), row=1, col=1)

    # --- 3D: G - plane, so the tangent plane is z = 0 ---
    zmax = 120.0
    hover3d = "x<sub>Ag</sub>=%{x:.4f}<br>x<sub>Cu</sub>=%{y:.4f}<br>G-plane=%{z:.2f} J/mol"
    for name, phase in near_eq.phases.items():
        x = phase.xdata
        z = phase.Gdata - plane(x)
        m = z <= zmax
        fig.add_trace(go.Scatter3d(
            x=x[m, 0], y=x[m, 1], z=z[m], mode="markers", name=f"{name} data",
            legendgroup=name, showlegend=False, marker=dict(color=color_of(name), size=2),
            hovertemplate=hover3d + f"<extra>{name} TDB</extra>"), row=1, col=2)

        q = quadratic.phases[name]
        xdist = xdists[name]
        r, th = np.meshgrid(np.linspace(0, xdist, 25), np.linspace(0, 2 * np.pi, 61))
        X = q.c_ref[0] + r * np.cos(th)
        Y = q.c_ref[1] + r * np.sin(th)
        XY = np.column_stack([X.ravel(), Y.ravel()])
        Z = (q.free_energy(XY) - plane(XY)).reshape(X.shape)
        Z = np.where((X >= 0) & (Y >= 0) & (X + Y <= 1) & (Z <= zmax), Z, np.nan)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, name=f"{name} quadratic", legendgroup=name, showlegend=True,
            colorscale=[[0, color_of(name)], [1, color_of(name)]], showscale=False,
            opacity=0.55, hovertemplate=hover3d + f"<extra>{name} quadratic</extra>"),
            row=1, col=2)

    fig.add_trace(go.Mesh3d(
        x=eq_comps[:, 0], y=eq_comps[:, 1], z=np.zeros(3), i=[0], j=[1], k=[2],
        color="silver", opacity=0.35, name="common tangent plane", showlegend=True,
        hoverinfo="skip"), row=1, col=2)

    fig.update_layout(
        title=f"Al-Ag-Cu eutectic quadratic fit (ammber), T = {T:.2f} K  |  "
              f"phases: {', '.join(quadratic.phases)}",
        template="plotly_white", height=750,
        ternary=dict(sum=1, aaxis=dict(title="Cu"), baxis=dict(title="Al"),
                     caxis=dict(title="Ag")),
        scene=dict(xaxis=dict(title="x<sub>Ag</sub>"), yaxis=dict(title="x<sub>Cu</sub>"),
                   zaxis=dict(title="G - plane (J/mol)", range=[-20, zmax]),
                   camera=dict(eye=dict(x=1.4, y=-1.6, z=0.8))),
    )
    return fig


def plot_cuts(quadratic, uncapped, energies, T, potential, intercept,
              xdists, path):
    """Cuts through c_ref along each eigenvector of the (uncapped) A."""
    names = list(quadratic.phases)
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 7), squeeze=False)
    for col, name in enumerate(names):
        q = quadratic.phases[name]
        A0 = uncapped[name]
        w, V = np.linalg.eigh(A0)
        xdist = xdists[name]
        for row in range(2):
            ax = axes[row, col]
            u = V[:, row]
            # step s along u moves the full composition by |s| * |(u, -sum u)|
            norm = np.sqrt(np.sum(u ** 2) + u.sum() ** 2)
            s = np.linspace(-1.5 * xdist, 1.5 * xdist, 301) / norm
            x = q.c_ref + s[:, None] * u
            plane = x @ potential + intercept
            G_tdb = energies[name](x, T) - plane
            G_fit = 0.5 * w[row] * s ** 2 + (q.C - (q.c_ref @ potential + intercept))
            G_cap = q.free_energy(x) - plane
            t = s * norm
            ax.plot(t, G_tdb, color="k", lw=2, label="TDB")
            ax.plot(t, G_fit, color=color_of(name), lw=1.5, ls="--", label="quadratic")
            if not np.allclose(A0, q.A):
                ax.plot(t, G_cap, color=color_of(name), lw=1.5, ls=":", label="capped")
            ax.axvspan(-xdist, xdist, color=color_of(name), alpha=0.08)
            ax.axhline(0, color="gray", lw=0.8)
            finite = G_tdb[np.isfinite(G_tdb)]
            top = max(np.percentile(finite, 90) if len(finite) else 1, 5)
            ax.set_ylim(min(-2, finite.min() - 2 if len(finite) else -2), top)
            ax.set_title(f"{name}: eigenvalue {w[row]:.3g}\n"
                         f"direction (Ag, Cu) = ({u[0]:+.2f}, {u[1]:+.2f})", fontsize=9)
            ax.set_xlabel("distance from c_ref (mole fraction)")
            if col == 0:
                ax.set_ylabel("G - tangent plane (J/mol)")
            ax.xaxis.set_major_locator(MaxNLocator(5))
        axes[0, col].text(0.02, 0.97, PHASE_DESCRIPTION[name], transform=axes[0, col].transAxes,
                          va="top", fontsize=8, color=color_of(name))
    handles = [Line2D([], [], color="k", lw=2, label="TDB"),
               Line2D([], [], color="gray", lw=1.5, ls="--", label="quadratic (uncapped)")]
    if any(not np.allclose(uncapped[n], quadratic.phases[n].A) for n in names):
        handles.append(Line2D([], [], color="gray", lw=1.5, ls=":", label="quadratic (capped A)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
    fig.suptitle(f"Al-Ag-Cu quadratic fits along the eigenvectors of A, T = {T:.2f} K "
                 f"(shaded: fit radius)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
