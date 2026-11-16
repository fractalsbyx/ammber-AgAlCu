"""
Compare two quadratic fits of the Al-Ag-Cu eutectic, e.g. the full pipeline
(fit_quadratic.py) and the simple ammber-only fit (fit_simple.py).

Top row, one panel per phase: the iso-energy contour 5 J/mol above each fit's
minimum relative to its own tangent plane (an ellipse whose size and shape are
set by A and whose centre is c_ref), with the TDB contour (relative to the
full fit's plane) for reference.
Bottom row: composition offset of c_ref, eigenvalues of A, liquid driving force.

Usage: python compare_fits.py [--full quadratic_fit_T771.4.json]
                              [--simple quadratic_fit_simple_T771.4.json] [--T 771.4]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tdb_sampling import PHASE_DESCRIPTION, PhaseEnergy, load_database

# Validated categorical pair (light surface); dashes and markers as secondary encoding
FULL = dict(color="#2a78d6", ls="-", marker="o", label="full pipeline")
SIMPLE = dict(color="#eb6834", ls="--", marker="s", label="simple, pdens 10000")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TDB_COLOR = "#c3c2b7"
LEVEL = 5.0  # J/mol above each fit's minimum
ORDER = ["LIQUID", "FCC_A1", "HCP_A3", "THETA"]


def load(path):
    """Phases by base name (FCC_A1_1 -> FCC_A1), and the solids' tangent plane."""
    data = json.load(open(path))
    phases = {}
    for p in data["phases"]:
        base = next(b for b in ORDER if p["name"] == b or p["name"].startswith(b + "_"))
        phases[base] = {k: np.array(p[k], dtype=float) for k in ("c_ref", "A", "B")}
        phases[base]["C"] = float(p["C"])
    s = phases["FCC_A1"]
    potential = s["B"]
    intercept = s["C"] - s["c_ref"] @ potential
    return phases, potential, intercept


def offset(p, potential, intercept):
    return p["C"] - (p["c_ref"] @ potential + intercept)


def ellipse(p, level, n=200):
    """Points where 0.5 dx.A.dx = level."""
    w, V = np.linalg.eigh(p["A"])
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.sqrt(2 * level / w[0]) * np.cos(t),
                       np.sqrt(2 * level / w[1]) * np.sin(t)])
    return p["c_ref"][:, None] + V @ circle


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TDB_COLOR)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", default="quadratic_fit_T771.4.json")
    parser.add_argument("--simple", default="quadratic_fit_simple_T771.4.json")
    parser.add_argument("--T", type=float, default=771.4)
    parser.add_argument("--out", default="compare_fits_T771.4.png")
    args = parser.parse_args()

    full, pot_f, int_f = load(args.full)
    simple, pot_s, int_s = load(args.simple)
    db = load_database("Ag-Al-Cu.TDB")

    fig = plt.figure(figsize=(16, 8.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 12, height_ratios=[1.15, 1], hspace=0.38, top=0.9, wspace=1.2)

    # --- Top row: iso-energy contours per phase ---
    for i, name in enumerate(ORDER):
        ax = fig.add_subplot(grid[0, 3 * i:3 * i + 3])
        style_axes(ax)
        ef, es = ellipse(full[name], LEVEL), ellipse(simple[name], LEVEL)
        pts = np.hstack([ef, es])
        lo, hi = pts.min(axis=1), pts.max(axis=1)
        pad = 0.15 * (hi - lo).max()
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo).max() + pad
        xs = np.linspace(centre[0] - half, centre[0] + half, 220)
        ys = np.linspace(centre[1] - half, centre[1] + half, 220)
        X, Y = np.meshgrid(xs, ys)
        G = PhaseEnergy(db, name)(np.column_stack([X.ravel(), Y.ravel()]), args.T)
        G = G.reshape(X.shape) - (X * pot_f[0] + Y * pot_f[1] + int_f)
        tdb_min = np.nanmin(G)
        ax.contour(X, Y, G, levels=[tdb_min + LEVEL], colors=TDB_COLOR, linewidths=3)
        for p, e, st in ((full[name], ef, FULL), (simple[name], es, SIMPLE)):
            ax.plot(e[0], e[1], color=st["color"], ls=st["ls"], lw=2)
            ax.plot(*p["c_ref"], marker=st["marker"], ms=8, color=st["color"],
                    mec=SURFACE, mew=2, ls="none")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel("x$_{Ag}$", color=INK2, fontsize=9)
        if i == 0:
            ax.set_ylabel("x$_{Cu}$", color=INK2, fontsize=9)
        ax.set_title(f"{name}  ·  {PHASE_DESCRIPTION[name]}", color=INK, fontsize=10,
                     loc="left")

    # --- Bottom left: composition offset of c_ref ---
    ax = fig.add_subplot(grid[1, 0:4])
    style_axes(ax)
    dx = [np.abs(simple[n]["c_ref"] - full[n]["c_ref"]).max() for n in ORDER]
    y = np.arange(len(ORDER))[::-1]
    ax.barh(y, dx, height=0.5, color=SIMPLE["color"])
    for yi, v in zip(y, dx):
        ax.text(v + 0.0003, yi, f"{v:.4f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, ORDER)
    ax.set_xlim(0, max(dx) * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("max |Δx| of c_ref, simple vs full (mole fraction)", color=INK2, fontsize=9)
    ax.set_title("Where the simple fit centres each phase", color=INK, fontsize=10, loc="left")

    # --- Bottom middle: eigenvalues of A ---
    ax = fig.add_subplot(grid[1, 4:8])
    style_axes(ax)
    for k, n in enumerate(ORDER):
        wf, ws = np.linalg.eigvalsh(full[n]["A"]), np.linalg.eigvalsh(simple[n]["A"])
        for j in range(2):
            yy = y[k] + (0.17 if j == 0 else -0.17)
            ax.plot([wf[j], ws[j]], [yy, yy], color=GRID, lw=2, zorder=1)
            # full: open ring drawn on top, so it stays visible where the two coincide
            ax.plot(ws[j], yy, marker="s", ms=7, color=SIMPLE["color"], mec=SURFACE, mew=1.5,
                    ls="none", zorder=2)
            ax.plot(wf[j], yy, marker="o", ms=11, mfc="none", mec=FULL["color"], mew=2,
                    ls="none", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, ORDER)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("eigenvalues of A (J/mol), soft (upper) and stiff (lower) per phase",
                  color=INK2, fontsize=9)
    ax.set_title("Curvature", color=INK, fontsize=10, loc="left")

    # --- Bottom right: liquid driving force ---
    ax = fig.add_subplot(grid[1, 8:12])
    style_axes(ax)
    df = [offset(full["LIQUID"], pot_f, int_f), offset(simple["LIQUID"], pot_s, int_s)]
    bars = ax.bar([0, 1], df, width=0.5, color=[FULL["color"], SIMPLE["color"]])
    for b, v in zip(bars, df):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center",
                fontsize=9, color=INK)
    ax.set_xticks([0, 1], ["full pipeline\n(= TDB)", "simple,\npdens 10000"])
    ax.set_ylim(0, max(df) * 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("G$_L$ − tangent plane at c$_{ref}$ (J/mol)", color=INK2, fontsize=9)
    ax.set_title(f"Liquid driving force, ΔT = 2 K", color=INK, fontsize=10, loc="left")

    handles = [Line2D([], [], color=FULL["color"], ls="-", lw=2, marker="o", ms=7,
                      mec=SURFACE, label=FULL["label"]),
               Line2D([], [], color=SIMPLE["color"], ls="--", lw=2, marker="s", ms=7,
                      mec=SURFACE, label=SIMPLE["label"]),
               Line2D([], [], color=TDB_COLOR, lw=3, label="TDB")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.98, 0.985))
    fig.suptitle(f"Simple vs full quadratic fit, T = {args.T:g} K", x=0.02, y=0.985,
                 ha="left", color=INK, fontsize=13)
    fig.text(0.02, 0.945, f"Top: contour {LEVEL:g} J/mol above each fit's minimum (relative "
             "to its own tangent plane); centre = c_ref, size and tilt = A.",
             color=INK2, fontsize=9)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
