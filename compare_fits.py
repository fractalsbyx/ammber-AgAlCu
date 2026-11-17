"""
Compare quadratic fits of the Al-Ag-Cu eutectic against the full pipeline
(fit_quadratic.py), e.g. the simple ammber fits from fit_simple.py,
fit_sample.py or fit_grid.py.

Top row, one panel per phase: the iso-energy contour 5 J/mol above each fit's
minimum relative to its own tangent plane (an ellipse whose size and shape are
set by A and whose centre is c_ref), with the TDB contour (relative to the
full fit's plane) for reference.
Bottom row: composition offset of c_ref, eigenvalues of A, liquid driving force.

Usage: python compare_fits.py [--full quadratic_fit_T771.4.json]
                              [--fit PATH LABEL]... [--T 771.4] [--out PNG]
Without --fit, compares the fromTDB pdens 10000 and sample_phase n 300 fits.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tdb_sampling import PHASE_DESCRIPTION, PhaseEnergy, load_database

# Validated categorical slots 1-4 (light surface), in fixed order; dashes and
# markers as secondary encoding (slots 3 and 4 are below 3:1 contrast, so every
# value is also labelled in text)
FULL = dict(color="#2a78d6", ls="-", marker="o", label="full pipeline")
STYLES = [dict(color="#eb6834", ls="--", marker="s"),
          dict(color="#1baf7a", ls=":", marker="^"),
          dict(color="#eda100", ls="-.", marker="D")]
DEFAULT_FITS = [("quadratic_fit_simple_T771.4.json", "simple, fromTDB pdens 10000"),
                ("quadratic_fit_sample_T771.4.json", "simple, sample_phase n 300")]
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
    parser.add_argument("--fit", nargs=2, action="append", metavar=("PATH", "LABEL"),
                        help=f"fit to compare, repeatable (up to {len(STYLES)})")
    parser.add_argument("--T", type=float, default=771.4)
    parser.add_argument("--out", default="compare_fits_T771.4.png")
    args = parser.parse_args()
    fit_args = args.fit or DEFAULT_FITS
    if len(fit_args) > len(STYLES):
        parser.error(f"at most {len(STYLES)} --fit")

    full, pot_f, int_f = load(args.full)
    others = []  # (phases, potential, intercept, style with label)
    for (path, label), style in zip(fit_args, STYLES):
        others.append((*load(path), dict(style, label=label)))
    db = load_database("Ag-Al-Cu.TDB")
    k_others = len(others)

    fig = plt.figure(figsize=(16, 8.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 12, height_ratios=[1.15, 1], hspace=0.38, top=0.87, wspace=1.2)

    # --- Top row: iso-energy contours per phase ---
    for i, name in enumerate(ORDER):
        ax = fig.add_subplot(grid[0, 3 * i:3 * i + 3])
        style_axes(ax)
        fits = [(full[name], FULL)] + [(o[0][name], o[3]) for o in others]
        curves = [ellipse(p, LEVEL) for p, _ in fits]
        pts = np.hstack(curves)
        lo, hi = pts.min(axis=1), pts.max(axis=1)
        pad = 0.15 * (hi - lo).max()
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo).max() + pad
        xs = np.linspace(centre[0] - half, centre[0] + half, 220)
        ys = np.linspace(centre[1] - half, centre[1] + half, 220)
        X, Y = np.meshgrid(xs, ys)
        G = PhaseEnergy(db, name)(np.column_stack([X.ravel(), Y.ravel()]), args.T)
        G = G.reshape(X.shape) - (X * pot_f[0] + Y * pot_f[1] + int_f)
        ax.contour(X, Y, G, levels=[np.nanmin(G) + LEVEL], colors=TDB_COLOR, linewidths=3)
        for (p, st), e in zip(fits, curves):
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

    y = np.arange(len(ORDER))[::-1]
    # vertical offsets of the k_others rows within one phase row
    spread = np.linspace(0.19, -0.19, k_others) if k_others > 1 else np.zeros(1)

    # --- Bottom left: composition offset of c_ref ---
    ax = fig.add_subplot(grid[1, 0:4])
    style_axes(ax)
    dmax = 0
    for (fit, _, _, st), dy in zip(others, spread):
        dx = [np.abs(fit[n]["c_ref"] - full[n]["c_ref"]).max() for n in ORDER]
        dmax = max(dmax, max(dx))
        yy = y + dy
        ax.barh(yy, dx, height=0.68 / k_others, color=st["color"], edgecolor=SURFACE,
                linewidth=1)
        for yi, val in zip(yy, dx):
            ax.text(val, yi, f" {val:.4f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, ORDER)
    ax.set_xlim(0, dmax * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("max |Δx| of c_ref vs full pipeline (mole fraction)", color=INK2, fontsize=9)
    ax.set_title("Where each fit centres each phase", color=INK, fontsize=10, loc="left")

    # --- Bottom middle: eigenvalues of A ---
    ax = fig.add_subplot(grid[1, 4:8])
    style_axes(ax)
    marker_dy = 0.5 * spread * 0.07 / 0.19 * 2 if k_others > 1 else np.zeros(1)
    for k, n in enumerate(ORDER):
        wf = np.linalg.eigvalsh(full[n]["A"])
        wo = [(np.linalg.eigvalsh(o[0][n]["A"]), o[3]) for o in others]
        for j in range(2):
            yy = y[k] + (0.17 if j == 0 else -0.17)
            vals = [wf[j]] + [w[j] for w, _ in wo]
            ax.plot([min(vals), max(vals)], [yy, yy], color=GRID, lw=2, zorder=1)
            # staggered so they don't hide each other; full is an open ring on top
            for (w, st), dy in zip(wo, marker_dy):
                ax.plot(w[j], yy + dy, marker=st["marker"], ms=7, color=st["color"],
                        mec=SURFACE, mew=1.5, ls="none", zorder=2)
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
    styles = [FULL] + [o[3] for o in others]
    df = [offset(full["LIQUID"], pot_f, int_f)] + [offset(o[0]["LIQUID"], o[1], o[2])
                                                   for o in others]
    xb = np.arange(len(df))
    bars = ax.bar(xb, df, width=0.55, color=[st["color"] for st in styles])
    for b, val in zip(bars, df):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center",
                fontsize=9, color=INK)
    ax.set_xticks(xb, ["full pipeline\n(= TDB)"]
                  + [st["label"].replace(", ", "\n") for st in styles[1:]], fontsize=8)
    ax.set_ylim(0, max(df) * 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("G$_L$ − tangent plane at c$_{ref}$ (J/mol)", color=INK2, fontsize=9)
    ax.set_title("Liquid driving force, ΔT = 2 K", color=INK, fontsize=10, loc="left")

    handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=2, marker=st["marker"],
                      ms=7, mec=SURFACE, label=st["label"]) for st in styles]
    handles.append(Line2D([], [], color=TDB_COLOR, lw=3, label="TDB"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.012, 0.94))
    fig.suptitle(f"Simple vs full quadratic fits, T = {args.T:g} K", x=0.02, y=0.985,
                 ha="left", color=INK, fontsize=13)
    fig.text(0.02, 0.95, f"Top: contour {LEVEL:g} J/mol above each fit's minimum (relative "
             "to its own tangent plane); centre = c_ref, size and tilt = A.",
             color=INK2, fontsize=9)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
