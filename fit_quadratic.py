"""
Fit parabolic free energies for the Al-Ag-Cu ternary eutectic
(liquid -> alpha-Al (FCC_A1) + Ag2Al (HCP_A3) + Al2Cu (THETA)) with ammber, and
write them as PRISMS-PF phase JSON.

Steps:
  1) locate the ternary eutectic temperature T_E with pycalphad (bisection) and
     report the four phase compositions at T_E
  2) at T = T_E - dT, sample every phase of the TDB (all sublattice site
     fractions) into an ammber IsothermalDiscreteSystem (independent Ag, Cu;
     solvent Al) and take the hull equilibrium at the eutectic liquid
     composition: the three-phase triangle FCC_A1 + HCP_A3 + THETA
  3) refine the triangle with pycalphad, sample each solid near its
     equilibrium composition and fit quadratics relative to the common tangent
     plane (IsothermalQuadraticSystem.from_discrete)
  4) fit the LIQUID around the composition where it is closest to that plane,
     against the same plane (same B), so its offset C - plane(c_ref) is the
     TDB driving force for solidification at this undercooling
  5) optionally cap the eigenvalues of every A at --max-curvature (c_ref, B, C
     and so the tangent plane are unchanged)
  6) check: fit errors vs the TDB, the equilibrium of the fitted quadratic
     system at the eutectic composition, and the liquid driving force

Writes quadratic_fit_T<T>.json (PRISMS-PF phase format, placeholder D) and plots.

Usage: python fit_quadratic.py [--dT 2] [--max-curvature 2e5] [--save]
  --save writes quadratic_fit_T<T>.html instead of opening a browser
  (the matplotlib figure quadratic_fit_T<T>_cuts.png is always written).
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ammber.Systems import (IsothermalDiscreteSystem, IsothermalEquilibrium,
                            IsothermalQuadraticSystem)
from tdb_sampling import (COMPONENTS, EUTECTIC_PHASES, LABELS, PHASE_DESCRIPTION,
                          SOLIDS, SOLVENT, PhaseEnergy, build_system, equilibrium_at,
                          find_eutectic, load_database, tangent_plane_from_mu)

# Placeholder diffusivities (m^2/s); the solids are effectively frozen
D_LIQUID = 2e-9
D_SOLID = 1e-15


def fmt(x):
    """(x_Ag, x_Cu) as a readable string, with x_Al."""
    x = np.asarray(x)
    return f"Ag {x[0]:.4f}, Cu {x[1]:.4f}, Al {1 - x.sum():.4f}"


def base_name(name):
    """Strip ammber's miscibility-gap suffix (FCC_A1_0 -> FCC_A1)."""
    for phase in EUTECTIC_PHASES:
        if name == phase or name.startswith(phase + "_"):
            return phase
    return name.rsplit("_", 1)[0]


def local_grid(center, xdist, npts):
    """Square grid of npts x npts points (center included) cut to a disc of radius
    xdist in full mole-fraction space (as ammber's resample_near_xpoint)."""
    extent = xdist / np.sqrt(2)
    axes = [np.linspace(c - extent, c + extent, npts) for c in center]
    X = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=-1)
    d = X - center
    dist = np.sqrt(np.sum(d ** 2, axis=1) + d.sum(axis=1) ** 2)
    keep = (dist <= xdist + 1e-12) & np.all(X >= 0, axis=1) & (X.sum(axis=1) <= 1)
    return X[keep]


def sample_local(energy, center, T, xdist, npts):
    x = local_grid(center, xdist, npts)
    G = energy(x, T)
    ok = np.isfinite(G)
    return x[ok], G[ok]


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

def locate_eutectic(db, x0, tol):
    print(f"Locating T_E by bisection at overall composition {fmt(x0)}")
    T_E, comps, (T_lo, T_hi) = find_eutectic(db, np.asarray(x0), tol=tol)
    print(f"\nT_E = {T_E:.2f} K  (bracket [{T_lo:.3f}, {T_hi:.3f}] K)")
    print("Phase compositions at T_E (liquid from just above, solids from just below):")
    for phase in EUTECTIC_PHASES:
        if phase in comps:
            print(f"  {phase:7s} {PHASE_DESCRIPTION[phase]:18s} {fmt(comps[phase])}")
    missing = set(EUTECTIC_PHASES) - set(comps)
    if missing:
        raise SystemExit(f"phases missing at the eutectic: {sorted(missing)}")
    return T_E, comps


def hull_equilibrium(db, T, query, n):
    """ammber lower-hull equilibrium of all TDB phases at the query composition."""
    system = build_system(db, T, n)
    eq = system.get_equilibrium(query)
    print(f"\nammber hull (all {len(system.phases)} phases, grid n={n}) at T = {T:.2f} K, "
          f"query {fmt(query)}:")
    for name, comp, G in eq:
        print(f"  {name:9s} {fmt(comp)}  G = {G:.1f} J/mol")
    found = sorted(base_name(name) for name, _, _ in eq)
    if found != sorted(SOLIDS):
        raise SystemExit(f"hull triangle is {found}, expected {sorted(SOLIDS)}; "
                         "try a smaller --dT or a finer --n")
    return system, eq


def refined_equilibrium(db, energies, T, query):
    """Three-solid triangle from pycalphad, as an ammber IsothermalEquilibrium."""
    stable, mu = equilibrium_at(db, T, query)
    names = sorted(s["phase"] for s in stable)
    if names != sorted(SOLIDS):
        raise SystemExit(f"pycalphad equilibrium at T={T:.2f} K is {names}")
    points = []
    for s in sorted(stable, key=lambda s: SOLIDS.index(s["phase"])):
        G = float(energies[s["phase"]](s["x"][None, :], T)[0])
        points.append((s["phase"], s["x"], G))
    equilibrium = IsothermalEquilibrium(points, query=query)
    potential, intercept = equilibrium.get_tangent_plane()
    pot_mu, int_mu = tangent_plane_from_mu(mu)
    print(f"\npycalphad equilibrium at T = {T:.2f} K:")
    for (name, comp, G), s in zip(points, sorted(stable, key=lambda s: SOLIDS.index(s["phase"]))):
        print(f"  {name:7s} {fmt(comp)}  phase fraction {s['NP']:.3f}")
    print(f"  tangent plane through the three solids: dG/dx = {np.round(potential, 2).tolist()}, "
          f"G(Al) = {intercept:.2f} J/mol")
    print(f"  from pycalphad chemical potentials:     dG/dx = {np.round(pot_mu, 2).tolist()}, "
          f"G(Al) = {int_mu:.2f} J/mol")
    return equilibrium, mu


def fit_solids(energies, equilibrium, T, xdists, npts):
    near_eq = IsothermalDiscreteSystem(components=COMPONENTS, solution_component=SOLVENT)
    for name, comp, _ in equilibrium:
        x, G = sample_local(energies[name], comp, T, xdists[name], npts)
        near_eq.add_phase(name, x, G, T=T)
    quadratic = IsothermalQuadraticSystem()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quadratic.from_discrete(near_eq, equilibrium=equilibrium)
    for w in caught:
        print(f"warning: {w.message}")
    return near_eq, quadratic


def fit_liquid(energy, equilibrium, T, x_guess, xdist, npts):
    """Fit the liquid against the solids' tangent plane, centred where G_L - plane is
    smallest (the parallel-tangent construction), so B is shared and C carries the
    driving force."""
    potential, intercept = equilibrium.get_tangent_plane()

    def above_plane(x):
        return float(energy(np.atleast_2d(x), T)[0] - (x @ potential + intercept))

    res = minimize(above_plane, x_guess, method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-10, maxiter=4000))
    c_L = res.x
    x, G = sample_local(energy, c_L, T, xdist, npts)
    data = IsothermalDiscreteSystem(components=COMPONENTS, solution_component=SOLVENT)
    data.add_phase("LIQUID", x, G, T=T)
    liquid = IsothermalQuadraticSystem().from_discrete(data, equilibrium=equilibrium)
    phase = liquid.phases["LIQUID"]
    if not np.allclose(phase.c_ref, c_L, atol=1e-12):
        print("warning: liquid c_ref is not the grid centre; refine --npts")
    return data.phases["LIQUID"], phase, res.fun


def cap_curvature(quadratic, max_curvature):
    """Clip the eigenvalues of each A at max_curvature, keeping c_ref, B and C."""
    capped = {}
    for name, phase in quadratic.phases.items():
        w, V = np.linalg.eigh(phase.A)
        if np.any(w > max_curvature):
            capped[name] = w.copy()
            A = V @ np.diag(np.minimum(w, max_curvature)) @ V.T
            phase.A = 0.5 * (A + A.T)
    return capped


def fit_errors(quadratic, near_eq, energies, T, potential, intercept):
    """Errors of each quadratic against the TDB on its fit points, and the radius
    within which the error stays below 1 J/mol."""
    rows = {}
    for name, phase in quadratic.phases.items():
        x = near_eq.phases[name].xdata
        err = phase.free_energy(x) - energies[name](x, T)
        d = x - phase.c_ref
        r = np.sqrt(np.sum(d ** 2, axis=1) + d.sum(axis=1) ** 2)
        bad = r[np.abs(err) > 1.0]
        rows[name] = dict(n=len(x), max=np.abs(err).max(), rms=np.sqrt(np.mean(err ** 2)),
                          r1=bad.min() if len(bad) else r.max())
    return rows


def check_fitted_equilibrium(quadratic, query, xdist=0.01):
    """Equilibrium of the fitted quadratic system on fine grids around every c_ref."""
    step = 2.5e-4
    xdata = np.vstack([local_grid(p.c_ref, xdist, int(round(2 * xdist / np.sqrt(2) / step)) | 1)
                       for p in quadratic.phases.values()])
    xdata = np.unique(np.round(xdata, 12), axis=0)
    return quadratic.to_discrete(xdata=xdata).get_equilibrium(query)


def report(quadratic, errors, capped, max_curvature):
    print("\nFitted quadratic phases, G = 0.5 dx.A.dx + B.dx + C with dx = x - c_ref "
          "(x = (x_Ag, x_Cu), J/mol):")
    for name, phase in quadratic.phases.items():
        e = errors[name]
        w = np.linalg.eigvalsh(phase.A)
        print(f"  {name}  ({PHASE_DESCRIPTION[name]}, {e['n']} points)")
        print(f"    c_ref = {np.round(phase.c_ref, 5).tolist()}")
        print(f"    A     = {np.round(phase.A, 1).tolist()}  eigenvalues {np.round(w, 0).tolist()}")
        if name in capped:
            print(f"            (capped at {max_curvature:g}; uncapped eigenvalues "
                  f"{np.round(capped[name], 0).tolist()})")
        print(f"    B     = {np.round(phase.B, 3).tolist()}")
        print(f"    C     = {phase.C:.3f}")
        print(f"    error vs TDB on fit points: max |dG| = {e['max']:.2f}, rms = {e['rms']:.2f} J/mol;"
              f" |dG| < 1 J/mol within {e['r1']:.4f} of c_ref")


def write_json(quadratic, T):
    json_path = Path(f"quadratic_fit_T{T:.1f}.json")
    # Update an existing file in place so hand-added entries (e.g. edited "D") are kept
    system_dict = json.loads(json_path.read_text()) if json_path.exists() else {}
    quadratic.add_to_dict(system_dict)
    system_dict["solution_component"] = LABELS[quadratic.solution_component]
    system_dict["components"] = [LABELS[c] for c in quadratic.components]
    n = len(quadratic.components)
    for entry in system_dict["phases"]:
        D = D_LIQUID if entry["name"] == "LIQUID" else D_SOLID
        entry.setdefault("D", (D * np.eye(n)).tolist())
    json_path.write_text(json.dumps(system_dict, indent=2))
    print(f"\nSaved fitted phases to {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tdb", default="Ag-Al-Cu.TDB", help="thermodynamic database")
    parser.add_argument("--dT", type=float, default=2.0,
                        help="undercooling below T_E at which the fit is made (K)")
    parser.add_argument("--x0", type=float, nargs=2, default=[0.18, 0.14],
                        metavar=("X_AG", "X_CU"),
                        help="overall composition inside the three-solid triangle, "
                             "used to bracket T_E")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="bisection tolerance for T_E (K)")
    parser.add_argument("--n", type=int, default=100,
                        help="grid divisions per site-fraction axis for the hull")
    parser.add_argument("--xdist-solid", type=float, default=0.01,
                        help="fit radius around each solid's equilibrium composition")
    parser.add_argument("--xdist-liquid", type=float, default=0.03,
                        help="fit radius around the liquid composition")
    parser.add_argument("--xdist-phase", nargs="*", default=["THETA=0.005"], metavar="PHASE=R",
                        help="per-phase fit radius overrides (default THETA=0.005: Al2Cu sits "
                             "next to the y_Ag = 0 and y_Al = 0 site-fraction limits, where G "
                             "is logarithmic and a wide quadratic fit degrades)")
    parser.add_argument("--npts", type=int, default=21,
                        help="resampling points per axis within that radius (odd)")
    parser.add_argument("--max-curvature", type=float, default=None,
                        help="cap the eigenvalues of A at this value (J/mol)")
    parser.add_argument("--save", action="store_true",
                        help="write the plotly figure to HTML instead of opening a browser")
    args = parser.parse_args()
    if args.npts % 2 == 0:
        parser.error("--npts must be odd so the grid contains its centre")

    db = load_database(args.tdb)
    energies = {p: PhaseEnergy(db, p) for p in EUTECTIC_PHASES}

    # 1) eutectic
    xdists = {"LIQUID": args.xdist_liquid, **{p: args.xdist_solid for p in SOLIDS}}
    for item in args.xdist_phase:
        name, r = item.split("=")
        if name not in xdists:
            parser.error(f"--xdist-phase: unknown phase {name!r}")
        xdists[name] = float(r)

    T_E, comps_E = locate_eutectic(db, args.x0, args.tol)
    T_E = round(T_E, 2)
    T = T_E - args.dT
    query = comps_E["LIQUID"]

    # 2) ammber hull at T
    system, hull_eq = hull_equilibrium(db, T, query, args.n)

    # 3) refined triangle and solid fits
    equilibrium, mu = refined_equilibrium(db, energies, T, query)
    potential, intercept = equilibrium.get_tangent_plane()
    for name, comp, _ in equilibrium:
        hull_comp = next(c for n, c, _ in hull_eq if base_name(n) == name)
        print(f"  {name:7s} hull vs pycalphad: |dx| = {np.abs(hull_comp - comp).max():.4f}"
              f" (grid spacing {1 / args.n:.4f})")
    near_eq, quadratic = fit_solids(energies, equilibrium, T, xdists, args.npts)

    # 4) liquid against the same plane
    liq_data, liq_phase, offset_tdb = fit_liquid(
        energies["LIQUID"], equilibrium, T, query, args.xdist_liquid, args.npts)
    quadratic.phases = {"LIQUID": liq_phase, **quadratic.phases}
    near_eq.phases = {"LIQUID": liq_data, **near_eq.phases}
    uncapped = {name: p.A.copy() for name, p in quadratic.phases.items()}

    # 5) curvature cap
    capped = {}
    if args.max_curvature is not None:
        capped = cap_curvature(quadratic, args.max_curvature)

    # 6) checks
    errors = fit_errors(quadratic, near_eq, energies, T, potential, intercept)
    report(quadratic, errors, capped, args.max_curvature)

    print(f"\nLiquid driving force at T = {T:.2f} K (dT = {args.dT:g} K below T_E):")
    pot_mu, int_mu = tangent_plane_from_mu(mu)
    c_L = liq_phase.c_ref
    df_tdb_mu = float(energies["LIQUID"](c_L[None], T)[0] - (c_L @ pot_mu + int_mu))
    df_fit = liq_phase.C - (c_L @ potential + intercept)
    print(f"  parallel-tangent liquid composition c_ref: {fmt(c_L)}"
          f" (eutectic liquid at T_E: {fmt(query)})")
    print(f"  G_L - plane at c_ref: TDB {offset_tdb:.3f} J/mol "
          f"(plane from pycalphad mu: {df_tdb_mu:.3f}), quadratic {df_fit:.3f} J/mol")
    dG_E_tdb = float(energies["LIQUID"](query[None], T)[0] - (query @ potential + intercept))
    dG_E_fit = float(liq_phase.free_energy(query) - (query @ potential + intercept))
    print(f"  G_L - plane at the eutectic composition: TDB {dG_E_tdb:.3f}, "
          f"quadratic {dG_E_fit:.3f} J/mol")

    check = check_fitted_equilibrium(quadratic, query)
    print(f"\nEquilibrium of the fitted quadratic system at {fmt(query)}:")
    ok = True
    for name, comp, G in check:
        ref = next(c for n, c, _ in equilibrium if n == base_name(name))
        dev = np.abs(comp - ref).max()
        ok &= dev < 1e-3
        print(f"  {name:7s} {fmt(comp)}  |dx| vs TDB = {dev:.5f}")
    if sorted(base_name(n) for n, _, _ in check) != sorted(SOLIDS) or not ok:
        print("  WARNING: fitted system does not reproduce the TDB three-solid triangle")
    else:
        print("  reproduces the TDB three-solid triangle (within the 2.5e-4 check grid)")

    write_json(quadratic, T)

    from plot_fit import plot_cuts, plot_fit
    png = plot_cuts(quadratic, uncapped, energies, T, potential, intercept,
                    xdists, Path(f"quadratic_fit_T{T:.1f}_cuts.png"))
    print(f"Saved figure to {png}")
    fig = plot_fit(system, equilibrium, near_eq, quadratic, check, query, T,
                   potential, intercept, xdists)
    if args.save:
        html_path = Path(f"quadratic_fit_T{T:.1f}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Saved figure to {html_path}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
