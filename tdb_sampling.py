"""
pycalphad helpers for the Al-Ag-Cu database (Witusiewicz et al. 2005).

Composition coordinates follow ammber: independent components (Ag, Cu), solvent Al.
A composition is the array x = (x_Ag, x_Cu); x_Al = 1 - x_Ag - x_Cu. Energies are
molar Gibbs energies (J/mol of atoms) with the database's SER reference.

  sample_phase       grid over every sublattice's site fractions (internal degrees
                     of freedom included), reduced to the lowest G per composition
  build_system       all phases sampled into an ammber IsothermalDiscreteSystem
  PhaseEnergy        G(x) of one phase at given compositions (phases whose site
                     fractions follow uniquely from x, e.g. LIQUID, FCC_A1, HCP_A3,
                     THETA = (Al)2(Ag,Al,Cu)1)
  equilibrium_at     pycalphad equilibrium at (T, x)
  find_eutectic      bisection for the ternary eutectic temperature
"""

import itertools
import warnings
from functools import lru_cache

import numpy as np
from pycalphad import Database, Model, calculate, equilibrium, variables as v

from ammber.Systems import IsothermalDiscreteSystem

COMPONENTS = ["AG", "CU"]          # independent components (pycalphad names)
SOLVENT = "AL"
ELEMENTS = COMPONENTS + [SOLVENT, "VA"]
LABELS = {"AG": "Ag", "CU": "Cu", "AL": "Al"}
P = 101325.0

# Eutectic phases in the TDB and what they are
EUTECTIC_PHASES = ["LIQUID", "FCC_A1", "HCP_A3", "THETA"]
PHASE_DESCRIPTION = {
    "LIQUID": "liquid",
    "FCC_A1": "alpha-Al (fcc)",
    "HCP_A3": "Ag2Al (hcp, zeta)",
    "THETA": "Al2Cu (theta)",
}
SOLIDS = ["FCC_A1", "HCP_A3", "THETA"]


@lru_cache(maxsize=None)
def load_database(path):
    return Database(str(path))


def stable_phase_names(db):
    """All phases of the database except the gas."""
    return sorted(p for p in db.phases if p != "GAS")


def _simplex_grid(k, n):
    """All points of a regular n-division grid on the (k-1)-simplex, shape (m, k)."""
    if k == 1:
        return np.ones((1, 1))
    pts = [c for c in itertools.product(range(n + 1), repeat=k - 1) if sum(c) <= n]
    pts = np.array(pts, dtype=float) / n
    return np.column_stack([pts, 1 - pts.sum(axis=1)])


def _sublattice_species(db, phase):
    """Constituents of each sublattice, in pycalphad's internal (sorted) order."""
    mod = Model(db, ELEMENTS, phase)
    return [sorted(str(s.name) for s in sub) for sub in mod.constituents]


def sample_phase(db, phase, T, n=100, max_points=60000):
    """
    Sample a phase over all its sublattice site fractions at temperature T.

    Each sublattice gets a regular simplex grid; the grids are combined as a
    Cartesian product (coarsened until the product has at most max_points
    points). Points that map to the same overall composition keep the lowest G,
    which is the phase's free energy after minimising over internal degrees of
    freedom. Returns x (npts, 2) and G (npts,).
    """
    species = _sublattice_species(db, phase)
    sizes = [len(s) for s in species]
    m = n
    while True:
        grids = [_simplex_grid(k, m) for k in sizes]
        total = np.prod([len(g) for g in grids])
        if total <= max_points or m <= 4:
            break
        m = int(m * 0.8)
    combos = [np.concatenate(parts) for parts in itertools.product(*grids)]
    points = np.array(combos)
    res = calculate(db, ELEMENTS, phase, P=P, T=T, points=points, output="GM")
    G = res.GM.values.reshape(-1)
    X = np.column_stack([res.X.sel(component=c).values.reshape(-1) for c in COMPONENTS])
    ok = np.isfinite(G) & np.all(np.isfinite(X), axis=1)
    X, G = X[ok], G[ok]
    # Lowest G for each (rounded) composition
    key = np.round(X, 10)
    order = np.lexsort((G, key[:, 1], key[:, 0]))
    key, X, G = key[order], X[order], G[order]
    first = np.ones(len(G), dtype=bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    return X[first], G[first]


def build_system(db, T, n=100, phases=None):
    """Sample every phase (default: all but GAS) into an ammber discrete system."""
    system = IsothermalDiscreteSystem(components=COMPONENTS, solution_component=SOLVENT)
    for phase in phases or stable_phase_names(db):
        x, G = sample_phase(db, phase, T, n)
        system.add_phase(phase, x, G, T=T)
    return system


class PhaseEnergy:
    """
    G(x) for a phase whose site fractions are fixed by the overall composition:
    every sublattice holds one species except one mixing sublattice (vacancy-only
    sublattices are ignored). Compositions outside the phase's range give NaN.
    """

    def __init__(self, db, phase):
        self.db, self.phase = db, phase
        species = _sublattice_species(db, phase)
        ratios = [float(r) for r in db.phases[phase].sublattices]
        self.layout = list(zip(species, ratios))
        atoms = [(s, a) for s, a in self.layout if s != ["VA"]]
        mixing = [i for i, (s, a) in enumerate(self.layout) if len(s) > 1]
        if len(mixing) != 1 or "VA" in self.layout[mixing[0]][0]:
            raise ValueError(f"{phase}: site fractions are not a unique function of x")
        self.mixing = mixing[0]
        self.n_atoms = sum(a for _, a in atoms)
        self.fixed = {e: 0.0 for e in COMPONENTS + [SOLVENT]}
        for i, (s, a) in enumerate(self.layout):
            if i != self.mixing and s != ["VA"]:
                self.fixed[s[0]] += a

    def site_fractions(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        xfull = {"AG": x[:, 0], "CU": x[:, 1], "AL": 1 - x[:, 0] - x[:, 1]}
        cols = []
        for i, (species, a) in enumerate(self.layout):
            for s in species:
                if i == self.mixing:
                    cols.append((xfull[s] * self.n_atoms - self.fixed[s]) / a)
                else:
                    cols.append(np.ones(len(x)))
        y = np.column_stack(cols)
        # Elements absent from the mixing sublattice must match the fixed part exactly
        mix_species = self.layout[self.mixing][0]
        valid = np.all(y >= -1e-12, axis=1)
        for e in COMPONENTS + [SOLVENT]:
            if e not in mix_species:
                valid &= np.abs(xfull[e] * self.n_atoms - self.fixed[e]) < 1e-9
        return np.clip(y, 0, 1), valid

    def __call__(self, x, T):
        y, valid = self.site_fractions(x)
        G = np.full(len(y), np.nan)
        if valid.any():
            res = calculate(self.db, ELEMENTS, self.phase, P=P, T=T, points=y[valid],
                            output="GM")
            G[valid] = res.GM.values.reshape(-1)
        return G


def equilibrium_at(db, T, x, phases=None):
    """
    pycalphad equilibrium at temperature T and composition x = (x_Ag, x_Cu).

    Returns (stable, mu): stable is a list of dicts with 'phase', 'NP', 'x' for
    each phase present; mu maps element -> chemical potential (J/mol).
    """
    phases = phases or stable_phase_names(db)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, ELEMENTS, phases,
                         {v.T: T, v.P: P, v.N: 1, v.X("AG"): x[0], v.X("CU"): x[1]})
    names = eq.Phase.values.squeeze()
    NP = eq.NP.values.squeeze()
    X = eq.X.values.squeeze()
    comps = list(eq.component.values)
    stable = []
    for i, name in enumerate(names):
        if name and NP[i] > 1e-10:
            stable.append(dict(phase=str(name), NP=float(NP[i]),
                               x=np.array([X[i][comps.index(c)] for c in COMPONENTS])))
    mu = {c: float(m) for c, m in zip(comps, eq.MU.values.squeeze())}
    return stable, mu


def tangent_plane_from_mu(mu):
    """Plane G = x . potential + intercept in (x_Ag, x_Cu) coordinates."""
    potential = np.array([mu[c] - mu[SOLVENT] for c in COMPONENTS])
    return potential, mu[SOLVENT]


def find_eutectic(db, x, T_lo=760.0, T_hi=785.0, tol=0.05, verbose=True):
    """
    Bisect for the ternary eutectic temperature at an overall composition x
    inside the three-solid triangle: below T_E there is no liquid, above it
    there is. Returns T_E and the four phase compositions at T_E: the liquid
    from the equilibrium just above T_E (its composition there converges to the
    eutectic point as tol -> 0) and the three solids from just below T_E.
    """
    def has_liquid(T):
        stable, _ = equilibrium_at(db, T, x)
        return any(s["phase"] == "LIQUID" for s in stable)

    if has_liquid(T_lo) or not has_liquid(T_hi):
        raise RuntimeError(f"eutectic not bracketed by [{T_lo}, {T_hi}] K at x={x}")
    while T_hi - T_lo > tol:
        T_mid = 0.5 * (T_lo + T_hi)
        if has_liquid(T_mid):
            T_hi = T_mid
        else:
            T_lo = T_mid
        if verbose:
            print(f"  T_E in [{T_lo:.3f}, {T_hi:.3f}] K")
    above, _ = equilibrium_at(db, T_hi, x)
    below, _ = equilibrium_at(db, T_lo, x)
    compositions = {s["phase"]: s["x"] for s in below}
    compositions.update({s["phase"]: s["x"] for s in above if s["phase"] == "LIQUID"})
    return 0.5 * (T_lo + T_hi), compositions, (T_lo, T_hi)
