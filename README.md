# Al-Ag-Cu ternary eutectic: parabolic free energies for PRISMS-PF

Fits parabolic (quadratic) free energies to the CALPHAD description of the
Al-Ag-Cu ternary eutectic

    LIQUID -> alpha-Al (FCC_A1) + Ag2Al (HCP_A3) + Al2Cu (THETA)

with ammber and writes them as PRISMS-PF phase JSON for an
isothermal, isotropic phase-field run from an undercooled melt.

## Usage

Python env: pycalphad 0.11, ammber 0.1.8, numpy, scipy, plotly, matplotlib.

```bash
python fit_quadratic.py --save
```

```bash
python fit_quadratic.py --dT 2 --max-curvature 2e5 --save
```

Main options (`python fit_quadratic.py -h` lists them all):

| option | default | meaning |
|---|---|---|
| `--dT` | 2 | undercooling below T_E at which the fit is made (K) |
| `--max-curvature` | off | cap the eigenvalues of every A (J/mol); c_ref, B, C unchanged |
| `--xdist-solid` | 0.01 | fit radius around each solid's equilibrium composition |
| `--xdist-liquid` | 0.03 | fit radius around the liquid composition |
| `--xdist-phase` | `THETA=0.005` | per-phase fit radius overrides |
| `--n` | 100 | site-fraction grid for the ammber hull of all TDB phases |
| `--tol` | 0.02 | bisection tolerance for T_E (K) |
| `--save` | | write the plotly figure to HTML instead of opening a browser |

`check_eutectic.py` is the quick pycalphad scan used to bracket T_E:
`python check_eutectic.py Ag-Al-Cu.TDB`.

## What the pipeline does

1. **T_E.** Bisection on "is there liquid?" at an overall composition inside the
   three-solid triangle (`--x0`, default Ag 0.18, Cu 0.14) gives
   T_E = 773.42 K (literature about 773 K). Phase compositions at T_E (mole fractions):

   | phase | Ag | Cu | Al |
   |---|---|---|---|
   | LIQUID | 0.1823 | 0.1276 | 0.6900 |
   | FCC_A1 (alpha-Al) | 0.1833 | 0.0452 | 0.7716 |
   | HCP_A3 (Ag2Al) | 0.5846 | 0.0294 | 0.3860 |
   | THETA (Al2Cu) | 0.0004 | 0.3229 | 0.6767 |

2. **ammber hull.** At T = T_E - dT every TDB phase except GAS is sampled over
   all its sublattice site fractions (`tdb_sampling.sample_phase`; internal degrees
   of freedom are reduced to the lowest G per composition) into an
   `IsothermalDiscreteSystem` with solvent Al and independent components Ag, Cu.
   The hull equilibrium at the eutectic liquid composition is the three-solid
   triangle; the script stops if it is not.
3. **Solids.** The triangle is refined with pycalphad (the hull is limited to
   its grid spacing), each solid is resampled on a fine grid around its
   equilibrium composition, and `IsothermalQuadraticSystem.from_discrete` fits A
   with c_ref at the equilibrium composition and the common tangent plane as B, C.
4. **Liquid.** The liquid is fitted against the same tangent plane: c_ref is
   where G_L - plane is smallest (parallel tangent), B is the shared plane slope,
   and C - plane(c_ref) is the TDB driving force for solidification at this
   undercooling (29.2 J/mol at dT = 2 K).
5. **Curvature cap** (`--max-curvature`). Eigenvalues of A above the cap are
   clipped. c_ref, B and C are untouched, so the tangent plane, the three-phase
   equilibrium and the liquid driving force stay exact; only the curvature (and
   so the fit away from c_ref) changes.
6. **Checks** printed by the script: fit error vs the TDB on the fit points,
   the equilibrium of the fitted quadratic system at the eutectic composition
   (must reproduce the three solid compositions), and the liquid offset vs the
   TDB driving force computed from pycalphad's chemical potentials.

## Results at T = 771.42 K (dT = 2 K, uncapped)

| phase | eigenvalues of A (J/mol) | max / rms error vs TDB on fit points (J/mol) |
|---|---|---|
| LIQUID | 2.3e4, 1.1e5 | 0.57 / 0.18 (radius 0.03) |
| FCC_A1 | 2.3e3, 8.1e4 | 0.20 / 0.07 (radius 0.01) |
| HCP_A3 | 2.1e5, 1.4e6 | 0.71 / 0.27 (radius 0.01) |
| THETA | 8.8e5, 6.8e6 | 3.3 / 1.5 (radius 0.005) |

THETA is (Al)2(Ag,Al,Cu)1 and its equilibrium composition sits next to the
y_Ag = 0 and y_Al = 0 site-fraction limits, where G is logarithmic: no quadratic
fits it well over a wide radius, hence the smaller default radius. HCP_A3 and
THETA are stiff (up to 7e6 J/mol); `--max-curvature` trades fit accuracy away
from c_ref for a less stiff phase-field problem.

## Outputs

- `quadratic_fit_T<T>.json`: PRISMS-PF phase JSON in the same format as the
  Ga-In-Pb example (`solution_component`, `components`, phases with `c_ref`, `A`,
  `B`, `C`, `D`). Energies are J/mol of atoms in the TDB's SER reference, with
  `x = (x_Ag, x_Cu)`. Phase names are the TDB names (LIQUID, FCC_A1, HCP_A3,
  THETA). `D` is a **placeholder**: 2e-9 m^2/s for the liquid and 1e-15 m^2/s
  (effectively frozen) for the solids. An existing JSON is updated in place, so
  edited `D` entries are kept.
- `quadratic_fit_T<T>.html`: plotly figure, ternary view of the hull, fit points and
  equilibrium triangle, and G - tangent plane in 3D (data and fitted quadratics).
- `quadratic_fit_T<T>_cuts.png`: matplotlib cuts through each c_ref along the
  eigenvectors of A (TDB, quadratic, capped quadratic).

## Files

- `fit_quadratic.py`: the pipeline (CLI)
- `tdb_sampling.py`: pycalphad sampling, single-phase G(x), equilibrium and T_E search
- `plot_fit.py`: plotly and matplotlib figures
- `check_eutectic.py`: quick pycalphad equilibrium scan
- `Ag-Al-Cu.TDB`: thermodynamic database

## Database source and license

`Ag-Al-Cu.TDB` is the assessment of Witusiewicz et al.,
J. Alloys Compd. 387 (2005) 217. The file was taken from the
MaterialsMap GitHub mirror (`mfkiwl/MaterialsMap`, `tasks/Ag-Al-Cu.TDB`). Its
license is **not stated** there; check the terms before redistributing the
TDB, and cite the paper when using results from it.

## Limits

- Isothermal fit at one temperature. Directional solidification with
  temperature-dependent coefficients would need fits at several temperatures.
- The quadratic of each phase is accurate only near its c_ref (see the fit radii);
  far from it, especially for the capped phases, it is not the TDB energy.
