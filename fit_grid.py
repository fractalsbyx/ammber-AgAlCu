"""
Minimal ammber fit of the Al-Ag-Cu ternary eutectic for PRISMS-PF, using
fromTDB's default grid sampling.

Same method as fit_simple.py: only the eutectic composition and the phase names
are needed. The phases are sampled on a regular site-fraction grid, the lower
convex hull gives the three-solid equilibrium below T_E, and from_discrete fits
every phase (the solids and the metastable liquid) against that common
tangent plane. The fit points around each phase are either
  resample: interpolated onto a regular composition grid (resample_near_*), or
  mask:     the original grid points themselves (mask_near_*).

Usage: python fit_grid.py [--select resample|mask] [--n 400] [--T 771.4] [--xdist 0.02]
"""

import argparse
import json

import numpy as np
from pycalphad import Database

from ammber.Systems import IsothermalDiscreteSystem, IsothermalQuadraticSystem

X_EUTECTIC = [0.1823, 0.1276]                     # (x_Ag, x_Cu) of the eutectic liquid
PHASES = ["LIQUID", "FCC_A1", "HCP_A3", "THETA"]  # liquid, alpha-Al, Ag2Al, Al2Cu

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--select", choices=["resample", "mask"], default="mask",
                    help="how the fit points near each phase are chosen")
parser.add_argument("--n", type=int, default=400, help="site-fraction grid divisions")
parser.add_argument("--T", type=float, default=771.4, help="temperature, below T_E = 773.4 K")
parser.add_argument("--xdist", type=float, default=0.02, help="fit radius around each phase")
args = parser.parse_args()

system = IsothermalDiscreteSystem()
# max_points large enough that no single-sublattice ternary phase is coarsened
system.fromTDB(Database("Ag-Al-Cu.TDB"), ["AG", "CU"], "AL", args.T, phase_list=PHASES,
               n=args.n, max_points=(args.n + 1) * (args.n + 2) // 2)

# Below T_E the hull at the eutectic composition is the three-solid triangle
equilibrium = system.get_equilibrium(X_EUTECTIC)
print("Equilibrium:", [(name, np.round(x, 4).tolist()) for name, x, _ in equilibrium])

# Fit the solids near their equilibrium compositions and the liquid near the
# eutectic composition, all against the same tangent plane (shared B)
liquid = system.phases["LIQUID"]
if args.select == "resample":
    data = system.resample_near_equilibrium(X_EUTECTIC, xdist=args.xdist)
    data.phases["LIQUID"] = liquid.resample_near_xpoint(X_EUTECTIC, xdist=args.xdist)
else:
    data = system.mask_near_equilibrium(X_EUTECTIC, xdist=args.xdist)
    data.phases["LIQUID"] = liquid.mask_near_xpoint(X_EUTECTIC, xdist=args.xdist)
quadratic = IsothermalQuadraticSystem().from_discrete(data, equilibrium=equilibrium)

potential, intercept = equilibrium.get_tangent_plane()
for name, phase in quadratic.phases.items():
    err = phase.free_energy(data.phases[name].xdata) - data.phases[name].Gdata
    print(f"{name}: {len(data.phases[name].Gdata)} points, "
          f"c_ref={np.round(phase.c_ref, 4).tolist()}, "
          f"eig(A)={np.round(np.linalg.eigvalsh(phase.A), -2).tolist()}, "
          f"G-plane at c_ref={phase.C - phase.c_ref @ potential - intercept:.1f} J/mol, "
          f"max fit error={np.abs(err).max():.1f} J/mol")

out = quadratic.add_to_dict({})
out["solution_component"], out["components"] = "Al", ["Ag", "Cu"]
for name, entry in out["phases"].items():  # placeholder diffusivities (m^2/s); solids frozen
    entry["D"] = ((2e-9 if name == "LIQUID" else 1e-15) * np.eye(2)).tolist()
path = f"quadratic_fit_grid_{args.select}_n{args.n}_r{args.xdist:g}_T{args.T:g}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("Saved", path)
