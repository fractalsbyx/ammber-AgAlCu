"""
Combine a quadratic fit with the phase-field parameters into the system JSON
read by AMMBER-PRISMS-PF's QuadraticSystem.

From the fit (e.g. quadratic_fit_grid_mask_n400_r0.005_T771.4.json) only the
fitted quantities are taken: solution_component, components,
convert_fractional_to_volumetric_energy, and each phase's name, c_ref, A, B, C.
Everything else comes from the parameter file (phase-field/system_parameters.json):
dimensions, Vm, l_int, sigma, mu_int, order_parameters, and each phase's c0
and D.

The phases are written in the parameter file's "phase_order", which is the
row/column order of its sigma and mu_int matrices, so the matrices are used
as they are.

Usage: python combine_system.py [--fit FIT.json] [--params PARAMS.json] [--out OUT.json]
"""

import argparse
import json

FIT_SYSTEM_KEYS = ["solution_component", "components", "convert_fractional_to_volumetric_energy"]
FIT_PHASE_KEYS = ["c_ref", "A", "B", "C"]
PARAM_SYSTEM_KEYS = ["dimensions", "Vm", "l_int", "sigma", "mu_int", "order_parameters"]
PARAM_PHASE_KEYS = ["c0", "D"]


def fmt(obj, indent=0):
    """JSON with numeric vectors and matrix rows on one line."""
    pad, pad1 = "  " * indent, "  " * (indent + 1)
    if isinstance(obj, dict):
        items = [f"{pad1}{json.dumps(k)}: {fmt(v, indent + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return "[" + ", ".join(json.dumps(v) for v in obj) + "]"
        return "[\n" + ",\n".join(pad1 + fmt(v, indent + 1) for v in obj) + "\n" + pad + "]"
    return json.dumps(obj)


def by_name(phases, source):
    out = {}
    for phase in phases:
        if phase["name"] in out:
            raise ValueError(f"{source}: duplicate phase {phase['name']!r}")
        out[phase["name"]] = phase
    return out


def combine(fit, params):
    order = params["phase_order"]
    fit_phases = by_name(fit["phases"], "fit")
    param_phases = by_name(params["phases"], "parameters")

    for source, names in (("fit", fit_phases), ("parameters", param_phases)):
        if sorted(names) != sorted(order):
            raise ValueError(f"{source} phases {sorted(names)} do not match "
                             f"phase_order {sorted(order)}")
    for key in ("sigma", "mu_int"):
        matrix = params[key]
        if len(matrix) != len(order) or any(len(row) != len(order) for row in matrix):
            raise ValueError(f"{key} must be {len(order)}x{len(order)} (phase_order)")
    unknown = set(params["order_parameters"]) - set(order)
    if unknown:
        raise ValueError(f"order_parameters not in phase_order: {sorted(unknown)}")
    n_comps = len(fit["components"])
    for name, phase in param_phases.items():
        if len(phase["c0"]) != n_comps:
            raise ValueError(f"{name}: c0 must have {n_comps} entries")

    system = {key: params[key] for key in ("dimensions",)}
    system.update({key: fit[key] for key in FIT_SYSTEM_KEYS})
    system.update({key: params[key] for key in ("Vm", "l_int")})
    system["phases"] = [
        {"name": name,
         **{key: fit_phases[name][key] for key in FIT_PHASE_KEYS},
         **{key: param_phases[name][key] for key in PARAM_PHASE_KEYS}}
        for name in order
    ]
    system.update({key: params[key] for key in ("sigma", "mu_int", "order_parameters")})
    system["phase_order"] = order
    if "notes" in params:
        system["notes"] = params["notes"]
    return system


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default="quadratic_fit_grid_mask_n400_r0.005_T771.4.json")
    parser.add_argument("--params", default="phase-field/system_parameters.json")
    parser.add_argument("--out", default="phase-field/system.json")
    args = parser.parse_args()

    with open(args.fit) as f:
        fit = json.load(f)
    with open(args.params) as f:
        params = json.load(f)
    system = combine(fit, params)
    with open(args.out, "w") as f:
        f.write(fmt(system) + "\n")
    print(f"Saved {args.out}: phases {system['phase_order']} from {args.fit} + {args.params}")


if __name__ == "__main__":
    main()
