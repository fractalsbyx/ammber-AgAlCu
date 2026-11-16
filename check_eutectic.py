import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from pycalphad import Database, equilibrium, variables as v
db = Database(sys.argv[1])
comps = ['AG','AL','CU','VA']
phases = [p for p in db.phases if p != 'GAS']
print("phases:", phases)
xag, xcu = 0.18, 0.14   # literature ternary eutectic approx (at.)
for T in [770, 775, 778, 780, 785, 790]:
    eq = equilibrium(db, comps, phases, {v.T:T, v.P:101325, v.N:1, v.X('AG'):xag, v.X('CU'):xcu})
    ph = eq.Phase.values.squeeze(); npf = eq.NP.values.squeeze()
    X = eq.X.values.squeeze()
    out = []
    for i,p in enumerate(ph):
        if p: out.append(f"{p}:{npf[i]:.2f} (Ag {X[i][0]:.3f}, Cu {X[i][2]:.3f})")
    print(T, " | ".join(out))
