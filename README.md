
# nomopyomo - no more pyomo

This script takes an unsolved PyPSA network and solves a linear
optimal power flow (LOPF) using a custom handler for the optimisation
solver, rather than pyomo.

The script builds:

- `network.variables`, a pandas.DataFrame of variables along with their upper and lower bounds
- `network.constraints`, a pandas.DataFrame of constraints along with their sense (<=,>=,==) and constant right-hand-side terms
- `network.constraint_matrix`, a scipy.sparse.coo_matrix for the constraint-variable coefficients

Then writes a .lp file for the problem, solves it, and reads back in
the solution.

The script currently works for Load, Generator and Link components,
and with the solver cbc/clp.

TODO:

- integrate Store component
- integrate Line component
- integrate kirchhoff formulation for LPF
- implement gurobi solver
- implement glpk solver