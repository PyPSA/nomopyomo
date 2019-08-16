
# nomopyomo - no more pyomo

This script takes an unsolved [PyPSA](https://github.com/PyPSA/PyPSA)
network and solves a linear optimal power flow (LOPF) using a custom
handler for the optimisation solver, rather than pyomo.

The script builds:

- `network.variables`, a pandas.DataFrame of variables along with their upper and lower bounds
- `network.constraints`, a pandas.DataFrame of constraints along with their sense (<=,>=,==) and constant right-hand-side terms
- `network.constraint_matrix`, a dictionary of the sparse constraint-variable coefficients

Then writes a .lp file for the problem, solves it, and reads back in
the solution.

The script currently works for Load, Generator, Link, Line,
Transformer, Store and GlobalConstraint components, and with the
solver cbc/clp.

TODO:

- implement gurobi solver
- implement glpk solver
- allow extra functionality
- constant term in objective function
- handle non-optimal solutions
- extract dual variables

No planned support for StorageUnit (replace with Store and Links
following [this
example](https://pypsa.org/examples/replace-generator-storage-units-with-store.html)).

# Usage

The usage is similar to PyPSA's `network.lopf()`:

```python
import nomopyomo

nomopyomo.network_lopf(network, solver_name="cbc")
```



# Licence


Copyright 2019 Tom Brown (KIT)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either [version 3 of the
License](./LICENSE.txt), or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU
General Public License](./LICENSE.txt) for more details.
