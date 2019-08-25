
# nomopyomo - no more pyomo

This script takes an unsolved [PyPSA](https://github.com/PyPSA/PyPSA)
network and solves a linear optimal power flow (LOPF) using a custom
handler for the optimisation solver, rather than pyomo.

nomopyomo is both faster than pyomo and uses considerably less
memory. Here is an example of the memory usage when solving
investments and operation for a zero-net-emission sector-coupled
50-node model of Europe:

![pyomo-nomopyomo comparison](https://www.nworbmot.org/pyomo-versus-nomopyomo-190825.png)

However, nomopyomo is harder to customise and cannot currently be used
for non-linear problems.

nomopyomo works by writing an .lp file for the problem, solving it, and
reading back in the solution.

The script currently works for Load, Generator, Link, Line,
Transformer, Store and GlobalConstraint components, and with the
solvers cbc/clp and gurobi.

It has been tested against the standard PyPSA examples.

TODO:

- implement glpk solver
- allow extra functionality
- constant term in objective function
- handle non-optimal solutions
- extract dual variables
- extract voltage angles

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
