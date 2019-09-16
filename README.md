
# nomopyomo - no more pyomo

This script takes an unsolved [PyPSA](https://github.com/PyPSA/PyPSA)
network and solves an investment and operation linear optimisation
problem with power flow (LOPF) using a custom handler for the
optimisation solver, rather than the [pyomo](http://www.pyomo.org)
optimisation framework used by PyPSA.

nomopyomo is both faster than pyomo and uses considerably less
memory.

Here is an example of the memory usage when solving investments and
operation for a zero-net-emission sector-coupled 50-node model of
Europe 3-hourly for a year (9 million variables, 5 milion constraints,
22 million non-zero values in the constraint matrix). Pyomo takes up
more than three quarters of the total memory used, whereas nomopyomo
is so memory-efficient that the solver gurobi dominates the memory
usage.

![pyomo-nomopyomo comparison](https://www.nworbmot.org/pyomo-versus-nomopyomo-190828-1100.png)

The final memory surge can probably be eliminated by managing the
memory in pyomo's GUROBI_RUN.py script better (nomopyomo uses this
script to run gurobi and return the values of dual variables).

The fact that the gurobi solving time is shorter with nomopyomo than
pyomo is not a general feature, but a random speed-up for this problem
that comes from a different ordering of the variables and
constraints. It will not hold for all problems.

The results are identical (within the solver tolerances) and have been
tested with a variety of standard PyPSA test cases.

On the negative side, nomopyomo is harder to customise and cannot
currently be used for non-linear problems.

nomopyomo works by writing an .lp file for the problem, solving it, and
reading back in the solution.

The script currently works for Load, Generator, Link, Line,
Transformer, Store and GlobalConstraint components, and with the
solvers cbc/clp and gurobi.

It has been tested against the standard PyPSA examples.

TODO:

- constant term in objective function
- handle non-optimal solutions
- calculate nodal imbalances
- calculate voltage angles
- implement glpk solver
- logfile for cbc

No planned support for StorageUnit (replace with Store and Links
following [this
example](https://pypsa.org/examples/replace-generator-storage-units-with-store.html)).

# Usage

The usage is similar to PyPSA's `network.lopf()`:

```python
import nomopyomo

nomopyomo.network_lopf(network, solver_name="cbc")
```

# Customisation

Just like PyPSA's `network.lopf()` you can add `extra_functionality`
and `extra_postprocessing` arguments, however the code must be
modified from the PyPSA case to use nomopyomo instead of pyomo. For an
example, see the [changes made for the model.energy
code](https://github.com/PyPSA/whobs-server/commit/0908ed45d7758bb75f2f52ad028a170093e1a8a0).



# How it works

nomopyomo gives each variable and constraint a unique integer label,
then writes the linear objective function, variable bounds and
constraints to a .lp problem file. It is solved, then the result is
read back in. nomopyomo stores very little in memory beyond the
original pypsa.Network.

The integer assignments are determined by an implicit ordering of the
variables and constraints. Within each group (shown below), the
variables are indexed in order by several index sets. The start and
finish integers for each group are stored in the pandas.DataFrames
`network.variable_positions` and `network.constraint_positions`.

The variables are organised into the following groups:

| group name | variables | index by |
| --- | --- | --- |
| Generator-p | generator dispatch | network.generators.index, snapshots |
| Generator-p_nom | extendable generator capacity | network.generators.index[network.generator.p_nom_extendable] |
| Link-p | link dispatch | network.links.index, snapshots |
| Link-p_nom | extendable link capacity | network.links.index[network.link.p_nom_extendable] |
| Store-p | store dispatch | network.stores.index, snapshots |
| Store-e | store state of charge | network.stores.index, snapshots |
| Store-e_nom | extendable store capacity | network.stores.index[network.store.e_nom_extendable] |
| Line-p | line dispatch | network.lines.index, snapshots |
| Line-s_nom | extendable line capacity | network.lines.index[network.line.s_nom_extendable] |
| Transformer-p | transformer dispatch | network.transformers.index, snapshots |
| Transformer-s_nom | extendable transformer capacity | network.transformers.index[network.transformer.s_nom_extendable] |



The constraints are organised into the following groups:

| group name | constraints | index by |
| --- | --- | --- |
| Generator-p_lower | dispatch limit for extendable generators | network.generators.index[network.generator.p_nom_extendable], snapshots |
| Generator-p_upper | dispatch limit for extendable generators | network.generators.index[network.generator.p_nom_extendable], snapshots |
| etc for other components | |
| Cycle | Kirchhoff Voltage Law for passive branches | cycles, snapshots |
| Store | store state of charge consistency | network.stores.index, snapshots |
| nodal_balance | energy conservation at each bus | network.buses.index, snapshots |
| global_constraints | constraints on e.g. CO2 emissions | network.global_constraints.index |

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
