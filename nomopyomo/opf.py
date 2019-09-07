## Copyright 2019 Tom Brown (KIT), Fabian Hofmann (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""nomopyomo: build optimisation problems from PyPSA networks without
Pyomo. nomopyomo = no more Pyomo."""

from .opt import (get_as_dense, get_bounds_pu, get_extendable_i,
                  get_non_extendable_i, write_bound, write_constraint,
                  write_objective, numerical_to_string, set_conref, set_varref,
                  df_var, df_con, pnl_var, pnl_con, lookup, prefix)

from pypsa.pf import find_cycles as find_cycles, _as_snapshots


import pandas as pd
import numpy as np

import gc, string, random, time, pyomo, os

import logging
logger = logging.getLogger(__name__)


# =============================================================================
#  var and con defining functions
# =============================================================================

def define_nominal_for_extendable_variables(n, c):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    lbound = n.df(c)[f'{prefix[c]}_nom_min'][ext_i]
    ubound = n.df(c)[f'{prefix[c]}_nom_max'][ext_i]
    variables = write_bound(n, lbound, ubound)
    set_varref(n, variables, c, f'{prefix[c]}_nom', pnl=False)


def define_dispatch_for_extendable_variables(n, sns, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    lbound = pd.DataFrame(-np.inf, index=sns, columns=ext_i)
    ubound = pd.DataFrame(np.inf, index=sns, columns=ext_i)
    variables = write_bound(n, lbound, ubound)
    set_varref(n, variables, c, attr, pnl=True)


def define_dispatch_for_non_extendable_variables(n, sns, c, attr):
    fix_i = get_non_extendable_i(n, c)
    if fix_i.empty: return
    nominal_fix = n.df(c)[f'{attr}_nom'][fix_i]
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lbound = min_pu.mul(nominal_fix)
    ubound = max_pu.mul(nominal_fix)
    variables = write_bound(n, lbound, ubound)
    set_varref(n, variables, c, attr, pnl=True)


def define_dispatch_for_extendable_constraints(n, sns, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    min_pu, max_pu = get_bounds_pu(n, c, sns, ext_i, attr)
    operational_ext_v = pnl_var(n, c, attr)[ext_i]
    nominal_v = df_var(n, c, prefix[c] + '_nom')[ext_i]
    wo_prefactor = nominal_v + '\n' + '-1.0 ' +  operational_ext_v
    rhs = '0'

    lhs = numerical_to_string(max_pu) + wo_prefactor
    constraints = write_constraint(n, lhs, '>=', rhs)
    set_conref(n, constraints, c, 'mu_upper')

    lhs = numerical_to_string(min_pu) + wo_prefactor
    constraints = write_constraint(n, lhs, '<=', rhs)
    set_conref(n, constraints, c, 'mu_lower')


def define_nodal_balance_constraints(n, sns):

    def bus_injection(c, attr, groupcol='bus', sign=1):
        #additional sign only necessary for branches in reverse direction
        if 'sign' in n.df(c):
            sign = sign * n.df(c).sign
        return (numerical_to_string(sign) + pnl_var(n, c, attr))\
                .rename(columns=n.df(c)[groupcol])

    # one might reduce this a bit by using n.branches and lookup
    arg_list = [['Generator', 'p', 'bus', 1],
                ['Store', 'e', 'bus', 1],
                ['StorageUnit', 'p_dispatch', 'bus', 1],
                ['StorageUnit', 'p_store', 'bus', -1],
                ['Line', 's', 'bus0', -1],
                ['Line', 's', 'bus1', 1],
                ['Transformer', 's', 'bus0', -1],
                ['Transformer', 's', 'bus1', 1],
                ['Link', 'p', 'bus0', -1],
                ['Link', 'p', 'bus1', n.links.efficiency]]
    arg_list = [arg for arg in arg_list if not n.df(arg[0]).empty]

    lhs = (pd.concat([bus_injection(*args) for args in arg_list], axis=1)
           .groupby(axis=1, level=0)
           .agg(lambda x: '\n'.join(x.values)))
    sense = '='
    rhs = ((- n.loads_t.p_set * n.loads.sign)
           .groupby(n.loads.bus, axis=1).sum()
           .pipe(numerical_to_string, append_space=False)
           .reindex(columns=n.buses.index, fill_value='0.0'))
    constraints = write_constraint(n, lhs, sense, rhs)
    set_conref(n, constraints, 'Bus', 'nodal_balance')


def define_kirchhoff_constraints(n):
    n.calculate_dependent_values()
    n.determine_network_topology()
    n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)
    weightings = n.lines.x_pu_eff.where(n.lines.carrier == 'AC', n.lines.r_pu_eff)

    def cycle_flow(ds):
        ds = ds[lambda ds: ds!=0.].dropna()
        return (numerical_to_string(ds) +
                n.lines_t.s_varref[ds.index] + '\n').sum(1)

    constraints = []
    for sub in n.sub_networks.obj:
        find_cycles(sub)
        C = pd.DataFrame(sub.C.todense(), index=sub.lines_i())
        if C.empty:
            continue
        C_weighted = 1e5 * C.mul(weightings[sub.lines_i()], axis=0)
        con = write_constraint(n, C_weighted.apply(cycle_flow), '=', '0')
        constraints.append(con)
    constraints = pd.concat(constraints, axis=1, ignore_index=True)
    set_conref(n, constraints, 'Line', 'kirchhoff_voltage')


def define_store_constraints(n):
    return


def define_storage_units_constraints(n):
    return

def define_global_constraints(n, sns):

    def join_entries(df): return '\n'.join(df.values.flatten())

    glcs = n.global_constraints.query('type == "primary_energy"')
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]
        if emissions.empty: continue
        gens = n.generators.query('carrier in @emissions.index')
        em_pu = gens.carrier.map(emissions)/gens.efficiency
        em_pu = n.snapshot_weightings.to_frame() @ em_pu.to_frame('weightings').T
        lhs = em_pu.pipe(numerical_to_string) \
                   .add(pnl_var(n, 'Generator', 'p')[gens.index]) \
                   .pipe(join_entries)

        sus = n.storage_units.query('carrier in @emissions.index and '
                                    'not cyclic_state_of_charge')
        if not sus.empty:
            lhs += (sus.carrier.map(emissions).mul(sus.state_of_charge_initial)
                    .pipe(numerical_to_string).pipe(join_entries))
            lhs += (sus.carrier.map(n.emissions).mul(-1)
                    .pipe(numerical_to_string)
                    .add(pnl_var(n, 'StorageUnit', 'state_of_charge')
                        .loc[sns[-1], sus.index])
                    .pipe(join_entries))

        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        if not stores.empty:
            lhs += (stores.carrier.map(emissions).mul(stores.e_initial)
                    .pipe(numerical_to_string).pipe(join_entries))
            lhs += (sus.stores.map(n.emissions).mul(-1)
                    .pipe(numerical_to_string)
                    .add(pnl_var(n, 'Store', 'e').loc[sns[-1], stores.index])
                    .pipe(join_entries))

        glcs.loc[name, 'lhs'] = lhs
    constraints = write_constraint(n, glcs.lhs, glcs.sense.replace('==', '='),
                                   glcs.constant.pipe(numerical_to_string))
    set_conref(n, constraints, 'GlobalConstraint', 'mu', pnl=False)


def define_objective(n):
    for c, attr in lookup.query('marginal_cost').index:
        cost = (get_as_dense(n, c, 'marginal_cost')
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(n.snapshot_weightings, axis=0))
        if cost.empty: continue
        terms = numerical_to_string(cost) + pnl_var(n, c, attr)[cost.columns]
        write_objective(n, terms)
    #investment
    for c, attr in prefix.items():
        cost = n.df(c)['capital_cost'][get_extendable_i(n, c)]
        if cost.empty: continue
        terms = numerical_to_string(cost) + df_var(n, c, attr+'_nom')[cost.index]
        write_objective(n, terms)


def prepare_lopf(n, snapshots=None, keep_files=False,
                 extra_functionality=None):
    snapshots = n.snapshots if snapshots is None else snapshots
    time_log = pd.Series({'start': time.time()})
    n.storage_units = n.storage_units.eval('p_store_nom = p_nom')\
                                     .eval('p_dispatch_nom = p_nom')

    n.identifier = ''.join(random.choice(string.ascii_lowercase)
                        for i in range(8))
    n.objective_fn = "/tmp/objective-{}.txt".format(n.identifier)
    n.constraints_fn = "/tmp/constraints-{}.txt".format(n.identifier)
    n.bounds_fn = "/tmp/bounds-{}.txt".format(n.identifier)
    n.problem_fn = "/tmp/test-{}.lp".format(n.identifier)

    print('\* LOPF *\n\nmin\nobj:\n', file=open(n.objective_fn, 'w'))
    print("\n\ns.t.\n\n", file=open(n.constraints_fn, "w"))
    print("\nbounds\n", file=open(n.bounds_fn, "w"))


    for c, attr in lookup.query('nominal').index:
        define_nominal_for_extendable_variables(n, c)
    for c, attr in lookup.query('not nominal').index:
        define_dispatch_for_non_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_constraints(n, snapshots, c, attr)

    time_log['define nominal and dispatch variables'] = time.time()

    define_kirchhoff_constraints(n)
    time_log['define_kirchhoff_constraints'] = time.time()

    define_nodal_balance_constraints(n, n.snapshots)
    time_log['define_nodal_balance_constraints'] = time.time()

    define_global_constraints(n, n.snapshots)
    time_log['define_global_constraints'] = time.time()

    define_objective(n)
    time_log['define_objective'] = time.time()

    print("end\n", file=open(n.bounds_fn, "a"))

    os.system(f"cat {n.objective_fn} {n.constraints_fn} {n.bounds_fn} "
              f"> {n.problem_fn}")

    time_log['writing out lp file'] = time.time()
    time_log = time_log.diff()
    time_log.drop('start', inplace=True)
    time_log['total prepartion time'] = time_log.sum()
    logger.info(f'\n{time_log}')

    if not keep_files:
        for fn in [n.objective_fn, n.constraints_fn, n.bounds_fn]:
            os.system("rm "+ fn)


# =============================================================================
# solvers, solving, assigning
# =============================================================================

def run_cbc(filename, solution_fn, solver_logfile, solver_options, keep_files):
    options = "" #-dualsimplex -primalsimplex
    #printingOptions is about what goes in solution file
    command = (f"cbc -printingOptions all -import {filename}"
               f" -stat=1 -solve {options} -solu {solution_fn}")
    logger.info("Running command:")
    logger.info(command)
    os.system(command)
    #logfile = open(solver_logfile, 'w')
    #status = subprocess.run(["cbc",command[4:]], bufsize=0, stdout=logfile)
    #logfile.close()

    if not keep_files:
       os.system("rm "+ filename)

def run_gurobi(n, filename, solution_fn, solver_logfile,
               solver_options, keep_files):

    solver_options["logfile"] = solver_logfile

    script_fn = "/tmp/gurobi-{}.script".format(n.identifier)
    script_f = open(script_fn,"w")
    script_f.write('import sys\n')
    script_f.write('from gurobipy import *\n')
    script_f.write(f'sys.path.append("{os.path.dirname(pyomo.__file__)}'
                   '/solvers/plugins/solvers")\n')
    #script_f.write('sys.path.append("{}")\n'.format(os.path.dirname(__file__)))
    script_f.write('from GUROBI_RUN import *\n')
    #2nd argument is warmstart
    script_f.write(f'gurobi_run("{filename}",None,"{solution_fn}",None,'
                   f'{solver_options},["dual"],)\n')
    script_f.write('quit()\n')
    script_f.close()

    command = "gurobi.sh {}".format(script_fn)

    logger.info("Running command:")
    logger.info(command)
    os.system(command)

    if not keep_files:
        os.system("rm "+ filename)
        os.system("rm "+ script_fn)


def read_cbc(n, solution_fn, keep_files):
    f = open(solution_fn,"r")
    data = f.readline()
    logger.info(data)
    f.close()

    status = "ok"

    if data[:len("Optimal - objective value ")] == "Optimal - objective value ":
        termination_condition = "optimal"
        n.objective = float(data[len("Optimal - objective value "):])
    elif "Infeasible" in data:
        termination_condition = "infeasible"
    else:
        termination_condition = "other"

    if termination_condition != "optimal":
        return status, termination_condition, None, None

    sol = pd.read_csv(solution_fn, header=None, skiprows=[0],
                      sep=r'\s+', usecols=[1,2,3], index_col=0)
    variables_b = sol.index.str[0] == 'x'
    variables_sol = sol[variables_b][2]
    constraints_dual = sol[~variables_b][3]

    if not keep_files:
       os.system("rm "+ solution_fn)

    return status, termination_condition, variables_sol, constraints_dual


def read_gurobi(n, solution_fn, keep_files):
    f = open(solution_fn,"r")
    for i in range(23):
        data = f.readline()
        logger.info(data)
    f.close()

    sol = pd.read_csv(solution_fn, header=None, skiprows=range(23),
                      sep=' ', index_col=0, usecols=[1,3])[3]
    variables_b = sol.index.str[0] == 'x'
    variables_sol = sol[variables_b]
    constraints_dual = sol[~variables_b]

    if not keep_files:
       os.system("rm "+ solution_fn)

    status = "ok"
    termination_condition = "optimal"

    return status, termination_condition, variables_sol, constraints_dual


def assign_solution(n, sns, variables_sol, constraints_dual,
                    extra_postprocessing):
    non_empty_components = [c for c in prefix.index if not n.df(c).empty]
    #solutions
    def map_solution(c, attr, pnl=True):
        if pnl:
            values = (pnl_var(n, c, attr).stack()
                      .map(variables_sol).unstack())
            if c in n.passive_branch_components ^ {'Link'}:
                n.pnl(c)['p0'] = values
                n.pnl(c)['p1'] = -values
            else:
                n.pnl(c)[attr] = values
        else:
            n.df(c)[attr+'_opt'] = (df_var(n, c, attr)
                                            .map(variables_sol))

    for c, attr in prefix[non_empty_components].items():
        map_solution(c, attr  + '_nom', pnl=False)
    for c, attr in lookup.loc[non_empty_components].index:
        map_solution(c, attr, pnl=True)

    #duals
    def map_dual(c, attr, name=None, pnl=True):
        if name is None: name = attr
        if pnl:
            n.pnl(c)[attr] = (pnl_con(n, c, attr).stack()
                                      .map(-constraints_dual).unstack())
        else:
            n.df(c)[attr] = (df_con(n, c, attr)
                                     .map(-constraints_dual))

    map_dual('Bus', 'nodal_balance', 'marginal_price')
    map_dual('GlobalConstraint', 'mu', pnl=False)
    for c in non_empty_components:
        map_dual(c, 'mu_upper')
        map_dual(c, 'mu_lower')

    return

def lopf(n, snapshots=None, solver_name="cbc",
         solver_logfile=None, skip_pre=False,
         extra_functionality=None,extra_postprocessing=None,
         formulation="kirchhoff",
         solver_options={},keep_files=False):
    """
    Linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    solver_logfile : None|string
        If not None, sets the logfile option of the solver.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    formulation : string
        Formulation of the linear power flow equations to use; only "kirchhoff"
        is currently supported
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user to
        extract further information about the solution, such as additional shadow prices.

    Returns
    -------
    None
    """
    supported_solvers = ["cbc","gurobi"]
    if solver_name not in supported_solvers:
        raise NotImplementedError(f"Solver {solver_name} not in "
                                  f"supported solvers: {supported_solvers}")

    if formulation != "kirchhoff":
        raise NotImplementedError("Only the kirchhoff formulation is supported")


    snapshots = _as_snapshots(n, snapshots)

    if solver_logfile is None:
        solver_logfile = "test.log"

    logger.info("Prepare linear problem")
    prepare_lopf(n, snapshots, keep_files, extra_functionality)
    gc.collect()
    solution_fn = "/tmp/test-{}.sol".format(n.identifier)

    logger.info("Prepare linear problem")

    if solver_name == "cbc":
        run_cbc(n.problem_fn, solution_fn, solver_logfile,
                solver_options, keep_files=True)
        res = read_cbc(n, solution_fn, keep_files)
    elif solver_name == "gurobi":
        run_gurobi(n, n.problem_fn, solution_fn, solver_logfile,
                   solver_options, keep_files)
        res = read_gurobi(n, solution_fn, keep_files)
    status, termination_condition, variables_sol, constraints_dual = res

    if termination_condition != "optimal":
        return status,termination_condition

    gc.collect()
    assign_solution(n, snapshots, variables_sol,
                    constraints_dual, extra_postprocessing)
    gc.collect()

    return status,termination_condition


