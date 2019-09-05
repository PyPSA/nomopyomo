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


from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.pf import find_cycles as find_cycles, _as_snapshots

import pandas as pd
import numpy as np

import os, gc, string, random, time, csv, pyomo

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# writing functions
# =============================================================================
def write_objective(n, df):
    df.to_csv(n.objective_fn, sep='\n', index=False, header=False, mode='a')


def write_bound(n, lower, upper):
    shape = max([lower.shape, upper.shape])
    axes = lower.axes if shape == lower.shape else upper.axes
    var_index, variables = var_array(shape)
    (lower.astype(str) + ' <= ' + variables + ' <= ' + upper.astype(str))\
        .to_csv(n.bounds_fn, sep='\n', index=False, header=False, mode='a')
    if len(shape) > 1:
        return pd.DataFrame(variables, *axes)
    else:
        return pd.Series(variables, *axes)

def write_constraint(n, lhs, sense, rhs):
    shape = max([df.shape for df in [lhs, rhs]
                if isinstance(df, (pd.Series, pd.DataFrame))])
    axes = lhs.axes if shape == lhs.shape else rhs.axes
    con_index, constraints = con_array(shape)
    char = '\n'
    (constraints + ':' + char + lhs + char + sense + char + rhs + char)\
        .to_csv(n.constraints_fn, sep='\n', index=False, header=False,
                mode='a', quoting=csv.QUOTE_NONE, escapechar=' ')
    if len(shape) > 1:
        return pd.DataFrame(constraints, *axes)
    else:
        return pd.Series(constraints, *axes)


# =============================================================================
# helpers, helper functions
# =============================================================================
prefixes = pd.Series({'Generator': 'p',
                      #'StorageUnit': 'p',
                      #'Store': 'e',
                      'Link': 'p',
                      'Line': 's',
                      'Transformer':'s'})

var_ref_suffix = '_varref' # after solving replace with '_opt'
con_ref_suffix = '_conref' # after solving replace with ''

def numerical_to_string(val):
    if isinstance(val, (float, int)):
        return f'+{float(val)}' if val >= 0 else f'{float(val)}'
    else:
        return val.pipe(np.sign).replace([0, 1, -1], ['+', '+', '-'])\
                  .add(abs(val).astype(str)).astype(str)

x_counter = 0
def var_array(shape):
    length = np.prod(shape)
    global x_counter
    index = pd.RangeIndex(x_counter, x_counter + length)
    x_counter += length
    array = 'x' + index.astype(str).values.reshape(shape)
    return index, array

c_counter = 0
def con_array(shape):
    length = np.prod(shape)
    global c_counter
    index = pd.RangeIndex(c_counter, c_counter + length)
    c_counter += length
    array = 'c' + index.astype(str).values.reshape(shape)
    return index, array

# references to vars and cons, rewrite this part to not store every reference
def _add_reference(n, df, component, attr, suffix, pnl=True):
    attr_name = attr + suffix
    if pnl:
        if attr_name in n.pnl(component):
            n.pnl(component)[attr_name][df.columns] = df
        else:
            n.pnl(component)[attr_name] = df
    else:
        n.df(component).loc[df.index, attr_name] = df

def set_varref(n, variables, component, attr, pnl=True):
    _add_reference(n, variables, component, attr, var_ref_suffix, pnl=pnl)

def set_conref(n, constraints, component, attr, pnl=True):
    _add_reference(n, constraints, component, attr, con_ref_suffix, pnl=pnl)

def pnl_var(n, component, attr):
    return n.pnl(component)[attr + var_ref_suffix]

def df_var(n, component, attr):
    return n.df(component)[attr + var_ref_suffix]

def pnl_con(n, component, attr):
    return n.pnl(component)[attr + con_ref_suffix]

def df_con(n, component, attr):
    return n.df(component)[attr + con_ref_suffix]


# 'getter' functions
def get_extendable_i(n, component):
    return n.df(component)[lambda ds:
        ds[f'{prefixes[component]}_nom_extendable']].index

def get_non_extendable_i(n, component):
    return n.df(component)[lambda ds:
            ~ds[f'{prefixes[component]}_nom_extendable']].index

def get_bounds_pu(n, component, snapshots, index=None):
    max_pu = get_as_dense(n, component,
                          f'{prefixes[component]}_max_pu', snapshots)
    if component in n.passive_branch_components:
        min_pu = - max_pu
    else:
        min_pu = get_as_dense(n, component,
                              f'{prefixes[component]}_min_pu', snapshots)
    return (min_pu, max_pu) if index is None else (min_pu[index], max_pu[index])

# =============================================================================
#  var and con defining functions
# =============================================================================

def define_nominal_for_extendable_variables(n, component, attr='p'):
    ext_i = get_extendable_i(n, component)
    if ext_i.empty: return
    lbound = n.df(component)[f'{attr}_nom_min'][ext_i]
    ubound = n.df(component)[f'{attr}_nom_max'][ext_i]
    variables = write_bound(n, lbound, ubound)
    set_varref(n, variables, component, f'{attr}_nom', pnl=False)


def define_dispatch_for_extendable_variables(n, snapshots, component, attr='p'):
    ext_i = get_extendable_i(n, component)
    if ext_i.empty: return
    lbound = pd.DataFrame(-np.inf, index=snapshots, columns=ext_i)
    ubound = pd.DataFrame(np.inf, index=snapshots, columns=ext_i)
    variables = write_bound(n, lbound, ubound)
    set_varref(n, variables, component, attr, pnl=True)


def define_dispatch_for_non_extendable_variables(n, snapshots, component, attr='p'):
    fix_i = get_non_extendable_i(n, component)
    if fix_i.empty: return
    nominal_fix = n.df(component)[f'{attr}_nom'][fix_i]
    min_pu, max_pu = get_bounds_pu(n, component, snapshots, index=fix_i)
    lbound = min_pu.mul(nominal_fix)
    ubound = max_pu.mul(nominal_fix)
    variables = write_bound(n, lbound, ubound)
    set_varref(n, variables, component, attr, pnl=True)


def define_dispatch_for_extendable_constraints(n, snapshots, component, attr):
    ext_i = get_extendable_i(n, component)
    if ext_i.empty: return
    min_pu, max_pu = get_bounds_pu(n, component, snapshots, index=ext_i)
    operational_ext_v = pnl_var(n, component, attr)[ext_i]
    nominal_v = df_var(n, component, attr + '_nom')[ext_i]
    wo_prefactor = nominal_v + '\n' + '-1.0 ' +  operational_ext_v
    rhs = '0'

    lhs = numerical_to_string(max_pu) + ' ' + wo_prefactor
    constraints = write_constraint(n, lhs, '>=', rhs)
    set_conref(n, constraints, component, 'mu_upper')

    lhs = numerical_to_string(min_pu) + ' ' + wo_prefactor
    constraints = write_constraint(n, lhs, '<=', rhs)
    set_conref(n, constraints, component, 'mu_lower')


def define_nodal_balance_constraints(n, snapshots):

    def bus_injection(component, attr, groupcol='bus', sign=1):
        #additional sign only necessary for branches in reverse direction
        if 'sign' in n.df(component):
            sign = sign * n.df(component).sign
        return (numerical_to_string(sign) + ' ' + pnl_var(n, component, attr))\
                .rename(columns=n.df(component)[groupcol])

    arg_list = [['Generator', 'p', 'bus', 1],
#                ['Store', 'e', 'bus', 1],
#                ['StorageUnite', 'p_dispatch', 'bus', 1],
#                ['StorageUnit', 'p_charge', 'bus', -1],
                ['Line', 's', 'bus0', -1],
                ['Line', 's', 'bus1', 1],
                ['Transformer', 's', 'bus0', -1],
                ['Transformer', 's', 'bus1', 1],
                ['Link', 'p', 'bus0', -1],
                ['Link', 'p', 'bus1', n.links.efficiency]]
    arg_list = [arg for arg in arg_list if arg[0] in n.non_empty_components]

    lhs = (pd.concat([bus_injection(*args) for args in arg_list], axis=1)
           .groupby(axis=1, level=0)
           .agg(lambda x: '\n'.join(x.values)))
    sense = '='
    rhs = ((- n.loads_t.p_set * n.loads.sign)
           .groupby(n.loads.bus, axis=1).sum()
           .pipe(numerical_to_string)
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
        return (numerical_to_string(ds) + ' ' +
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


def define_objective(n):
    operating = prefixes[n.non_empty_components - n.passive_branch_components]
    for comp, attr in operating.items():
        cost = (get_as_dense(n, comp, 'marginal_cost')
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(n.snapshot_weightings, axis=0))
        if cost.empty: continue
        terms = numerical_to_string(cost) + ' ' + pnl_var(n, comp, attr)[cost.columns]
        write_objective(n, terms)
    #investment
    for comp, attr in prefixes[n.non_empty_components].items():
        cost = n.df(comp)['capital_cost'][get_extendable_i(n, comp)]
        if cost.empty: continue
        terms = numerical_to_string(cost) + ' ' + df_var(n, comp, attr+'_nom')[cost.index]
        write_objective(n, terms)


def prepare_lopf(n, snapshots=None, keep_files=False,
                 extra_functionality=None):
    snapshots = n.snapshots if snapshots is None else snapshots
    time_log = pd.Series({'start': time.time()})
    n.non_empty_components = set(c for c in prefixes.index if not n.df(c).empty)

    n.objective_fn = "/tmp/objective-{}.txt".format(n.identifier)
    n.constraints_fn = "/tmp/constraints-{}.txt".format(n.identifier)
    n.bounds_fn = "/tmp/bounds-{}.txt".format(n.identifier)

    print('\* LOPF *\n\nmin\nobj:\n', file=open(n.objective_fn, 'w'))
    print("\n\ns.t.\n\n", file=open(n.constraints_fn, "w"))
    print("\nbounds\n", file=open(n.bounds_fn, "w"))

    for c, attr in prefixes[n.non_empty_components].items():
        define_nominal_for_extendable_variables(n, c, attr)
    time_log['define_nominal_for_extendable_variables'] = time.time()


    for c, attr in prefixes[n.non_empty_components].items():
        define_dispatch_for_extendable_variables(n, n.snapshots, c, attr)
    time_log['define_dispatch_for_extendable_variables'] = time.time()

    for c, attr in prefixes[n.non_empty_components].items():
        define_dispatch_for_non_extendable_variables(n, n.snapshots, c, attr)
    time_log['define_dispatch_for_non_extendable_variables'] = time.time()

    for c, attr in prefixes[n.non_empty_components].items():
        define_dispatch_for_extendable_constraints(n, n.snapshots, c, attr)
    time_log['define_dispatch_for_extendable_constraints'] = time.time()


    define_kirchhoff_constraints(n)
    time_log['define_kirchhoff_constraints'] = time.time()

    define_nodal_balance_constraints(n, n.snapshots)
    time_log['define_nodal_balance_constraints'] = time.time()

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

    return status,termination_condition,variables_sol,constraints_dual


def assign_solution(n, snapshots, variables_sol, constraints_dual,
                    extra_postprocessing):
    #solutions
    def map_solution(component, attr, pnl=True):
        if pnl:
            values = (pnl_var(n, component, attr).stack()
                      .map(variables_sol).unstack())
            if component in n.passive_branch_components ^ {'Link'}:
                n.pnl(component)['p0'] = values
                n.pnl(component)['p1'] = -values
            else:
                n.pnl(component)[attr] = values
        else:
            n.df(component)[attr+'_opt'] = (df_var(n, component, attr)
                                            .map(variables_sol))

    for component, attr in prefixes[n.non_empty_components].items():
        map_solution(component, attr, pnl=True)
    for component, attr in (prefixes[n.non_empty_components] + '_nom').items():
        map_solution(component, attr, pnl=False)

    #duals
    def map_dual(component, attr, name=None, pnl=True):
        if name is None: name = attr
        if pnl:
            n.pnl(component)[attr] = (pnl_con(n, component, attr).stack()
                                      .map(variables_sol).unstack())
        else:
            n.df(component)[attr] = df_con(n, component, attr).map(variables_sol)

    map_dual('Bus', 'nodal_balance', 'marginal_price')
    for component in n.non_empty_components:
        map_dual(component, 'mu_upper')
        map_dual(component, 'mu_lower')

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

    n.identifier = ''.join(random.choice(string.ascii_lowercase)
                        for i in range(8))
    n.problem_fn = "/tmp/test-{}.lp".format(n.identifier)
    solution_fn = "/tmp/test-{}.sol".format(n.identifier)
    if solver_logfile is None:
        solver_logfile = "test.log"

    logger.info("Prepare linear problem")
    prepare_lopf(n, snapshots, keep_files, extra_functionality)
    gc.collect()

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


