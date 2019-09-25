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
                  numerical_to_string, set_conref, set_varref,
                  get_con, get_var, lookup, nominals, scat, reset_counter,
                  expand_series, join_entries, align_frame_function_getter,
                  run_and_read_cbc, run_and_read_gurobi)

from pypsa.pf import find_cycles as find_cycles, _as_snapshots


import pandas as pd
import numpy as np

import gc, string, random, time, os

import logging
logger = logging.getLogger(__name__)

def define_nominal_for_extendable_variables(n, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    lower = n.df(c)[attr+'_min'][ext_i]
    upper = n.df(c)[attr+'_max'][ext_i]
    variables = write_bound(n, lower, upper)
    set_varref(n, variables, c, attr, pnl=False)


def define_dispatch_for_extendable_variables(n, sns, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    variables = write_bound(n, -np.inf, np.inf, axes=[sns, ext_i])
    set_varref(n, variables, c, attr, pnl=True, spec='extendables')


def define_dispatch_for_non_extendable_variables(n, sns, c, attr):
    fix_i = get_non_extendable_i(n, c)
    if fix_i.empty: return
    nominal_fix = n.df(c)[nominals.at[c]][fix_i]
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)
    variables = write_bound(n, lower, upper)
    set_varref(n, variables, c, attr, pnl=True, spec='nonextendables')


def define_dispatch_for_extendable_constraints(n, sns, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    min_pu, max_pu = get_bounds_pu(n, c, sns, ext_i, attr)
    operational_ext_v = get_var(n, c, attr)[ext_i]
    nominal_v = get_var(n, c, nominals.at[c])[ext_i]
    wo_prefactor, *axes = scat(nominal_v, '\n-1.0 ', operational_ext_v,
                               return_axes=True)
    rhs = '0'

    lhs, *axes = scat(max_pu, wo_prefactor, return_axes=True)
    constraints = write_constraint(n, lhs, '>=', rhs, axes)
    set_conref(n, constraints, c, 'mu_upper', pnl=True, spec=attr)

    lhs, *axes = scat(min_pu, wo_prefactor, return_axes=True)
    constraints = write_constraint(n, lhs, '<=', rhs, axes)
    set_conref(n, constraints, c, 'mu_lower', pnl=True, spec=attr)


def define_fixed_variariable_constraints(n, sns, c, attr, pnl=True):
    if pnl:
        if attr + '_set' not in n.pnl(c): return
        fix = n.pnl(c)[attr + '_set'].unstack().dropna()
        if fix.empty: return
        lhs = scat(1, get_var(n, c, attr).unstack()[fix.index])
        constraints = write_constraint(n, lhs, '=', fix).unstack().T
    else:
        if attr + '_set' not in n.df(c): return
        fix = n.df(c)[attr + '_set'].dropna()
        if fix.empty: return
        lhs = scat(1, get_var(n, c, attr)[fix.index])
        constraints = write_constraint(n, lhs, '=', fix)
    set_conref(n, constraints, c, f'mu_{attr}_set', pnl)


def define_ramp_limit_constraints(n, sns):
    c = 'Generator'
    rup_i = n.df(c).query('ramp_limit_up == ramp_limit_up').index
    rdown_i = n.df(c).query('ramp_limit_down == ramp_limit_down').index
    if rup_i.empty & rdown_i.empty:
        return
    p = get_var(n, c, 'p').loc[sns[1:]]
    p_prev = get_var(n, c, 'p').shift(1).loc[sns[1:]]

    #fix up
    gens_i = rup_i & get_non_extendable_i(n, c)
    lhs = pd.DataFrame(*scat(1, p[gens_i], -1, p_prev[gens_i], return_axes=True))
    rhs = n.df(c).loc[gens_i].eval('ramp_limit_up * p_nom')
    constraints = write_constraint(n, lhs, '<=', rhs)
    set_conref(n, constraints, c, 'mu_ramp_limit_up', spec='nonextendables')

    #ext up
    gens_i = rup_i & get_extendable_i(n, c)
    limit_pu = n.df(c)['ramp_limit_up'][gens_i]
    p_nom = get_var(n, c, 'p_nom')[gens_i]
    lhs = pd.DataFrame(*scat(1, p[gens_i], -1, p_prev[gens_i],
                             -limit_pu, p_nom, return_axes=True))
    constraints = write_constraint(n, lhs, '<=', 0)
    set_conref(n, constraints, c, 'mu_ramp_limit_up', spec='extendables')

    #fix down
    gens_i = rdown_i & get_non_extendable_i(n, c)
    lhs = pd.DataFrame(*scat(1, p[gens_i], -1, p_prev[gens_i], return_axes=True))
    rhs = n.df(c).loc[gens_i].eval('-1 * ramp_limit_down * p_nom')
    constraints = write_constraint(n, lhs, '>=', rhs)
    set_conref(n, constraints, c, 'mu_ramp_limit_down', spec='nonextendables')

    #ext down
    gens_i = rdown_i & get_extendable_i(n, c)
    limit_pu = n.df(c)['ramp_limit_down'][gens_i]
    p_nom = get_var(n, c, 'p_nom')[gens_i]
    lhs = pd.DataFrame(*scat(1, p[gens_i], -1, p_prev[gens_i],
                             limit_pu, p_nom, return_axes=True))
    constraints = write_constraint(n, lhs, '>=', 0)
    set_conref(n, constraints, c, 'mu_ramp_limit_down', spec='extendables')


def define_nodal_balance_constraints(n, sns):

    def bus_injection(c, attr, groupcol='bus', sign=1):
        #additional sign only necessary for branches in reverse direction
        if 'sign' in n.df(c):
            sign = sign * n.df(c).sign
        vals = scat(sign, get_var(n, c, attr), return_axes=True)
        return pd.DataFrame(*vals).rename(columns=n.df(c)[groupcol])

    # one might reduce this a bit by using n.branches and lookup
    args = [['Generator', 'p'], ['Store', 'p'], ['StorageUnit', 'p_dispatch'],
            ['StorageUnit', 'p_store', 'bus', -1], ['Line', 's', 'bus0', -1],
            ['Line', 's', 'bus1', 1], ['Transformer', 's', 'bus0', -1],
            ['Transformer', 's', 'bus1', 1], ['Link', 'p', 'bus0', -1],
            ['Link', 'p', 'bus1', n.links.efficiency]]
    args = [arg for arg in args if not n.df(arg[0]).empty]

    lhs = (pd.concat([bus_injection(*args) for args in args], axis=1)
           .groupby(axis=1, level=0)
           .agg(lambda x: '\n'.join(x.values))
           .reindex(columns=n.buses.index))
    sense = '='
    rhs = ((- n.loads_t.p_set * n.loads.sign)
           .groupby(n.loads.bus, axis=1).sum()
           .pipe(numerical_to_string, append_space=False)
           .reindex(columns=n.buses.index, fill_value='0.0'))
    constraints = write_constraint(n, lhs, sense, rhs)
    set_conref(n, constraints, 'Bus', 'nodal_balance')


def define_kirchhoff_constraints(n):
    weightings = n.lines.x_pu_eff.where(n.lines.carrier == 'AC', n.lines.r_pu_eff)

    def cycle_flow(ds):
        ds = ds[lambda ds: ds!=0.].dropna()
        vals = scat(ds, get_var(n, 'Line', 's')[ds.index], '\n')
        return vals.sum(1)

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


def define_storage_unit_constraints(n, sns):
    sus_i = n.storage_units.index
    if sus_i.empty: return
    c = 'StorageUnit'
    #spillage
    upper = get_as_dense(n, c, 'inflow').loc[:, lambda df: df.max() > 0]
    spill = write_bound(n, 0, upper)
    set_varref(n, spill, 'StorageUnit', 'spill')

    #soc constraint previous_soc + p_store - p_dispatch + inflow - spill == soc
    align_frame = align_frame_function_getter(n, c, sns)

    eh = expand_series(n.snapshot_weightings, sus_i) #elapsed hours

    eff_stand = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)
    eff_dispatch = expand_series(n.df(c).efficiency_dispatch, sns).T
    eff_store = expand_series(n.df(c).efficiency_store, sns).T

    soc = get_var(n, c, 'state_of_charge')
    cyclic_i = n.df(c).query('cyclic_state_of_charge').index
    noncyclic_i = n.df(c).query('~cyclic_state_of_charge').index

    prev_soc_cyclic = soc.shift().fillna(soc.loc[sns[-1]])

    coeff_var = [(-1, soc),
                 (-1/eff_dispatch * eh, get_var(n, c, 'p_dispatch')),
                 (-eh, get_var(n, c, 'spill'), spill.columns),
                 (eff_store * eh, get_var(n, c, 'p_store')),
                 (eff_stand, prev_soc_cyclic, cyclic_i),
                 (eff_stand.loc[sns[1:]], soc.shift().loc[sns[1:]], noncyclic_i)]
    aligned_coeff_var = [align_frame(*args) for args in coeff_var]
    lhs = scat(*aligned_coeff_var)

    rhs = -get_as_dense(n, c, 'inflow').mul(eh)
    rhs.loc[sns[0], noncyclic_i] -= n.df(c).state_of_charge_initial[noncyclic_i]

    constraints = write_constraint(n, lhs, '==', rhs)
    set_conref(n, constraints, c, 'soc')


def define_store_constraints(n, sns):
    stores_i = n.stores.index
    if stores_i.empty: return
    c = 'Store'
    variables = write_bound(n, -np.inf, np.inf, axes=[sns, stores_i])
    set_varref(n, variables, c, 'p')

    #previous_e - p == e
    align_frame = align_frame_function_getter(n, c, sns)
    eh = expand_series(n.snapshot_weightings, stores_i)  #elapsed hours
    eff_stand = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)

    e = get_var(n, c, 'e')
    cyclic_i = n.df(c).query('e_cyclic').index
    noncyclic_i = n.df(c).query('~e_cyclic').index

    previous_e_cyclic = e.shift().fillna(e.loc[sns[-1]])

    coeff_var = [(-eh, get_var(n, c, 'p')),
                 (-1, e),
                 (eff_stand, previous_e_cyclic, cyclic_i),
                 (eff_stand.loc[sns[1:]], e.shift().loc[sns[1:]], noncyclic_i)]
    aligned_coeff_var = [align_frame(*args) for args in coeff_var]
    lhs = scat(*aligned_coeff_var)

    rhs = pd.DataFrame(0, sns, stores_i)
    rhs.loc[sns[0], noncyclic_i] -= n.df(c)['e_initial'][noncyclic_i]

    constraints = write_constraint(n, lhs, '==', rhs)
    set_conref(n, constraints, c, 'soc')


def define_global_constraints(n, sns):
    glcs = n.global_constraints.query('type == "primary_energy"')
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]
        if emissions.empty: continue
        gens = n.generators.query('carrier in @emissions.index')
        em_pu = gens.carrier.map(emissions)/gens.efficiency
        em_pu = n.snapshot_weightings.to_frame() @ em_pu.to_frame('weightings').T
        vals = scat(em_pu, get_var(n, 'Generator', 'p')[gens.index])
        lhs = join_entries(vals)
        rhs = glc.constant

        #storage units
        sus = n.storage_units.query('carrier in @emissions.index and '
                                    'not cyclic_state_of_charge')
        sus_i = sus.index
        if not sus.empty:
            vals = scat(-sus.carrier.map(emissions),
                get_var(n, 'StorageUnit', 'state_of_charge').loc[sns[-1], sus_i])
            lhs = scat(lhs, join_entries(vals))
            rhs -= sus.carrier.map(emissions) @ sus.state_of_charge_initial

        #stores
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        if not stores.empty:
            vals = scat(-stores.carrier.map(n.emissions),
                        get_var(n, 'Store', 'e').loc[sns[-1], stores.index])
            lhs = scat(lhs, join_entries(vals))
            rhs -= stores.carrier.map(emissions) @ stores.state_of_charge_initial


        con = write_constraint(n, lhs, glc.sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)

    #expansion limits
    glcs = n.global_constraints.query('type == '
                                      '"transmission_volume_expansion_limit"')
    for name, glc in glcs.iterrows():
        carattr = [c.strip() for c in glc.carrier_attribute.split(',')]
        lines_ext_i = n.lines.query(f'carrier in @carattr '
                                    'and s_nom_extendable').index
        links_ext_i = n.links.query(f'carrier in @carattr '
                                    'and p_nom_extendable').index
        linevars = scat(n.lines.length[lines_ext_i],
                          get_var(n, 'Line', 's_nom')[lines_ext_i])
        linkvars = scat(n.links.length[links_ext_i],
                          get_var(n, 'Link', 'p_nom')[links_ext_i])
        lhs = scat(join_entries(linevars), join_entries(linkvars))
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)

    #expansion cost limits
    glcs = n.global_constraints.query('type == '
                                      '"transmission_expansion_cost_limit"')
    for name, glc in glcs.iterrows():
        carattr = [c.strip() for c in glc.carrier_attribute.split(',')]
        lines_ext_i = n.lines.query(f'carrier in @carattr '
                                    'and s_nom_extendable').index
        links_ext_i = n.links.query(f'carrier in @carattr '
                                    'and p_nom_extendable').index
        linevars = scat(n.lines.capital_cost[lines_ext_i],
                        get_var(n, 'Line', 's_nom')[lines_ext_i])
        linkvars = scat(n.links.capital_cost[links_ext_i],
                        get_var(n, 'Link', 'p_nom')[links_ext_i])
        lhs = scat(join_entries(linevars), join_entries(linkvars))
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)


def define_objective(n):
    for c, attr in lookup.query('marginal_cost').index:
        cost = (get_as_dense(n, c, 'marginal_cost')
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(n.snapshot_weightings, axis=0))
        if cost.empty: continue
        terms = scat(cost, get_var(n, c, attr)[cost.columns], '\n')
        for t in terms.flatten():
            n.objective_f.write(t)
    #investment
    for c, attr in nominals.items():
        cost = n.df(c)['capital_cost'][get_extendable_i(n, c)]
        if cost.empty: continue
        terms = scat(cost, get_var(n, c, attr)[cost.index], '\n')
        for t in terms.flatten():
            n.objective_f.write(t)



def prepare_lopf(n, snapshots=None, keep_files=False,
                 extra_functionality=None):
    reset_counter()

    #used in kirchhoff and globals
    n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)

    cols = ['component', 'name', 'pnl', 'specification']
    n.variables = pd.DataFrame(columns=cols).set_index(cols[:2])
    n.constraints = pd.DataFrame(columns=cols).set_index(cols[:2])

    snapshots = n.snapshots if snapshots is None else snapshots
    start = time.time()
    def time_info(message):
        logger.info(f'{message} {round(time.time()-start, 2)}s')

    n.identifier = ''.join(random.choice(string.ascii_lowercase)
                        for i in range(8))
    objective_fn = f"/tmp/objective-{n.identifier}.txt"
    constraints_fn = f"/tmp/constraints-{n.identifier}.txt"
    bounds_fn = f"/tmp/bounds-{n.identifier}.txt"
    n.problem_fn = f"/tmp/test-{n.identifier}.lp"

    n.objective_f = open(objective_fn, mode='w')
    n.constraints_f = open(constraints_fn, mode='w')
    n.bounds_f = open(bounds_fn, mode='w')

    n.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
    n.constraints_f.write("\n\ns.t.\n\n")
    n.bounds_f.write("\nbounds\n")


    for c, attr in lookup.query('nominal and not handle_separately').index:
        define_nominal_for_extendable_variables(n, c, attr)
        define_fixed_variariable_constraints(n, snapshots, c, attr, pnl=False)
    for c, attr in lookup.query('not nominal and not handle_separately').index:
        define_dispatch_for_non_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_constraints(n, snapshots, c, attr)
        define_fixed_variariable_constraints(n, snapshots, c, attr)

    define_ramp_limit_constraints(n, snapshots)
    define_storage_unit_constraints(n, snapshots)
    define_store_constraints(n, snapshots)
    define_kirchhoff_constraints(n)
    define_nodal_balance_constraints(n, snapshots)
    define_global_constraints(n, snapshots)
    define_objective(n)

    n.objective_f.close()
    n.constraints_f.close()
    n.bounds_f.write("end\n")
    n.bounds_f.close()

    del n.objective_f
    del n.constraints_f
    del n.bounds_f

    os.system(f"cat {objective_fn} {constraints_fn} {bounds_fn} "
              f"> {n.problem_fn}")

    time_info('Total preparation time:')

    if not keep_files:
        for fn in [objective_fn, constraints_fn, bounds_fn]:
            os.system("rm "+ fn)


def assign_solution(n, sns, variables_sol, constraints_dual,
                    extra_postprocessing, remove_references=True):
    pop = remove_references
    #solutions
    def map_solution(c, attr, pnl):
        if pnl:
            variables = get_var(n, c, attr, pop=pop)
            if variables.empty: return
            values = variables.stack().map(variables_sol).unstack()
            if c in n.passive_branch_components:
                n.pnl(c)['p0'] = values
                n.pnl(c)['p1'] = - values
            elif c == 'Link':
                n.pnl(c)['p0'] = values
                n.pnl(c)['p1'] = - values * n.df(c).efficiency
            else:
                n.pnl(c)[attr] = values
        elif not get_extendable_i(n, c).empty:
            n.df(c)[attr+'_opt'] = get_var(n, c, attr, pop=pop)\
                                    .map(variables_sol).fillna(n.df(c)[attr])
        else:
            n.df(c)[attr+'_opt'] = n.df(c)[attr]

    for (c, attr), pnl in n.variables.pnl.items():
        map_solution(c, attr, pnl)

    if not n.df('StorageUnit').empty:
        c = 'StorageUnit'
        n.pnl(c)['p'] = n.pnl(c)['p_dispatch'] - n.pnl(c)['p_store']

    #duals
    def map_dual(c, attr, pnl):
        if pnl:
            n.pnl(c)[attr] = (get_con(n, c, attr, pop=pop).stack()
                                      .map(-constraints_dual).unstack())
        else:
            n.df(c)[attr] = get_con(n, c, attr, pop=pop).map(-constraints_dual)

    for (c, attr), pnl in n.constraints.pnl.items():
        map_dual(c, attr, pnl)

    #load
    n.loads_t.p = n.loads_t.p_set

    #injection, why does it include injection in hvdc 'network'
    ca = [('Generator', 'p', 'bus' ), ('Store', 'p', 'bus'),
          ('Load', 'p', 'bus'), ('StorageUnit', 'p', 'bus'),
          ('Link', 'p0', 'bus0'), ('Link', 'p1', 'bus1')]
    sign = lambda c: n.df(c).sign if 'sign' in n.df(c) else -1 #sign for 'Link'
    n.buses_t.p = pd.concat(
            [n.pnl(c)[attr].mul(sign(c)).rename(columns=n.df(c)[group])
             for c, attr, group in ca], axis=1).groupby(level=0, axis=1).sum()

    def v_ang_for_(sub):
        buses_i = sub.buses_o
        if len(buses_i) == 1: return
        sub.calculate_B_H(skip_pre=True)
        if len(sub.buses_i()) == 1: return
        Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
        Z -= Z[sub.slack_bus]
        return n.buses_t.p[buses_i] @ Z
    n.buses_t.v_ang = (pd.concat(
                       [v_ang_for_(sub) for sub in n.sub_networks.obj], axis=1)
                      .reindex(columns=n.buses.index, fill_value=0))




def lopf(n, snapshots=None, solver_name="cbc",
         solver_logfile=None, skip_pre=False,
         extra_functionality=None, extra_postprocessing=None,
         formulation="kirchhoff", remove_references=True,
         solver_options={}, keep_files=False):
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
        extract further information about the solution, such as additional
        shadow prices.

    Returns
    -------
    None
    """
    supported_solvers = ["cbc", "gurobi"]
    if solver_name not in supported_solvers:
        raise NotImplementedError(f"Solver {solver_name} not in "
                                  f"supported solvers: {supported_solvers}")

    if formulation != "kirchhoff":
        raise NotImplementedError("Only the kirchhoff formulation is supported")


    snapshots = _as_snapshots(n, snapshots)
    n.calculate_dependent_values()
    n.determine_network_topology()
    for sub in n.sub_networks.obj:
        sub.find_bus_controls()

    if solver_logfile is None:
        solver_logfile = "test.log"

    logger.info("Prepare linear problem")
    prepare_lopf(n, snapshots, keep_files, extra_functionality)
    gc.collect()
    solution_fn = "/tmp/test-{}.sol".format(n.identifier)

    logger.info("Solve linear problem")

    if solver_name == "cbc":
        res = run_and_read_cbc(n.problem_fn, solution_fn, solver_logfile,
                               solver_options, keep_files=True)
    elif solver_name == "gurobi":
        res = run_and_read_gurobi(n.problem_fn, solution_fn, solver_logfile,
                                  solver_options, keep_files)
    status, termination_condition, variables_sol, constraints_dual, obj = res
    del n.problem_fn

    if termination_condition != "optimal":
        return status,termination_condition

    #adjust objective value
    for c, attr in nominals.items():
        obj -= n.df(c)[attr] @ n.df(c).capital_cost
    n.objective = obj
    gc.collect()
    assign_solution(n, snapshots, variables_sol, constraints_dual,
                    extra_postprocessing, remove_references=remove_references)
    gc.collect()

    return status,termination_condition
