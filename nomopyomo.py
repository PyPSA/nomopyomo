
## Copyright 2019 Tom Brown (KIT)

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
Pyomo. No more Pyomo."""



from pypsa.descriptors import get_switchable_as_dense, allocate_series_dataframes
from pypsa.pf import (calculate_dependent_values, find_slack_bus,
                 find_bus_controls, calculate_B_H, find_cycles, _as_snapshots)

import pandas as pd
import numpy as np
import datetime as dt

import os

import gc

import logging
logger = logging.getLogger(__name__)


now = dt.datetime.now()

def add_variables(network,group,variables):
    """
    Add variables to the network.variables.

    Parameters
    ----------
    network: pypsa.Network
    group : string
        Name of variables group.
    variables : pandas.DataFrame
        Multiindex has component name, snapshot. Columns for upper and lower bounds.
    """

    variables["i"] = range(len(network.variables),len(network.variables)+len(variables))

    if type(variables.index) is not pd.MultiIndex:
        variables = pd.concat([variables], keys=[pd.NaT], names=['snapshot']).reorder_levels(["name","snapshot"])

    variables = pd.concat([variables], keys=[group], names=['group'])

    network.variables = pd.concat((network.variables,variables),
                                  sort=False)

    #if variables is empty, it can mangle the dtype
    network.variables["i"] = network.variables["i"].astype(int)

    #network.variables = network.variables.sort_index()

def add_constraints(network,group,constraints):
    """
    Add constraints to the network.constraints.

    Parameters
    ----------
    network: pypsa.Network
    group : string
        Name of constraints group.
    constraints : pandas.DataFrame
        Multiindex has component name, snapshot. Columns for sense and rhs.
    """

    constraints["i"] = range(len(network.constraints),len(network.constraints)+len(constraints))

    if type(constraints.index) is not pd.MultiIndex:
        constraints = pd.concat([constraints], keys=[pd.NaT], names=['snapshot']).reorder_levels(["name","snapshot"])

    constraints = pd.concat([constraints], keys=[group], names=['group'])

    network.constraints = pd.concat((network.constraints,constraints),
                                  sort=False)

    #if constraints is empty, it can mangle the dtype
    network.constraints["i"] = network.constraints["i"].astype(int)

    #network.constraints = network.constraints.sort_index()


def extendable_attribute_constraints(network,snapshots,component,attr,marginal_cost=True):

    df = getattr(network,network.components[component]["list_name"])

    if len(df) == 0:
        return

    ext = df.index[df[attr+"_nom_extendable"]]
    ext.name = "name"
    fix = df.index[~df[attr+"_nom_extendable"]]
    fix.name = "name"

    max_pu = get_switchable_as_dense(network, component, attr+'_max_pu', snapshots)
    min_pu = get_switchable_as_dense(network, component, attr+'_min_pu', snapshots)

    variables = pd.DataFrame(index=pd.MultiIndex.from_product([df.index,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["lower","upper","obj"],
                             dtype=float)

    if len(fix) > 0:
        variables.loc[(fix,snapshots),"lower"] = min_pu.loc[snapshots,fix].multiply(df.loc[fix,attr+'_nom']).stack().swaplevel()
        variables.loc[(fix,snapshots),"upper"] = max_pu.loc[snapshots,fix].multiply(df.loc[fix,attr+'_nom']).stack().swaplevel()

    variables.loc[(ext,snapshots),"lower"] = -np.inf
    variables.loc[(ext,snapshots),"upper"] = np.inf

    if marginal_cost:
        mc = get_switchable_as_dense(network, component, 'marginal_cost', snapshots).multiply(network.snapshot_weightings[snapshots],axis=0)
        variables["obj"] = mc.stack().swaplevel()
    else:
        variables["obj"] = 0.

    add_variables(network,"{}-{}".format(component,attr),variables)


    if len(ext) > 0:
        variables = pd.DataFrame(index=ext,
                                 columns=["lower","upper","obj"],
                                 dtype=float)

        variables.loc[ext,"lower"] = df.loc[ext,attr+"_nom_min"]
        variables.loc[ext,"upper"] = df.loc[ext,attr+"_nom_max"]
        variables.loc[ext,"obj"] = df.loc[ext,"capital_cost"]

        add_variables(network,"{}-{}_nom".format(component,attr),variables)

        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([ext,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = ">="
        constraints["i"] = range(len(constraints))

        for unit in ext:
            i = len(network.constraints) + constraints.at[(unit,snapshots[0]),"i"]
            j = network.variables.at[("{}-{}".format(component,attr),unit,snapshots[0]),"i"]
            j_nom = network.variables.at[("{}-{}_nom".format(component,attr),unit,pd.NaT),"i"]
            for k,sn in enumerate(snapshots):
                network.constraint_matrix[i+k] = {j+k : 1., j_nom : -min_pu.at[sn, unit]}

        add_constraints(network,"{}-{}_lower".format(component,attr),constraints)

        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([ext,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = "<="
        constraints["i"] = range(len(constraints))

        for unit in ext:
            i = len(network.constraints) + constraints.at[(unit,snapshots[0]),"i"]
            j = network.variables.at[("{}-{}".format(component,attr),unit,snapshots[0]),"i"]
            j_nom = network.variables.at[("{}-{}_nom".format(component,attr),unit,pd.NaT),"i"]
            for k,sn in enumerate(snapshots):
                network.constraint_matrix[i+k] = {j+k : 1., j_nom : -max_pu.at[sn, unit]}

        add_constraints(network,"{}-{}_upper".format(component,attr),constraints)


def define_generator_constraints(network,snapshots):

    if network.generators.committable.any():
        logger.warning("Committable generators are currently not supported")

    extendable_attribute_constraints(network,snapshots,"Generator","p")

def define_link_constraints(network,snapshots):

    extendable_attribute_constraints(network,snapshots,"Link","p")


def define_store_constraints(network,snapshots):

    variables = pd.DataFrame(index=pd.MultiIndex.from_product([network.stores.index,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["lower","upper","obj"],
                             dtype=float)

    variables["lower"] = -np.inf
    variables["upper"] = np.inf
    mc = get_switchable_as_dense(network, "Store", 'marginal_cost', snapshots).multiply(network.snapshot_weightings[snapshots],axis=0)
    if len(network.stores) > 0:
        variables["obj"] = mc.stack().swaplevel()

    add_variables(network,"Store-p",variables)

    extendable_attribute_constraints(network,snapshots,"Store","e",marginal_cost=False)

    ## Builds the constraint -e_now + e_previous - p == 0 ##

    constraints = pd.DataFrame(index=pd.MultiIndex.from_product([network.stores.index,snapshots],
                                                                names=["name","snapshot"]),
                              columns=["sense","rhs"])
    constraints["sense"] = "=="
    constraints["rhs"] = 0.
    constraints["i"] = range(len(constraints))

    stores = network.stores

    for store in stores.index:
        i = len(network.constraints) + constraints.at[(store,snapshots[0]),"i"]
        j_e = network.variables.at[("Store-e",store,snapshots[0]),"i"]
        j_p = network.variables.at[("Store-p",store,snapshots[0]),"i"]
        standing_loss = stores.at[store,"standing_loss"]
        for k,sn in enumerate(snapshots):
            network.constraint_matrix[i+k] = {j_e+k : -1.}

            elapsed_hours = network.snapshot_weightings[sn]

            if k == 0:
                if stores.at[store,"e_cyclic"]:
                    network.constraint_matrix[i+k][j_e+len(snapshots)-1] = (1-standing_loss)**elapsed_hours
                else:
                    constraints.at[(store,sn),"rhs"] = -((1-standing_loss)**elapsed_hours
                                                         * stores.at[store,"e_initial"])
            else:
                network.constraint_matrix[i+k][j_e+k-1] = (1-standing_loss)**elapsed_hours

            network.constraint_matrix[i+k][j_p+k] =  -elapsed_hours

    add_constraints(network,"Store",constraints)


def define_nodal_balance_constraints(network,snapshots):

    constraints = pd.DataFrame(index=pd.MultiIndex.from_product([network.buses.index,snapshots],
                                                                names=["name","snapshot"]),
                              columns=["sense","rhs"])

    constraints["rhs"] = -get_switchable_as_dense(network, 'Load', 'p_set', snapshots).multiply(network.loads.sign).groupby(network.loads.bus,axis=1).sum().reindex(columns=network.buses.index,fill_value=0.).stack().swaplevel()
    constraints["sense"] = "=="
    constraints["i"] = range(len(constraints))

    for bus in network.buses.index:
        i = len(network.constraints) + constraints.at[(bus,snapshots[0]),"i"]
        for k,sn in enumerate(snapshots):
            network.constraint_matrix[i+k] = {}

    for component in ["Generator","Store"]:
        df = getattr(network,network.components[component]["list_name"])
        for unit in df.index:
            bus = df.at[unit,"bus"]
            sign = df.at[unit,"sign"]
            i = len(network.constraints) + constraints.at[(bus,snapshots[0]),"i"]
            j = network.variables.at[(component+"-p",unit,snapshots[0]),"i"]
            for k,sn in enumerate(snapshots):
                network.constraint_matrix[i+k][j+k] = sign

    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

    for link in network.links.index:
        bus0 = network.links.at[link,"bus0"]
        bus1 = network.links.at[link,"bus1"]
        i0 = len(network.constraints) + constraints.at[(bus0,snapshots[0]),"i"]
        i1 = len(network.constraints) + constraints.at[(bus1,snapshots[0]),"i"]
        j = network.variables.at[("Link-p",link,snapshots[0]),"i"]
        for k,sn in enumerate(snapshots):
            network.constraint_matrix[i0+k][j+k] = -1.
            network.constraint_matrix[i1+k][j+k] = efficiency.at[sn,link]

    #Add any other buses to which the links are attached
    for i in [int(col[3:]) for col in network.links.columns if col[:3] == "bus" and col not in ["bus0","bus1"]]:
        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency{}'.format(i), snapshots)
        for link in network.links.index[network.links["bus{}".format(i)] != ""]:
            bus = network.links.at[link, "bus{}".format(i)]
            ii = len(network.constraints) + constraints.at[(bus,snapshots[0]),"i"]
            j = network.variables.at[("Link-p",link,snapshots[0]),"i"]
            for k,sn in enumerate(snapshots):
                network.constraint_matrix[ii+k][j+k] = efficiency.at[sn,link]

    add_constraints(network,"nodal_balance",constraints)



def define_global_constraints(network,snapshots):

    gcs = network.global_constraints.index
    if len(gcs) == 0:
        return

    gcs.name = "name"
    constraints = pd.DataFrame(index=gcs,
                               columns=["sense","rhs"])
    constraints["i"] = range(len(constraints))

    for k,gc in enumerate(gcs):
        if network.global_constraints.loc[gc,"type"] == "primary_energy":
            constraints.at[gc,"sense"] = network.global_constraints.loc[gc,"sense"]
            constraints.at[gc,"rhs"] = network.global_constraints.loc[gc,"constant"]

            i = len(network.constraints) + k
            network.constraint_matrix[i] = {}

            carrier_attribute = network.global_constraints.loc[gc,"carrier_attribute"]

            for carrier in network.carriers.index:
                attribute = network.carriers.at[carrier,carrier_attribute]
                if attribute == 0.:
                    continue
                #for generators, use the prime mover carrier
                gens = network.generators.index[network.generators.carrier == carrier]
                for gen in gens:
                    j = network.variables.at[("Generator-p",gen,snapshots[0]),"i"]
                    for k,sn in enumerate(snapshots):
                        network.constraint_matrix[i][j+k] =(attribute
                                                            * (1/network.generators.at[gen,"efficiency"])
                                                            * network.snapshot_weightings[sn])

                #for stores, inherit the carrier from the bus
                #take difference of energy at end and start of period
                stores = network.stores.index[(network.stores.bus.map(network.buses.carrier) == carrier) & (~network.stores.e_cyclic)]
                for store in stores:
                    j = network.variables.at[("Store-e",store,snapshots[-1]),"i"]
                    network.constraint_matrix[i][j] = -attribute
                    constraints.at[gc,"rhs"] -= attribute*network.stores.at[store,"e_initial"]

    add_constraints(network,"global_constraints",constraints)


def problem_to_lp(network,filename):

    f = open(filename, 'w')
    f.write('\\* LOPF \*\\n\nmin\nobj:\n')

    obj = network.variables.obj.values
    for i in range(len(network.variables)):
        coeff = obj[i]
        #if coeff == 0:
        #    continue
        f.write("{}{} x{}\n".format("+" if coeff >= 0 else "",coeff,i))

    f.write("\n\ns.t.\n\n")


    sense = network.constraints.sense.str.replace("==","=").values
    rhs = network.constraints.rhs.values
    for i in range(len(network.constraints)):
        f.write("c{}:\n".format(i))
        for j,coeff in network.constraint_matrix[i].items():
            if coeff == 0:
                continue
            f.write("{}{} x{}\n".format("+" if coeff >= 0 else "",coeff,j))
        f.write("{} {}\n\n".format(sense[i],rhs[i]))

    f.write("\nbounds\n")

    lower = network.variables.lower.values
    upper = network.variables.upper.values
    for i in range(len(network.variables)):
        f.write("{} <= x{} <= {}\n".format(lower[i],i,upper[i]))

    f.write("end\n")
    f.close()


def run_cbc(filename,sol_filename):
    options = "" #-dualsimplex -primalsimplex
    command = "cbc -printingOptions all -import {} -stat=1 -solve {} -solu {}".format(filename,options,sol_filename)
    logger.info("Running command:")
    logger.info(command)
    os.system(command)


def read_cbc(network,sol_filename):
    f = open(sol_filename,"r")
    data = f.readline()
    logger.info(data)
    f.close()
    sol = pd.read_csv(sol_filename,header=None,skiprows=1,sep=r"\s+")

    variables = sol.index[sol[1].str[:1] == "x"]
    variables_sol = sol.loc[variables,2].astype(float)
    variables_sol.index = sol.loc[variables,1].str[1:].astype(int)

    network.variables["sol"] = network.variables["i"].map(variables_sol)


def assign_solution(network,snapshots):

    allocate_series_dataframes(network, {'Generator': ['p'],
                                         'Load': ['p'],
                                         'StorageUnit': ['p', 'state_of_charge', 'spill'],
                                         'Store': ['p', 'e'],
                                         'Bus': ['p', 'v_ang', 'v_mag_pu', 'marginal_price'],
                                         'Line': ['p0', 'p1', 'mu_lower', 'mu_upper'],
                                         'Transformer': ['p0', 'p1', 'mu_lower', 'mu_upper'],
                                         'Link': ["p"+col[3:] for col in network.links.columns if col[:3] == "bus"]
                                                  +['mu_lower', 'mu_upper']})

    if len(network.generators) > 0:
        network.generators_t.p.loc[snapshots] = network.variables.loc["Generator-p","sol"].unstack(level=0)

    if len(network.stores) > 0:
        network.stores_t.p.loc[snapshots] = network.variables.loc["Store-p","sol"].unstack(level=0)
        network.stores_t.e.loc[snapshots] = network.variables.loc["Store-e","sol"].unstack(level=0)

    if len(network.links) > 0:
        network.links_t.p0.loc[snapshots] = network.variables.loc["Link-p","sol"].unstack(level=0)
        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)
        network.links_t.p1.loc[snapshots] = -network.links_t.p0.loc[snapshots]*efficiency.loc[snapshots,:]

    for component in ["Generator","Link","Store"]:
        df = getattr(network,network.components[component]["list_name"])
        attr = "e" if component == "Store" else "p"
        df[attr+"_nom_opt"] = df[attr+"_nom"]
        if len(df) > 0:
            df.loc[df[attr+"_nom_extendable"],attr+"_nom_opt"] = network.variables.loc["{}-{}_nom".format(component,attr),"sol"].groupby(level=0).sum()


def prepare_lopf_problem(network,snapshots):
   network.variables = pd.DataFrame()
   network.constraints = pd.DataFrame()
   network.constraint_matrix = {}

   logger.info("before gen %s",dt.datetime.now()-now)
   define_generator_constraints(network,snapshots)
   logger.info("before link %s",dt.datetime.now()-now)
   define_link_constraints(network,snapshots)
   logger.info("before store %s",dt.datetime.now()-now)
   define_store_constraints(network,snapshots)
   logger.info("before nodal %s",dt.datetime.now()-now)
   define_nodal_balance_constraints(network,snapshots)
   logger.info("before global %s",dt.datetime.now()-now)
   define_global_constraints(network,snapshots)

def network_lopf(network, snapshots=None, solver_name="cbc",skip_pre=False,formulation="kirchhoff"):
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
    solver_io : string, default None
        Solver Input-Output option, e.g. "python" to use "gurobipy" for
        solver_name="gurobi"
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
        Formulation of the linear power flow equations to use; must be
        one of ["angles","cycles","kirchhoff","ptdf"]
    ptdf_tolerance : float
        Value below which PTDF entries are ignored
    free_memory : set, default {'pyomo'}
        Any subset of {'pypsa', 'pyomo'}. Allows to stash `pypsa` time-series
        data away while the solver runs (as a pickle to disk) and/or free
        `pyomo` data after the solution has been extracted.
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user to
        extract further information about the solution, such as additional shadow prices.

    Returns
    -------
    None
    """

    supported_solvers = ["cbc"]
    if solver_name not in supported_solvers:
        raise NotImplementedError("Solver {} not in supported solvers: {}".format(solver_name,supported_solvers))

    if formulation != "kirchhoff":
        raise NotImplementedError("Only the kirchhoff formulation is supported")

    if not skip_pre:
        network.determine_network_topology()
        calculate_dependent_values(network)
        for sub_network in network.sub_networks.obj:
            find_slack_bus(sub_network)
        logger.info("Performed preliminary steps")

    snapshots = _as_snapshots(network, snapshots)

    logger.info("before prep %s",dt.datetime.now()-now)
    prepare_lopf_problem(network,snapshots)
    gc.collect()
    logger.info("before write file %s",dt.datetime.now()-now)
    problem_to_lp(network,"test.lp")
    gc.collect()
    logger.info("before run %s",dt.datetime.now()-now)

    if solver_name == "cbc":
        run_cbc("test.lp","test.sol")
        logger.info("before read %s",dt.datetime.now()-now)
        read_cbc(network,"test.sol")

    gc.collect()
    logger.info("before assign %s",dt.datetime.now()-now)
    assign_solution(network,snapshots)
    logger.info("end %s",dt.datetime.now()-now)
    gc.collect()
