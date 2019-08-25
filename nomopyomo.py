
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
Pyomo. nomopyomo = no more Pyomo."""



from pypsa.descriptors import get_switchable_as_dense, allocate_series_dataframes
from pypsa.pf import (calculate_dependent_values, find_slack_bus,
                 find_bus_controls, calculate_B_H, find_cycles, _as_snapshots)

import pandas as pd
import numpy as np
import datetime as dt

import os, gc, string, random

import logging
logger = logging.getLogger(__name__)


now = dt.datetime.now()

def write_objective(network,coeff,position):
    if coeff == 0:
        coeff = abs(coeff)
    network.objective_f.write("{}{} x{}\n".format("+" if coeff >= 0 else "",coeff,position))

def write_bounds(network,lower,upper,position):
    network.bounds_f.write("{} <= x{} <= {}\n".format(lower,position,upper))

def write_constraint(network,constraint_matrix_row,sense,rhs,position):
    network.constraints_f.write("c{}:\n".format(position))
    for j,coeff in constraint_matrix_row.items():
        if coeff == 0:
            continue
        network.constraints_f.write("{}{} x{}\n".format("+" if coeff >= 0 else "",coeff,j))
    network.constraints_f.write("{} {}\n\n".format("=" if sense == "==" else sense,rhs))

def add_group(network,sort,group,length):
    """sort is "variable" or "constraint"""""
    df = getattr(network,sort + "_positions")
    df.at[group,"start"] = 0 if len(df) == 0 else df["finish"][-1]
    df.at[group,"finish"] = df.at[group,"start"] + length
    setattr(network,sort+ "_positions",df.astype(int))

def extendable_attribute_constraints(network,snapshots,component,attr,marginal_cost=True):

    df = getattr(network,network.components[component]["list_name"])
    pnl = getattr(network,network.components[component]["list_name"]+"_t")

    if len(df) == 0:
        return

    group = "{}-{}".format(component,attr)
    add_group(network,"variable",group,len(df.index)*len(snapshots))

    for i,unit in enumerate(df.index):
        if not marginal_cost:
            mc = 0.*network.snapshot_weightings[snapshots]
        else:
            if unit in pnl.marginal_cost:
                mc = pnl.marginal_cost.loc[snapshots,unit]*network.snapshot_weightings[snapshots]
            else:
                mc = df.at[unit,"marginal_cost"]*network.snapshot_weightings[snapshots]
        mc = mc.values

        start = network.variable_positions.at[group,"start"] + i*len(snapshots)

        for k in range(len(snapshots)):
            write_objective(network,mc[k],start+k)

        if df.at[unit,attr+"_nom_extendable"]:
            for k in range(len(snapshots)):
                write_bounds(network,-np.inf,np.inf,start+k)
        else:
            if unit in pnl[attr+"_max_pu"]:
                upper = pnl[attr+"_max_pu"].loc[snapshots,unit].values
            else:
                upper = df.at[unit,attr+"_max_pu"]*np.ones(len(snapshots))
            if component in network.passive_branch_components:
                lower = -upper
            else:
                if unit in pnl[attr+"_min_pu"]:
                    lower = pnl[attr+"_min_pu"].loc[snapshots,unit].values
                else:
                    lower = df.at[unit,attr+"_min_pu"]*np.ones(len(snapshots))
            upper = upper*df.at[unit,attr+"_nom"]
            lower = lower*df.at[unit,attr+"_nom"]
            for k in range(len(snapshots)):
                write_bounds(network,lower[k],upper[k],start+k)

    ext = df.index[df[attr+"_nom_extendable"]]
    group = "{}-{}_nom".format(component,attr)
    add_group(network,"variable",group,len(ext))

    start = network.variable_positions.at[group,"start"]
    for i,unit in enumerate(ext):
        write_objective(network,df.at[unit,"capital_cost"],start+i)
        write_bounds(network,df.at[unit,attr+"_nom_min"],df.at[unit,attr+"_nom_max"],start+i)

    group = "{}-{}_lower".format(component,attr)
    add_group(network,"constraint",group,len(ext)*len(snapshots))
    start = network.constraint_positions.at[group,"start"]

    for i_unit,unit in enumerate(ext):
        i = start + i_unit*len(snapshots)
        j = network.variable_positions.at["{}-{}".format(component,attr),"start"] + df.index.get_loc(unit)*len(snapshots)
        j_nom = network.variable_positions.at["{}-{}_nom".format(component,attr),"start"] + i_unit
        if component in network.passive_branch_components:
            if unit in pnl[attr+"_max_pu"]:
                lower = -pnl[attr+"_max_pu"].loc[snapshots,unit].values
            else:
                lower = -df.at[unit,attr+"_max_pu"]*np.ones(len(snapshots))
        else:
            if unit in pnl[attr+"_min_pu"]:
                lower = pnl[attr+"_min_pu"].loc[snapshots,unit].values
            else:
                lower = df.at[unit,attr+"_min_pu"]*np.ones(len(snapshots))
        for k,sn in enumerate(snapshots):
            write_constraint(network,{j+k : 1., j_nom : -lower[k]},">=","0",i+k)


    group = "{}-{}_upper".format(component,attr)
    add_group(network,"constraint",group,len(ext)*len(snapshots))
    start = network.constraint_positions.at[group,"start"]

    for i_unit,unit in enumerate(ext):
        i = start + i_unit*len(snapshots)
        j = network.variable_positions.at["{}-{}".format(component,attr),"start"] + df.index.get_loc(unit)*len(snapshots)
        j_nom = network.variable_positions.at["{}-{}_nom".format(component,attr),"start"] + i_unit
        if unit in pnl[attr+"_max_pu"]:
            upper = pnl[attr+"_max_pu"].loc[snapshots,unit].values
        else:
            upper = df.at[unit,attr+"_max_pu"]*np.ones(len(snapshots))
        for k,sn in enumerate(snapshots):
            write_constraint(network,{j+k : 1., j_nom : -upper[k]},"<=","0",i+k)

def define_generator_constraints(network,snapshots):

    if network.generators.committable.any():
        logger.warning("Committable generators are currently not supported")

    extendable_attribute_constraints(network,snapshots,"Generator","p")

def define_link_constraints(network,snapshots):

    extendable_attribute_constraints(network,snapshots,"Link","p")


def define_passive_branch_constraints(network,snapshots):

    passive_branches = network.passive_branches()

    if len(passive_branches) == 0:
        return

    extendable_attribute_constraints(network,snapshots,"Line","s",marginal_cost=False)
    extendable_attribute_constraints(network,snapshots,"Transformer","s",marginal_cost=False)

    for sub_network in network.sub_networks.obj:
        find_cycles(sub_network)

        #following is necessary to calculate angles post-facto
        find_bus_controls(sub_network)
        if len(sub_network.branches_i()) > 0:
            calculate_B_H(sub_network)

    c = 0
    constraint_matrix = {}
    for subnetwork in network.sub_networks.obj:

        branches = subnetwork.branches()
        attribute = "r_pu_eff" if network.sub_networks.at[subnetwork.name,"carrier"] == "DC" else "x_pu_eff"
        matrix = subnetwork.C.tocsc()

        for col_j in range(matrix.shape[1]):
            cycle_is = matrix.getcol(col_j).nonzero()[0]

            if len(cycle_is) == 0:  continue

            i = c*len(snapshots)
            for k,sn in enumerate(snapshots):
                constraint_matrix[i+k] = {}

            for cycle_i in cycle_is:
                branch_idx = branches.index[cycle_i]
                attribute_value = 1e5 * branches.at[ branch_idx, attribute] * subnetwork.C[ cycle_i, col_j]
                j = network.variable_positions.at["{}-s".format(branch_idx[0]),"start"] + getattr(network,network.components[branch_idx[0]]["list_name"]).index.get_loc(branch_idx[1])*len(snapshots)
                for k,sn in enumerate(snapshots):
                    constraint_matrix[i+k][j+k] = attribute_value
            c+=1

    group = "Cycle"
    add_group(network,"constraint",group,c*len(snapshots))
    start = network.constraint_positions.at[group,"start"]

    for i_c in range(c):
        i = i_c*len(snapshots)
        for k in range(len(snapshots)):
            write_constraint(network,constraint_matrix[i+k],"==","0",start+i+k)

def define_store_constraints(network,snapshots):


    group = "Store-p"
    add_group(network,"variable",group,len(network.stores.index)*len(snapshots))

    for i,unit in enumerate(network.stores.index):
        if unit in network.stores_t.marginal_cost:
            mc = network.stores_t.marginal_cost.loc[snapshots,unit]*network.snapshot_weightings[snapshots]
        else:
            mc = network.stores.at[unit,"marginal_cost"]*network.snapshot_weightings[snapshots]
        mc = mc.values

        start = network.variable_positions.at[group,"start"] + i*len(snapshots)

        for k in range(len(snapshots)):
            write_objective(network,mc[k],start+k)
            write_bounds(network,-np.inf,np.inf,start+k)

    extendable_attribute_constraints(network,snapshots,"Store","e",marginal_cost=False)

    ## Builds the constraint -e_now + e_previous - p == 0 ##

    group = "Store"
    add_group(network,"constraint",group,len(network.stores.index)*len(snapshots))
    start = network.constraint_positions.at[group,"start"]

    stores = network.stores

    for i_store,store in enumerate(stores.index):
        i = start+i_store*len(snapshots)
        j_e = network.variable_positions.at["Store-e","start"] + network.stores.index.get_loc(store)*len(snapshots)
        j_p = network.variable_positions.at["Store-p","start"] + network.stores.index.get_loc(store)*len(snapshots)
        standing_loss = stores.at[store,"standing_loss"]
        for k,sn in enumerate(snapshots):
            constraint_matrix_row = {j_e+k : -1.}
            rhs = 0.
            elapsed_hours = network.snapshot_weightings[sn]

            if k == 0:
                if stores.at[store,"e_cyclic"]:
                    constraint_matrix_row[j_e+len(snapshots)-1] = (1-standing_loss)**elapsed_hours
                else:
                    rhs = -((1-standing_loss)**elapsed_hours
                            * stores.at[store,"e_initial"])
            else:
                constraint_matrix_row[j_e+k-1] = (1-standing_loss)**elapsed_hours

            constraint_matrix_row[j_p+k] =  -elapsed_hours

            write_constraint(network,constraint_matrix_row,"==",rhs,i+k)

def define_nodal_balance_constraints(network,snapshots):

    constraint_matrix = {}


    for i_bus,bus in enumerate(network.buses.index):
        i = i_bus*len(snapshots)
        for k in range(len(snapshots)):
            constraint_matrix[i+k] = {}

    for component in ["Generator","Store"]:
        df = getattr(network,network.components[component]["list_name"])
        for unit in df.index:
            bus = df.at[unit,"bus"]
            sign = df.at[unit,"sign"]
            i = network.buses.index.get_loc(bus)*len(snapshots)
            j = network.variable_positions.at["{}-p".format(component),"start"] + df.index.get_loc(unit)*len(snapshots)
            for k,sn in enumerate(snapshots):
                constraint_matrix[i+k][j+k] = sign

    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

    for link in network.links.index:
        bus0 = network.links.at[link,"bus0"]
        bus1 = network.links.at[link,"bus1"]
        i0 = network.buses.index.get_loc(bus0)*len(snapshots)
        i1 = network.buses.index.get_loc(bus1)*len(snapshots)
        j = network.variable_positions.at["Link-p","start"] + network.links.index.get_loc(link)*len(snapshots)
        for k,sn in enumerate(snapshots):
            constraint_matrix[i0+k][j+k] = -1.
            constraint_matrix[i1+k][j+k] = efficiency.at[sn,link]

    #Add any other buses to which the links are attached
    for i in [int(col[3:]) for col in network.links.columns if col[:3] == "bus" and col not in ["bus0","bus1"]]:
        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency{}'.format(i), snapshots)
        for link in network.links.index[network.links["bus{}".format(i)] != ""]:
            bus = network.links.at[link, "bus{}".format(i)]
            ii = network.buses.index.get_loc(bus)*len(snapshots)
            j = network.variable_positions.at["Link-p","start"] + network.links.index.get_loc(link)*len(snapshots)
            for k,sn in enumerate(snapshots):
                constraint_matrix[ii+k][j+k] = efficiency.at[sn,link]

    for component in network.passive_branch_components:
        df = getattr(network,network.components[component]["list_name"])
        for unit in df.index:
            bus0 = df.at[unit,"bus0"]
            bus1 = df.at[unit,"bus1"]
            i0 = network.buses.index.get_loc(bus0)*len(snapshots)
            i1 = network.buses.index.get_loc(bus1)*len(snapshots)
            j = network.variable_positions.at["{}-s".format(component),"start"] + df.index.get_loc(unit)*len(snapshots)
            for k,sn in enumerate(snapshots):
                constraint_matrix[i0+k][j+k] = -1.
                constraint_matrix[i1+k][j+k] = 1.


    group = "nodal_balance"
    add_group(network,"constraint",group,len(network.buses.index)*len(snapshots))
    start = network.constraint_positions.at[group,"start"]

    rhs = -get_switchable_as_dense(network, 'Load', 'p_set', snapshots).multiply(network.loads.sign).groupby(network.loads.bus,axis=1).sum().reindex(columns=network.buses.index,fill_value=0.)
    for i_bus,bus in enumerate(network.buses.index):
        i = i_bus*len(snapshots)
        rhs_i = rhs[bus]
        for k in range(len(snapshots)):
            write_constraint(network,constraint_matrix[i+k],"==",rhs_i[k],start+i+k)


def define_global_constraints(network,snapshots):

    gcs = network.global_constraints.index
    if len(gcs) == 0:
        return

    group = "global_constraints"
    add_group(network,"constraint",group,len(gcs))
    start = network.constraint_positions.at[group,"start"]

    for i,gc in enumerate(gcs):
        if network.global_constraints.loc[gc,"type"] == "primary_energy":

            rhs = network.global_constraints.loc[gc,"constant"]
            constraint_matrix_row = {}

            carrier_attribute = network.global_constraints.loc[gc,"carrier_attribute"]

            for carrier in network.carriers.index:
                attribute = network.carriers.at[carrier,carrier_attribute]
                if attribute == 0.:
                    continue
                #for generators, use the prime mover carrier
                gens = network.generators.index[network.generators.carrier == carrier]
                for gen in gens:
                    j = network.variable_positions.at["Generator-p","start"] + network.generators.index.get_loc(gen)*len(snapshots)
                    for k,sn in enumerate(snapshots):
                        constraint_matrix_row[j+k] = (attribute
                                                      * (1/network.generators.at[gen,"efficiency"])
                                                      * network.snapshot_weightings[sn])

                #for stores, inherit the carrier from the bus
                #take difference of energy at end and start of period
                stores = network.stores.index[(network.stores.bus.map(network.buses.carrier) == carrier) & (~network.stores.e_cyclic)]
                for store in stores:
                    j = network.variable_positions.at["Store-e","start"] + network.stores.index.get_loc(store)*len(snapshots) + len(snapshots)-1
                    constraint_matrix_row[j] = -attribute
                    rhs -= attribute*network.stores.at[store,"e_initial"]

            write_constraint(network,constraint_matrix_row,network.global_constraints.loc[gc,"sense"],rhs,start+i)


def run_cbc(filename,sol_filename,solver_options,keep_files):
    options = "" #-dualsimplex -primalsimplex
    command = "cbc -printingOptions all -import {} -stat=1 -solve {} -solu {}".format(filename,options,sol_filename)
    logger.info("Running command:")
    logger.info(command)
    os.system(command)

    if not keep_files:
       os.system("rm "+ filename)

def run_gurobi(filename,sol_filename,solver_logfile,solver_options,keep_files):
    options = "" #-dualsimplex -primalsimplex
    command = "gurobi_cl"

    if solver_logfile is not None:
        command += " logfile=" + solver_logfile

    for k,v in solver_options.items():
        command += " {}={}".format(k,v)

    command += " QCPDual=1 resultfile={} {}".format(sol_filename,filename)

    logger.info("Running command:")
    logger.info(command)
    os.system(command)

    if not keep_files:
        os.system("rm "+ filename)

def read_cbc(network,sol_filename,keep_files):
    f = open(sol_filename,"r")
    data = f.readline()
    logger.info(data)
    f.close()
    sol = pd.read_csv(sol_filename,header=None,skiprows=1,sep=r"\s+")

    variables = sol.index[sol[1].str[:1] == "x"]
    variables_sol = sol.loc[variables,2].astype(float)
    variables_sol.index = sol.loc[variables,1].str[1:].astype(int)

    if not keep_files:
       os.system("rm "+ sol_filename)

    return variables_sol


def read_gurobi(network,sol_filename,keep_files):
    f = open(sol_filename,"r")
    for i in range(2):
        data = f.readline()
        logger.info(data)
    f.close()
    sol = pd.read_csv(sol_filename,header=None,skiprows=2,sep=" ")

    variables_sol = sol[1].astype(float)
    variables_sol.index = sol[0].str[1:].astype(int)

    if not keep_files:
       os.system("rm "+ sol_filename)

    return variables_sol



def assign_solution(network,snapshots,variables_sol):

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
        start,finish = network.variable_positions.loc["Generator-p"]
        network.generators_t.p.loc[snapshots,network.generators.index] = pd.Series(data=variables_sol[start:finish].values,
                                                                                   index=pd.MultiIndex.from_product([network.generators.index,snapshots])).unstack(level=0)[network.generators.index]

    if len(network.stores) > 0:
        start,finish = network.variable_positions.loc["Store-p"]
        network.stores_t.p.loc[snapshots,network.stores.index] = pd.Series(data=variables_sol[start:finish].values,
                                                                           index=pd.MultiIndex.from_product([network.stores.index,snapshots])).unstack(level=0)[network.stores.index]
        start,finish = network.variable_positions.loc["Store-e"]
        network.stores_t.e.loc[snapshots,network.stores.index] = pd.Series(data=variables_sol[start:finish].values,
                                                                           index=pd.MultiIndex.from_product([network.stores.index,snapshots])).unstack(level=0)[network.stores.index]

    if len(network.links) > 0:
        start,finish = network.variable_positions.loc["Link-p"]
        network.links_t.p0.loc[snapshots,network.links.index] = pd.Series(data=variables_sol[start:finish].values,
                                                          index=pd.MultiIndex.from_product([network.links.index,snapshots])).unstack(level=0)[network.links.index]
        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)
        network.links_t.p1.loc[snapshots,network.links.index] = -network.links_t.p0.loc[snapshots,network.links.index]*efficiency.loc[snapshots,network.links.index]
        for i in [int(col[3:]) for col in network.links.columns if col[:3] == "bus" and col not in ["bus0","bus1"]]:
            efficiency = get_switchable_as_dense(network, 'Link', 'efficiency{}'.format(i), snapshots)
            network.links_t["p"+str(i)].loc[snapshots,network.links.index] = -network.links_t.p0.loc[snapshots,network.links.index]*efficiency.loc[snapshots,network.links.index]

    for c in network.iterate_components(network.passive_branch_components):
        start,finish = network.variable_positions.loc["{}-s".format(c.name)]
        c.pnl.p0.loc[snapshots,c.df.index] = pd.Series(data=variables_sol[start:finish].values,
                                            index=pd.MultiIndex.from_product([c.df.index,snapshots])).unstack(level=0)[c.df.index]
        c.pnl.p1.loc[snapshots,c.df.index] = - c.pnl.p0.loc[snapshots,c.df.index]

    for component in ["Generator","Link","Store","Line","Transformer"]:
        df = getattr(network,network.components[component]["list_name"])
        if component == "Store":
            attr ="e"
        elif component in ["Line","Transformer"]:
            attr = "s"
        else:
            attr = "p"
        df[attr+"_nom_opt"] = df[attr+"_nom"]
        ext = df.index[df[attr+"_nom_extendable"]]
        if len(ext) > 0:
            start,finish = network.variable_positions.loc["{}-{}_nom".format(component,attr)]
            df.loc[ext,attr+"_nom_opt"] = variables_sol[start:finish].values


def prepare_lopf_problem(network,snapshots,problem_file,keep_files):

   network.variable_positions = pd.DataFrame(columns=["start","finish"])
   network.constraint_positions = pd.DataFrame(columns=["start","finish"])

   objective_fn = "/tmp/objective-{}.txt".format(network.identifier)
   network.objective_f = open(objective_fn,"w")
   network.objective_f.write('\\* LOPF \*\\n\nmin\nobj:\n')

   constraints_fn = "/tmp/constraints-{}.txt".format(network.identifier)
   network.constraints_f = open(constraints_fn,"w")
   network.constraints_f.write("\n\ns.t.\n\n")

   bounds_fn = "/tmp/bounds-{}.txt".format(network.identifier)
   network.bounds_f = open(bounds_fn,"w")
   network.bounds_f.write("\nbounds\n")

   logger.info("before gen %s",dt.datetime.now()-now)
   define_generator_constraints(network,snapshots)
   logger.info("before link %s",dt.datetime.now()-now)
   define_link_constraints(network,snapshots)
   logger.info("before passive %s",dt.datetime.now()-now)
   define_passive_branch_constraints(network,snapshots)
   logger.info("before store %s",dt.datetime.now()-now)
   define_store_constraints(network,snapshots)
   logger.info("before nodal %s",dt.datetime.now()-now)
   define_nodal_balance_constraints(network,snapshots)
   logger.info("before global %s",dt.datetime.now()-now)
   define_global_constraints(network,snapshots)

   network.bounds_f.write("end\n")

   network.objective_f.close()
   network.constraints_f.close()
   network.bounds_f.close()

   os.system("cat {} {} {} > {}".format(objective_fn,constraints_fn,bounds_fn,problem_file))

   if not keep_files:
       for fn in [objective_fn,constraints_fn,bounds_fn]:
           os.system("rm "+ fn)


def network_lopf(network, snapshots=None, solver_name="cbc",skip_pre=False,formulation="kirchhoff",solver_logfile=None,
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

    supported_solvers = ["cbc","gurobi"]
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

    network.identifier = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    problem_file = "/tmp/test-{}.lp".format(network.identifier)
    solution_file = "/tmp/test-{}.sol".format(network.identifier)
    log_file = "/tmp/test-{}.log".format(network.identifier)

    logger.info("before prep %s",dt.datetime.now()-now)
    prepare_lopf_problem(network,snapshots,problem_file,keep_files)
    gc.collect()

    logger.info("before run %s",dt.datetime.now()-now)

    if solver_name == "cbc":
        run_cbc(problem_file,solution_file,solver_options,keep_files)
        logger.info("before read %s",dt.datetime.now()-now)
        variables_sol = read_cbc(network,solution_file,keep_files)
    elif solver_name == "gurobi":
        run_gurobi(problem_file,solution_file,log_file,solver_options,keep_files)
        logger.info("before read %s",dt.datetime.now()-now)
        variables_sol = read_gurobi(network,solution_file,keep_files)

    gc.collect()
    logger.info("before assign %s",dt.datetime.now()-now)
    assign_solution(network,snapshots,variables_sol)
    logger.info("end %s",dt.datetime.now()-now)
    gc.collect()
