
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
from pypsa.pf import _as_snapshots

import pandas as pd
import numpy as np

from scipy.sparse import lil_matrix, vstack, hstack, coo_matrix

import os

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

def add_constraints(network,group,constraints,constraint_matrix):
    """
    Add constraints to the network.{constraints,constraint_matrix}.

    Parameters
    ----------
    network: pypsa.Network
    group : string
        Name of constraints group.
    constraints : pandas.DataFrame
        Multiindex has component name, snapshot. Columns for sense and rhs.
    """

    constraints["i"] = range(len(network.constraints),len(network.constraints)+len(constraints))

    constraints = pd.concat([constraints], keys=[group], names=['group'])

    network.constraints = pd.concat((network.constraints,constraints),
                                  sort=False)

    if network.constraint_matrix.shape[1] < len(network.variables):
        network.constraint_matrix = hstack((network.constraint_matrix,lil_matrix((network.constraint_matrix.shape[0],len(network.variables)-network.constraint_matrix.shape[1]))))

    network.constraint_matrix = vstack((network.constraint_matrix,constraint_matrix))



def define_generator_constraints(network,snapshots):

    if network.generators.committable.any():
        logger.warning("Committable generators are currently not supported")

    extendable_gens_i = network.generators.index[network.generators.p_nom_extendable]
    extendable_gens_i.name = "name"
    fixed_gens_i = network.generators.index[~network.generators.p_nom_extendable]


    variables = pd.DataFrame(index=pd.MultiIndex.from_product([network.generators.index,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["lower","upper"],
                             dtype=float)

    p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu', snapshots)
    p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu', snapshots)

    if len(fixed_gens_i) > 0:
        variables.loc[(fixed_gens_i,snapshots),"lower"] = p_min_pu.loc[snapshots,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom']).stack().swaplevel()
        variables.loc[(fixed_gens_i,snapshots),"upper"] = p_max_pu.loc[snapshots,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom']).stack().swaplevel()

    variables.loc[(extendable_gens_i,snapshots),"lower"] = -np.inf

    variables.loc[(extendable_gens_i,snapshots),"upper"] = np.inf

    add_variables(network,"gen-p",variables)


    ## Define generator capacity variables if generator is extendable ##

    if len(extendable_gens_i) > 0:
        variables = pd.DataFrame(index=extendable_gens_i,
                                 columns=["lower","upper"],
                                 dtype=float)

        variables.loc[extendable_gens_i,"lower"] = network.generators.loc[extendable_gens_i,"p_nom_min"]

        variables.loc[extendable_gens_i,"upper"] = network.generators.loc[extendable_gens_i,"p_nom_max"]

        add_variables(network,"gen-p_nom",variables)


        ## Define generator dispatch constraints for extendable generators ##

        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([extendable_gens_i,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = ">="
        constraints["i"] = range(len(constraints))
        constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

        for gen in extendable_gens_i:
            for sn in snapshots:
                constraint_matrix[constraints.at[(gen,sn),"i"],network.variables.at[("gen-p",gen,sn),"i"]] = 1.
                constraint_matrix[constraints.at[(gen,sn),"i"],network.variables.at[("gen-p_nom",gen,pd.NaT),"i"]] = -p_min_pu.at[sn, gen]

        add_constraints(network,"gen-p_lower",constraints,constraint_matrix)


        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([extendable_gens_i,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = "<="
        constraints["i"] = range(len(constraints))
        constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

        for gen in extendable_gens_i:
            for sn in snapshots:
                constraint_matrix[constraints.at[(gen,sn),"i"],network.variables.at[("gen-p",gen,sn),"i"]] = 1.
                constraint_matrix[constraints.at[(gen,sn),"i"],network.variables.at[("gen-p_nom",gen,pd.NaT),"i"]] = -p_max_pu.at[sn, gen]

        add_constraints(network,"gen-p_upper",constraints,constraint_matrix)

def define_link_constraints(network,snapshots):

    extendable_links_i = network.links.index[network.links.p_nom_extendable]
    extendable_links_i.name = "name"
    fixed_links_i = network.links.index[~network.links.p_nom_extendable]


    variables = pd.DataFrame(index=pd.MultiIndex.from_product([network.links.index,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["lower","upper"],
                             dtype=float)

    p_min_pu = get_switchable_as_dense(network, 'Link', 'p_min_pu', snapshots)
    p_max_pu = get_switchable_as_dense(network, 'Link', 'p_max_pu', snapshots)

    if len(fixed_links_i) > 0:
        variables.loc[(fixed_links_i,snapshots),"lower"] = p_min_pu.loc[snapshots,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom']).stack().swaplevel()
        variables.loc[(fixed_links_i,snapshots),"upper"] = p_max_pu.loc[snapshots,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom']).stack().swaplevel()

    variables.loc[(extendable_links_i,snapshots),"lower"] = -np.inf

    variables.loc[(extendable_links_i,snapshots),"upper"] = np.inf

    add_variables(network,"link-p",variables)


    ## Define link capacity variables if link is extendable ##

    if len(extendable_links_i) > 0:
        variables = pd.DataFrame(index=extendable_links_i,
                                 columns=["lower","upper"],
                                 dtype=float)

        variables.loc[extendable_links_i,"lower"] = network.links.loc[extendable_links_i,"p_nom_min"]

        variables.loc[extendable_links_i,"upper"] = network.links.loc[extendable_links_i,"p_nom_max"]

        add_variables(network,"link-p_nom",variables)


        ## Define link dispatch constraints for extendable links ##

        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([extendable_links_i,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = ">="
        constraints["i"] = range(len(constraints))
        constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

        for link in extendable_links_i:
            for sn in snapshots:
                constraint_matrix[constraints.at[(link,sn),"i"],network.variables.at[("link-p",link,sn),"i"]] = 1.
                constraint_matrix[constraints.at[(link,sn),"i"],network.variables.at[("link-p_nom",link,pd.NaT),"i"]] = -p_min_pu.at[sn, link]

        add_constraints(network,"link-p_lower",constraints,constraint_matrix)


        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([extendable_links_i,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = "<="
        constraints["i"] = range(len(constraints))
        constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

        for link in extendable_links_i:
            for sn in snapshots:
                constraint_matrix[constraints.at[(link,sn),"i"],network.variables.at[("link-p",link,sn),"i"]] = 1.
                constraint_matrix[constraints.at[(link,sn),"i"],network.variables.at[("link-p_nom",link,pd.NaT),"i"]] = -p_max_pu.at[sn, link]

        add_constraints(network,"link-p_upper",constraints,constraint_matrix)

def define_nodal_balance_constraints(network,snapshots):

    constraints = pd.DataFrame(index=pd.MultiIndex.from_product([network.buses.index,snapshots],
                                                                names=["name","snapshot"]),
                              columns=["sense","rhs"])

    constraints["rhs"] = -get_switchable_as_dense(network, 'Load', 'p_set', snapshots).multiply(network.loads.sign).groupby(network.loads.bus,axis=1).sum().reindex(columns=network.buses.index,fill_value=0.).stack().swaplevel()
    constraints["sense"] = "=="
    constraints["i"] = range(len(constraints))
    constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

    for gen in network.generators.index:
        bus = network.generators.at[gen,"bus"]
        sign = network.generators.at[gen,"sign"]
        for sn in snapshots:
            constraint_matrix[constraints.at[(bus,sn),"i"],network.variables.at[("gen-p",gen,sn),"i"]] = sign

    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

    for link in network.links.index:
        bus0 = network.links.at[link,"bus0"]
        bus1 = network.links.at[link,"bus1"]
        for sn in snapshots:
            constraint_matrix[constraints.at[(bus0,sn),"i"],network.variables.at[("link-p",link,sn),"i"]] = -1.
            constraint_matrix[constraints.at[(bus1,sn),"i"],network.variables.at[("link-p",link,sn),"i"]] = efficiency.at[sn,link]

    add_constraints(network,"nodal_balance",constraints,constraint_matrix)

    #TODO include multi-link

def define_linear_objective(network,snapshots):

    network.variables["obj"] = 0.
    extendable_generators_i = network.generators.index[network.generators.p_nom_extendable]
    extendable_generators_i.name = "name"
    extendable_links_i = network.links.index[network.links.p_nom_extendable]
    extendable_links_i.name = "name"

    mc = get_switchable_as_dense(network, 'Generator', 'marginal_cost', snapshots).multiply(network.snapshot_weightings[snapshots],axis=0)
    network.variables.loc[("gen-p",network.generators.index,snapshots),"obj"] = pd.concat([mc.stack().swaplevel()], keys=["gen-p"], names=["group"])

    mc = get_switchable_as_dense(network, 'Link', 'marginal_cost', snapshots).multiply(network.snapshot_weightings[snapshots],axis=0)
    network.variables.loc[("link-p",network.links.index,snapshots),"obj"] = pd.concat([mc.stack().swaplevel()], keys=["link-p"], names=["group"])

    cc = pd.concat([network.links.loc[extendable_links_i,"capital_cost"]], keys = [pd.NaT], names=["snapshot"]).reorder_levels(["name","snapshot"])
    cc = pd.concat([cc], keys = ["link-p_nom"], names=["group"])
    network.variables.loc[("link-p_nom",extendable_links_i),"obj"] = cc

    cc = pd.concat([network.generators.loc[extendable_generators_i,"capital_cost"]], keys = [pd.NaT], names=["snapshot"]).reorder_levels(["name","snapshot"])
    cc = pd.concat([cc], keys = ["gen-p_nom"], names=["group"])
    network.variables.loc[("gen-p_nom",extendable_generators_i),"obj"] = cc

    #TODO include constant term


def problem_to_lp(network,filename):

    f = open(filename, 'w')
    f.write('\\* LOPF \*\\n\nmin\nobj:\n')
    for var in network.variables.index:
        coeff = network.variables.at[var,"obj"]
        f.write("{}{} x{}\n".format("+" if coeff >= 0 else "",coeff,network.variables.at[var,"i"]))

    f.write("\n\ns.t.\n\n")

    for c in network.constraints.index:
        i = network.constraints.at[c,"i"]
        f.write("c{}:\n".format(i))
        sel = network.constraint_matrix.row == i
        for j,coeff in zip(network.constraint_matrix.col[sel],network.constraint_matrix.data[sel]):
            f.write("{}{} x{}\n".format("+" if coeff >= 0 else "",coeff,j))
        f.write("{} {}\n\n".format(network.constraints.at[c,"sense"].replace("==","="),network.constraints.at[c,"rhs"]))

    f.write("\nbounds\n")

    for var in network.variables.index:
        f.write("{} <= x{} <= {}\n".format(network.variables.at[var,"lower"],
                                           network.variables.at[var,"i"],
                                           network.variables.at[var,"upper"]))

    f.write("end\n")
    f.close()


def run_cbc(filename,sol_filename):
    options = "" #-dualsimplex -primalsimplex
    command = "cbc -printingOptions all -import {} -stat=1 -solve {} -solu {}".format(filename,options,sol_filename)
    print("Running command:")
    print(command)
    os.system(command)


def read_cbc(network,sol_filename):
    f = open(sol_filename,"r")
    data = f.readline()
    print(data)
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

    network.generators_t.p.loc[snapshots] = network.variables.loc["gen-p","sol"].unstack(level=0)

    network.links_t.p0.loc[snapshots] = network.variables.loc["link-p","sol"].unstack(level=0)
    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)
    network.links_t.p1.loc[snapshots] = -network.links_t.p0.loc[snapshots]*efficiency.loc[snapshots,:]


    network.generators.p_nom_opt = network.generators.p_nom
    network.generators.loc[network.generators.p_nom_extendable,"p_nom_opt"] = network.variables.loc["gen-p_nom","sol"].groupby(level=0).sum()

    network.links.p_nom_opt = network.links.p_nom
    network.links.loc[network.links.p_nom_extendable,"p_nom_opt"] = network.variables.loc["link-p_nom","sol"].groupby(level=0).sum()



def prepare_lopf_problem(network,snapshots):
   network.variables = pd.DataFrame()
   network.constraints = pd.DataFrame()
   network.constraint_matrix =lil_matrix((0,0))

   define_generator_constraints(network,snapshots)
   define_link_constraints(network,snapshots)
   define_nodal_balance_constraints(network,snapshots)
   define_linear_objective(network,snapshots)


def network_lopf(network, snapshots=None, solver_name="cbc"):

    snapshots = _as_snapshots(network, snapshots)

    prepare_lopf_problem(network,snapshots)

    problem_to_lp(network,"test.lp")

    if solver_name == "cbc":
        run_cbc("test.lp","test.sol")
        read_cbc(network,"test.sol")

    assign_solution(network,snapshots)
