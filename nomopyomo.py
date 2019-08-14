
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

    #if variables is empty, it can mangle the dtype
    network.variables["i"] = network.variables["i"].astype(int)

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

    #if constraints is empty, it can mangle the dtype
    network.constraints["i"] = network.constraints["i"].astype(int)

    if network.constraint_matrix.shape[1] < len(network.variables):
        network.constraint_matrix = hstack((network.constraint_matrix,lil_matrix((network.constraint_matrix.shape[0],len(network.variables)-network.constraint_matrix.shape[1]))))

    network.constraint_matrix = vstack((network.constraint_matrix,constraint_matrix))

def extendable_attribute_constraints(network,snapshots,component,attr):

    df = getattr(network,network.components[component]["list_name"])

    ext = df.index[df[attr+"_nom_extendable"]]
    ext.name = "name"
    fix = df.index[~df[attr+"_nom_extendable"]]
    fix.name = "name"

    max_pu = get_switchable_as_dense(network, component, attr+'_max_pu', snapshots)
    min_pu = get_switchable_as_dense(network, component, attr+'_min_pu', snapshots)

    variables = pd.DataFrame(index=pd.MultiIndex.from_product([df.index,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["lower","upper"],
                             dtype=float)

    if len(fix) > 0:
        variables.loc[(fix,snapshots),"lower"] = min_pu.loc[snapshots,fix].multiply(df.loc[fix,attr+'_nom']).stack().swaplevel()
        variables.loc[(fix,snapshots),"upper"] = max_pu.loc[snapshots,fix].multiply(df.loc[fix,attr+'_nom']).stack().swaplevel()

    variables.loc[(ext,snapshots),"lower"] = -np.inf
    variables.loc[(ext,snapshots),"upper"] = np.inf

    add_variables(network,"{}-{}".format(component,attr),variables)

    if len(ext) > 0:
        variables = pd.DataFrame(index=ext,
                                 columns=["lower","upper"],
                                 dtype=float)

        variables.loc[ext,"lower"] = df.loc[ext,attr+"_nom_min"]
        variables.loc[ext,"upper"] = df.loc[ext,attr+"_nom_max"]

        add_variables(network,"{}-{}_nom".format(component,attr),variables)


        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([ext,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = ">="
        constraints["i"] = range(len(constraints))
        constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

        for unit in ext:
            for sn in snapshots:
                constraint_matrix[constraints.at[(unit,sn),"i"],network.variables.at[("{}-{}".format(component,attr),unit,sn),"i"]] = 1.
                constraint_matrix[constraints.at[(unit,sn),"i"],network.variables.at[("{}-{}_nom".format(component,attr),unit,pd.NaT),"i"]] = -min_pu.at[sn, unit]

        add_constraints(network,"{}-{}_lower".format(component,attr),constraints,constraint_matrix)


        constraints = pd.DataFrame(index=pd.MultiIndex.from_product([ext,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["sense","rhs"])
        constraints["rhs"] = 0.
        constraints["sense"] = "<="
        constraints["i"] = range(len(constraints))
        constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

        for unit in ext:
            for sn in snapshots:
                constraint_matrix[constraints.at[(unit,sn),"i"],network.variables.at[("{}-{}".format(component,attr),unit,sn),"i"]] = 1.
                constraint_matrix[constraints.at[(unit,sn),"i"],network.variables.at[("{}-{}_nom".format(component,attr),unit,pd.NaT),"i"]] = -max_pu.at[sn, unit]

        add_constraints(network,"{}-{}_upper".format(component,attr),constraints,constraint_matrix)


def define_generator_constraints(network,snapshots):

    if network.generators.committable.any():
        logger.warning("Committable generators are currently not supported")

    extendable_attribute_constraints(network,snapshots,"Generator","p")

def define_link_constraints(network,snapshots):

    extendable_attribute_constraints(network,snapshots,"Link","p")


def define_store_constraints(network,snapshots):

    variables = pd.DataFrame(index=pd.MultiIndex.from_product([network.stores.index,snapshots],
                                                              names=["name","snapshot"]),
                             columns=["lower","upper"],
                             dtype=float)

    variables["lower"] = -np.inf
    variables["upper"] = np.inf

    add_variables(network,"Store-p",variables)

    extendable_attribute_constraints(network,snapshots,"Store","e")

    ## Builds the constraint -e_now + e_previous - p == 0 ##

    constraints = pd.DataFrame(index=pd.MultiIndex.from_product([network.stores.index,snapshots],
                                                                names=["name","snapshot"]),
                              columns=["sense","rhs"])
    constraints["sense"] = "=="
    constraints["rhs"] = 0.
    constraints["i"] = range(len(constraints))
    constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

    stores = network.stores
    for store in stores.index:
        for i,sn in enumerate(snapshots):
            constraint_matrix[constraints.at[(store,sn),"i"],network.variables.at[("Store-e",store,sn),"i"]] = -1.

            elapsed_hours = network.snapshot_weightings[sn]

            if i == 0 and not stores.at[store,"e_cyclic"]:
                constraints.at[(store,sn),"rhs"] = -((1-stores.at[store,"standing_loss"])**elapsed_hours
                                         * stores.at[store,"e_initial"])
            else:
                constraint_matrix[constraints.at[(store,sn),"i"],network.variables.at[("Store-e",store,snapshots[i-1]),"i"]] = (1-stores.at[store,"standing_loss"])**elapsed_hours

            constraint_matrix[constraints.at[(store,sn),"i"],network.variables.at[("Store-p",store,sn),"i"]] = -elapsed_hours

    add_constraints(network,"Store",constraints,constraint_matrix)


def define_nodal_balance_constraints(network,snapshots):

    constraints = pd.DataFrame(index=pd.MultiIndex.from_product([network.buses.index,snapshots],
                                                                names=["name","snapshot"]),
                              columns=["sense","rhs"])

    constraints["rhs"] = -get_switchable_as_dense(network, 'Load', 'p_set', snapshots).multiply(network.loads.sign).groupby(network.loads.bus,axis=1).sum().reindex(columns=network.buses.index,fill_value=0.).stack().swaplevel()
    constraints["sense"] = "=="
    constraints["i"] = range(len(constraints))
    constraint_matrix = lil_matrix((len(constraints),len(network.variables)))

    for component in ["Generator","Store"]:
        df = getattr(network,network.components[component]["list_name"])
        for unit in df.index:
            bus = df.at[unit,"bus"]
            sign = df.at[unit,"sign"]
            for sn in snapshots:
                constraint_matrix[constraints.at[(bus,sn),"i"],network.variables.at[(component+"-p",unit,sn),"i"]] = sign

    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

    for link in network.links.index:
        bus0 = network.links.at[link,"bus0"]
        bus1 = network.links.at[link,"bus1"]
        for sn in snapshots:
            constraint_matrix[constraints.at[(bus0,sn),"i"],network.variables.at[("Link-p",link,sn),"i"]] = -1.
            constraint_matrix[constraints.at[(bus1,sn),"i"],network.variables.at[("Link-p",link,sn),"i"]] = efficiency.at[sn,link]

    add_constraints(network,"nodal_balance",constraints,constraint_matrix)

    #TODO include multi-link

def define_linear_objective(network,snapshots):

    network.variables["obj"] = 0.

    for component in ["Generator","Link","Store"]:
        df = getattr(network,network.components[component]["list_name"])
        attr = "e" if component == "Store" else "p"
        ext = df.index[df[attr+"_nom_extendable"]]
        ext.name = "name"
        cc = pd.concat([df.loc[ext,"capital_cost"]], keys = [pd.NaT], names=["snapshot"]).reorder_levels(["name","snapshot"])
        cc = pd.concat([cc], keys = ["{}-{}_nom".format(component,attr)], names=["group"])
        if len(ext) > 0:
            network.variables.loc[("{}-{}_nom".format(component,attr),ext),"obj"] = cc

        mc = get_switchable_as_dense(network, component, 'marginal_cost', snapshots).multiply(network.snapshot_weightings[snapshots],axis=0)
        if len(df) > 0:
            network.variables.loc[("{}-{}".format(component,attr),df.index,snapshots),"obj"] = pd.concat([mc.stack().swaplevel()], keys=["{}-{}".format(component,attr)], names=["group"])

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
   network.constraint_matrix =lil_matrix((0,0))

   define_generator_constraints(network,snapshots)
   define_link_constraints(network,snapshots)
   define_store_constraints(network,snapshots)
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
