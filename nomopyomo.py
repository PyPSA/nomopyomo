
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
import datetime as dt

import os

import gc

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

    constraints = pd.concat([constraints], keys=[group], names=['group'])

    network.constraints = pd.concat((network.constraints,constraints),
                                  sort=False)

    #if constraints is empty, it can mangle the dtype
    network.constraints["i"] = network.constraints["i"].astype(int)

    #network.constraints = network.constraints.sort_index()


def extendable_attribute_constraints(network,snapshots,component,attr,marginal_cost=True):

    df = getattr(network,network.components[component]["list_name"])

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

    if marginal_cost and len(df) > 0:
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

    add_constraints(network,"nodal_balance",constraints)

    #TODO include multi-link

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
   network.constraint_matrix = {}

   print("before gen",dt.datetime.now()-now)
   define_generator_constraints(network,snapshots)
   print("before link",dt.datetime.now()-now)
   define_link_constraints(network,snapshots)
   print("before store",dt.datetime.now()-now)
   define_store_constraints(network,snapshots)
   print("before nodal",dt.datetime.now()-now)
   define_nodal_balance_constraints(network,snapshots)


def network_lopf(network, snapshots=None, solver_name="cbc"):

    snapshots = _as_snapshots(network, snapshots)

    print("before prep",dt.datetime.now()-now)
    prepare_lopf_problem(network,snapshots)
    gc.collect()
    print("before write file",dt.datetime.now()-now)
    problem_to_lp(network,"test.lp")
    gc.collect()
    print("before run",dt.datetime.now()-now)

    if solver_name == "cbc":
        run_cbc("test.lp","test.sol")
        print("before read",dt.datetime.now()-now)
        read_cbc(network,"test.sol")

    gc.collect()
    print("before assign",dt.datetime.now()-now)
    assign_solution(network,snapshots)
    print("end",dt.datetime.now()-now)
    gc.collect()
