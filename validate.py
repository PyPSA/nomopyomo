#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:04:16 2019

@author: fabian
"""

import pypsa
import nomopyomo
from pathlib import Path

network_path = str(Path(pypsa.__file__).parent.parent.joinpath('examples')
                .joinpath('ac-dc-meshed').joinpath('ac-dc-data'))
n = pypsa.Network(network_path)

#set one generator to non-exdendable
n.generators.loc[n.generators.carrier == 'gas', 'ramp_limit_down'] = 0.005
n.generators.loc[n.generators.carrier == 'gas', 'ramp_limit_up'] = 0.005
n.generators.loc[n.generators.carrier == 'gas', 'p_nom_extendable'] = False

#fix one generator
#n.generators_t.p_set.loc[n.snapshots[:5], 'Norway Gas'] = 200
#n.generators_t.p_set.loc[n.snapshots[8:], 'Manchester Gas'] = 200
# add additional storage unit
#n.add('StorageUnit', 'su', bus='Manchester', marginal_cost=10, inflow=50,
#      p_nom_extendable=True, capital_cost=10, p_nom=2000,
#      efficiency_dispatch=0.5,
#      cyclic_state_of_charge=True, state_of_charge_initial=1000)
#
##fix one soc timestep
#n.storage_units_t.state_of_charge_set.loc[n.snapshots[7], 'su'] = -100

# add another storage unit
#n.add('StorageUnit', 'methanization', bus='Manchester', marginal_cost=10,
#      p_nom_extendable=True, capital_cost=50, p_nom=2000,
#      efficiency_dispatch=0.5, carrier='gas',
#      cyclic_state_of_charge=False, state_of_charge_initial=1000)

# add additional store
#n.add('Bus', 'storebus', carrier = 'hydro', x=-5, y=55)
#n.madd('Link', ['go', 'return'], 'storelink',
#       bus0=['Manchester', 'storebus'], bus1=['storebus', 'Manchester'],
#       p_nom=100, efficiency=.9, p_nom_extendable=True, p_nom_max=1000)
#n.madd('Store', ['store'], bus='storebus', e_nom=2000,
#       e_nom_extendable=True, marginal_cost=10,
#       capital_cost=20, e_nom_max=5000, e_initial=100,
#       e_cyclic=True)

#nomopyomo.prepare_lopf(n, working_mode=True)

#solve it with gurobi and validate
nomopyomo.lopf(n, solver_name='gurobi', remove_references=True)

nomopyomo.test.check_nominal_bounds(n)
nomopyomo.test.check_nodal_balance_constraint(n)
nomopyomo.test.check_storage_unit_contraints(n)
nomopyomo.test.check_store_contraints(n)

##solve it with cbc and validate
#nomopyomo.lopf(n, solver_name='cbc')
#
#nomopyomo.test.check_nominal_bounds(n)
#nomopyomo.test.check_nodal_balance_constraint(n)
#nomopyomo.test.check_storage_unit_contraints(n)
#nomopyomo.test.check_store_contraints(n)
