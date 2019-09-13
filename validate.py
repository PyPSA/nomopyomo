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

# add additional storage unit
n.add('StorageUnit', 'su', bus='Manchester', marginal_cost=10, inflow=50,
      p_nom_extendable=True, capital_cost=10, p_nom=2000,
      efficiency_dispatch=0.5,
      cyclic_state_of_charge=True, state_of_charge_initial=1000)

# add additional store
n.add('Bus', 'storebus', carrier = 'hydro', x=-5, y=55)
n.madd('Link', ['go', 'return'], 'storelink',
       bus0=['Manchester', 'storebus'], bus1=['storebus', 'Manchester'],
       p_nom=200, efficiency=.9)
n.madd('Store', ['store'], bus='storebus', e_nom=2000,
       e_nom_extendable=True, marginal_cost=10,
       capital_cost=20, e_nom_max=5000, e_initial=100,
       e_cyclic=True)

#solve it with gurobi and validate
nomopyomo.lopf(n, solver_name='gurobi')

nomopyomo.test.check_nominal_bounds(n)
nomopyomo.test.check_nodal_balance_constraint(n)
nomopyomo.test.check_storage_unit_contraints(n)
nomopyomo.test.check_store_contraints(n)

#solve it with cbc and validate
nomopyomo.lopf(n, solver_name='cbc')

nomopyomo.test.check_nominal_bounds(n)
nomopyomo.test.check_nodal_balance_constraint(n)
nomopyomo.test.check_storage_unit_contraints(n)
nomopyomo.test.check_store_contraints(n)
