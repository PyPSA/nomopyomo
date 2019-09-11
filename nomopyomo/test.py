#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:56:34 2019

@author: fabian
"""

import pypsa
import pandas as pd
from .opt import expand_series, prefix


def check_storage_units_contraints(n, tol=1e-3):
    """
    Checks whether all storage units are balanced over time. This function
    requires the network to contain the separate variables p_store and
    p_dispatch, since they cannot be reconstructed from p. The latter results
    from times tau where p_store(tau) > 0 **and** p_dispatch(tau) > 0, which
    is allowed (even though not economic). Therefor p_store is necessarily
    equal to negative entries of p, vice versa for p_dispatch.
    """
    sus = n.storage_units
    sus_i = sus.index
    if sus_i.empty: return
    sns = n.snapshots
    c = 'StorageUnit'
    pnl = n.pnl(c)

    eh = expand_series(n.snapshot_weightings, sus_i)
    stand_eff = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)
    dispatch_eff = expand_series(n.df(c).efficiency_dispatch, sns).T.mul(eh)
    store_eff = expand_series(n.df(c).efficiency_store, sns).T.mul(eh)

    soc = pnl.state_of_charge

    store = store_eff * pnl.p_store#.clip(upper=0)
    dispatch = 1/dispatch_eff * pnl.p_dispatch#(lower=0)
    inflow = pypsa.descriptors.get_switchable_as_dense(n, c, 'inflow') * eh
    spill = eh[pnl.spill.columns] * pnl.spill
    start = soc.iloc[-1].where(sus.cyclic_state_of_charge,
                               sus.state_of_charge_initial)
    previous_soc = stand_eff * soc.shift().fillna(start)

    assert (spill.round(4) <= inflow[spill.columns].round(4)).all().all()


    reconstructed = (previous_soc.add(store, fill_value=0)
                    .add(inflow, fill_value=0)
                    .add(-dispatch, fill_value=0)
                    .add(-spill, fill_value=0))
    assert (reconstructed - soc).abs().max().max() < tol


def Incidence(n, branch_components=['Link', 'Line']):
    buses = n.buses.index
    return pd.concat([(n.df(c).assign(K=1).set_index('bus0', append=True)['K']
                     .unstack().reindex(columns=buses).fillna(0).T)
                     - (n.df(c).assign(K=1).set_index('bus1', append=True)['K']
                     .unstack().reindex(columns=buses).fillna(0).T)
                     for c in branch_components],
                     keys=branch_components, axis=1, sort=False)\
            .reindex(columns=n.branches().loc[branch_components].index)\
            .rename_axis(columns=['component', 'branch_i'])

def check_nodal_balance_constraint(n, tol=1e-9):
    """
    Helper function to double check whether network flow is balanced
    """
    K = Incidence(n)
    f = pd.concat([n.lines_t.p0, n.links_t.p0], axis=1,
                  keys=['Line', 'Link']).T
    assert (K.dot(f)).sum(0).max() < tol

def check_nominal_bounds(n, tol=1e-3):
    for c, attr in prefix.items():
        dispatch_attr = 'p0' if c in ['Line', 'Transformer', 'Link'] else attr
        assert (n.pnl(c)[dispatch_attr].abs().max().add(-tol)
                <= n.df(c)[attr + '_nom_opt']).all(), ('Test for nominal bounds'
                f' for component {c} failed')

