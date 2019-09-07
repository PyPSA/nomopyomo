#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:38:10 2019

@author: fabian
"""

import pandas as pd
import csv, os
import numpy as np
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

lookup = pd.read_csv(os.path.dirname(__file__) + '/variables.csv',
                        index_col=['component', 'variable'])
prefix = lookup.droplevel(1).prefix[lambda ds: ~ds.index.duplicated()]

# =============================================================================
# writing functions
# =============================================================================
def write_objective(n, df):
    df.to_csv(n.objective_fn, sep='\n', index=False, header=False, mode='a')

xCounter = 0
def write_bound(n, lower, upper, axes=None):
    if (isinstance(lower, (pd.Series, pd.DataFrame)) &
        isinstance(upper, (pd.Series, pd.DataFrame))):
        shape = max([lower.shape, upper.shape])
        axes = lower.axes if shape == lower.shape else upper.axes
        lower = lower.astype(str)
        upper = upper.astype(str)
    elif isinstance(lower, (float, int)) & isinstance(upper, (float, int)):
        lower, upper = str(lower), str(upper)
        shape = tuple(map(len, axes))
    ser_or_frame = pd.DataFrame if len(shape) > 1 else pd.Series
    length = np.prod(shape)
    global xCounter
    xCounter += length
    variables = ('x' + np.array([str(x) for x in
                range(xCounter - length, xCounter)], dtype=object).reshape(shape))
    ser_or_frame(lower + ' <= ' + variables + ' <= ' + upper)\
        .to_csv(n.bounds_fn, sep='\n', index=False, header=False, mode='a')
    return ser_or_frame(variables, *axes)

cCounter = 0
def write_constraint(n, lhs, sense, rhs):
    shape = max([df.shape for df in [lhs, rhs]
                if isinstance(df, (pd.Series, pd.DataFrame))])
    axes = lhs.axes if shape == lhs.shape else rhs.axes
    length = np.prod(shape)
    global cCounter
    cCounter += length
    constraints = ('c' + np.array([str(x) for x in
                range(cCounter - length, cCounter)], dtype=object).reshape(shape))
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

var_ref_suffix = '_varref' # after solving replace with '_opt'
con_ref_suffix = '_conref' # after solving replace with ''

def numerical_to_string(val, append_space=True):
    if isinstance(val, (float, int)):
        s = f'+{float(val)}' if val >= 0 else f'{float(val)}'
    else:
        s = val.pipe(np.sign).replace([0, 1, -1], ['+', '+', '-'])\
                  .add(abs(val).astype(str)).astype(str)
    return s + ' ' if append_space else s

# references to vars and cons, rewrite this part to not store every reference
def _add_reference(n, df, c, attr, suffix, pnl=True):
    attr_name = attr + suffix
    if pnl:
        if attr_name in n.pnl(c):
            n.pnl(c)[attr_name][df.columns] = df
        else:
            n.pnl(c)[attr_name] = df
    else:
        n.df(c).loc[df.index, attr_name] = df

def set_varref(n, variables, c, attr, pnl=True):
    _add_reference(n, variables, c, attr, var_ref_suffix, pnl=pnl)

def set_conref(n, constraints, c, attr, pnl=True):
    _add_reference(n, constraints, c, attr, con_ref_suffix, pnl=pnl)

def pnl_var(n, c, attr):
    return n.pnl(c)[attr + var_ref_suffix]

def df_var(n, c, attr):
    return n.df(c)[attr + var_ref_suffix]

def pnl_con(n, c, attr):
    return n.pnl(c)[attr + con_ref_suffix]

def df_con(n, c, attr):
    return n.df(c)[attr + con_ref_suffix]

# 'getter' functions
def get_extendable_i(n, c):
    return n.df(c)[lambda ds:
        ds[f'{prefix[c]}_nom_extendable']].index

def get_non_extendable_i(n, c):
    return n.df(c)[lambda ds:
            ~ds[f'{prefix[c]}_nom_extendable']].index

def get_bounds_pu(n, c, sns, index=None, attr=None):
    max_pu = get_as_dense(n, c, f'{prefix[c]}_max_pu', sns)
    if c in n.passive_branch_components:
        min_pu = - max_pu
    elif c == 'StorageUnit':
        min_pu = pd.DataFrame(0, max_pu.index, max_pu.columns)
        if attr == 'p_store':
            max_pu = - get_as_dense(n, c, f'{prefix[c]}_min_pu', sns)
    else:
        min_pu = get_as_dense(n, c, f'{prefix[c]}_min_pu', sns)
    return (min_pu, max_pu) if index is None else (min_pu[index], max_pu[index])
