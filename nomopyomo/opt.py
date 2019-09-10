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
nominals = lookup.query('nominal')

# =============================================================================
# writing functions
# =============================================================================

xCounter = 0
def write_bound(n, lower, upper, axes=None):
    """
    Writer function for writing out mutliple variables to the corresponding
    bounds file. If lower and upper are floats it demands to give pass axes,
    a tuple of (index, columns) or (index), for creating the variable of same
    upper and lower bounds.
    Return a series or frame with variable references.
    """
    if axes is None:
        axes, shape = broadcasted_axes(lower, upper)
    else:
        shape = tuple(map(len, axes))
    ser_or_frame = pd.DataFrame if len(shape) > 1 else pd.Series
    length = np.prod(shape)
    global xCounter
    xCounter += length
    variables = np.array([f'x{x}' for x in range(xCounter - length, xCounter)],
                          dtype=object).reshape(shape)

    for s in stringadd(lower, ' <= ', variables, ' <= ', upper, '\n').flatten():
        n.bounds_f.write(s)
    return ser_or_frame(variables, *axes)

cCounter = 0
def write_constraint(n, lhs, sense, rhs, axes=None):
    """
    Writer function for writing out mutliple constraints to the corresponding
    constraints file. If lower and upper are numpy.ndarrays it axes must not be
    None but a tuple of (index, columns) or (index).
    Return a series or frame with constraint references.
    """
    if axes is None:
        axes, shape = broadcasted_axes(lhs, rhs)
    else:
        shape = tuple(map(len, axes))
    ser_or_frame = pd.DataFrame if len(shape) > 1 else pd.Series
    length = np.prod(shape)
    global cCounter
    cCounter += length
    cons = np.array([f'c{x}' for x in range(cCounter - length, cCounter)],
                            dtype=object).reshape(shape)
    for c in stringadd(cons, ':\n', lhs, '\n', sense, '\n', rhs, '\n\n').flatten():
        n.constraints_f.write(c)
    return ser_or_frame(cons, *axes)


# =============================================================================
# helpers, helper functions
# =============================================================================

var_ref_suffix = '_varref' # after solving replace with '_opt'
con_ref_suffix = '_conref' # after solving replace with ''

def broadcasted_axes(*dfs):
    """
    Helper function which, from a collection of arrays, series, frames and other
    values, retrieves the axes of series and frames which result from
    broadcasting operations. It checks whether index and columns of given
    series and frames, repespectively, are aligned. Using this function allows
    to subsequently use pure numpy operations and keep the axes in the
    background.
    """
    axes = []
    shape = ()
    for df in dfs:
        if isinstance(df, (pd.Series, pd.DataFrame)):
            if len(axes):
                assert (axes[-1] == df.axes[-1]).all()
            axes = df.axes if len(df.axes) > len(axes) else axes
            shape = tuple(map(len, axes))
    return axes, shape


def stringadd(*vals, return_axes=False):
    """
    Fast way to join arrays, series, frames of strings together. Returns
    a np.ndarray of strings. If return_axes is set to True and a pd.Series or
    pd.DataFrame was past the corresponding axes is past.
    """
    axes, shape = broadcasted_axes(*vals)
    vals = [val.values if isinstance(val, (pd.Series, pd.DataFrame)) else val
            for val in vals]
    vals = [numerical_to_string(val) for val in vals]
    start = np.repeat('', np.prod(shape)).reshape(shape).astype(object)
    if return_axes:
        return sum(vals, start), axes
    else:
        return sum(vals, start)


def numerical_to_string(val, append_space=True):
    """
    Converts arrays, series or frames of string to strings of numericals with
    with related sign (necessary for the lp file).
    """
    if isinstance(val, str):
        return val
    if isinstance(val, (float, int)):
        s = f'+{float(val)}' if val >= 0 else f'{float(val)}'
        return s + ' ' if append_space else s
    if isinstance(val, np.ndarray):
        if val.dtype == object:
            return val
        signs = pd.Series(val) if val.ndim == 1 else pd.DataFrame(val)
        signs = signs.pipe(np.sign).replace([0, 1, -1], ['+', '+', '-']).values
    else:
        if val.values.dtype == object:
            return val
        signs = val.pipe(np.sign).replace([0, 1, -1], ['+', '+', '-']).values
    s = charprepend(signs, abs(val).astype(str))
    return charappend(s, ' ') if append_space else s


#weigh faster than adding string
def charappend(df, char):
    """
    Fast way to append a char or string to a large pd.DataFrame.
    """
    if isinstance(df, np.ndarray):
        return df + char
    d = df.copy()
    d[:] = d.values + char
    return d

def charprepend(df, char):
    """
    Fast way to prepend a char or string to a large pd.DataFrame.
    """
    if isinstance(df, np.ndarray):
        return df + char
    d = df.copy()
    d[:] = char + d.values
    return d


# =============================================================================
#  'getter' functions
# =============================================================================
def get_extendable_i(n, c):
    """
    Getter function. Get the index of extendable elements of a given component.
    """
    return n.df(c)[lambda ds:
        ds[f'{prefix[c]}_nom_extendable']].index

def get_non_extendable_i(n, c):
    """
    Getter function. Get the index of non-extendable elements of a given
    component.
    """
    return n.df(c)[lambda ds:
            ~ds[f'{prefix[c]}_nom_extendable']].index

def get_bounds_pu(n, c, sns, index=None, attr=None):
    """
    Getter function to retrieve the per unit bounds of a given compoent for
    given snapshots and possible subset of elements (e.g. non-extendables).
    Depending on the attr you can further specify the bounds of the variable
    you are looking at, e.g. p_store for storage units.

    Parameters
    ----------
    n : pypsa.Network
    c : string
        Component name, e.g. "Generator", "Line".
    sns : pandas.Index/pandas.DateTimeIndex
        set of snapshots for the bounds
    index : pd.Index, default None
        Subset of the component elements. If None (default) bounds of all
        elements are returned.
    attr : string, default None
        attribute name for the bounds, e.g. "p", "s", "p_store"

    """
    max_pu = get_as_dense(n, c, f'{prefix[c]}_max_pu', sns)
    if c in n.passive_branch_components:
        min_pu = - max_pu
    elif c == 'StorageUnit':
        min_pu = pd.DataFrame(0, max_pu.index, max_pu.columns)
        if attr == 'p_store':
            max_pu = - get_as_dense(n, c, f'{prefix[c]}_min_pu', sns)
        if attr == 'state_of_charge':
            max_pu = pd.concat([n.df(c).max_hours]*len(sns), keys=sns, axis=1).T
            min_pu = - max_pu
    else:
        min_pu = get_as_dense(n, c, f'{prefix[c]}_min_pu', sns)
    return (min_pu, max_pu) if index is None else (min_pu[index], max_pu[index])


# =============================================================================
#  references to vars and cons, rewrite this part to not store every reference
# =============================================================================
def _add_reference(n, df, c, attr, suffix, pnl=True):
    attr_name = attr + suffix
    if pnl:
        if attr_name in n.pnl(c):
            n.pnl(c)[attr_name][df.columns] = df
        else:
            n.pnl(c)[attr_name] = df
        if n.pnl(c)[attr_name].shape[1] == n.df(c).shape[0]:
            n.pnl(c)[attr_name] = n.pnl(c)[attr_name].reindex(columns=n.df(c).index)
    else:
        n.df(c).loc[df.index, attr_name] = df

def set_varref(n, variables, c, attr, pnl=True):
    """
    Sets variable references to the network.
    If pnl is False it stores a series of variable names in the static
    dataframe of the given component. The columns name is then given by the
    attribute name attr and the globally define var_ref_suffix.
    If pnl is True if stores the given frame of references in the component
    dict of time-depending quantities, e.g. network.generators_t .
    """
    _add_reference(n, variables, c, attr, var_ref_suffix, pnl=pnl)

def set_conref(n, constraints, c, attr, pnl=True):
    """
    Sets constraint references to the network.
    If pnl is False it stores a series of constraints names in the static
    dataframe of the given component. The columns name is then given by the
    attribute name attr and the globally define con_ref_suffix.
    If pnl is True if stores the given frame of references in the component
    dict of time-depending quantities, e.g. network.generators_t .
    """
    _add_reference(n, constraints, c, attr, con_ref_suffix, pnl=pnl)

def pnl_var(n, c, attr):
    """
    Retrieves variable references for a given time-depending attribute attr
    of a give component c.

    Example
    -------
    pnl_var(n, 'Generator', 'p')
    """
    return n.pnl(c)[attr + var_ref_suffix]

def df_var(n, c, attr):
    """
    Retrieves variable references for a given static attribute attr
    of a give component c.

    Example
    -------
    df_var(n, 'Line', 's_nom')
    """
    return n.df(c)[attr + var_ref_suffix]

def pnl_con(n, c, attr):
    """
    Retrieves constraint references for a given time-depending attribute attr
    of a give component c.

    Example
    -------
    pnl_con(n, 'Generator', 'mu_upper')
    """
    return n.pnl(c)[attr + con_ref_suffix]

def df_con(n, c, attr):
    """
    Retrieves contraint references for a given static attribute attr
    of a give component c.

    Example
    -------
    df_con(n, 'GlobalConstraint', 'mu')
    """
    return n.df(c)[attr + con_ref_suffix]