# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:22:57 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np


def missing_fill_with_mean(X, copy=True):
    """Fill missing values by column mean of non missing.

    Parameters
    ----------
    X array

    copy: boolean
        fill inplace or copy

    Return
    ------
    X, missing. X is the array with missing values filled by non missing.
    missing is a list of (col_index, mean_value, row_indices), where
    - col_index are columns with missing value,
    - mean_value is the mean computed over non missing
    - row_indices are the row indices of missing values
    """
    if copy:
        X = X.copy()
    missing = list()
    for j in xrange(X.shape[1]):
        x = X[:, j]
        nans = np.isnan(x)
        if nans.sum() == 0:
            continue
        mean = x[np.logical_not(nans)].mean()
        X[nans, j] = mean
        missing.append((j, mean, np.where(nans)[0]))
    return X, missing