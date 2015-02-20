# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:09:36 2015

@author: fh235918
"""
import numpy as np
def fdr(p):
    """ FDR correction for multiple comparisons.

    Computes fdr corrected p-values from an array o of multiple-test false
    positive levels (uncorrected p-values) a set after removing nan values,
    following Benjamin & Hockenberg procedure.

    Parameters
    ----------
    p : 1D numpy.array
        Input uncorrected p-values.

    Returns
    -------
    p_fdr : 1D numpy.array
        FDR corrected p-values.
    """
    if p.ndim > 1:
        raise ValueError('Expect 1D array')

    not_nan = p.shape[0]
    p_fdr = np.nan + np.ones(p.shape)
    indices = p.argsort()
    p_sorted = p[indices]
    not_nan = np.sum(np.logical_not(np.isnan(p)))
    if not_nan > 0:
        qt = np.minimum(
            1, not_nan * p_sorted[:not_nan] / (np.arange(not_nan) + 1))
        minimum = np.inf
        for n in range(not_nan - 1, -1, -1):
            minimum = min(minimum, qt[n])
            p_fdr[indices[n]] = minimum
    return p_fdr