# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:15:22 2014

@author: edouard.duchesnay@cea.fr
"""
import sklearn.feature_selection
import numpy as np

class SelectPvalue(sklearn.feature_selection.SelectFwe):
    """Filter: Select the p-values corresponding to Family-wise error rate

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).

    alpha : float, optional
        The highest uncorrected p-value for features to keep.

    Attributes
    ----------
    `scores_` : array-like, shape=(n_features,)
        Scores of features.

    `pvalues_` : array-like, shape=(n_features,)
        p-values of feature scores.
    """

    def __init__(self, alpha=5e-2):
        super(SelectPvalue, self).__init__(alpha=alpha)
    def _get_support_mask(self):
        alpha = self.alpha
        print np.sum(self.pvalues_ < alpha)
        return (self.pvalues_ < alpha)