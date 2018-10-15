# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:35:27 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np
import mulm
import statsmodels.api as sm

n = 100
px = 5
py_info = 2
py_noize = 100

beta = np.array([1, 0, .5] + [0] * (px - 4) + [2]).reshape((px, 1))
X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
Y = np.random.randn(n, py_info + py_noize)
# Causal model: add X on the first py_info variable
Y[:, :py_info] += np.dot(X, beta)

# t-test all the regressors (by default mulm and sm do two-tailed tests)
contrasts = np.identity(X.shape[1])

## OLS with statmodels, need to iterate over Y columns
sm_tvals = list()
sm_pvals = list()
for j in xrange(Y.shape[1]):
    mod = sm.OLS(Y[:, j], X)
    sm_ttest = mod.fit().t_test(contrasts)
    sm_tvals.append(sm_ttest.tvalue)
    sm_pvals.append(sm_ttest.pvalue)

sm_tvals = np.asarray(sm_tvals).T
sm_pvals = np.asarray(sm_pvals).T

## OLS with MULM
mod = mulm.MUOLS(Y, X)
mulm_tvals, mulm_pvals, mulm_df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

# Check that results ar similar
np.allclose(mulm_tvals, sm_tvals)
np.allclose(mulm_pvals, sm_pvals)
