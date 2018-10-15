# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:35:27 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:35:27 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np
import mulm
import pylab as plt

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

mod = mulm.MUOLS(Y, X)
tvals, rawp, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)
tvals, maxT, df2 = mod.t_test_maxT(contrasts, two_tailed=True)
tvals3, minP, df3 = mod.t_test_minP(contrasts, two_tailed=True)

n, bins, patches = plt.hist([rawp[0,:], maxT[0,:], minP[0,:]],
                            color=['blue', 'red', 'green'],
                            label=['rawp','maxT', 'minP'])
plt.legend()
plt.show()