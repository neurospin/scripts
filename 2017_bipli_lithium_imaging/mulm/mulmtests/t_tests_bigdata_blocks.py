# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:03:13 2015

@author: cp243490
"""
import numpy as np
import time
import tempfile
from mulm import MUOLS

start_time = time.time()

n = 1000
px = 4
py_noize = 2000
py_info = 100

## Build data
beta = np.array([1, 0, .5] + [0] * (px - 4) + [2]).reshape((px, 1))
X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
f = tempfile.mktemp(dir="C:\\Users\\js247994\\Downloads\\pylearn-mulm")
Y = np.random.randn(n, py_info + py_noize)
# Causal model: add X on the first py_info variable
Y[:, :py_info] += np.dot(X, beta)
np.save(f, Y)
del Y

contrasts = np.identity(X.shape[1])

## Read data Y as a memory map
##############################
Y_memmap = np.load(f + '.npy', mmap_mode='r')

# univariate analysis: fit by blocks of 2**26 elements
muols = MUOLS(Y=Y_memmap, X=X)
time1 = time.time()
muols.fit(block=True, max_elements=2 ** 26)
time2 = time.time()
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
print(time2 - time1)
del muols

# univariate analysis: fit by blocks of 2**27 elements
muols = MUOLS(Y=Y_memmap, X=X)
time3 = time.time()
muols.fit(block=True, max_elements=2 ** 27)
time4 = time.time()
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
print(time4 - time3)
del muols

# univariate analysis: fit in one go
muols = MUOLS(Y=Y_memmap, X=X)
time5 = time.time()
muols.fit(block=False)
time6 = time.time()
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)

print(time6 - time5)
del muols
del Y_memmap

## Load Y
##########
Y = np.load(f + '.npy')

# univariate analysis: fit by blocks of 2**26 elements
muols = MUOLS(Y=Y, X=X)
time7 = time.time()
muols.fit(block=True, max_elements=2 ** 26)
time8 = time.time()
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
print(time8 - time7)
del muols

# univariate analysis: fit by blocks of 2**27 elements
muols = MUOLS(Y=Y, X=X)
time9 = time.time()
muols.fit(block=True, max_elements=2 ** 27)
time10 = time.time()
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
print(time10 - time9)
del muols

# univariate analysis: fit in one go
muols = MUOLS(Y=Y, X=X)
time11 = time.time()
muols.fit(block=False)
time12 = time.time()
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
print(time12 - time11)
del muols
del Y