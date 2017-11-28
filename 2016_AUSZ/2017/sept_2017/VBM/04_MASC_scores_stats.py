#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:45:00 2017

@author: ad247405
"""


import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/MASCtot.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/y.npy'


X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
DX = np.load(INPUT_DATA_DX)

X = X[np.logical_not(np.isnan(y)).ravel(),:]
DX = DX[np.logical_not(np.isnan(y))]
y = y[np.logical_not(np.isnan(y))]

assert sum(DX==0) == 30 #controls
assert sum(DX==1) == 30 #ASD
assert sum(DX==2) == 18 #SCZ-ASD
assert sum(DX==3) == 32 #SCZ


print("HC mean MASC score: " + str(np.mean(y[DX==0])))
print("ASD mean MASC score: " + str(np.mean(y[DX==1])))
print("SCZ-ASD mean MASC score: " + str(np.mean(y[DX==2])))
print("SCZ mean MASC score: " + str(np.mean(y[DX==3])))


#df = pd.DataFrame()
#df["DX"] = DX
#df["MASCtot"] = y

f, p = stats.f_oneway(y[DX==0],y[DX==1],y[DX==2],y[DX==3])
print ("Overall, Anova Test; F = %s and p = %s" %(f,p))


mc = MultiComparison(df['MASCtot'], df['DX'])
result = mc.tukeyhsd()

print(result)
print(mc.groupsunique)