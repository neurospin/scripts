# -*- coding: utf-8 -*-

"""
Created on Mon august 25 11:55:01 2014

@author: christophe
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


import utilities_plots as UP


################
# Input/Output #
################

INPUTDIR = "/volatile/share/2014_bd_dwi"
INPUT_RESULTS = os.path.join(INPUTDIR, "bd_dwi_enettv_csi_old/results_CSI.csv")

OUTPUTDIR = os.path.join(INPUTDIR, "figures")
if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)


# Parameters

krange = [100, 1000, 10000, 100000, -1]
alphas = [.01, .05, .1 , .5, 1.]
ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])
                       
                       
                      
# script
results = pd.read_csv(INPUT_RESULTS)
#x_data = results[["k", "a", "l1", "l2", "tv", ]]
#y_data = results[["k", "a", "l1", "l2", "recall_mean", ]]
#
#x_data.
#for k in krange :
#    plt.figure()
#    for a in alphas :

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def l1_ratio(l1, l2, tv):
    pos_val = np.array([1., 0., .5, .1, .9, .01, .99, .001, .999])
    if (tv == 1.0):
        return np.nan
    if ((l1 + l2) != 0.0):
        approx_ratio = l1/(l1 + l2)
        return find_nearest(pos_val, approx_ratio) 

results['l1_ratio'] = np.vectorize(l1_ratio)(results['l1'], results['l2'], results['tv'])

r1 = results.loc[results.k == 1000]
UP.f(r1, 'recall_mean', tv_pen='tv')
plt.show()