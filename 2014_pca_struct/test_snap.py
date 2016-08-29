# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:25:20 2016

@author: ad247405
"""


import os, sys
import json
import time
import numpy as np
import pandas as pd
import nibabel
import argparse
import parsimony.utils.consts as consts
import pca_tv

import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.functions.nesterov.l1tv as l1tv

import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils.start_vectors as start_vectors
import brainomics.mesh_processing as mesh_utils

from parsimony.algorithms.utils import AlgorithmSnapshot
#import sklearn.decomposition




config_filenane = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0/config.json"

     
###############################################################################
## Dataset
###############################################################################
os.chdir(os.path.dirname(config_filenane))
config = json.load(open(config_filenane))

# Data
X = np.load(config["data"]["X"])
assert X.shape == (500,10000)
####################################################################
## Algo parameters
###############################################################################
# a, l1, l2, tv penalties
global_pen = 0.01
tv_ratio = 0.5#1e-05
l1_ratio = 0.5

ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))

#Compute A and mask
masks = []
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"
for i in range(3):
    filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
    masks.append(np.load(filename))
im_shape = config["im_shape"]
Atv = nesterov_tv.A_from_shape(im_shape)


# PARSIMONY
########################################
snapshot = AlgorithmSnapshot('/neurospin/brainomics/2014_pca_struct/lambda_max/',saving_period=1).save_conesta
mod = pca_tv.PCA_L1_L2_TV(n_components=3,
                                l1=ll1, l2=ll2, ltv=ltv,
                                Atv=Atv,
                                criterion="frobenius",
                                eps=1e-4,
                                max_iter=100,
                                inner_max_iter=int(1e4),
                                output=True,callback=snapshot)  
mod.fit(X[0:250])                                
 


# sklearn
###########################################
os.system("cd /home/ad247405/git/scripts/2014_pca_struct")
import sklearn_modified.decomposition
prefix = '/neurospin/brainomics/2014_pca_struct/test_snapshot_tv/'
snapshot = AlgorithmSnapshot(prefix, saving_period=1).save        

mod = sklearn_modified.decomposition.SparsePCA(n_components=3,alpha=1,tol=1e-20,callback=snapshot,verbose=True)                            
mod.fit(X[0:])
                      
 

   #
class AlgorithmSnapshot:
    """
    Snapshot the algorithm state to disk. Its save_* methods should be provided
    as callback argument to FISTA or CONESTA. This callback will be called at
    each iteration."""

 
    def __init__(self, output_prefix, saving_period=1):
        self.output_prefix = output_prefix
        self.saving_period = saving_period
        self.cpt = 0
        self.continuation_ite_nb = list()  # ite nb where continuation occured

    def save(self, locals):
        self.cpt += 1
        if (self.cpt % self.saving_period) != 0:
            return
        snapshot = dict(cost=locals["current_cost"],code = locals["code"], dict = locals["dictionary"], errors = locals["errors"],time=locals["dt"],iter=locals["ii"])
        
        cpt_str = str(self.cpt)
        output_filename = self.output_prefix + 'conesta_ite:%s.npz' % (cpt_str)
        #print "AlgorithmSnapshot.save_conesta: save in ", output_filename
        np.savez_compressed(output_filename, **snapshot)







#
from parsimony.utils import plot_map2d
import os, sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import itertools
import scipy.ndimage as ndimage

INPUT = '/neurospin/brainomics/2014_pca_struct/test_snapshot_tv'

INPUT = '/neurospin/brainomics/2014_pca_struct/lambda_max'
######################PARSIMONY
for i in range(1,100):
    filename = os.path.join(INPUT, os.listdir(INPUT)[i])
    d = np.load(filename)
    r = {k.replace('Info.', ''):d[k] for k in d} 
    fig = plot_map2d(r["beta"].reshape(100,100))
#    r["time"] = r["time"][1:]
#    r["func_val"] = r["func_val"][1:]
    
    print i
   
   
y_label = r"$f(\beta^{k}) - f(\beta^{*} )$"
plt.plot(np.cumsum((r["time"])),np.log(r["func_val"][:]-r["func_val"][-1]),color='black')
plt.xlabel("Time [s]")
plt.ylabel(y_label)

############################################ SKLEARN 
INPUT = '/neurospin/brainomics/2014_pca_struct/test_snapshot_sparse_pca'

time = []
f = []
for i in range(1,429):
    filename = os.path.join(INPUT, os.listdir(INPUT)[i])
    d = np.load(filename)
    time.append(d['time'])
    f.append(d['cost'])
    
y_label = r"$f(\beta^{k}) - f(\beta^{*} )$"   
plt.plot(time,np.log(f-f[-1]))   
plt.xlabel("Time [s]")
plt.ylabel(y_label)


plt.plot(t2,f2)
plt.xlabel("Time [s]")
plt.ylabel(y_label)

f=f2
t=t2
#Time required to reach a given level of precision
arg = sum(f-f[-1]>10e-06)
(f-f[-1])[arg]

time0 = t[arg]
print "time to achieve such a precision", t[arg]


#Time required to reach a given level of precision
arg = sum(f0-f0[-1]>1e-06)
(f0-f0[-1])[arg]
time0 = t0[arg]

arg = sum(f1-f1[-1]>1e-06)
(f1-f1[-1])[arg]
time1 = t1[arg]

arg = sum(f2-f2[-1]>1e-06)
(f2-f2[-1])[arg]
time2 = t2[arg]

print "Total time to acquired a precision of 10e-05;", time0 + time1 + time2


t0 = np.load('/neurospin/brainomics/2014_pca_struct/time_enet/t0.npy')
f0 = np.load('/neurospin/brainomics/2014_pca_struct/time_enet/f0.npy')

t1 = np.load('/neurospin/brainomics/2014_pca_struct/time_enet/t1.npy')
f1 = np.load('/neurospin/brainomics/2014_pca_struct/time_enet/f1.npy')

t2 = np.load('/neurospin/brainomics/2014_pca_struct/time_enet/t2.npy')
f2 = np.load('/neurospin/brainomics/2014_pca_struct/time_enet/f2.npy')

from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy import interpolate


#ius = InterpolatedUnivariateSpline(t0, f0-f0[-1])
#x= np.linspace(0,50,10000)
#y = ius(x)
#plt.plot(t0,f0-f0[-1],'o',x,y)
#


plt.plot(t2,f2-f2[-1])

#Interpolate to find time necesarry to achieve a given precision
#########################################################################################
#ius = interpolate.Rbf(np.abs(f0-f0[-1]),t0)
ius = interpolate.Rbf(np.abs(f0-f0[-1]),t0,smooth=0.1)
x= np.linspace(0,np.abs(f0-f0[-1]).max())
y = ius(x)
plt.plot(np.abs(f0-f0[-1]),t0,'o',x,y)
time0 = ius(1e-03)

#ius = interp1d(np.abs(f1-f1[-1]),t1,kind='linear')
ius = interpolate.Rbf(np.abs(f1-f1[-1]),t1,smooth=2)
x= np.linspace(0,np.abs(f1-f1[-1]).max())
y = ius(x)
plt.plot(np.abs(f1-f1[-1]),t1,'o',x,y)
time1 = ius(1e-03)

#ius = interp1d(np.abs(f2-f2[-1]),t2)
ius = interpolate.Rbf(np.abs(f2-f2[-1]),t2,smooth=0.8)
x= np.linspace(0,np.abs(f2-f2[-1]).max())
y = ius(x)
plt.plot(np.abs(f2-f2[-1]),t2,'o',x,y)
time2 = ius(1e-03)

print "Total time necessary in order to achieve a precision of 1e-6;", time0 + time1 + time2



t0 = np.load('/neurospin/brainomics/2014_pca_struct/time_tv/t0.npy')
f0 = np.load('/neurospin/brainomics/2014_pca_struct/time_tv/f0.npy')

t1 = np.load('/neurospin/brainomics/2014_pca_struct/time_tv/t1.npy')
f1 = np.load('/neurospin/brainomics/2014_pca_struct/time_tv/f1.npy')

t2 = np.load('/neurospin/brainomics/2014_pca_struct/time_tv/t2.npy')
f2 = np.load('/neurospin/brainomics/2014_pca_struct/time_tv/f2.npy')



t = np.load('/neurospin/brainomics/2014_pca_struct/time_sparse/t.npy')
f = np.load('/neurospin/brainomics/2014_pca_struct/time_sparse/f.npy')

#ius = interp1d(np.abs(f-f[-1]),t)
ius = interpolate.Rbf(np.abs(f-f[-1]),t,smooth=0.1)
x= np.linspace(0,np.abs(f-f[-1]).max())
y = ius(x)
plt.plot(np.abs(f-f[-1]),t,'o',x,y)
time = ius(1e-04)
print "Total time necessary in order to achieve a precision of 1e-6 with Sparse PCA (Sklearn);", time

print ius(1e-03)
print ius(1e-04)
print ius(1e-05)
print ius(1e-06)



