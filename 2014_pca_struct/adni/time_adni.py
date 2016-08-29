# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:28:02 2016

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



INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs"    
TEMPLATE_PATH = os.path.join(INPUT_BASE_DIR, "freesurfer_template")                      
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_10comp"


BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
X=np.load(os.path.join(INPUT_BASE_DIR,'X.npy'))   
   
   
global_pen = 0.1
tv_ratio =1e-05# 0.5
l1_ratio = 0.1

ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))



mesh_coord, mesh_triangles = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))
mask = np.load(os.path.join(INPUT_BASE_DIR, "mask.npy"))
import parsimony.functions.nesterov.tv as tv_helper
Atv = tv_helper.linear_operator_from_mesh(mesh_coord, mesh_triangles, mask=mask)
     

# PARSIMONY
########################################
from parsimony.algorithms.utils import AlgorithmSnapshot
snapshot = AlgorithmSnapshot('/neurospin/brainomics/2014_pca_struct/adni/adni_time/enet_1e-6/',saving_period=1).save_conesta

mod = pca_tv.PCA_L1_L2_TV(n_components=3,
                                l1=ll1, l2=ll2, ltv=ltv,
                                Atv=Atv,
                                criterion="frobenius",
                                eps=1e-6,
                                inner_eps=1e-1,
                                max_iter=100,
                                inner_max_iter=int(1e4),
                                output=True,callback=snapshot)  
mod.fit(X)                                
    
    
    
    
    
 
  # Plot time and precision
#########################################################################################

import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from scipy.interpolate import Rbf

a = np.load('/neurospin/brainomics/2014_pca_struct/adni/adni_time/enet_1e-6/nipals_ite_for_comp:2.npz')
eps_0 = a['func'][0] - a['func'][0][-1]
eps_1 = a['func'][1] - a['func'][1][-1]
eps_2 = a['func'][2] - a['func'][2][-1]

time_0 = a['time'][0]
time_1 = a['time'][1]
time_2 = a['time'][2]

plt.plot(np.cumsum(time_0),np.abs(eps_0))
plt.xlabel('Time')
plt.ylabel('f - f*')
plt.title('ADNI - component 0')
plt.yscale('log')

plt.plot(np.cumsum(time_1),np.abs(eps_1))
plt.xlabel('Time')
plt.ylabel('f - f*')
plt.title('ADNI - component 1')
plt.yscale('log')

plt.plot(np.cumsum(time_2),np.abs(eps_2))
plt.xlabel('Time')
plt.ylabel('f - f*')
plt.title('ADNI - component 2')
plt.yscale('log')
#########################################################################################


#Interpolate to find time necesarry to achieve a given precision
#########################################################################################

ius_0 = Rbf(np.abs(eps_0),np.cumsum(time_0),smooth=0.1)
x= np.linspace(0,np.abs(eps_0).max())
y = ius_0(x)
plt.plot(np.abs(eps_0),np.cumsum(time_0),'o',x,y)


ius_1 = Rbf(np.abs(eps_1),np.cumsum(time_1),smooth=0.1)
x= np.linspace(0,np.abs(eps_1).max())
y = ius_1(x)
plt.plot(np.abs(eps_1),np.cumsum(time_1),'o',x,y)


ius_2 = Rbf(np.abs(eps_2),np.cumsum(time_2),smooth=0.1)
x= np.linspace(0,np.abs(eps_2).max())
y = ius_2(x)
plt.plot(np.abs(eps_2),np.cumsum(time_2),'o',x,y)


print "Total time necessary in order to achieve a precision of 1e-3;", ius_0(1e-03) + ius_1(1e-03) + ius_2(1e-03) 
print "Total time necessary in order to achieve a precision of 1e-4;", ius_0(1e-04) + ius_1(1e-04) + ius_2(1e-04) 
print "Total time necessary in order to achieve a precision of 1e-5;", ius_0(1e-05) + ius_1(1e-05) + ius_2(1e-05) 
print "Total time necessary in order to achieve a precision of 1e-6;", ius_0(1e-06) + ius_1(1e-06) + ius_2(1e-06) 
  
    