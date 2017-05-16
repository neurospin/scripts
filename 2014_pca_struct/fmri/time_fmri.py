# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:21:46 2016

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
import nibabel as nib

import parsimony.functions.nesterov.tv
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.functions.nesterov.l1tv as l1tv

import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils.start_vectors as start_vectors
import brainomics.mesh_processing as mesh_utils

from parsimony.algorithms.utils import AlgorithmSnapshot
INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/fmri"
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,"T.npy")
INPUT_MASK = os.path.join(INPUT_BASE_DIR,"mask.nii.gz")

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
X=np.load(os.path.join(BASE_PATH,'toward_on','svm','T.npy'))
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))



global_pen = 0.1
tv_ratio = 1e-5#0.5
l1_ratio = 0.5

ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))



nib_mask = nib.load(INPUT_MASK)
Atv = parsimony.functions.nesterov.tv.A_from_mask(nib_mask.get_data())


# PARSIMONY
################################################################################
from parsimony.algorithms.utils import AlgorithmSnapshot
snapshot = AlgorithmSnapshot('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/enet_1e-8/').save_nipals

mod = pca_tv.PCA_L1_L2_TV(n_components=3,
                                l1=ll1, l2=ll2, ltv=ltv,
                                Atv=Atv,
                                criterion="frobenius",
                                eps=1e-8,
                                max_iter=100,
                                inner_max_iter=int(1e4),
                                output=True,callback=snapshot)
mod.fit(X)
###############################################################################



  # Plot time and precision
###############################################################################

comp1 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/enet_1e-8/component:1.npz')
comp2 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/enet_1e-8/component:2.npz')
comp3 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/enet_1e-8/component:3.npz')


comp1 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/tv_1e-8/component:1.npz')
comp2 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/tv_1e-8/component:2.npz')
comp3 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/tv_1e-8/component:3.npz')


comp1_V = (comp1['v'] - comp1['v'][-1,:])
eps_comp1 = np.zeros(comp1_V.shape[0]-1)
time_comp1 = np.zeros(comp1_V.shape[0]-1)
for i in range(0,comp1_V.shape[0]-1):
    eps_comp1[i] = np.linalg.norm(comp1_V[i,:])
    time_comp1[i] = comp1["time"][i]

comp2_V = (comp2['v'] - comp2['v'][-1,:])
eps_comp2 = np.zeros(comp2_V.shape[0]-1)
time_comp2 = np.zeros(comp2_V.shape[0]-1)
for i in range(0,comp2_V.shape[0]-1):
    eps_comp2[i] = np.linalg.norm(comp2_V[i,:])
    time_comp2[i] = comp2["time"][i]

comp3_V = (comp3['v'] - comp3['v'][-1,:])
eps_comp3 = np.zeros(comp3_V.shape[0]-1)
time_comp3 = np.zeros(comp3_V.shape[0]-1)
for i in range(0,comp3_V.shape[0]-1):
    eps_comp3[i] = np.linalg.norm(comp3_V[i,:])
    time_comp3[i] = comp3["time"][i]


import matplotlib.pyplot as plt
plt.plot(np.cumsum(time_comp1),eps_comp1,'o')
plt.plot(np.cumsum(time_comp2),eps_comp2,'o')
plt.plot(np.cumsum(time_comp3),eps_comp3,'o')
plt.yscale('log')
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.title('synthetic dataset -Enet PCA')
#########################################################################################
import scipy
import scipy.interpolate
from scipy.interpolate import Rbf


#Interpolate to find time necesarry to achieve a given precision
#########################################################################################
#for tv, O.1 smoothing
ius_1 = Rbf(np.abs(eps_comp1),np.cumsum(time_comp1),smooth=0.1)
x= np.linspace(0,np.abs(eps_comp1).max(),100)
y = ius_1(x)
plt.plot(np.abs(eps_comp1),np.cumsum(time_comp1),'o',x,y)


ius_2 = Rbf(np.abs(eps_comp2),np.cumsum(time_comp2),smooth=0.1)
x= np.linspace(0,np.abs(eps_comp2).max(),100)
y = ius_2(x)
plt.plot(np.abs(eps_comp2),np.cumsum(time_comp2),'o',x,y)


ius_3 = Rbf(np.abs(eps_comp3),np.cumsum(time_comp3),smooth=0.1)
x= np.linspace(0,np.abs(eps_comp3).max(),100)
y = ius_3(x)
plt.plot(np.abs(eps_comp3),np.cumsum(time_comp3),'o',x,y)


print ("Total time necessary in order to achieve a precision of 1e-4;", ius_1(1e1) + ius_2(1e1) + ius_3(1e1) )
print ("Total time necessary in order to achieve a precision of 1e-5;", ius_1(1e-0) + ius_2(1e-0) + ius_3(1e-0) )
print ("Total time necessary in order to achieve a precision of 1e-6;", ius_1(1e-01) + ius_2(1e-01) + ius_3(1e-01) )
print ("Total time necessary in order to achieve a precision of 1e-2;", ius_1(1e-02) + ius_2(1e-02) + ius_3(1e-02) )
print ("Total time necessary in order to achieve a precision of 1e-3;", ius_1(1e-03) + ius_2(1e-03) + ius_3(1e-03) )



#############################################################################
#####GraphNet
import scipy.sparse as sparse
from parsimony.algorithms.utils import AlgorithmSnapshot
import pca_struct
import parsimony.functions.nesterov.tv as tv_helper

global_pen = 0.1
gn_ratio = 0.5
l1_ratio = 0.5

lgn = global_pen * gn_ratio
ll1 = l1_ratio * global_pen * (1 - gn_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - gn_ratio)
assert(np.allclose(ll1 + ll2 + lgn, global_pen))



nib_mask = nib.load(INPUT_MASK)
Agn = sparse.vstack(tv_helper.linear_operator_from_mask(nib_mask.get_data()))

################################################################################
snapshot = AlgorithmSnapshot('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/gn_1e-8/').save_nipals

mod = pca_struct.PCAGraphNet(n_components=3,
                                l1=ll1, l2=ll2, lgn=lgn,
                                Agn=Agn,
                                criterion="frobenius",
                                eps=1e-8,
                                max_iter=500,
                                output=False,callback = snapshot)
mod.fit(X)




comp1 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/gn_1e-8/component:1.npz')
comp2 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/gn_1e-8/component:2.npz')
comp3 = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/gn_1e-8/component:3.npz')


comp1_V = (comp1['v'] - comp1['v'][-1,:])
eps_comp1 = np.zeros(comp1_V.shape[0]-1)
time_comp1 = np.zeros(comp1_V.shape[0]-1)
for i in range(0,comp1_V.shape[0]-1):
    eps_comp1[i] = np.linalg.norm(comp1_V[i,:])
    time_comp1[i] = comp1["time"][i]

comp2_V = (comp2['v'] - comp2['v'][-1,:])
eps_comp2 = np.zeros(comp2_V.shape[0]-1)
time_comp2 = np.zeros(comp2_V.shape[0]-1)
for i in range(0,comp2_V.shape[0]-1):
    eps_comp2[i] = np.linalg.norm(comp2_V[i,:])
    time_comp2[i] = comp2["time"][i]

comp3_V = (comp3['v'] - comp3['v'][-1,:])
eps_comp3 = np.zeros(comp3_V.shape[0]-1)
time_comp3 = np.zeros(comp3_V.shape[0]-1)
for i in range(0,comp3_V.shape[0]-1):
    eps_comp3[i] = np.linalg.norm(comp3_V[i,:])
    time_comp3[i] = comp3["time"][i]


import matplotlib.pyplot as plt
plt.plot(np.cumsum(time_comp1),eps_comp1,'o')
plt.plot(np.cumsum(time_comp2),eps_comp2,'o')
plt.plot(np.cumsum(time_comp3),eps_comp3,'o')
plt.yscale('log')
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.title('synthetic dataset -Graph PCA')


import scipy
import scipy.interpolate
from scipy.interpolate import Rbf


#Interpolate to find time necesarry to achieve a given precision
#########################################################################################
#for tv, O.1 smoothing
ius_1 = Rbf(np.abs(eps_comp1),np.cumsum(time_comp1),smooth=0.1)
x= np.linspace(0,np.abs(eps_comp1).max(),100)
y = ius_1(x)
plt.plot(np.abs(eps_comp1),np.cumsum(time_comp1),'o',x,y)


ius_2 = Rbf(np.abs(eps_comp2),np.cumsum(time_comp2),smooth=0.1)
x= np.linspace(0,np.abs(eps_comp2).max(),100)
y = ius_2(x)
plt.plot(np.abs(eps_comp2),np.cumsum(time_comp2),'o',x,y)


ius_3 = Rbf(np.abs(eps_comp3),np.cumsum(time_comp3),smooth=0.1)
x= np.linspace(0,np.abs(eps_comp3).max(),100)
y = ius_3(x)
plt.plot(np.abs(eps_comp3),np.cumsum(time_comp3),'o',x,y)


print ("Total time necessary in order to achieve a precision of 10;", ius_1(1e1) + ius_2(1e1) + ius_3(1e1) )
print ("Total time necessary in order to achieve a precision of 1;", ius_1(1e-0) + ius_2(1e-0) + ius_3(1e-0) )
print ("Total time necessary in order to achieve a precision of 1e-1;", ius_1(1e-01) + ius_2(1e-01) + ius_3(1e-01) )
print ("Total time necessary in order to achieve a precision of 1e-2;", ius_1(1e-02) + ius_2(1e-02) + ius_3(1e-02) )
print ("Total time necessary in order to achieve a precision of 1e-3;", ius_1(1e-03) + ius_2(1e-03) + ius_3(1e-03) )







#SPARSE MINI BATCH

###############################################################################
import sklearn.decomposition
from sklearn import metrics


#MiniBatch Sparse PCA - no modifications
snapshot =  AlgorithmSnapshot(output_prefix='/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/sparse_miniBatch/').save
mod = sklearn.decomposition.MiniBatchSparsePCA(n_components=3,verbose=10,alpha=10,n_iter=100)#,callback=snapshot)
mod.fit


#MiniBatch Sparse PCA - with modifications
cd "git/scripts/2014_pca_struct/"
import sklearn_modified
from sklearn_modified import decomposition
snapshot =  AlgorithmSnapshot(output_prefix='/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/minibatch_sparse/').save

mod = sklearn_modified.decomposition.MiniBatchSparsePCA(n_components=3,verbose=10,alpha=10.0,n_iter=1000,callback=snapshot)
mod.fit(X)


###############################################################################

class AlgorithmSnapshot:

    def __init__(self, output_prefix, saving_period=1):
        self.output_prefix = output_prefix
        self.saving_period = saving_period
        self.cpt = 0

    def save(self, locals):
        self.cpt += 1
        if (self.cpt % self.saving_period) != 0:
            return
        snapshot = dict(dict_ = locals["dictionary"],time_iter = locals['dt_iter'],code = locals['code'],time=locals["dt"],iter=locals["ii"],X = locals['this_X'])
        cpt_str = str(self.cpt)
        output_filename = self.output_prefix + 'ite:%s.npz' % (cpt_str)
        #print "AlgorithmSnapshot.save_conesta: save in ", output_filename
        np.savez_compressed(output_filename, **snapshot)
###############################################################################
cpt=0
code = np.zeros((1000,63966,3))
time = np.zeros((1000))
time_original = np.zeros((1000))
for i in range (1,1001,1):
    if i!= 0:
        a = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/minibatch_sparse/ite:%s.npz' %(i))
        code[cpt,:,:] = a['code']
        time[cpt] = a['time_iter']
        time_original[cpt] = a['time']
        cpt = cpt+1

beta_star_sparse = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/sparse_1e-10/ite:66.npz')['code']
comp = code - code[-1,:]

comp = code - beta_star_sparse


eps = np.zeros(999)
for i in range(0,999):
    eps[i] = np.linalg.norm(comp[i,:])


import matplotlib.pyplot as plt
plt.plot(np.cumsum(time[:-1]),np.abs(eps),'o')
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.yscale('log')

###############################################################################

import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from scipy.interpolate import Rbf

ius = Rbf(np.abs(eps),np.cumsum(time[:-1]),smooth=0.1)
x= np.linspace(np.abs(eps).max(),1e-6,1e3)
y = ius(x)
plt.plot(np.cumsum(time[:-1]),np.abs(eps),'o',y,x)
plt.yscale('log')
plt.ylabel('precision')
plt.xlabel('Time in second')
plt.title('fMRI dataset -Sparse PCA')

print "Total time necessary in order to achieve a precision of 10;", ius(10)
print "Total time necessary in order to achieve a precision of 1;", ius(1)
print "Total time necessary in order to achieve a precision of 1e-2;", ius(1e-01)
print "Total time necessary in order to achieve a precision of 1e-3;", ius(1e-02)
print "Total time necessary in order to achieve a precision of 1e-4;", ius(1e-03)
###############################################################################




#SPARSE PCA

###############################################################################

snapshot =  AlgorithmSnapshot(output_prefix='/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/sparse_1e-10/').save
#mod = sklearn.decomposition.MiniBatchSparsePCA(n_components=3,verbose=10,alpha=1,n_iter=10000,callback=snapshot)
cd "git/scripts/2014_pca_struct/"

import sklearn_modified
from sklearn_modified import decomposition

mod = sklearn_modified.decomposition.SparsePCA(n_components=3,alpha=10.0,tol=1e-10,verbose=100,callback=snapshot)
mod.fit(X)
###############################################################################

class AlgorithmSnapshot:

    def __init__(self, output_prefix, saving_period=1):
        self.output_prefix = output_prefix
        self.saving_period = saving_period
        self.cpt = 0

    def save(self, locals):
        self.cpt += 1
        if (self.cpt % self.saving_period) != 0:
            return
        snapshot = dict(dict_ = locals["dictionary"],time_iter = locals['dt_iter'],code = locals['code'],time=locals["dt"],iter=locals["ii"],error= locals['errors'],cost=locals["current_cost"])
        cpt_str = str(self.cpt)
        output_filename = self.output_prefix + 'ite:%s.npz' % (cpt_str)
        #print "AlgorithmSnapshot.save_conesta: save in ", output_filename
        np.savez_compressed(output_filename, **snapshot)




cpt=0
frob = np.zeros((66))
dict_ = np.zeros((66,165,3))
code = np.zeros((66,63966,3))
time = np.zeros((66))
costs= np.zeros((66))
for i in range (1,67,1):
    if i!= 0:
        a = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/sparse_1e-10/ite:%s.npz' %(i))
        time[cpt] = a['time_iter']
        costs[cpt] = a["cost"]
        cpt = cpt+1

eps = costs[:-1] - costs[-1]

import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from scipy.interpolate import Rbf
plt.plot(np.cumsum(time[:-1]),np.abs(eps),'o')
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.yscale('log')
plt.title('fMRI dataset -Sparse PCA')



ius = Rbf(np.abs(eps),np.cumsum(time[:-1]),smooth=1,function='linear')
x= np.linspace(0,1e3,100)
y = ius(x)
plt.plot(np.abs(eps),np.cumsum(time[:-1]),'o',x,y)
plt.yscale('log')
plt.title('fMRI dataset -Sparse PCA')

print "Total time necessary in order to achieve a precision of 10;", ius(10)
print "Total time necessary in order to achieve a precision of 1;", ius(1)
print "Total time necessary in order to achieve a precision of 1e-1;", ius(1e-01)
print "Total time necessary in order to achieve a precision of 1e-2;", ius(1e-02)
print "Total time necessary in order to achieve a precision of 1e-3;", ius(1e-03)
