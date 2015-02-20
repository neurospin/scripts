# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 13:51:04 2015

@author: fh235918
"""

#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import pickle
import os
import optparse


#load prepared data for the project (see exemple_pw.py)
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synaptic10.pickle'
f = open(fname)
genodata = pickle.load(f)
f.close()

# read x data
x = genodata.data
x_subj = ["%012d" % int(i) for i in genodata.fid]

# read y data
y = open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/Hippocampus_L.csv').read().split('\n')[1:-1]
y_subj = [i.split('\t')[0] for i in y]
y = [float(i.split('\t')[2]) for i in y]

#intersect subject list
soi = list(set(x_subj).intersection(set(y_subj)))

# build daatset with X and Y
X = numpy.zeros((len(soi), x.shape[1]))
Y = numpy.zeros(len(soi))
for i, s in enumerate(soi):
    X[i, :] = x[x_subj.index(s), :]
    Y[i] = y[y_subj.index(s)]

groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]
#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np
import pickle
import os
import optparse
import sklearn
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.metrics import r2_score
from sklearn import cross_validation
import parsimony.algorithms as algorithms
import parsimony.utils.consts as consts
import parsimony.estimators as estimators
import parsimony.functions.nesterov.gl as gl
import matplotlib.pyplot as plt

#load prepared data for the project (see exemple_pw.py)
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synaptic10.pickle'
f = open(fname)
genodata = pickle.load(f)
f.close()

# read x data
x = genodata.data
x_subj = ["%012d" % int(i) for i in genodata.fid]

# read y data
y = open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/Hippocampus_L.csv').read().split('\n')[1:-1]
y_subj = [i.split('\t')[0] for i in y]
y = [float(i.split('\t')[2]) for i in y]

#intersect subject list
soi = list(set(x_subj).intersection(set(y_subj)))

# build daatset with X and Y
X_ = np.zeros((len(soi), x.shape[1]))
Y_ = np.zeros(len(soi))
for i, s in enumerate(soi):
    X_[i, :] = x[x_subj.index(s), :]
    Y_[i] = y[y_subj.index(s)]

groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]

#######################
#selecting individuals with respect to hippo volume
#############################
hyp_vol_max=4000
hyp_vol_min=2800
ind = [i for i,temp in enumerate(Y_) if temp < hyp_vol_max and temp>hyp_vol_min]

X = np.zeros((len(ind), x.shape[1]))
Y = np.zeros(len(ind))
for i, s in enumerate(ind):
    X[i, :] = X_[s, :]
    Y[i] = Y_[s]
########################################

############################
#scalin and selecting train and test
#############"""
n,p=X.shape
X = sklearn.preprocessing.scale(X,
                                axis=0,
                                with_mean=True,
                                with_std=False)
Y=Y-Y.mean()
n_train =  int(X.shape[0]/1.3)

Xtr = X[:n_train, :]
ytr = Y[:n_train]
Xte = X[n_train:, :]
yte = Y[n_train:]








####################################################################
#test with simple linear correlation and corrected p value
################################################################










from scipy.stats import pearsonr
p_vect=np.array([])
cor_vect=np.array([])
for i in range(3201):
    r_row, p_value = pearsonr(X[:,i], Y)
#    print r_row, p_value
    p_vect = np.hstack((p_vect,p_value))
    cor_vect = np.hstack((cor_vect,r_row))
plt.plot(p_vect)    
#plt.plot(p_vect[p_vect<0.05])
plt.show()
p_vect.argesort()
import p_value_correction as p_c
print p_c.fdr(p_vect.reshape(3201,1))
#############################################################################
## sklearn
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

## Enet
enet = ElasticNetCV()
enet.fit(Xtr, ytr)
y_pred_enet = enet.predict(Xte)
r2_score(yte, y_pred_enet)
cross_validation.cross_val_score(enet, X, Y, cv=5)
#array([ -8.60231114e-05,  -6.41673920e-03,  -8.69694214e-03,
#        -6.83330203e-02,  -7.77224429e-03])

## SVM
svmlin = svm.SVR(kernel='linear')
svmlin.fit(Xtr, ytr)
y_pred_svmlin = svmlin.predict(Xte)
r2_score(yte, y_pred_svmlin)
cross_validation.cross_val_score(svmlin, X, Y, cv=5)
#array([-0.683538  , -0.58925786, -0.74613231, -1.04904344, -1.03914531])

## RF
rf = RandomForestRegressor()
rf.fit(Xtr, ytr)
y_pred_rf = svmlin.predict(Xte)
r2_score(yte, y_pred_rf)
cross_validation.cross_val_score(rf, X, Y, cv=5)
#array([-0.09635575, -0.20115192, -0.07531322, -0.21041208, -0.04896116])

#Xte = sklearn.preprocessing.scale(Xte,
#                                axis=0,
#                                with_mean=True,
#                                with_std=False)
#Xtr = sklearn.preprocessing.scale(Xtr,
#                                axis=0,
#                                with_mean=True,
#                                with_std=False)
#ytr=ytr-ytr.mean()
#yte=yte-yte.mean()
#########


#####
###################
#estimation of the maximal l1 possible penalisation, 
#this will help to estimate the order of the others penalisations
###########


s= [np.linalg.norm(np.dot(Xtr[:,i],ytr)) for i in range(Xtr.shape[1])]
l1_max =0.1* np.max(s)/Xtr.shape[0]
print "l1 max is", l1_max
#################################

l1, l2, lgl =l1_max * np.array((0.01, 0.1, 0.01))



weights = [np.sqrt(len(group)) for group in groups]
weights = 1./np.sqrt(np.asarray(weights))



Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)
algorithm = algorithms.proximal.CONESTA(eps=consts.TOLERANCE, max_iter=15000)
enet_gl = estimators.LinearRegressionL1L2GL(l1, l2,  lgl , Agl, algorithm=algorithm)
yte_pred_enetgl = enet_gl.fit(Xtr, ytr).predict(Xte)
print "lr r carré vaut",  r2_score(yte, yte_pred_enetgl)

p1=plt.plot(yte,marker='o')
p2=plt.plot(yte_pred_enetgl,marker='v')
plt.title("group_lasso_test") 
#plt.legend([p1, p2], ["beta", "beta_star"])
plt.show()
cf = np.corrcoef(yte.reshape(1,n-n_train),yte_pred_enetgl.reshape(1,n-n_train))[0,1]

print "le coef de correlation vaut ", cf 

