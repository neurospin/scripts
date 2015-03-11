# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:42:24 2015

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
import pandas
import pickle
import numpy as np

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
from sklearn.linear_model import LinearRegression

#######################
# get Enigma2 dataset
#######################
fin = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
       'imagen_subcortCov_NP.csv')
df = pandas.DataFrame.from_csv(fin, sep=' ', index_col=False)
iid_fid = ["%012d" % int(i) for i in df['IID']]
iid_fid = pandas.DataFrame(np.asarray([iid_fid, iid_fid]).T,
                           columns=['FID', 'IID'])
#######################
# get phenotype Lhippo
#######################
Lhippo = df[['Lhippo']].join(iid_fid)
Lhippo = Lhippo.set_index(iid_fid['IID'])

#######################
# get phenotype Lhippo
#######################
covariate = iid_fid
covariate = covariate.join(pandas.get_dummies(df['ScanningCentre'],
                                              prefix='Centre')[range(7)])
covariate = covariate.join(df[['Age', 'Sex', 'ICV', 'AgeSq']])
covariate = covariate.set_index(iid_fid['IID'])

#######################
# get phenotype Lhippo
#######################
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synapticAll.pickle'
f = open(fname)
genodata = pickle.load(f)
f.close()
#######################
# read geno data
########################
iid_fid = ["%012d" % int(i) for i in genodata.fid]
iid_fid = pandas.DataFrame(np.asarray([iid_fid, iid_fid]).T,
                           columns=['FID', 'IID'])
rsname = genodata.get_meta()[0].tolist()
geno = pandas.DataFrame(genodata.data, columns=rsname)
geno = geno.join(iid_fid)
geno = geno.set_index(iid_fid['IID'])


#######################
# Perform subseting
########################
indx = list(set(Lhippo['IID']).intersection(
            set(covariate['IID'])).intersection(
            set(geno['IID'])))

covariate = covariate.loc[indx]
geno = geno.loc[indx]
Lhippo = Lhippo.loc[indx]


#######################
# get the usual matrices
########################
Y = Lhippo['Lhippo'].as_matrix()
tmp = list(covariate.columns)
#tmp.remove('FID')
#tmp.remove('IID')
#tmp.remove('AgeSq')
mycol = [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
Cov = covariate[mycol].as_matrix()
tmp = list(geno.columns)
tmp.remove('FID')
tmp.remove('IID')
X = geno[tmp].as_matrix()

X[X[:,4343]==128, 4343] = np.median(X[X[:,4343]!=128, 4343])
X[X[:,7554]==128, 7554] = np.median(X[X[:,7554]!=128, 7554])
X[X[:,7797]==128, 7797] = np.median(X[X[:,7797]!=128, 7797])
X[X[:,8910]==128, 8910] = np.median(X[X[:,8910]!=128, 8910])






groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]






























#######################
#selecting individuals with respect to hippo volume
#############################
hyp_vol_max=6500
hyp_vol_min=2500


ind = [i for i,temp in enumerate(Y) if temp < hyp_vol_max and temp>hyp_vol_min]


X_ = np.zeros((len(ind), X.shape[1]))
Y_ = np.zeros(len(ind))
Xnon_res = np.zeros((len(ind), X.shape[1]+Cov.shape[1]))
Ynon_res = np.zeros(len(ind))
Cov_ = np.zeros((len(ind), Cov.shape[1]))

for i, s in enumerate(ind):
    Xnon_res[i, :] = np.hstack((Cov[s, :], X[s, :]))
    Ynon_res[i] = Y[s]
    Cov_[i,:] = Cov[s,:]
    X_[i, :] =  X[s, :]
    Y_[i] = Y[s]
    
    


n,p=X_.shape
X_ = sklearn.preprocessing.scale(X_,
                                axis=0,
                                with_mean=True,
                                with_std=False)

Cov_ = sklearn.preprocessing.scale(Cov_,
                                axis=0,
                                with_mean=True,
                                with_std=False)


Y_=Y_-Y_.mean()




Y_ =Y_ - LinearRegression().fit(Cov_,Y_).predict(Cov_) 




n_train =  int(X_.shape[0]/1.75)
Xtr = X_[:n_train, :]
ytr = Y_[:n_train]
Xte = X_[n_train:, :]
yte = Y_[n_train:]




from scipy.stats import pearsonr
p_vect=np.array([])
cor_vect=np.array([])
for i in range(9266):
    r_row, p_value = pearsonr(X_[:,i], Y_)
    p_vect = np.hstack((p_vect,p_value))
    cor_vect = np.hstack((cor_vect,r_row))
p2=np.sort(p_vect)
plt.plot(p_vect)    
plt.show()   

n, bins, patches =plt. hist(p2, 20)


n, bins, patches =plt. hist(p2, 20, normed=1)

indices = np.where(p_vect <= 0.05)
print len(indices[0])


import p_value_correction as p_c
p_corrected = p_c.fdr(p_vect)
indices = np.where(p_corrected <= 0.5)

plt.plot(p_corrected)    
plt.show() 

n, bins, patches =plt. hist(p_corrected, 20)


from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

## Enet
enet = ElasticNetCV()
enet.fit(Xtr, ytr)
y_pred_enet = enet.predict(Xte)
r2_score(yte, y_pred_enet)
print cross_validation.cross_val_score(enet, X_, Y_, cv=5)



svmlin = svm.SVR(kernel='linear')
svmlin.fit(Xtr, ytr)
y_pred_svmlin = svmlin.predict(Xte)
print r2_score(yte, y_pred_svmlin)
print cross_validation.cross_val_score(svmlin, X_, Y_, cv=5)
#array([-0.683538  , -0.58925786, -0.74613231, -1.04904344, -1.03914531])

## Randoom Forests
rf = RandomForestRegressor()
rf.fit(Xtr, ytr)
y_pred_rf = svmlin.predict(Xte)
print r2_score(yte, y_pred_rf)
print cross_validation.cross_val_score(rf, X_, Y_, cv=5)















s= [np.linalg.norm(np.dot(Xtr[:,i],ytr)) for i in range(Xtr.shape[1])]
l1_max =0.1* np.max(s)/Xtr.shape[0]
print "l1 max is", l1_max
#################################

l1, l2, lgl =l1_max * np.array((0.1, 0.1, 0.01))



weights = [np.sqrt(len(group)) for group in groups]
weights = 1./np.sqrt(np.asarray(weights))



Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)
algorithm = algorithms.proximal.CONESTA(eps=consts.TOLERANCE, max_iter=20000)
enet_gl = estimators.LinearRegressionL1L2GL(l1, l2,  lgl , Agl, algorithm=algorithm)
yte_pred_enetgl = enet_gl.fit(Xtr, ytr).predict(Xte)
print " r carré vaut",  r2_score(yte, yte_pred_enetgl)





















Xnon_res = sklearn.preprocessing.scale(Xnon_res,
                                axis=0,
                                with_mean=True,
                                with_std=False)
Ynon_res = Ynon_res-Ynon_res.mean()


n_train =  int(X_.shape[0]/1.5)
Xtr_res = Xnon_res[:n_train, :]
ytr_res = Ynon_res[:n_train]
Xte_res = Xnon_res[n_train:, :]
yte_res = Ynon_res[n_train:]
######################




s= [np.linalg.norm(np.dot(Xtr_res[:,i],ytr_res)) for i in range(Xtr_res.shape[1])]
l1_max =0.1* np.max(s)/Xtr_res.shape[0]
print "l1 max is", l1_max





Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)
algorithm = algorithms.proximal.CONESTA(eps=consts.TOLERANCE, max_iter=1200)
enet_gl = estimators.LinearRegressionL1L2GL(0.2, 0.2,  0.2 , Agl, algorithm=algorithm, penalty_start=10)
yte_pred_enetgl_res = enet_gl.fit(Xtr_res, ytr_res).predict(Xte_res)
print " r carré vaut",  r2_score(yte_res, yte_pred_enetgl_res)



#r carré vaut 0.147265167498

#test without group lasso penalty when using the matrix design as predictors


ridge_es = estimators.RidgeRegression(0.05, penalty_start=10)

yte_pred_ridge_res = ridge_es.fit(Xtr_res, ytr_res).predict(Xte_res)
print " r carré vaut",  r2_score(yte_res,yte_pred_ridge_res)


#r carré vaut 0.140534187938


#test without group lasso penalty when using the residualized matrix 


ridge_es = estimators.RidgeRegression(0.05, penalty_start=10)

yte_pred_ridge = ridge_es.fit(Xtr, ytr).predict(Xte)
print " r carré vaut",  r2_score(yte,yte_pred_ridge)

#r carré vaut -0.520837542958


#Use of permutation and cross validation for the parsimony estimatord

# Prepare permutation
from sklearn import cross_validation
from sklearn.utils import check_random_state
rnd = check_random_state(None)
#rnd.permutation(len(y))
n_perms = 2

train_res = list()
test_res = list()
for i in xrange(n_perms+1):
    print i
    if i == 0:
        perms = np.arange(len(Y_))
    else:
        perms = rnd.permutation(len(Y_))
    yperm = Y_[perms]
    train_perm = list()
    test_perm = list()
    for train,test in cross_validation.StratifiedKFold(y=yperm, n_folds=2):
        Xtrain = Xnon_res[train, :]
        Xtest = Xnon_res[test, :]
        ytrain = yperm[train]
        ytest = yperm[test]
        ridge_es.fit(Xtrain, ytrain)
        y_pred_train = ridge_es.predict(Xtrain)
        y_pred_test = ridge_es.predict(Xtest)
        train_acc = r2_score(y_pred_train, ytrain)
        test_acc = r2_score(y_pred_test, ytest)
        train_perm.append(train_acc)
        test_perm.append(test_acc)
    train_res.append(train_perm)
    test_res.append(test_perm)

train_res_ar = np.array(train_res)
test_res_ar = np.array(test_res)

train_acc_mean = train_res_ar.mean(1)
train_acc_sd = train_res_ar.std(1)

test_acc_mean = test_res_ar.mean(1)
test_acc_sd = test_res_ar.std(1)

pval_train = np.sum(train_acc_mean[1:] > train_acc_mean[0])
pval_test = np.sum(test_acc_mean[1:] > test_acc_mean[0])



