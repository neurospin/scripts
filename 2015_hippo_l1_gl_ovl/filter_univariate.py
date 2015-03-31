# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:09:12 2015

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
from sklearn.utils import check_random_state
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import itertools

rnd = check_random_state(None)

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
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/new_synapticAll.pickle'
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
Y_all = Lhippo['Lhippo'].as_matrix()
tmp = list(covariate.columns)
#tmp.remove('FID')
#tmp.remove('IID')
#tmp.remove('AgeSq')
mycol = [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
#mycol = [u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
#mycol = [u'Sex',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']

Cov = covariate[mycol].as_matrix()
tmp = list(geno.columns)
tmp.remove('FID')
tmp.remove('IID')
X_all = geno[tmp].as_matrix()


#Y_.shape

plt.hist(Y_all)
plt.show()

#######################
#selecting individuals with respect to hippo volume
#############################

hyp_vol_max=7600
hyp_vol_min=100


select_mask = (Y_all > hyp_vol_min) & (Y_all < hyp_vol_max)
X_ = X_all[select_mask, :].astype(float)
Y_ = Y_all[select_mask]
Cov_ = Cov[select_mask, :]

assert X_.shape == (1701, 8787)

p = X_.shape[1]

###################
# Remove Covariates
y = Y_ - LinearRegression().fit(Cov_,Y_).predict(Cov_)


X_ = sklearn.preprocessing.scale(X_,
                                axis=0,
                                with_mean=True,
                                with_std=False)

X = np.c_[np.ones((X_.shape[0], 1)), X_]
assert X.shape == (1701, 8788) and np.all(X[:, 0]==1) and np.all(X[:, 1:]==X_)


groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]    
weights = [len(group) for group in groups]
weights = np.sqrt(np.asarray(weights))


#weights = [np.sqrt(len(group)) for group in groups]
#weights = 1./np.sqrt(np.asarray(weights))

N_FOLDS_EXT = 10
N_FOLDS_INT = 10
N_PERMS = 1
K = 200
cv_ext = cross_validation.KFold(X.shape[0], n_folds=N_FOLDS_EXT)
algorithm = algorithms.proximal.FISTA(eps=0.000001, max_iter=3000)
L1 = [0.001,0.05, 1, 5,50]
L2 = [0.001,0.05, 1, 5,50]
LGL = [0.0001,0.05, 1, 5,50]
#for l1, l2, lgl in itertools.product(L1, L2, LGL):
#    print l1, l2, lgl

#{for l1 in L1 for l2 in L2}
#j=0
#={}
#for l1 in L1:
#    for l2 in L2:
#        for lgl in LGL:
#            dict[j]=(l1,l2,lgl)
#            j = j +1
train_res = list()
test_res = list()

for i in xrange(N_PERMS + 1):
    # i = 0
    if i == 0:
        perms = np.arange(len(y))
    else:
        perms = rnd.permutation(len(y))
    yperm = y[perms]
    train_perm = list()
    test_perm = list()
    for train, test in cv_ext:    
        Xtrain = X[train, :]
        Xtest = X[test, :]
        ytrain = yperm[train]
        ytest = yperm[test]
        ## Inner loop
        cv_int = cross_validation.KFold(len(ytrain), n_folds=N_FOLDS_INT)
        inner_param = dict()
        for l1, l2, lgl in itertools.product(L1, L2, LGL):
            inner_param[(l1, l2, lgl)] = []
        for tr, val in cv_int:
            Xtr = Xtrain[tr, :]
            Xval = Xtrain[val, :]
            ytr = ytrain[tr]
            yval = ytrain[val]
            test_perm = list()
            for l1, l2, lgl in itertools.product(L1, L2, LGL):
                print l1, l2, lgl
                filter_univ = SelectKBest(f_regression, k=K)
                filter_univ.fit(Xtr, ytr)
                filter_ = filter_univ.get_support()
                filter_[0] = True
                Xtr_filtered = Xtr[:, filter_]
                Xval_filtered = Xval[:, filter_]
                # map form full to filtered, -1 means not selected
                map_full_to_filtered = -np.ones(Xtrain.shape[1], dtype=int)
                map_full_to_filtered[filter_] = np.arange((K+1))
                groups_filtered = [map_full_to_filtered[g] for g in groups]
                groups_filtered = [g[g != -1] for g in groups_filtered]
                groups_filtered = [g for g in groups_filtered if len(g) >= 1]
                weights_filtered = [len(g) for g in groups_filtered]
                weights_filtered = np.sqrt(np.asarray(weights_filtered))
                Agl = gl.linear_operator_from_groups(Xval_filtered.shape[1],
                                                     groups=groups_filtered,
                                                     weights=weights_filtered,
                                                     penalty_start=1)
                enet_gl = estimators.LinearRegressionL1L2GL(l1=l1, l2=l2, gl=lgl,
                                                            A=Agl,
                                                            algorithm=algorithm,
                                                            penalty_start=1)
                enet_gl.fit(Xtr_filtered, ytr)
                y_pred_test = enet_gl.predict(Xval_filtered)
                test_acc = r2_score(yval, y_pred_test)
                print test_acc
                inner_param[(l1, l2, lgl)].append(test_acc)
        inner_param_mean = {k:np.mean(inner_param[k]) for k in inner_param.keys()}
        print inner_param_mean
        

                
            
            
        
   
#        train_acc = np.sum(y_pred_train == ytrain) / float(len(ytrain))
#        test_acc = np.sum(y_pred_test == ytest) / float(len(ytest))
#        train_perm.append(train_acc)
#        test_perm.append(test_acc)
#    train_res.append(train_perm)
#    test_res.append(test_perm)










#Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)













#index = pandas.MultiIndex.from_product([L1, L2, LGL],
#                                       names=['l1', 'l2', 'lgl'])
#train_res = pandas.DataFrame(index=index,
#                             columns=range(N_FOLDS))
#test_res = pandas.DataFrame(index=index,
#                            columns=range(N_FOLDS))
#
#for l1 in L1:
#    for l2 in L2:
#        for lgl in LGL:
#            for fold, (train,test) in enumerate(cv):
#                print "fold", fold, "for l1=", l1,"l2 =",l2, "lgl=", lgl
##                enet_gl = estimators.LinearRegressionL1L2TV(l1, l2, lgl, A, algorithm=algo, penalty_start=10)
#
#                enet_gl = estimators.LinearRegressionL1L2GL(l1=l1, l2=l2, gl=lgl,
#                                                            A=Agl,
#                                                            algorithm=algorithm,
#                                                            penalty_start=8)
##                enet_gl = estimators.RidgeRegression(0.0001,
##                                                    algorithm=algorithm,
##                                                            penalty_start=10)
#
#                Xtrain = Xnon_res[train, :]
#                Xtest = Xnon_res[test, :]
#                ytrain = Ynon_res[train]
#                ytest = Ynon_res[test]
#                enet_gl.fit(Xtrain, ytrain)
##                print (len(np.where(enet_gl.beta==0))[0])
#                plt.plot(enet_gl.beta)
#                plt.show()
#                y_pred_train = enet_gl.predict(Xtrain)
#                y_pred_test = enet_gl.predict(Xtest)
#                train_acc = r2_score(ytrain, y_pred_train)
#                train_res.loc[l1, l2, lgl][fold] = train_acc
#                test_acc = r2_score(ytest, y_pred_test)
#                test_res.loc[l1, l2, lgl][fold] = test_acc
#                print train_acc, test_acc
#            train_res.append(train_cv)
#            test_res.append(test_cv)
#            print "train_res", train_res, "for l1=", l1,"l2 =",l2, "lgl=", lgl
#            print "test_res", test_res, "for l1=", l1,"l2 =",l2, "lgl=", lgl

#            print "test_res", test_res


#train_res_ar = np.array(train_res)
#test_res_ar = np.array(test_res)
#
#train_acc_mean = train_res_ar.mean(1)
#train_acc_sd = train_res_ar.std(1)
#
#test_acc_mean = test_res_ar.mean(1)
#test_acc_sd = test_res_ar.std(1)

#from sklearn.linear_model import LinearRegression
##

