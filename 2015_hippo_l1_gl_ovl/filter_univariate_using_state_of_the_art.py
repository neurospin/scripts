# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:55:21 2015

@author: fh235918
"""

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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import itertools
import operator
rnd = check_random_state(None)


from sklearn import cross_validation
from sklearn.linear_model import  ElasticNetCV, ElasticNet
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
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
#fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/new_synapticAll.pickle'
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/kegg.pickle'

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


#######################
#selecting individuals with respect to hippo volume
#############################

hyp_vol_max=7600
hyp_vol_min=100


select_mask = (Y_all > hyp_vol_min) & (Y_all < hyp_vol_max)
X_ = X_all[select_mask, :].astype(float)
Y_ = Y_all[select_mask]
Cov_ = Cov[select_mask, :]

#assert X_.shape == (1701, 8787)

p = X_.shape[1]

###################
# Remove Covariates
y = Y_ - LinearRegression().fit(Cov_,Y_).predict(Cov_)


X_ = sklearn.preprocessing.scale(X_,
                                axis=0,
                                with_mean=True,
                                with_std=False)

#X = np.c_[np.ones((X_.shape[0], 1)), X_]
#assert X.shape == (1701, 8788) and np.all(X[:, 0]==1) and np.all(X[:, 1:]==X_)
X = X_
#X = SelectKBest(f_regression, k=1000).fit_transform(X_, y)

################################################################################################################



#ridge_es = estimators.RidgeRegression(0.05, penalty_start=1)
#ridge_es.fit(X_, y)
#beta = ridge_es.beta

#import genomic_plot
#genomic_plot.genomic_plot(beta, genodata)






#
#
#
from sklearn import cross_validation
from sklearn.linear_model import  ElasticNetCV, ElasticNet
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


### Enet
enet = ElasticNetCV()
X_new = SelectKBest(f_regression, k=10000).fit_transform(X_, y)
#
print cross_validation.cross_val_score(enet, X_new, y, cv=5)
#[ 0.21822996  0.19626531  0.15985638  0.16047595  0.12360559]
##svmlin = svm.SVR(kernel='linear')
##print cross_validation.cross_val_score(svmlin, X_new, y, cv=5)
##[-0.15075714 -0.15068342 -0.17763498 -0.24752618 -0.33530044]
#
################################################################################
anova_filter = SelectKBest(f_regression, k=5)
enet = ElasticNetCV()
anova_enetcv = Pipeline([('anova', anova_filter), ('enet', enet)])
cv_res = cross_validation.cross_val_score(anova_enetcv, X_, y, cv=20)
np.mean(cv_res)

# [-0.20394014 -0.1661658  -0.29431243 -0.22668731 -0.31640065]
#svr = svm.SVR(kernel="linear")
svr = svm.SVR(kernel="rbf")
anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])
cv_res = cross_validation.cross_val_score(anova_svr, X_, y, cv=10)
np.mean(cv_res)
#[-0.0009754  -0.00804201 -0.00999459 -0.00966989 -0.00933321]

parameters = {'svr__C': (.001, .01, .1, 1., 10., 100)}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
print cross_validation.cross_val_score(anova_svrcv, X_, y, cv=50)
#[-0.00113985 -0.00789315 -0.00962538 -0.00940644 -0.01980303]



parameters = {'svr__C': (.001, .01, .1, 1., 10., 100), 
              'svr__kernel': ("linear", "rbf"),
              'anova' : [5, 10, 20, 50, 100]}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
res_cv =  cross_validation.cross_val_score(anova_svrcv, X_, y, cv=50)
#[-0.00113985 -0.00789315 -0.00962538 -0.00940644 -0.01980303]

#LeNear and NLN
#array([ -6.37568438e-03,  -2.05293782e-02,  -1.52730828e-01,
#        -8.35234238e-03,   7.92095599e-04,  -8.55421547e-04,
#        -2.00368974e-01,  -6.98279846e-03,  -1.52070553e-03,
#        -6.17469948e-03,  -3.65137449e-04,  -3.00574037e-04,
#        -3.60139842e-03,  -3.81577422e-03,  -6.83820154e-03,
#        -4.43401443e-02,  -2.13681005e-04,  -3.32317355e-02,
#        -4.42426063e-02,  -8.49429908e-02,  -3.13626466e-02,
#        -9.07780769e-03,  -1.27396560e-03,  -1.13584268e-02,
#        -7.76355951e-03,  -2.39003086e-01,  -8.31242067e-04,
#        -1.55961248e-02,  -5.30185960e-02,  -4.50291803e-04,
#        -3.67313208e-02,  -2.99432417e-02,  -3.61989391e-02,
#        -4.72122953e-02,  -4.95644440e-04,  -1.01539116e-03,
#        -1.74901813e-03,   3.25169678e-04,  -6.97815342e-02,
#        -5.10003964e-02,  -1.46567202e-01,  -1.22999700e-03,
#        -2.25409887e-04,  -3.45267409e-04,  -4.15490011e-04,
#        -9.34191234e-02,  -5.42613489e-02,  -2.50334333e-02,
#        -5.39852507e-04,  -1.97629333e-02])
#>>> 































################################################################################################""
N_FOLDS_EXT = 50
N_FOLDS_INT = 5
N_PERMS = 50
K = 1000
cv_ext = cross_validation.KFold(X.shape[0], n_folds=N_FOLDS_EXT)
Alpha = [5,10]
L1_ratio = [0,0.000001, 1]


#import genomic_plot
#genomic_plot.genomic_plot(beta, genodata)

train_res = list()
test_res = list()

for i in xrange(N_PERMS + 1):
    print 'la permutation', i
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
        for alpha, l1_ratio in itertools.product(Alpha, L1_ratio):
            inner_param[(alpha, l1_ratio)] = []
        for tr, val in cv_int:
            Xtr = Xtrain[tr, :]
            Xval = Xtrain[val, :]
            ytr = ytrain[tr]
            yval = ytrain[val]
#            test_perm = list()
            for alpha, l1_ratio in itertools.product(Alpha, L1_ratio):
#                print alpha, l1_ratio
                filter_univ = SelectKBest(f_regression, k=K)
                filter_univ.fit(Xtr, ytr)
                filter_ = filter_univ.get_support()
#                filter_[0] = True
                Xtr_filtered = Xtr[:, filter_]
                Xval_filtered = Xval[:, filter_]
                enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=6000)                                            
                enet.fit(Xtr_filtered, ytr)
                y_pred_test = enet.predict(Xval_filtered)
                test_acc = r2_score(yval, y_pred_test)
#                print test_acc
                inner_param[(alpha, l1_ratio)].append(test_acc)
        inner_param_mean = {k:np.mean(inner_param[k]) for k in inner_param.keys()}
#        print inner_param_mean
        alpha, l1_ratio = max(inner_param_mean.iteritems(), key=operator.itemgetter(1))[0]
        print 'selected: ',alpha, l1_ratio
        filter_univ = SelectKBest(f_regression, k=K)
        filter_univ.fit(Xtrain, ytrain)
        filter_ = filter_univ.get_support()
#        filter_[0] = True
        Xtrain_filtered = Xtrain[:, filter_]
        Xtest_filtered = Xtest[:, filter_]
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=6000)
        enet.fit(Xtrain_filtered, ytrain)
        y_pred_test = enet.predict(Xtest_filtered)
        test_acc = r2_score(ytest, y_pred_test)
        print 'the obtained score is ', test_acc
        test_perm.append(test_acc)
    test_res.append(test_perm)
test_res_ar = np.array(test_res)
test_acc_mean = test_res_ar.mean(1)
test_acc_sd = test_res_ar.std(1)
pval_test = np.sum(test_acc_mean[1:] > test_acc_mean[0])/float(len(test_acc_mean)-1)
print 'pval = ',pval_test






####################################################################################################""





























#
#
#from sklearn import svm, cross_validation
#
#from sklearn.feature_selection import SelectKBest, f_regression
#from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
#from sklearn.utils import check_random_state
##cv = StratifiedKFold(y, n_folds=6)
#clf = svm.SVR(kernel='linear')
#nfeats = np.exp2(np.arange(np.int(np.floor(np.log2(X.shape[1])))+1)).astype(int)
#if X.shape[1] > nfeats[-1]: nfeats = np.concatenate((nfeats, [X.shape[1]]))
## ANOVA SVM-C
## -----------
## Whole pipeline with cv for selecting the number of best ranked input feature
#filter_anova = SelectKBest(f_regression, k=30)
#clf_svm = svm.SVR(kernel='linear')
#anova_svm = Pipeline([('anova', filter_anova), ('svm', clf_svm)])
#anovacv_svm = GridSearchCV(estimator=anova_svm, param_grid=dict(anova__k=nfeats))
#clf = anovacv_svm
#
#
#
## Prepare permutation
#from sklearn.utils import check_random_state
#rnd = check_random_state(None)
##rnd.permutation(len(y))
#n_perms = 10
#
#train_res = list()
#test_res = list()
#for i in xrange(n_perms+1):
#    if i == 0:
#        perms = np.arange(len(y))
#    else:
#        perms = rnd.permutation(len(y))
#    yperm = y[perms]
#    train_perm = list()
#    test_perm = list()
#    for train,test in cross_validation.KFold(X.shape[0], n_folds=10):
#        Xtrain = X[train, :]
#        Xtest = X[test, :]
#        ytrain = yperm[train]
#        ytest = yperm[test]
#        clf.fit(Xtrain, ytrain)
#        y_pred_train = clf.predict(Xtrain)
#        y_pred_test = clf.predict(Xtest)
#        train_acc = r2_score(ytrain, y_pred_train)
#        test_acc = r2_score(ytest,y_pred_test)
#        print 'train',train_acc ,'test', test_acc
#        train_perm.append(train_acc)
#        test_perm.append(test_acc)
#    train_res.append(train_perm)
#    test_res.append(test_perm)
#    print 'train',train_res ,'test', test_res
#train_res_ar = np.array(train_res)
#test_res_ar = np.array(test_res)
#
#train_acc_mean = train_res_ar.mean(1)
#train_acc_sd = train_res_ar.std(1)
#
#test_acc_mean = test_res_ar.mean(1)
#test_acc_sd = test_res_ar.std(1)
#
#pval_train = np.sum(train_acc_mean[1:] > train_acc_mean[0])
#pval_test = np.sum(test_acc_mean[1:] > test_acc_mean[0])
#print 'pval = ',pval_test

