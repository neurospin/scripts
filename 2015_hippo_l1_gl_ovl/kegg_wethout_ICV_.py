# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:17:57 2015

@author: fh235918
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:43:55 2015

@author: fh235918
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:04:35 2015

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
mycol = [u'Age', u'Sex', u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
#mycol = [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
#mycol = [u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
#mycol = [u'Sex',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']

Cov = covariate[mycol].as_matrix()
tmp = list(geno.columns)
tmp.remove('FID')
tmp.remove('IID')
X_all = geno[tmp].as_matrix()






###############"








print 'we consider the residualized approach'
#####################
#################""
Cov = sklearn.preprocessing.scale(Cov,
                                axis=0,
                                with_mean=True,
                                with_std=False)

Y_all = Y_all - Y_all.mean()
Y_all = Y_all - LinearRegression().fit(Cov,Y_all).predict(Cov)

X_all = X_all - X_all.mean(axis=0)
#X_all = sklearn.preprocessing.scale(X_all,
#                                axis=0,
#                                with_mean=True,
#                                with_std=False)

#from scipy.stats import pearsonr
#p_vect_res=np.array([])
#cor_vect_res=np.array([])
#p = X_all.shape[1]
#for i in range(p):
#    r_row, p_value = pearsonr(X_all[:,i],  Y_all)
#    p_vect_res = np.hstack((p_vect_res,p_value))
#    cor_vect_res = np.hstack((cor_vect_res,r_row))
#
#
#
#
#   
#indices_res = np.where(p_vect_res <= 0.05)
#print 'numbers of significant p values', len(indices_res[0]), 'over ', p
#


#
#plt.figure(4)
#plt.subplot(211)
#plt.hist(p_vect_res,20)
#plt.title('uncorrected p values ')
#plt.subplot(212)
#plt.plot(cor_vect_res)
#plt.title('correlation p coef')
#plt.show()


#
#import p_value_correction as p_c
#p_corrected_res = p_c.fdr(p_vect_res)
#indices_c_res = np.where(p_corrected_res  <= 0.05)
#
#
#print 'numbers of significant corrected p values', len(indices_c_res[0]), 'over ', p
#
#plt.figure(5)
#plt.hist(p_corrected_res,20)
#plt.show() 

#
#
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import itertools
import operator
rnd = check_random_state(None)
#
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.linear_model import  ElasticNetCV, ElasticNet
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

y = Y_all
#K = 12000
#enet = ElasticNetCV()
#X_new = SelectKBest(f_regression, k=K).fit_transform(X_all, y)
##
#print cross_validation.cross_val_score(enet, X_all, y, cv=5)
##[-0.00597387 -0.00074044  0.00018395 -0.00211909  0.00239543]
#
##
################################
#anova_filter = SelectKBest(f_regression, k=K)
#enet = ElasticNetCV()
#anova_enetcv = Pipeline([('anova', anova_filter), ('enet', enet)])
#cv_res = cross_validation.cross_val_score(anova_enetcv, X_all, y, cv=5)
#np.mean(cv_res)
#
#svr = svm.SVR(kernel="rbf")
#anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])
#cv_res = cross_validation.cross_val_score(anova_svr, X_all, y, cv=5)
#print 'mean(cv_res)', np.mean(cv_res)
##test sur 500 000 snp array([-0.00022171,  0.00021368, -0.0045152 , -0.0053676 , -0.00029959])
#parameters = {'svr__C': (.001, .01, .1, 1., 10., 100)}
#anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
#print cross_validation.cross_val_score(anova_svrcv, X_all, y, cv=5)
#
#parameters = {'svr__C': (.001, .01, .1, 1., 10., 100),
#              'svr__kernel': ("linear", "rbf"),
#              'anova': [5, 10, 20, 50, 100]}
#anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
#res_cv = cross_validation.cross_val_score(anova_svrcv, X_all, y, cv=5)
#
#




















X = np.c_[np.ones((X_all.shape[0], 1)), X_all]
#assert X.shape == (1701, 8788) and np.all(X[:, 0]==1) and np.all(X[:, 1:]==X_)


groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]
weights = [len(group) for group in groups]
weights = np.sqrt(np.asarray(weights))



#weights = [np.sqrt(len(group)) for group in groups]
#weights = 1./np.sqrt(np.asarray(weights))

N_FOLDS_EXT = 5
N_FOLDS_INT = 5
N_PERMS = 50
K = 1000
cv_ext = cross_validation.KFold(X.shape[0], n_folds=N_FOLDS_EXT)
Alpha = [0.01,5,100]
L1_ratio = [0,0.000001, 1,100]


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










