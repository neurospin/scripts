# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:41:11 2015

@author: fh235918
"""

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

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.linear_model import  ElasticNetCV, ElasticNet
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
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

#X = SelectKBest(f_regression, k=1000).fit_transform(X_, y)

##############################################################
### Enet
K = 12000
enet = ElasticNetCV()
X_new = SelectKBest(f_regression, k=K).fit_transform(X_, y)
#
print cross_validation.cross_val_score(enet, X_new, y, cv=5)

#
###############################
anova_filter = SelectKBest(f_regression, k=K)
enet = ElasticNetCV()
anova_enetcv = Pipeline([('anova', anova_filter), ('enet', enet)])
cv_res = cross_validation.cross_val_score(anova_enetcv, X_new, y, cv=5)
np.mean(cv_res)

svr = svm.SVR(kernel="rbf")
anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])
cv_res = cross_validation.cross_val_score(anova_svr, X_new, y, cv=5)
np.mean(cv_res)

parameters = {'svr__C': (.001, .01, .1, 1., 10., 100)}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
print cross_validation.cross_val_score(anova_svrcv, X_new, y, cv=5)

parameters = {'svr__C': (.001, .01, .1, 1., 10., 100),
              'svr__kernel': ("linear", "rbf"),
              'anova': [5, 10, 20, 50, 100]}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
res_cv = cross_validation.cross_val_score(anova_svrcv, X_new, y, cv=5)







K=12500

groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]
filter_univ = SelectKBest(f_regression, k=K)
filter_univ.fit(X_, y)
filter_ = filter_univ.get_support()
#filter_[0] = True
X_filtered = X_[:, filter_]
#Xtest_filtered = Xtest[:, filter_]
print cross_validation.cross_val_score(enet, X_filtered, y, cv=5)




map_full_to_filtered = -np.ones(X_.shape[1], dtype=int)
map_full_to_filtered[filter_] = np.arange((K))
groups_filtered = [map_full_to_filtered[g] for g in groups]
groups_filtered = [g[g != -1] for g in groups_filtered]
#groups_filtered = [g for g in groups_filtered if len(g) >= 1]
a = [float(len(groups_filtered[i])) / float(len(groups[i])) for i in range(len(groups))]
b = (a - np.mean(a)) / np.std(a)
#plt.plot(a)
#plt.show()
ind = np.where(b > 1.96)
#[i for i in ind[0]]
#plt.hist(b,20)
#plt.show()
#bb = np.array(b)

res = list(np.array(groups)[ind])

c = list(set([int(ind) for gr in res for ind in gr]))


X_new2 = X_[:,c]
enet = ElasticNetCV()
#X_new = SelectKBest(f_regression, k=K).fit_transform(X_, y)
#
print cross_validation.cross_val_score(enet, X_new2, y, cv=5)


from scipy.stats import pearsonr
p_vect=np.array([])
cor_vect=np.array([])
p = X_.shape[1]
for i in range(p):
    r_row, p_value = pearsonr(X_[:,i],  y)
    p_vect = np.hstack((p_vect,p_value))
    cor_vect = np.hstack((cor_vect,r_row))

p_vect[p_vect > 0.25] = 0
len(np.where(abs(cor_vect) > 0.05)[0])

cor_vect[abs(cor_vect) < 0.05] = 0
p_vect[p_vect != 0] = 1 / p_vect[p_vect != 0]
import genomic_plot
genomic_plot.genomic_plot(p_vect, genodata)
genomic_plot.genomic_plot(cor_vect, genodata)

   
indices = np.where(p_vect <= 0.05)
print 'numbers of significant p values', len(indices[0]), 'over ', p









####################################################################################################""





























