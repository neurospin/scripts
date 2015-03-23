# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:47:55 2015

@author: fh235918
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:31:13 2015

@author: fh235918
"""

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
#mycol = [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
mycol = [u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
#mycol = [u'Sex',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']

Cov = covariate[mycol].as_matrix()
tmp = list(geno.columns)
tmp.remove('FID')
tmp.remove('IID')
X = geno[tmp].as_matrix()

#X[X[:,4343]==128, 4343] = np.median(X[X[:,4343]!=128, 4343])
#X[X[:,7554]==128, 7554] = np.median(X[X[:,7554]!=128, 7554])
#X[X[:,7797]==128, 7797] = np.median(X[X[:,7797]!=128, 7797])
#X[X[:,8910]==128, 8910] = np.median(X[X[:,8910]!=128, 8910])



###########################################
# defining groups of futures tu use a group LASSo model
##########################################"


groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]



#######################
#selecting individuals with respect to hippo volume
#############################

hyp_vol_max=5500
hyp_vol_min=3000

ind = [i for i,temp in enumerate(Y) if temp < hyp_vol_max and temp>hyp_vol_min]

X_ = np.zeros((len(ind), 1+ X.shape[1]+Cov.shape[1]))

#X_ = np.zeros((len(ind), X.shape[1]))
Y_ = np.zeros(len(ind))
Cov_ = np.zeros((len(ind), Cov.shape[1]))


for i, s in enumerate(ind):
    Cov_[i,:] = Cov[s,:]
    X_[i, :] = np.hstack((1,Cov[s, :], X[s, :]))
    Y_[i] = Y[s]

p=X.shape[1]

###############"
##select groups of individuals following the sex variable
ind_sex_1=np.where(Cov_[:,0]==1)
ind_sex_0=np.where(Cov_[:,0]==0)



age_max=5700
age_min=5000
age = covariate[u'Age'].as_matrix()[ind]

ind_age=np.intersect1d(np.where(age<age_max)[0], np.where(age>age_min)[0])

ind_sex_0_age = np.intersect1d(ind_age,ind_sex_0[0])
ind_sex_1_age = np.intersect1d(ind_age,ind_sex_1[0])
#ind_age_sex_1= np.intersect1d()
#ind_age_sex2=
Y_sex_0= Y_[ind_sex_0_age]
Y_sex_1= Y_[ind_sex_1_age]

X_sex_1=X_[ind_sex_1_age,:]
X_sex_0=X_[ind_sex_0_age,:]

X_sex_1 = sklearn.preprocessing.scale(X_sex_1,
                                axis=0,
                                with_mean=True,
                                with_std=False)



Y_sex_1=Y_sex_1-Y_sex_1.mean()


weights = [np.sqrt(len(group)) for group in groups]
weights = 1./np.sqrt(np.asarray(weights))



n_sex_1=X_sex_1.shape[0]
N_FOLDS = 5
cv = cross_validation.KFold(n_sex_1, n_folds=N_FOLDS, shuffle =True)
#train_res = list()
#test_res = list()
Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)
algorithm = algorithms.proximal.FISTA(eps=0.000001, max_iter=4000)

L1 = [1500]
L2 = [300]
LGL = [300]

index = pandas.MultiIndex.from_product([L1, L2, LGL],
                                       names=['l1', 'l2', 'lgl'])
train_res = pandas.DataFrame(index=index,
                             columns=range(N_FOLDS))
test_res = pandas.DataFrame(index=index,
                            columns=range(N_FOLDS))

for l1 in L1:
    for l2 in L2:
        for lgl in LGL:
            for fold, (train,test) in enumerate(cv):
                print "fold", fold, "for l1=", l1,"l2 =",l2, "lgl=", lgl
#                enet_gl = estimators.LinearRegressionL1L2TV(l1, l2, lgl, A, algorithm=algo, penalty_start=10)

                enet_gl = estimators.LinearRegressionL1L2GL(l1=l1, l2=l2, gl=lgl,
                                                            A=Agl,
                                                            algorithm=algorithm,
                                                            penalty_start=10)
#               enet_gl = estimators.RidgeRegression(0.0001,
#                                                    algorithm=algorithm,
#                                                            penalty_start=10)

                Xtrain = X_sex_1[train, :]
                Xtest = X_sex_1[test, :]
                ytrain = Y_sex_1[train]
                ytest = Y_sex_1[test]
                enet_gl.fit(Xtrain, ytrain)
#                print (len(np.where(enet_gl.beta==0))[0])
                print "beta dim", len(enet_gl.beta)
                plt.plot(enet_gl.beta)
                plt.show()
                print " age coef", enet_gl.beta[1], "meann of weights", np.mean(enet_gl.beta[11:])
                y_pred_train = enet_gl.predict(Xtrain)
                y_pred_test = enet_gl.predict(Xtest)
                train_acc = r2_score(ytrain, y_pred_train)
                train_res.loc[l1, l2, lgl][fold] = train_acc
                test_acc = r2_score(ytest, y_pred_test)
                test_res.loc[l1, l2, lgl][fold] = test_acc
                print train_acc, test_acc
                ###########
                
                lm=LinearRegression(fit_intercept=False)
                lm.fit(Xtrain,ytrain)  
                beta=lm.coef_  
                plt.plot(beta)
                plt.show() 
                print " age coef lin model ", beta[1], "meann of weights", np.mean(beta[11:])

                ##########




#y = np.array( [0]*5 + [1]*10 + [3] * 5)
#cv2 = cross_validation.StratifiedKFold(y,n_folds=3)
#for train, test in cv2: 
#    print train, y[np.asarray(train)]
    
#from sklearn.linear_model import LinearRegression
  
plt.hist(X_sex_1[:,1],20)
plt.show()
    