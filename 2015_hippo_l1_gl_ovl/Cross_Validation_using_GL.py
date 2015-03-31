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

#plt.hist(Y_all)
#plt.show()

#######################
#selecting individuals with respect to hippo volume
#############################

hyp_vol_max=4000
hyp_vol_min=3500

ind = [i for i,temp in enumerate(Y_all) if temp < hyp_vol_max and temp>hyp_vol_min]

X_ = np.zeros((len(ind),  X_all.shape[1]))

Y_ = np.zeros(len(ind))
Cov_ = np.zeros((len(ind), Cov.shape[1]))


for i, s in enumerate(ind):
    Cov_[i,:] = Cov[s,:]
    X_[i, :] = X_all[s, :]
    Y_[i] = Y_all[s]

p=X_.shape[1]

###############"

Y_res=Y_

#Y_res = Y_ - LinearRegression().fit(Cov_,Y_).predict(Cov_)
print Y_res.mean()

X_res = X_
#X_res = sklearn.preprocessing.scale(X_res,
#                                axis=0,
#                                with_mean=True,
#                                with_std=False)


#X_res = np.zeros((X_.shape[0], 1+X_.shape[1]))
#for i in range(X_.shape[0]):
#    X_res[i, :] = np.hstack((1,X_[i, :]))




groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]    
weights = [len(group) for group in groups]
weights = np.sqrt(np.asarray(weights))








    
n_=X_res.shape[0]
N_FOLDS = 5
cv = cross_validation.KFold(n_, n_folds=N_FOLDS, shuffle =True)
#train_res = list()
#test_res = list()
Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)
algorithm = algorithms.proximal.CONESTA(eps=0.0000001, max_iter=1900)

mean = True
##################
#estimation of maaximal penalty
#n = float(X_res.shape[0])       

###################"
from sklearn.utils import check_random_state
rnd = check_random_state(None)



#perms = rnd.permutation(len(Y_res))
#yperm = Y_res[perms]
#
#Y_res=yperm

L1 = [0.1,5]
L2 = [0.1,5]
LGL = [0.1,1,5]
#L1 = np.dot([0.25],l1_max)
#L2 = np.dot([0.25],l1_max)
#LGL = np.dot([0.25,0.5],l1_max)

index = pandas.MultiIndex.from_product([L1, L2, LGL],
                                       names=['l1', 'l2', 'lgl'])
train_res = pandas.DataFrame(index=index,
                             columns=range(N_FOLDS))
test_res = pandas.DataFrame(index=index,
                           columns=range(N_FOLDS))
beta_mat = np.zeros((N_FOLDS,1+X_res.shape[1]))
for l1 in L1:
    for l2 in L2:
        for lgl in LGL:
            for fold, (train,test) in enumerate(cv):
                print "fold", fold, "for l1=", l1,"l2 =",l2, "lgl=", lgl
#                enet_gl = estimators.LinearRegressionL1L2TV(l1, l2, lgl, A, algorithm=algo, penalty_start=10)

                enet_gl = estimators.LinearRegressionL1L2GL(l1=l1, l2=l2, gl=lgl,
                                                            A=Agl,
                                                            algorithm=algorithm,
                                                            penalty_start=1, mean=mean)
#               enet_gl = estimators.RidgeRegression(0.0001,
#                                                    algorithm=algorithm,
#                                                            penalty_start=10)
                                      
                Xtrain = X_res[train, :]
                Xtest = X_res[test, :]
                ytrain = Y_res[train]
                ytest = Y_res[test]
###################################
                Xtrain = sklearn.preprocessing.scale(Xtrain,
                                axis=0,
                                with_mean=True,
                                with_std=False)
                Xtest = sklearn.preprocessing.scale(Xtest,
                                axis=0,
                                with_mean=True,
                                with_std=False) 
                ytrain = ytrain-ytrain.mean()
                ytest = ytest - ytest.mean() 
                Xtr= np.concatenate((np.ones((Xtrain.shape[0],1)),Xtrain), axis=1)
                Xte= np.concatenate((np.ones((Xtest.shape[0],1)),Xtest), axis=1)
###################################                
                enet_gl.fit(Xtr, ytrain)
                print "card null", len(np.where(enet_gl.beta==0)[0])/np.float(p)
                print "beta dim", len(enet_gl.beta)
                beta_1 = enet_gl.beta
#                plt.plot(beta_1)
#                plt.show()
                beta_mat[fold,:]=enet_gl.beta.reshape(X_res.shape[1]+1)
#                plt.plot(enet_gl.beta[11:])
#                plt.show()
                
                print " intercept", enet_gl.beta[0], "meann of weights", np.mean(enet_gl.beta[1:])
                y_pred_train = enet_gl.predict(Xtr)
                y_pred_test = enet_gl.predict(Xte)
                train_acc = r2_score(ytrain, y_pred_train)
                train_res.loc[l1, l2, lgl][fold] = train_acc
                test_acc = r2_score(ytest, y_pred_test)
                test_res.loc[l1, l2, lgl][fold] = test_acc
                print "train", train_acc, "test" ,test_acc
                print  "test" ,test_res
                ###########
                
#                lm=LinearRegression(fit_intercept=False)
#                lm.fit(Xtrain,ytrain)
#                print "r carre lm", r2_score(ytest, lm.predict(Xtest))
#                print "r2 train lm", lm.score(Xtrain, ytrain)
#                beta=lm.coef_  
#                mask = (beta*beta>1e-8)
#                plt.plot(beta)
#                plt.show() 
#                print " age coef lin model ", beta[1], "meann of weights", np.mean(beta[1:])

                ##########



plt.plot(beta_mat[0,:])
plt.plot(beta_mat[1,:])
plt.plot(beta_mat[2,:])
plt.plot(beta_mat[3,:])
plt.plot(beta_mat[4,:])
plt.show()

#################
#save plot
###########""

#filename = '/tmp/figure.pdf'
#
## Save and crop the figure
#plt.savefig(filename)
#
#os.system("pdfcrop %s %s" % (filename, filename))      



################
#save csv table
############   
#
#pandas.DataFrame.to_csv(test_res, '/home/fh235918/git/scripts/2015_hippo_l1_gl_ovl/table_desin_complete_var_penalized_without_normali_icv_.csv')
#
#
#
#pandas.read_csv('/home/fh235918/git/scripts/2015_hippo_l1_gl_ovl/table.csv')
