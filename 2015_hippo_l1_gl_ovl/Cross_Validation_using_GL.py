# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:58:45 2015

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
#mycol = [u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
mycol = [u'Sex',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']

Cov = covariate[mycol].as_matrix()
tmp = list(geno.columns)
tmp.remove('FID')
tmp.remove('IID')
X = geno[tmp].as_matrix()

X[X[:,4343]==128, 4343] = np.median(X[X[:,4343]!=128, 4343])
X[X[:,7554]==128, 7554] = np.median(X[X[:,7554]!=128, 7554])
X[X[:,7797]==128, 7797] = np.median(X[X[:,7797]!=128, 7797])
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


X_ = np.zeros((len(ind), X.shape[1]))
Y_ = np.zeros(len(ind))
Xnon_res = np.zeros((len(ind), X.shape[1]+Cov.shape[1]))
Ynon_res = np.zeros(len(ind))
Cov_ = np.zeros((len(ind), Cov.shape[1]))

#Xnon_res is used when the design matrix variables are considered as predictors
#X_ is used when the y vector is residualized using the design matrix

Cov_ = sklearn.preprocessing.scale(Cov_,
                                axis=0,
                                with_mean=True,
                                with_std=False)
for i, s in enumerate(ind):
    Xnon_res[i, :] = np.hstack((Cov[s, :], X[s, :]))
    Ynon_res[i] = Y[s]
    Cov_[i,:] = Cov[s,:]
    X_[i, :] =  X[s, :]
    Y_[i] = Y[s]


n,p=X_.shape

##################
#here we  scale the data : first X_, then Cov_ because it is needed to 
#regress Y_ for the residualization, and finally the Y_ vector
##############""
X_ = sklearn.preprocessing.scale(X_,
                                axis=0,
                                with_mean=True,
                                with_std=False)



Y_=Y_-Y_.mean()



######################
###################
#the residualization
##############""

Y1 =Y_ - LinearRegression().fit(Cov_,Y_).predict(Cov_)







#####################################

#l1, l2, lgl = np.array((0.2, 0.2, 0.2))


Xnon_res = sklearn.preprocessing.scale(Xnon_res,
                                axis=0,
                                with_mean=True,
                                with_std=False)

#######################
#univariate approach
#################
#before normalization

#from scipy.stats import pearsonr
#p_vect=np.array([])
#cor_vect=np.array([])
#pp = Xnon_res.shape[1]
#for i in range(pp):
#    r_row, p_value = pearsonr(Xnon_res[:,i],  Ynon_res)
#    p_vect = np.hstack((p_vect,p_value))
#    cor_vect = np.hstack((cor_vect,r_row))
#p2=np.sort(p_vect)
#plt.plot(p_vect)    
#plt.show()   
#
#n, bins, patches =plt. hist(p2, 200)
#
#
#n, bins, patches =plt. hist(p2, 20, normed=1)
#
#indices = np.where(p_vect <= 0.05)
#print len(indices[0])
#
#
#import p_value_correction as p_c
#p_corrected = p_c.fdr(p_vect)
#indices = np.where(p_corrected <= 0.05)
#
#plt.plot(p_corrected)    
#plt.show() 
#
#n, bins, patches =plt. hist(p_corrected, 20)
#
#indices = np.where(p_corrected <= 0.1)
#p_corrected[indices]
##########################"
############################""
Ynon_res=Ynon_res/ covariate[u'ICV'].as_matrix()[ind]
Ynon_res = Ynon_res-Ynon_res.mean()







weights = [np.sqrt(len(group)) for group in groups]
weights = 1./np.sqrt(np.asarray(weights))


#################"
#shape = (p, 1, 1)
#import parsimony.functions.nesterov.tv as nesterov_tv
#
#A, n_compacts = nesterov_tv.linear_operator_from_shape(shape)
#algo = algorithms.proximal.CONESTA(max_iter=100000, eps = 0.0000000001, tau=0.2)

N_FOLDS = 2
cv = cross_validation.KFold(n, n_folds=N_FOLDS)
#train_res = list()
#test_res = list()
Agl = gl.linear_operator_from_groups(p, groups=groups, weights=weights)
algorithm = algorithms.proximal.FISTA(eps=0.000001, max_iter=2000)

L1 = [0.01]
L2 = [0.01,]
LGL = [0.0001]

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
                                                            penalty_start=8)
#               enet_gl = estimators.RidgeRegression(0.0001,
#                                                    algorithm=algorithm,
#                                                            penalty_start=10)

                Xtrain = Xnon_res[train, :]
                Xtest = Xnon_res[test, :]
                ytrain = Ynon_res[train]
                ytest = Ynon_res[test]
                enet_gl.fit(Xtrain, ytrain)
#                print (len(np.where(enet_gl.beta==0))[0])
                plt.plot(enet_gl.beta)
                plt.show()
                y_pred_train = enet_gl.predict(Xtrain)
                y_pred_test = enet_gl.predict(Xtest)
                train_acc = r2_score(ytrain, y_pred_train)
                train_res.loc[l1, l2, lgl][fold] = train_acc
                test_acc = r2_score(ytest, y_pred_test)
                test_res.loc[l1, l2, lgl][fold] = test_acc
                print train_acc, test_acc
         
         
         

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
pandas.DataFrame.to_csv(test_res, '/home/fh235918/git/scripts/2015_hippo_l1_gl_ovl/table_desin_complete_var_penalized_without_normali_icv_678.csv')
#
#
#
#pandas.read_csv('/home/fh235918/git/scripts/2015_hippo_l1_gl_ovl/table.csv')
