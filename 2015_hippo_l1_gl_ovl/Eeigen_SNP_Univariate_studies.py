# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:08:07 2015

@author: fh235918
"""


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
import operator
rnd = check_random_state(None)
from sklearn.decomposition import PCA

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


#Y_.shape

#plt.hist(Y_all)
#plt.show()

#######################
#selecting individuals with respect to hippo volume
#############################

hyp_vol_max = 7600
hyp_vol_min = 100


select_mask = (Y_all > hyp_vol_min) & (Y_all < hyp_vol_max)
X_ = X_all[select_mask, :].astype(float)
Y_ = Y_all[select_mask]
Cov_ = Cov[select_mask, :]

#assert X_.shape == (1701, 8787)

p = X_.shape[1]

###################
# Remove Covariates
y = Y_ - LinearRegression().fit(Cov_, Y_).predict(Cov_)


#X_ = sklearn.preprocessing.scale(X_,
#                                axis=0,
#                                with_mean=True,
#                                with_std=False)

#X = np.c_[np.ones((X_.shape[0], 1)), X_]
#assert X.shape == (1701, 8788) and np.all(X[:, 0]==1) and np.all(X[:, 1:]==X_)


groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]
weights = [len(group) for group in groups]
weights = np.sqrt(np.asarray(weights))


new = []
indices = []
j = 0
pca = PCA(copy=True, n_components=0.999, whiten=False)
for i in range(len(groups)):
    Xpca = X_[:, groups[i]]
    pca.fit(Xpca)
    P = pca.components_
    matrix = np.dot(Xpca, np.transpose(P))
    new = new + [vector for vector in matrix.transpose()]
    size = matrix.shape[-1]
    indices.append(range(j, j + size))
    j += size

X_new = np.transpose(np.array(new))
print X_new.shape






#################################################################################################"
#here we use the eigen snip approach from chen et al


from scipy.stats import pearsonr
p_vect_eig_SNP=np.array([])
cor_vect_eig_SNP=np.array([])
p_eig_SNP = X_new.shape[1]
for i in range(p_eig_SNP):
    r_row_eig_SNP, p_value_eig_SNP = pearsonr(X_new[:,i],  y)
    p_vect_eig_SNP = np.hstack((p_vect_eig_SNP,p_value_eig_SNP))
    cor_vect_eig_SNP = np.hstack((cor_vect_eig_SNP,r_row_eig_SNP))




   
indices_eig_SNP = np.where(p_vect_eig_SNP <= 0.05)
print 'numbers of significant p values for _eig_SNP', len(indices_eig_SNP[0]), 'over ', p_eig_SNP



plt.figure(1)
plt.subplot(211)
plt.hist(p_vect_eig_SNP,20)
plt.title('uncorrected p values _eig_SNP ')
plt.subplot(212)
plt.plot(cor_vect_eig_SNP)
plt.title('correlation p coef _eig_SNP')
plt.show()



import p_value_correction as p_c
p_corrected_eig_SNP = p_c.fdr(p_vect_eig_SNP)
indices_c_eig_SNP = np.where(p_corrected_eig_SNP  <= 0.05)


print 's0 : numbers of significant corrected p values', len(indices_c_eig_SNP[0]), 'over ', X_new.shape[1]
plt.figure(2)
plt.hist(p_corrected_eig_SNP)
plt.title('corrected p values for _eig_SNP')
plt.show() 
