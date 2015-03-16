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


Y_s= Y_all
X_s=X_all



#######################
#selecting individuals with respect to hippo volume
#############################

hyp_vol_max=5500
hyp_vol_min=3000

ind = [i for i,temp in enumerate(Y_s) if temp < hyp_vol_max and temp>hyp_vol_min]

X_ = np.zeros((len(ind),  X_s.shape[1]))

Y_ = np.zeros(len(ind))
Cov_ = np.zeros((len(ind), Cov.shape[1]))


for i, s in enumerate(ind):
    Cov_[i,:] = Cov[s,:]
    X_[i, :] = X_s[s, :]
    Y_[i] = Y_s[s]

p=X_s.shape[1]

###############"
##select groups of individuals following the sex variable
ind_sex_1=np.where(Cov_[:,1]==1)
ind_sex_0=np.where(Cov_[:,1]==0)


age_max=5600
age_min=5000
age = covariate[u'Age'].as_matrix()[ind]

ind_age=np.intersect1d(np.where(age<age_max)[0], np.where(age>age_min)[0])

ind_sex_0_age = np.intersect1d(ind_age,ind_sex_0[0])
ind_sex_1_age = np.intersect1d(ind_age,ind_sex_1[0])


Y_sex_0= Y_[ind_sex_0_age]
Y_sex_1= Y_[ind_sex_1_age]



##############################
#non résidualisé
#######################""
boxplots = [Y_sex_0,Y_sex_1]
plt.boxplot(boxplots)
plt.title('Hyppo volume distribution for both sexes')
plt.xticks([1, 2], ['sex 0 ', 'sex 1'])
plt.show() 


X_sex_1=X_[ind_sex_1_age,:]
X_sex_0=X_[ind_sex_0_age,:]


Cov_sex_1 = Cov_[ind_sex_1_age,:]
Cov_sex_0 = Cov_[ind_sex_0_age,:]






################
#non residualise et sexe separe
###########

from scipy.stats import pearsonr
p_vect_sex_1=np.array([])
cor_vect_sex_1=np.array([])
p_sex_1 = X_sex_1.shape[1]
for i in range(p_sex_1):
    r_row, p_value = pearsonr(X_sex_1[:,i],  Y_sex_1)
    p_vect_sex_1 = np.hstack((p_vect_sex_1,p_value))
    cor_vect_sex_1 = np.hstack((cor_vect_sex_1,r_row))







p_vect_sex_0=np.array([])
cor_vect_sex_0=np.array([])
p_sex_0 = X_sex_0.shape[1]
for i in range(p_sex_0):
    r_row, p_value = pearsonr(X_sex_0[:,i],  Y_sex_0)
    p_vect_sex_0 = np.hstack((p_vect_sex_0,p_value))
    cor_vect_sex_0 = np.hstack((cor_vect_sex_0,r_row))

   
indices_sex_1 = np.where(p_vect_sex_1 <= 0.05)
print 'numbers of significant p values', len(indices_sex_1[0]), 'over ', len(Y_sex_1)

indices_sex_0 = np.where(p_vect_sex_0 <= 0.05)
print 'numbers of significant p values', len(indices_sex_0[0]), 'over ', len(Y_sex_0)






plt.figure(1)
plt.subplot(221)
plt.hist(p_vect_sex_0,200)
plt.title('uncorrected p values for sex 0')
plt.subplot(222)
plt.hist(p_vect_sex_1,200)
plt.title('uncorrected p values for sex 1')
plt.subplot(223)
plt.plot(cor_vect_sex_0)
plt.title('correlation p coef sex 0')
plt.subplot(224)
plt.plot(cor_vect_sex_1)
plt.title('correlation coef for sex 1')
plt.show()



import p_value_correction as p_c
p_corrected_sex_0 = p_c.fdr(p_vect_sex_0)
indices_c_sex_0 = np.where(p_corrected_sex_0  <= 0.05)

p_corrected_sex_1 = p_c.fdr(p_vect_sex_1)
indices_c_sex_1 = np.where(p_corrected_sex_1  <= 0.05)

print 's0 : numbers of significant corrected p values', len(indices_c_sex_0[0]), 'over ', len(Y_sex_0)

print 's1 :numbers of significant corrected  p values', len(indices_c_sex_1[0]), 'over ', len(Y_sex_1)


plt.figure(2)
plt.subplot(211)
plt.hist(p_corrected_sex_0,20)
plt.title('corrected p values for sex 0')
plt.subplot(212)
plt.hist(p_corrected_sex_1,20)
plt.title('uncorrected p values for sex 1')
plt.show() 




##########################"
############################""







################
#non residualise et sexe non separe
###########

from scipy.stats import pearsonr
p_vect=np.array([])
cor_vect=np.array([])
p = X_.shape[1]
for i in range(p):
    r_row, p_value = pearsonr(X_sex_1[:,i],  Y_sex_1)
    p_vect = np.hstack((p_vect_sex_1,p_value))
    cor_vect = np.hstack((cor_vect_sex_1,r_row))




   
indices = np.where(p_vect <= 0.05)
print 'numbers of significant p values', len(indices[0]), 'over ', len(Y_)




plt.figure(3)
plt.subplot(211)
plt.hist(p_vect,20)
plt.title('uncorrected p values ')
plt.subplot(212)
plt.plot(cor_vect)
plt.title('correlation p coef')
plt.show()



import p_value_correction as p_c
p_corrected = p_c.fdr(p_vect)
indices_c = np.where(p_corrected  <= 0.05)


print 'numbers of significant corrected p values', len(indices_c[0]), 'over ', len(Y_)



plt.figure(4)
plt.hist(p_corrected,20)
plt.show() 




##########################"
############################""


##############################
#résidualisé
#######################""
###################
print 'now we consider the residualized approach'
#####################
#################""

X_res = X_[ind_age]
Y_res = Y_[ind_age]
Cov_res = Cov_[ind_age]

Y_res = Y_res - LinearRegression().fit(Cov_res,Y_res).predict(Cov_res)



from scipy.stats import pearsonr
p_vect_res=np.array([])
cor_vect_res=np.array([])
p = X_res.shape[1]
for i in range(p):
    r_row, p_value = pearsonr(X_res[:,i],  Y_res)
    p_vect_res = np.hstack((p_vect_res,p_value))
    cor_vect_res = np.hstack((cor_vect_res,r_row))




   
indices_res = np.where(p_vect_res <= 0.05)
print 'numbers of significant p values', len(indices_res[0]), 'over ', len(Y_res)




plt.figure(4)
plt.subplot(211)
plt.hist(p_vect_res,20)
plt.title('uncorrected p values ')
plt.subplot(212)
plt.plot(cor_vect_res)
plt.title('correlation p coef')
plt.show()



import p_value_correction as p_c
p_corrected_res = p_c.fdr(p_vect_res)
indices_c_res = np.where(p_corrected_res  <= 0.05)


print 'numbers of significant corrected p values', len(indices_c_res[0]), 'over ', len(Y_res)

plt.figure(5)
plt.hist(p_corrected_res,20)
plt.show() 



#########
#study of covariate correlation 
##########"

print 'before resudualisation'

Y_res = Y_[ind_age]

lm_age = LinearRegression().fit(Cov_res[:,0].reshape([len(Y_res),1]),Y_res)
lm_icv = LinearRegression().fit(Cov_res[:,2].reshape([len(Y_res),1]),Y_res)


y_icv=Cov_res[:,2]*lm_icv.coef_[0]+lm_icv.intercept_
y_age=Cov_res[:,0]*lm_age.coef_[0]+lm_age.intercept_


plt.figure(5)
plt.subplot(211)
plt.scatter(Cov_res[:,2],Y_res)
plt.plot(Cov_res[:,2],y_icv)
plt.title('ICV')
plt.subplot(212)
plt.scatter(Cov_res[:,0],Y_res)
plt.plot(Cov_res[:,0],y_age)
plt.title('age')
plt.show()


###############
#correlation with covariate matrix
##############"

from scipy.stats import pearsonr
p_vect_cov=np.array([])
cor_vect_cov=np.array([])
p_cov = Cov_res.shape[1]
for i in range(p_cov):
    r_row, p_value = pearsonr(Cov_res[:,i],  Y_res)
    p_vect_cov = np.hstack((p_vect_cov,p_value))
    cor_vect_cov = np.hstack((cor_vect_cov,r_row))





plt.figure(6)
plt.subplot(211)
plt.hist(p_vect_cov,20)
plt.title('p values for covariate')
plt.subplot(212)
plt.plot(cor_vect_cov)
plt.title('correlation for covariate')
plt.show()


##
#

print 'print after  resudualisation'

Y_res = Y_res - LinearRegression().fit(Cov_res,Y_res).predict(Cov_res)


lm_age = LinearRegression().fit(Cov_res[:,0].reshape([len(Y_res),1]),Y_res)
lm_icv = LinearRegression().fit(Cov_res[:,2].reshape([len(Y_res),1]),Y_res)


y_icv=Cov_res[:,2]*lm_icv.coef_[0]+lm_icv.intercept_
y_age=Cov_res[:,0]*lm_age.coef_[0]+lm_age.intercept_


plt.figure(5)
plt.subplot(211)
plt.scatter(Cov_res[:,2],Y_res)
plt.plot(Cov_res[:,2],y_icv)
plt.title('icv after resudualisation')
plt.subplot(212)
plt.scatter(Cov_res[:,0],Y_res)
plt.plot(Cov_res[:,0],y_age)
plt.title('age after resudualisation')
plt.show()


###############
#correlation with covariate matrix
##############"

from scipy.stats import pearsonr
p_vect_cov=np.array([])
cor_vect_cov=np.array([])
p_cov = Cov_res.shape[1]
for i in range(p_cov):
    r_row, p_value = pearsonr(Cov_res[:,i],  Y_res)
    p_vect_cov = np.hstack((p_vect_cov,p_value))
    cor_vect_cov = np.hstack((cor_vect_cov,r_row))





plt.figure(7)
plt.subplot(211)
plt.hist(p_vect_cov,20)
plt.title('p values for covariate after resudualisation')
plt.subplot(212)
plt.plot(cor_vect_cov)
plt.title('correlation for covariate after resudualisation')
plt.show()