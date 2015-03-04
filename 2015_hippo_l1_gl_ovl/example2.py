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
# get covariet info
#######################
covariate = iid_fid
covariate = covariate.join(pandas.get_dummies(df['ScanningCentre'],
                                              prefix='Centre')[range(7)])
covariate = covariate.join(df[['Age', 'Sex', 'ICV', 'AgeSq']])
covariate = covariate.set_index(iid_fid['IID'])

#######################
# get genotype information from the pathway c7 (read from pickle)
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

Lhippo.to_csv('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/testLhippo.phe',
              cols=['FID', 'IID', 'Lhippo'], header=True, sep=" ",index=False)
covariate.to_csv('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/test.cov',
                 cols=['FID', 'IID'] + mycol, header=True, sep=" ", index=False)
with open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/testList.snp', 'w') as fp:
    fp.write("\n".join(list(genodata.snpid))+'\n')

import plinkio

geno = plinkio.Genotype('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/qc_subjects_qc_genetics_all_snps_wave2')
geno.setOrderedSubsetIndiv(indx)
gt = geno.snpGenotypeByName('rs3755456')
from sklearn.linear_model import LinearRegression
design = np.hstack((gt, Cov))
lm = LinearRegression().fit(design,Y)
#bug fixed!
#X[X[:,4343]==128, 4343] = np.median(X[X[:,4343]!=128, 4343])
#X[X[:,7554]==128, 7554] = np.median(X[X[:,7554]!=128, 7554])
#X[X[:,7797]==128, 7797] = np.median(X[X[:,7797]!=128, 7797])
#X[X[:,8910]==128, 8910] = np.median(X[X[:,8910]!=128, 8910])