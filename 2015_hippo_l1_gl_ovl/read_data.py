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


def convert_data2XyCovGroup(covariate, phenotype, genotype, meta):
    #######################
    # get the usual matrices
    ########################
    Y = phenotype['Lhippo'].as_matrix()
    tmp = list(covariate.columns)
    #tmp.remove('FID')
    #tmp.remove('IID')
    #tmp.remove('AgeSq')
    mycol = [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
    Cov = covariate[mycol].as_matrix()
    tmp = list(genotype.columns)
    tmp.remove('FID')
    tmp.remove('IID')
    X = genotype[tmp].as_matrix()

    groups_name = meta.keys()
    groups = [list(meta[n]) for n in groups_name]

    return Cov, X, Y, groups_name, groups


def read_hippo_l1_gl_ovl(fgenotype=None, pname='Lhippo'):
    if fgenotype is None:
        fname = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
                 'synapticAll.pickle')
    else:
        fname = fgenotype
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
    # get phenotype pname
    #######################
    phen = df[[pname]].join(iid_fid)
    phen = phen.set_index(iid_fid['IID'])

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
    indx = list(set(phen['IID']).intersection(
                set(covariate['IID'])).intersection(
                set(geno['IID'])))

    covariate = covariate.loc[indx]
    genotype = geno.loc[indx]
    phenotype = phen.loc[indx]
    meta = genodata.get_meta_pws()

    return covariate, phenotype, genotype, meta
