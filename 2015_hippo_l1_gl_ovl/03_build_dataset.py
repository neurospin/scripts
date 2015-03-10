#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import pickle


from read_data import read_hippo_l1_gl_ovl

covariate, Lhippo, genotype, groups_descr = read_hippo_l1_gl_ovl(pname='Lhippo')

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
tmp = list(genotype.columns)
tmp.remove('FID')
tmp.remove('IID')
X = genotype[tmp].as_matrix()

groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]
