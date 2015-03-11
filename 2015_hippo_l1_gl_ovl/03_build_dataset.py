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


from read_data import read_hippo_l1_gl_ovl, convert_data2XyCovGroup

covariate, Lhippo, genotype, groups_descr = read_hippo_l1_gl_ovl(pname='Lhippo')

Cov, X, Y, groups_name, groups = \
            convert_data2XyCovGroup(covariate, Lhippo, genotype, groups_descr)
            
print "col of Cov are :"
print [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
print "cols of X ars snps"
print "cols of Y are value of lefthippo"
