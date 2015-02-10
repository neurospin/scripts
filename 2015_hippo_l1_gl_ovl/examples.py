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
import os

# read a pickle file containing the patways
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synaptic.pickle'

f = open(fname)
g = pickle.load(f)
f.close()

pw = g['genodata']
print pw.keys()

pw1_name = pw.keys()[0]

pw1_genotyping = pw[pw1_name].data
print pw1_genotyping

snpname, chrname, genename = pw[pw1_name].get_meta()