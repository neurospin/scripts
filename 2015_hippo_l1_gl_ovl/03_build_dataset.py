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
import optparse


#load prepared data for the project (see exemple_pw.py)
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synaptic10.pickle'
f = open(fname)
genodata = pickle.load(f)
f.close()

# read x data
x = genodata.data
x_subj = ["%012d" % int(i) for i in genodata.fid]

# read y data
y = open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/Hippocampus_L.csv').read().split('\n')[1:-1]
y_subj = [i.split('\t')[0] for i in y]
y = [float(i.split('\t')[2]) for i in y]

#intersect subject list
soi = list(set(x_subj).intersection(set(y_subj)))

# build daatset with X and Y
X = numpy.zeros((len(soi), x.shape[1]))
Y = numpy.zeros(len(soi))
for i, s in enumerate(soi):
    X[i, :] = x[x_subj.index(s), :]
    Y[i] = y[y_subj.index(s)]

groups_descr = genodata.get_meta_pws()
groups_name = groups_descr.keys()
groups = [list(groups_descr[n]) for n in groups_name]
