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

pws = g['genodata']
print pws.keys()

pw1_name = pws.keys()[0]
pw2_name = pws.keys()[1]

print pw1_name,pw2_name
# Pour le premier pathway
snpname, chrname, genename = pws[pw1_name].get_meta()
print snpname
print pws[pw1_name].data
print pws[pw1_name].fid

x = numpy.hstack((pws[pw1_name].data,pws[pw2_name].data) )
x_subj = ["%012d"% int(i) for i in pws[pw1_name].fid]

y = open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/Hippocampus_L.csv').read().split('\n')[1:-1]
y_subj = [i.split('\t')[0] for i in y]
y_train = [float(i.split('\t')[2]) for i in y]

soi = list(set(x_subj).intersection(set(y_subj)))

x_fin = numpy.zeros((len(soi), x.shape[1]))
y_fin = numpy.zeros(len(soi))
for i, s in enumerate(soi):
        x_fin[i, :]  = x[x_subj.index(s), :]
        y_fin[i] = y_train[y_subj.index(s)]
        
groups = [range(pws[pw1_name].data.shape[1])]
groups.append(range(pws[pw1_name].data.shape[1],pws[pw2_name].data.shape[1]+
                                                pws[pw1_name].data.shape[1]))