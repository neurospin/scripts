# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:16:43 2013

@author: ed203246

This script generate a workflow to perform a MULM cluster-level analysis with permutation.

"""

import os, sys
import numpy
import scipy, scipy.ndimage
import tables

from epac import BaseNode, Pipe
from mulm import LinearRegression
# X(nxp), y(nx1), mask => MULMStats => pval(px1) => ClusterStat() => sizes(kx1)

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

class MULMStats(BaseNode):
    def transform(self, **kwargs):
        lm = LinearRegression()
        X = kwargs['design_matrix']
        Y = kwargs['Y']
        contrast = kwargs['contrast']
        print 'Fitting LM'
        lm.fit(X=X, Y=Y)
        print 'Computing  p-values'
        tval, pval = lm.stats(X, Y, contrast=contrast, pval=True)
        kwargs["pval"] = pval
        return kwargs

class ClusterStats(BaseNode):
    def transform(self, **kwargs):
        print 'Clustering'
        mask   = kwargs['mask']
        pval   = kwargs['pval']
        thresh = kwargs['thresh']
        # Convert mask to numpy binary array
        binary_mask = mask!=0
        # Create an image and fill it with values
        img_data = numpy.zeros(mask.shape, dtype=bool)
        img_data[binary_mask] = pval < thresh;
        # Compute connected components
        (labeled_array, num_features) = scipy.ndimage.label(img_data)
        clust_sizes = scipy.ndimage.measurements.histogram(labeled_array, 1, num_features, num_features)
        out = dict(clust_sizes=clust_sizes)
        return out

h5filename = '/volatile/micro_subdepression.hdf5'

# Load the file
h5file = tables.openFile(h5filename, mode = "r")
images, regressors, mask, mask_affine = data_api.get_data(h5file)
design_mat = data_api.get_dummy(h5file)
n_useful_voxels = images.shape[1]
n_obs           = images.shape[0]
n_regressors    = design_mat.shape[1]
if regressors.shape[0] != n_obs:
    print 'You stupid'
    sys.exit()

pipeline = Pipe(MULMStats(), ClusterStats())
pipeline.run(Y=images, design_matrix=design_mat, mask=numpy.asarray(mask), thresh=0.001, contrast=[1, 0, 0, 0, 0, 0])

#h5file.close()