# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:16:43 2013

@author: ed203246

This script generate a workflow to perform a MULM cluster-level analysis with permutation.

"""

# Standard library modules
import os, sys, argparse, ast
import numpy
import scipy, scipy.ndimage
import sklearn.preprocessing
import tables

from epac import BaseNode, Pipe
from mulm import LinearRegression
# X(nxp), y(nx1), mask => MULMStats => pval(px1) => ClusterStat() => sizes(kx1)

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

DEFAULT_N_PERMS  = 1000
DEFAULT_THRESH   = 0.01
DEFAULT_CONTRAST = 'auto'

class MULMStats(BaseNode):
    def transform(self, **kwargs):
        lm = LinearRegression()
        X = kwargs['design_matrix']
        Y = kwargs['Y']
        contrast = kwargs['contrast']
        print 'Fitting LM'
        lm.fit(X=X, Y=Y)
        print 'Computing p-values'
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

def mulm_stat(h5filename, workflow_dir,
              n_perms=DEFAULT_N_PERMS, thresh=DEFAULT_THRESH,
              contrast=DEFAULT_CONTRAST):
    # Load the file
    h5file = tables.openFile(h5filename, mode = "r")
    images, regressors, mask, mask_affine = data_api.get_data(h5file)
    Y_dummy = data_api.get_dummy(h5file)
    n_useful_voxels = images.shape[1]
    n_obs           = images.shape[0]
    n_regressors    = Y_dummy.shape[1]
    if regressors.shape[0] != n_obs:
        print 'You stupid'
        sys.exit()

    # Create design matrix: add an intercept column to Y_dummy & normalize
    design_mat = numpy.ones((n_obs, n_regressors+1))
    design_mat[:, 0:-1] = Y_dummy
    design_mat = sklearn.preprocessing.scale(design_mat)

    if contrast == 'auto':
        contrast = numpy.zeros((n_regressors+1))
        contrast[0] = 1

    pipeline = Pipe(MULMStats(), ClusterStats())
    results = pipeline.run(Y=images, design_matrix=design_mat, mask=numpy.asarray(mask),
                           thresh=thresh, contrast=contrast)

    h5file.close()
    return pipeline, results

if __name__ == '__main__':
    # Stupid type convert for contrast
    def convert_contrast(arg):
        try:
            contrast = ast.literal_eval(arg)
        except:
            contrast = arg
        return contrast

    # Parse CLI
    parser = argparse.ArgumentParser(description='''Create a workflow for MULM and cluster level stat''')

    parser.add_argument('h5filename',
      type=str,
      help='Read from filename')

    parser.add_argument('workflow_dir',
      type=str,
      help='Directory on which to save the workflow')

    parser.add_argument('--n_perms',
      type=int, default=DEFAULT_N_PERMS,
      help='Number of permutations')

    parser.add_argument('--thresh',
      type=float, default=DEFAULT_THRESH,
      help='p-values threshold')

    parser.add_argument('--contrast',
      type=convert_contrast, default=DEFAULT_CONTRAST,
      help='Contrast (python list or ''auto''; by default 1 with as many 0 as needed')

    args = parser.parse_args()
    pipeline = mulm_stat(**vars(args))