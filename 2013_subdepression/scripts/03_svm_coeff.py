# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

Run some SVM and try to display the support vector in a brain-friendly way.
This is done to illustrate some caracteristics
Parameters C and k (# features) could be taken from the output of 02_svm_choose_param.py

"""

DEFAULT_C=1
DEFAULT_K=0
DEFAULT_PENALTY='l1'

# Standard library modules
import os, sys, argparse
# Numpy and friends
import numpy
import sklearn, sklearn.svm, sklearn.feature_selection
# For reading HDF5 files
import tables
# For images
import nibabel

import epac

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

if __name__ == '__main__':

    # Parse CLI
    parser = argparse.ArgumentParser(description='''Load the data in HDF5 and apply SVM''')

    parser.add_argument('h5filename',
      type=str,
      help='Read from filename')

    parser.add_argument('outfilename',
      type=str,
      help='Write to outfilename')

    parser.add_argument('--C',
      type=float, default=DEFAULT_C,
      help='Values for C (float)')

    parser.add_argument('--k',
      type=int, default=DEFAULT_K,
      help='Values for k (integer; 0 = all features)')

    parser.add_argument('--penalty',
      type=str, default=DEFAULT_PENALTY,
      help='Penalty function (''l1'' or ''l2)''')

    args = parser.parse_args()

    # Open the output file
    h5file = tables.openFile(args.h5filename, mode = "r")
    X, Y, mask, mask_affine = data_api.get_data(h5file)
    y = Y[:, 0]
    np_mask = numpy.asarray(mask)
    binary_mask = np_mask!=0
    if args.k==0:
        args.k = X.shape[1]

    # Create objects
    anova_svm = epac.Pipe(
                  sklearn.feature_selection.SelectKBest(k=args.k),
                  sklearn.preprocessing.StandardScaler(),
                  sklearn.svm.LinearSVC(class_weight='auto',
                                C=args.C, penalty=args.penalty,
                                dual=False))

    # Fit the SVM
    print 'Running'
    anova_svm.run(X=X, y=y)
    print 'Finished'

    selector = anova_svm.estimator
    leaves = [l for l in anova_svm.walk_leaves()]
    svm = leaves[-1].estimator
    
    # Put the selecetd features in an image
    feature_support = selector.get_support()
    selected_features = numpy.zeros(binary_mask.shape, dtype=bool)
    selected_features[binary_mask] = feature_support    

    # Plot the hyperplane parameters
    betas = svm.coef_
    betas_img = numpy.zeros(binary_mask.shape)
    betas_img[selected_features] = betas[0, :]
    outimg = nibabel.Nifti1Image(betas_img, mask_affine)
    nibabel.save(outimg, args.outfilename)

    h5file.close()

