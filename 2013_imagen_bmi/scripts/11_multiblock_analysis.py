# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:52:24 2013

@author: md238665

This script performs a parameter selection of the SGCCA model.
SGCCA is implemented using RGCCA + L1 regularization.

TODO:
 - save the weights (at least of each outer fold)
 - allow to use different data
 - output path?

"""

import os, fnmatch, itertools
import multiprocessing

import numpy as np
import pandas as pd
import sklearn, sklearn.preprocessing, sklearn.cross_validation

import nibabel

import structured.models as models
import structured.start_vectors as start_vectors
import structured.schemes as schemes
import structured.prox_ops as ops

import mulm

def rgcca_fit_predict(fold):
    '''This function will be called several times'''
    params = fold[0]
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = fold[1:]

    # Create estimator
    rgcca = models.RGCCA(num_comp=1)
    rgcca.set_prox_op(ops.L1(*params))
    rgcca.set_start_vector(start_vectors.OnesStartVector())
    rgcca.set_scheme(schemes.Factorial())
    rgcca.set_max_iter(10000)
    rgcca.set_tolerance(5e-12)
    rgcca.set_adjacency_matrix([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 0]])

    # Center and scale training data (in place)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train, copy=False)
    scaler.transform(X_test,  copy=False)

    scaler.fit(Y_train)
    scaler.transform(Y_train, copy=False)
    scaler.transform(Y_test,  copy=False)

    scaler.fit(Z_train)
    scaler.transform(Z_train, copy=False)
    scaler.transform(Z_test,  copy=False)

    # Fitting
    rgcca.fit(X_train, Y_train, Z_train)

    # Prediction
    lm = mulm.MUOLS()
    T = rgcca.transform(X_test, Y_test, Z_test)
    design_mat = np.concatenate((T[0], T[1]), axis=1)
    lm.fit(X=design_mat, Y=T[2])
    Zhat = lm.predict(X=design_mat)
    Z_test_bar = Z_test.mean()
    SS_tot =  Z_test.var()
    SS_reg = ((Zhat - Z_test_bar)**2).mean()
    R_squared = 1 - SS_reg / SS_tot
    return (rgcca, Zhat, R_squared)

if __name__ == '__main__':

    ##############
    # Parameters #
    ##############
    DATA_PATH = '/neurospin/brainomics/2013_imagen_bmi/data'
    #IMG_PATH='VBM/gaser_vbm8/'
    IMG_PATH='.'
    FULL_IMG_DIR = os.path.join(DATA_PATH, IMG_PATH)
    IMG_FILENAME_TEMPLATE = 'rsmwp1{subject_id:012}*.nii' # TODO: try full data

    MASK_PATH   = os.path.join(DATA_PATH, 'rmask.nii')
    babel_mask  = nibabel.load(MASK_PATH)
    mask        = babel_mask.get_data()
    binary_mask = mask!=0
    useful_voxels = np.ravel_multi_index(np.where(binary_mask), mask.shape)
    n_useful_voxels = len(useful_voxels)

    # TODO: change values
    N_OUTER_FOLDS = 10
    N_INNER_FOLDS = 5
    PARAM = np.arange(start=0.1, stop=1.0, step=0.1)
    param_set = list(itertools.product(PARAM, PARAM, [1]))

    N_PROCESSES = 5

    #############
    # Read data #
    #############

    # Read clinic data & SNPs
    SNPs   = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'SNPs.csv'), index_col=0)
    clinic = pd.io.parsers.read_csv(os.path.join(DATA_PATH, 'BMI.csv'), index_col=0)
    subject_indices = clinic.index
    n_subjects = subject_indices.shape[0]

    # Read images in the same order than subjects
    masked_images = np.zeros((n_subjects, n_useful_voxels))
    img_dir_files = os.listdir(FULL_IMG_DIR)
    for (index, subject_index) in enumerate(subject_indices):
        # Find filename
        pattern = IMG_FILENAME_TEMPLATE.format(subject_id=subject_index)
        filename = fnmatch.filter(img_dir_files, pattern)
        if len(filename) != 1:
            raise Exception
        else:
            filename = os.path.join(FULL_IMG_DIR, filename[0])
        # Load (as numpy array)
        image = nibabel.load(filename).get_data()
        # Apply mask (returns a flat image)
        masked_image = image[binary_mask]
        # Store in Y
        masked_images[index, :] = masked_image
    print "Data loaded"

    X = masked_images
    Y = SNPs.astype(np.float64).as_matrix()
    Z = clinic['BMI'].as_matrix()

#    N_SUB_SUBJECT = n_subjects
#    X = masked_images[0:N_SUB_SUBJECT, 0:200]
#    Y = SNPs.astype(np.float64).as_matrix()[0:N_SUB_SUBJECT, 0:50]
#    Z = clinic.as_matrix()[0:N_SUB_SUBJECT, :]

#    N_SUB_SUBJECT = 150
#    X = masked_images[0:N_SUB_SUBJECT, :]
#    Y = SNPs.astype(np.float64).as_matrix()[0:N_SUB_SUBJECT, :]
#    Z = clinic.as_matrix()[0:N_SUB_SUBJECT, :]

    ####################
    # Cross-validation #
    ####################
    # Create process pool
    process_pool = multiprocessing.Pool(processes=N_PROCESSES)

    levels = list(itertools.product(range(N_OUTER_FOLDS), itertools.product(PARAM, PARAM), range(N_INNER_FOLDS)))
    index = pd.MultiIndex.from_tuples(levels, names=['Outer_fold', 'L1_params', 'Inner_fold'])
    inner_results = pd.Series(name='Inner_Rsquared', index=index)
    outer_results = pd.DataFrame(columns=['Best_params', 'Outer_Rsquared'], index=range(N_OUTER_FOLDS))

    # Outer loop
    outer_folds = sklearn.cross_validation.KFold(len(X), n_folds=N_OUTER_FOLDS, indices=False)
    best_params = []
    outer_tasks = []
    for outer_fold_index, outer_fold_masks in enumerate(outer_folds):
        train_mask, test_mask = outer_fold_masks
        print "In outer fold %i" % outer_fold_index
        X_train, X_test = X[train_mask], X[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        Z_train, Z_test = Z[train_mask], Z[test_mask]
        # Inner loop
        inner_folds = sklearn.cross_validation.KFold(len(X_train), n_folds=N_INNER_FOLDS, indices=False)
        for inner_fold_index, inner_fold_masks in enumerate(inner_folds):
            print "\tIn inner fold %i" % inner_fold_index
            inner_train_mask, inner_test_mask = inner_fold_masks
            X_inner_train, X_inner_test = X_train[inner_train_mask], X_train[inner_test_mask]
            Y_inner_train, Y_inner_test = Y_train[inner_train_mask], Y_train[inner_test_mask]
            Z_inner_train, Z_inner_test = Z_train[inner_train_mask], Z_train[inner_test_mask]
            inner_tasks = []
            for l1_params in param_set:
                inner_tasks.append((l1_params, X_inner_train, X_inner_test, Y_inner_train, Y_inner_test, Z_inner_train, Z_inner_test))
                #print l1_params
            res = process_pool.map(rgcca_fit_predict, inner_tasks)
            # Put results of this inner fold in a dataframe
            for i, l1_params in enumerate(param_set):
                rsquare = res[i][2]
                inner_results[outer_fold_index, (l1_params[0], l1_params[1]), inner_fold_index] = rsquare
        # End inner loop

        # Find best parameters for this outer fold
        outer_fold_average = inner_results.loc[outer_fold_index].mean(level=0)
        best_param = list(outer_fold_average.idxmax()) # transform the tuple in list
        best_params.append(best_param)
        best_param.append(1.0) # Append parameter for Z block
        outer_tasks.append((best_param, X_train, X_test, Y_train, Y_test, Z_train, Z_test))
    out_res = process_pool.map(rgcca_fit_predict, outer_tasks)
    for i in range(N_OUTER_FOLDS):
        outer_results['Best_params'][i] = best_params[i]
        outer_results['Outer_Rsquared'][i] = out_res[i][2]
    inner_results.to_csv('inner_results.csv', header=True)
    outer_results.to_csv('outer_results.csv', header=True)