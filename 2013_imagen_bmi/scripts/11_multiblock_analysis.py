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
import tables
import sklearn, sklearn.preprocessing, sklearn.cross_validation

import nibabel

import structured.models as models
import structured.start_vectors as start_vectors
import structured.schemes as schemes
import structured.prox_ops as ops

import mulm

import bmi_utils

def scale_datasets(train, test):
    # Center and scale training and testing data
    # Don't use in place transformation because it would modify the datasets
    # in the calling environment
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test  = scaler.transform(test)

    return (scaled_train, scaled_test)

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
    #rgcca.set_tolerance(5e-12)
    rgcca.set_adjacency_matrix([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 0]])

    # Fitting
    rgcca.fit(X_train, Y_train, Z_train)

    X_W = rgcca.get_transform(0)
    if np.isnan(X_W).any():
        print "nan in X_W"
    if np.isinf(X_W).any():
        print "inf in X_W"

    Y_W = rgcca.get_transform(1)
    if np.isnan(Y_W).any():
        print "nan in Y_W"
    if np.isinf(Y_W).any():
        print "inf in Y_W"

    Z_W = rgcca.get_transform(2)
    if np.isnan(Z_W).any():
        print "nan in Z_W"
    if np.isinf(Z_W).any():
        print "inf in Z_W"

    # Prediction:
    # Warning: X is the design matrix (concatenation of the latent variables for X and Y)
    #          Y is the latent variable for Z
    Yhat = None
    R_squared = None
    try:
        lm = mulm.MUOLS()
        T = rgcca.transform(X_test, Y_test, Z_test)
        X = np.concatenate((T[0], T[1]), axis=1)
        Y = T[2]
        try:
            lm.fit(X=X, Y=Y)
        except:
            if np.isnan(X).any():
                print "nan in X"
            if np.isinf(X).any():
                print "inf in X"
            if np.isnan(Y).any():
                print "nan in Y"
            if np.isinf(Y).any():
                print "inf in Y"
        Yhat = lm.predict(X=X)
        Y_bar = Y.mean()
        SS_tot =  Y.var()
        SS_reg = ((Yhat - Y_bar)**2).mean()
        R_squared = 1 - SS_reg / SS_tot
    except:
        print "Ça n'a pas marché."
    return (rgcca, Z_W, Yhat, R_squared)

if __name__ == '__main__':

    ##############
    # Parameters #
    ##############
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    DATASET_FILE = os.path.join(DATA_PATH, 'dataset.hdf5')

    OUT_DIR = os.path.join(BASE_PATH, 'results', 'multiblock_analysis')

    # TODO: change values
    N_OUTER_FOLDS = 5
    N_INNER_FOLDS = 3
    #PARAM = np.arange(start=0.1, stop=1.0, step=0.1)
    PARAM = [0.3, 0.7]
    param_set = list(itertools.product(PARAM, PARAM, [1]))

    N_PROCESSES = 5

    #############
    # Read data #
    #############

    h5file = tables.openFile(DATASET_FILE)
    SNPs = bmi_utils.read_array(h5file, "/SNPs")
    BMI  = bmi_utils.read_array(h5file, "/BMI")
    masked_images = bmi_utils.read_array(h5file, "/smoothed_images_subsampled_residualized_gender_center_TIV_pds/masked_images")
    print "Data loaded"

    X = masked_images
    Y = SNPs
    Z = BMI[:, 2]

#    N_SUB_SUBJECT = 150
#    X = masked_images[0:N_SUB_SUBJECT, :]
#    Y = SNPs[0:N_SUB_SUBJECT, :]
#    Z = BMI[:, 2][0:N_SUB_SUBJECT, :]

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
    # TODO: reactivate multiprocessing
    #process_pool = multiprocessing.Pool(processes=N_PROCESSES)

    levels = list(itertools.product(range(N_OUTER_FOLDS), itertools.product(PARAM, PARAM), range(N_INNER_FOLDS)))
    index = pd.MultiIndex.from_tuples(levels, names=['Outer_fold', 'L1_params', 'Inner_fold'])
    inner_results = pd.Series(name='Inner_Rsquared', index=index)
    outer_results = pd.DataFrame(columns=['Best_params', 'Outer_Rsquared'], index=range(N_OUTER_FOLDS))

    # Outer loop
    outer_folds = sklearn.cross_validation.KFold(len(X), n_folds=N_OUTER_FOLDS, indices=False)
    best_params = []
    outer_tasks = []
    outer_res = []
    for outer_fold_index, outer_fold_masks in enumerate(outer_folds):
        train_mask, test_mask = outer_fold_masks
        print "In outer fold %i" % outer_fold_index
        X_train, X_test = X[train_mask], X[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        Z_train, Z_test = Z[train_mask], Z[test_mask]
        # Inner loop
        inner_folds = sklearn.cross_validation.KFold(len(X_train), n_folds=N_INNER_FOLDS, indices=False, shuffle=False)
        for inner_fold_index, inner_fold_masks in enumerate(inner_folds):
            print "\tIn inner fold %i" % inner_fold_index
            inner_train_mask, inner_test_mask = inner_fold_masks
            X_inner_train, X_inner_test = X_train[inner_train_mask], X_train[inner_test_mask]
            Y_inner_train, Y_inner_test = Y_train[inner_train_mask], Y_train[inner_test_mask]
            Z_inner_train, Z_inner_test = Z_train[inner_train_mask], Z_train[inner_test_mask]
            X_inner_train_std, X_inner_test_std = scale_datasets(X_inner_train, X_inner_test)
            Y_inner_train_std, Y_inner_test_std = scale_datasets(Y_inner_train, Y_inner_test)
            Z_inner_train_std, Z_inner_test_std = scale_datasets(Z_inner_train, Z_inner_test)
            # Store data
            inner_fold_dir = os.path.join(OUT_DIR, "{outer}/{inner}".format(outer=outer_fold_index, inner=inner_fold_index))
            if not os.path.exists(inner_fold_dir):
                os.makedirs(inner_fold_dir)
            # Raw data
            X_inner_train.tofile(os.path.join(inner_fold_dir, 'X_inner_train.bin'))
            X_inner_test.tofile(os.path.join(inner_fold_dir, 'X_inner_test.bin'))
            Y_inner_train.tofile(os.path.join(inner_fold_dir, 'Y_inner_train.bin'))
            Y_inner_test.tofile(os.path.join(inner_fold_dir, 'Y_inner_test.bin'))
            Z_inner_train.tofile(os.path.join(inner_fold_dir, 'Z_inner_train.bin'))
            Z_inner_test.tofile(os.path.join(inner_fold_dir, 'Z_inner_test.bin'))
            # Normalized data
            X_inner_train_std.tofile(os.path.join(inner_fold_dir, 'X_inner_train_std.bin'))
            X_inner_test_std.tofile(os.path.join(inner_fold_dir, 'X_inner_test_std.bin'))
            Y_inner_train_std.tofile(os.path.join(inner_fold_dir, 'Y_inner_train_std.bin'))
            Y_inner_test_std.tofile(os.path.join(inner_fold_dir, 'Y_inner_test_std.bin'))
            Z_inner_train_std.tofile(os.path.join(inner_fold_dir, 'Z_inner_train_std.bin'))
            Z_inner_test_std.tofile(os.path.join(inner_fold_dir, 'Z_inner_test_std.bin'))
            # Put tasks in list
            inner_tasks = []
            inner_res = []
            for l1_params in param_set:
                inner_tasks.append((l1_params, X_inner_train_std, X_inner_test_std, Y_inner_train_std, Y_inner_test_std, Z_inner_train_std, Z_inner_test_std))
                print '\t\t', l1_params
                inner_res.append(rgcca_fit_predict((l1_params, X_inner_train_std, X_inner_test_std, Y_inner_train_std, Y_inner_test_std, Z_inner_train_std, Z_inner_test_std)))
            # TODO: reactivate multiprocessing
            #inner_res = process_pool.map(rgcca_fit_predict, inner_tasks)
            # Put results of this inner fold in a dataframe
            for i, l1_params in enumerate(param_set):
                rsquare = inner_res[i][3]
                inner_results[outer_fold_index, (l1_params[0], l1_params[1]), inner_fold_index] = rsquare
        # End inner loop

        # Find best parameters for this outer fold
        outer_fold_average = inner_results.loc[outer_fold_index].mean(level=0)
        best_param = list(outer_fold_average.idxmax()) # transform the tuple in list
        best_params.append(best_param)
        best_param.append(1.0) # Append parameter for Z block
        outer_tasks.append((best_param, X_train, X_test, Y_train, Y_test, Z_train, Z_test))
        outer_res.append(rgcca_fit_predict((best_param, X_train, X_test, Y_train, Y_test, Z_train, Z_test)))
    # TODO: reactivate multiprocessing
    #outer_res = process_pool.map(rgcca_fit_predict, outer_tasks)
    for i in range(N_OUTER_FOLDS):
        outer_results['Best_params'][i] = best_params[i]
        outer_results['Outer_Rsquared'][i] = outer_res[i][3]
    inner_results.to_csv(os.path.join(OUT_DIR, 'inner_results.csv'), header=True)
    outer_results.to_csv(os.path.join(OUT_DIR, 'outer_results.csv'), header=True)
