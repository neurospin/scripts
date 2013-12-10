# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:52:24 2013

@author: md238665

This script create a workflow for parameter selection of the SGCCA model.

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

#import structured.models as models
#import structured.start_vectors as start_vectors
#import structured.schemes as schemes
#import structured.prox_ops as ops

from soma_workflow.client import Job, Workflow, Group, Helper

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

if __name__ == '__main__':

    ##############
    # Parameters #
    ##############
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    IMAGES_FILE = os.path.join(DATA_PATH, 'Archives', 'subsampled_smoothed_images.hdf5')
    SNPS_FILE = os.path.join(DATA_PATH, 'SNPs.csv')
    BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

    OUT_DIR = os.path.join(BASE_PATH, 'results', 'multiblock_analysis')

    # TODO: change values
    N_OUTER_FOLDS = 5
    N_INNER_FOLDS = 3
    #PARAM = np.arange(start=0.1, stop=1.0, step=0.1)
    PARAM = [0.3, 0.7]
    param_set = list(itertools.product(PARAM, PARAM, [1]))

    N_PROCESSES = 5

    C = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]

    global BASE_SGCCA_CMD, BASE_PREDICT_CMD
    SGCCA_PATH = "/home/md238665/Code/brainomics-team/SGCCA_py"
    BASE_SGCCA_CMD = ["python", os.path.join(SGCCA_PATH, "SGCCA.py")]
    BASE_PREDICT_CMD = ["python", os.path.join(SGCCA_PATH, "simple_predict.py")]

    # Return a fit command
    def fit_command(block_filenames, J, C, c, scheme, model_filename):
        cmd = BASE_SGCCA_CMD
        cmd = cmd + ["--input_files"] + block_filenames
        cmd = cmd + ["--J", str(J)]
        cmd = cmd + ["--C", str(C)]
        cmd = cmd + ["--c", str(c)]
        cmd = cmd + ["--scheme", scheme]
        # Output
        cmd = cmd + ["--model", model_filename]
        #print cmd
        return cmd

    # Return a transform command
    def transform_command(model_filename, block_filenames, transformed_filenames):
        cmd = BASE_SGCCA_CMD
        cmd = cmd + ["--model", model_filename]
        cmd = cmd + ["--input_files"] + block_filenames
        # Output
        cmd = cmd + ["--output_files"] + transformed_filenames
        #print cmd
        return cmd

    def predict_command(transformed_filenames, predict_output):
        cmd = BASE_PREDICT_CMD
        cmd = cmd + ["--input_files"] + transformed_filenames
        # Output
        # TODO: change that
        cmd = cmd + ["--output_file", predict_output]
        #print cmd
        return cmd

    #############
    # Read data #
    #############
#    # SNPs and BMI
#    SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
#    BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()
#
#    # Images
#    h5file = tables.openFile(IMAGES_FILE)
#    masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")
#    print "Data loaded"

    X = np.random.random((100, 1000))
    Y = np.random.random((100, 100))
    Z = np.random.random((100, 1))

    #################################
    # Create cross-validation tasks #
    #################################
    jobs = []
    dependencies = []
    group_elements = []

    # Outer loop
    outer_folds = sklearn.cross_validation.KFold(len(X), n_folds=N_OUTER_FOLDS, indices=False)
    for outer_fold_index, outer_fold_masks in enumerate(outer_folds):
        train_mask, test_mask = outer_fold_masks
        outer_fold_dir = os.path.join(OUT_DIR, str(outer_fold_index))
#        if not os.path.exists(outer_fold_dir):
#            os.makedirs(outer_fold_dir)
        X_train, X_test = X[train_mask], X[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        Z_train, Z_test = Z[train_mask], Z[test_mask]
        # Inner loop
        inner_folds = sklearn.cross_validation.KFold(len(X_train), n_folds=N_INNER_FOLDS, indices=False, shuffle=False)
        for inner_fold_index, inner_fold_masks in enumerate(inner_folds):
            print "\tIn inner fold %i" % inner_fold_index
            inner_train_mask, inner_test_mask = inner_fold_masks
            inner_fold_dir = os.path.join(outer_fold_dir, str(inner_fold_index))
#            if not os.path.exists(inner_fold_dir):
#                os.makedirs(inner_fold_dir)
            X_inner_train, X_inner_test = X_train[inner_train_mask], X_train[inner_test_mask]
            Y_inner_train, Y_inner_test = Y_train[inner_train_mask], Y_train[inner_test_mask]
            Z_inner_train, Z_inner_test = Z_train[inner_train_mask], Z_train[inner_test_mask]

            # Scale data
            X_inner_train_std, X_inner_test_std = scale_datasets(X_inner_train, X_inner_test)
            Y_inner_train_std, Y_inner_test_std = scale_datasets(Y_inner_train, Y_inner_test)
            Z_inner_train_std, Z_inner_test_std = scale_datasets(Z_inner_train, Z_inner_test)
            # Store normalized data
            full_inner_X_train = os.path.join(inner_fold_dir, 'X_inner_train_std.npy')
            #np.save(full_inner_X_train, X_inner_train_std)
            full_inner_X_test = os.path.join(inner_fold_dir, 'X_inner_test_std.npy')
            #np.save(full_inner_X_test, X_inner_test_std)

            full_inner_Y_train = os.path.join(inner_fold_dir, 'Y_inner_train_std.npy')
            #np.save(full_inner_Y_train, Y_inner_train_std)
            full_inner_Y_test = os.path.join(inner_fold_dir, 'Y_inner_test_std.npy')
            #np.save(full_inner_Y_test, Y_inner_test_std)

            full_inner_Z_train = os.path.join(inner_fold_dir, 'Z_inner_train_std.npy')
            #np.save(full_inner_Z_train, Z_inner_train_std)
            full_inner_Z_test = os.path.join(inner_fold_dir, 'Z_inner_test_std.npy')
            #np.save(full_inner_Z_test, Z_inner_test_std)
            # TODO: save scaling parameters
            inner_fold_training_files = [full_inner_X_train,
                                         full_inner_Y_train,
                                         full_inner_Z_train]
            inner_fold_testing_files = [full_inner_X_test,
                                        full_inner_Y_test,
                                        full_inner_Z_test]

            # Compute SVD of training data
            # TODO:
            # Store SVD
            # TODO:

            # Put tasks in WF
            inner_fold_jobs = []
            for l1_params in param_set:
                param_dir = os.path.join(inner_fold_dir,
                                         "{c[0]}_{c[1]}".format(c=l1_params))
#                if not os.path.exists(param_dir):
#                    os.makedirs(param_dir)
                model_filename = os.path.join(param_dir, "model.pkl")
                param_transform_files=[os.path.join(param_dir, f)
                                       for f in ["X.transformed.npy", "Y.transformed.npy", "Z.transformed.npy"]]
                param_prediction_file = os.path.join(param_dir, "prediction.npz")
                # Fitting task
                fit_cmd = fit_command(inner_fold_training_files,
                                      3, C, l1_params, "centroid",
                                      model_filename)
                job_name = "{out}/{inn}/{c}/fit".format(
                            out=outer_fold_index,
                            inn=inner_fold_index,
                            c=l1_params)
                fit = Job(command=fit_cmd, name=job_name)
                jobs.append(fit)
                # TODO: ajouter dépendance à transferts
                inner_fold_jobs.append(fit)  # Just for grouping

                # Transform task
                transform_cmd = transform_command(model_filename,
                                                  inner_fold_testing_files,
                                                  param_transform_files)
                job_name = "{out}/{inn}/{c}/transform".format(
                            out=outer_fold_index,
                            inn=inner_fold_index,
                            c=l1_params)
                transform = Job(command=transform_cmd, name=job_name)
                jobs.append(transform)
                dependencies.append((fit, transform))
                inner_fold_jobs.append(transform)  # Just for grouping
                # Predict task
                predict_cmd = predict_command(param_transform_files,
                                              param_prediction_file)
                job_name = "{out}/{inn}/{c}/predict".format(
                            out=outer_fold_index,
                            inn=inner_fold_index,
                            c=l1_params)
                predict = Job(command=predict_cmd, name=job_name)
                jobs.append(predict)
                dependencies.append((transform, predict))
                inner_fold_jobs.append(predict)  # Just for grouping
            # End loop on params
            group_elements.append(Group(elements=inner_fold_jobs,
                                        name="Outer fold {out}/Inner fold {inn}".format(
                                        out=outer_fold_index,
                                        inn=inner_fold_index)))
        # End inner loop
    # End outer loop

    workflow = Workflow(jobs=jobs,
                        dependencies=dependencies,
                        root_group=group_elements)

    # save the workflow into a file
    Helper.serialize("/tmp/workflow_example", workflow)
