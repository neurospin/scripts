# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:52:24 2013

@author: md238665

This script create a workflow for parameter selection of the SGCCA model.

Data are written to /neurospin/tmp which is accesible from clients and from
the gabriel cluster.
The excutable are from the python SGCCA implementation (see brainomics-team repository).

TODO:
 - file transfert?
 - find a way to store less data (for instance store only indices and generate
   dataset on nodes - requires a python script to do that)
  -compute SVD on nodes?

"""

import os, itertools

import numpy as np
import pandas as pd

import sklearn, sklearn.preprocessing, sklearn.cross_validation

from soma_workflow.client import Job, Workflow, Group, Helper

import tables

import bmi_utils

def scale_datasets(train, test):
    # Center and scale training and testing data
    # Don't use in place transformation because it would modify the datasets
    # in the calling environment
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test  = scaler.transform(test)

    return (scaler, scaled_train, scaled_test)


##############
# Parameters #
##############
# Input data
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
SNPS_FILE = os.path.join(DATA_PATH, 'SNPs.csv')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/md238665"
SHARED_DIR = os.path.join(BASE_SHARED_DIR, 'multiblock_analysis')
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)

# Results
OUT_DIR = os.path.join(BASE_PATH, 'results', 'multiblock_analysis')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
WF_NAME = "SGCCA_hyperparameter_selection_wf"

# CV parameters
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5

# Model parameters
L1_PARAM = np.arange(start=0.1, stop=1.0, step=0.1)
L1_PARAM_SET = list(itertools.product(L1_PARAM, L1_PARAM, [1]))

C_hier = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
C_complete = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
C_DICT = {"hierarchical": C_hier,
          "complete": C_complete}
C_PARAM_SET = ["hierarchical", "complete"]

PARAM_SET = list(itertools.product(C_PARAM_SET, L1_PARAM_SET))

#N_PROCESSES = 5

global BASE_SGCCA_CMD, BASE_PREDICT_CMD, BASE_SVD_CMD
SGCCA_PATH = "/home/md238665/Code/brainomics-team/SGCCA_py"
BASE_SGCCA_CMD = ["python", os.path.join(SGCCA_PATH, "SGCCA.py")]
BASE_PREDICT_CMD = ["python", os.path.join(SGCCA_PATH, "simple_predict.py")]
BASE_SVD_CMD = [os.path.join(SGCCA_PATH, "compute_first_SVD.py")]

# Return a command to compute the first SVD
def svd_command(scaled_block_file, init_file):
    cmd = BASE_SVD_CMD
    # Input
    cmd = cmd + ["--input_file", scaled_block_file]
    # Output
    cmd = cmd + ["--output_file", init_file]
    return cmd

# Return a fit command
def fit_command(block_files, J, C, c, scheme, init_files, model_filename):
    cmd = BASE_SGCCA_CMD
    cmd = cmd + ["--input_files"] + block_files
    cmd = cmd + ["--J", str(J)]
    cmd = cmd + ["--C", str(C)]
    cmd = cmd + ["--c", str(c)]
    cmd = cmd + ["--scheme", scheme]
    cmd = cmd + ["--init"] + init_files
    # Output
    cmd = cmd + ["--model", model_filename]
    #print cmd
    return cmd

# Return a transform command
def transform_command(model_filename, block_filenames, transformed_filenames):
    cmd = BASE_SGCCA_CMD
    cmd = cmd + ["--transform"]
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
    cmd = cmd + ["--output_file", predict_output]
    #print cmd
    return cmd

#############
# Read data #
#############
# SNPs and BMI
SNPs = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "SNPs.csv"), dtype='float64', index_col=0).as_matrix()
BMI = pd.io.parsers.read_csv(os.path.join(DATA_PATH, "BMI.csv"), index_col=0).as_matrix()

# Images
h5file = tables.openFile(IMAGES_FILE)
#masked_images = bmi_utils.read_array(h5file, "/standard_mask/residualized_images_gender_center_TIV_pds")
print "Data loaded"

#X = masked_images
#Y = SNPs
#Z = BMI

X = np.random.random((100, 1000))
Y = np.random.random((100, 100))
Z = np.random.random((100, 1))

####################################
# Create cross-validation workflow #
#  & data                          #
####################################
jobs = []
dependencies = []
group_elements = []

# Outer loop
outer_folds = sklearn.cross_validation.KFold(len(X), n_folds=N_OUTER_FOLDS, indices=False)
for outer_fold_index, outer_fold_masks in enumerate(outer_folds):
    print "In outer fold %i" % outer_fold_index
    train_mask, test_mask = outer_fold_masks
    outer_fold_dir = os.path.join(SHARED_DIR, str(outer_fold_index))
    if not os.path.exists(outer_fold_dir):
        os.makedirs(outer_fold_dir)
    X_train, X_test = X[train_mask], X[test_mask]
    Y_train, Y_test = Y[train_mask], Y[test_mask]
    Z_train, Z_test = Z[train_mask], Z[test_mask]
    # Inner loop
    inner_folds = sklearn.cross_validation.KFold(len(X_train), n_folds=N_INNER_FOLDS, indices=False, shuffle=False)
    for inner_fold_index, inner_fold_masks in enumerate(inner_folds):
        print "\tIn inner fold %i" % inner_fold_index
        inner_train_mask, inner_test_mask = inner_fold_masks
        inner_fold_dir = os.path.join(outer_fold_dir, str(inner_fold_index))
        if not os.path.exists(inner_fold_dir):
            os.makedirs(inner_fold_dir)
        X_inner_train, X_inner_test = X_train[inner_train_mask], X_train[inner_test_mask]
        Y_inner_train, Y_inner_test = Y_train[inner_train_mask], Y_train[inner_test_mask]
        Z_inner_train, Z_inner_test = Z_train[inner_train_mask], Z_train[inner_test_mask]

        # Scale data
        X_scaler, X_inner_train_std, X_inner_test_std = scale_datasets(X_inner_train,
                                                                       X_inner_test)
        del X_inner_train, X_inner_test
        Y_scaler, Y_inner_train_std, Y_inner_test_std = scale_datasets(Y_inner_train,
                                                                       Y_inner_test)
        del Y_inner_train, Y_inner_test
        Z_scaler, Z_inner_train_std, Z_inner_test_std = scale_datasets(Z_inner_train,
                                                                       Z_inner_test)
        del Z_inner_train, Z_inner_test

        # Store normalized data & scaling parameters
        full_X_inner_train = os.path.join(inner_fold_dir, 'X_inner_train_std.npy')
        np.save(full_X_inner_train, X_inner_train_std)
        full_X_inner_test = os.path.join(inner_fold_dir, 'X_inner_test_std.npy')
        np.save(full_X_inner_test, X_inner_test_std)
        np.savez(os.path.join(inner_fold_dir, 'X_inner_train_scaling.npz'),
                              mean_=X_scaler.mean_, std_=X_scaler.std_)
        del X_scaler

        full_Y_inner_train = os.path.join(inner_fold_dir, 'Y_inner_train_std.npy')
        np.save(full_Y_inner_train, Y_inner_train_std)
        full_Y_inner_test = os.path.join(inner_fold_dir, 'Y_inner_test_std.npy')
        np.save(full_Y_inner_test, Y_inner_test_std)
        np.savez(os.path.join(inner_fold_dir, 'Y_inner_train_scaling.npz'),
                              mean_=Y_scaler.mean_, std_=Y_scaler.std_)
        del Y_scaler

        full_Z_inner_train = os.path.join(inner_fold_dir, 'Z_inner_train_std.npy')
        np.save(full_Z_inner_train, Z_inner_train_std)
        full_Z_inner_test = os.path.join(inner_fold_dir, 'Z_inner_test_std.npy')
        np.save(full_Z_inner_test, Z_inner_test_std)
        np.savez(os.path.join(inner_fold_dir, 'Z_inner_train_scaling.npz'),
                              mean_=Z_scaler.mean_, std_=Z_scaler.std_)
        del Z_scaler

        inner_fold_training_files = [full_X_inner_train,
                                     full_Y_inner_train,
                                     full_Z_inner_train]
        inner_fold_testing_files = [full_X_inner_test,
                                    full_Y_inner_test,
                                    full_Z_inner_test]
        # Don't need scaled test data anymore
        del X_inner_test_std, Y_inner_test_std, Z_inner_test_std

        # Put tasks in WF
        inner_fold_jobs = []

        # Computation of init vectors
        common_job_name = "{out}/{inn}/init".format(
                               out=outer_fold_index,
                               inn=inner_fold_index)

        full_X_inner_init = os.path.join(inner_fold_dir, 'X_inner_init.npy')
        X_init_cmd = svd_command(full_X_inner_train,
                                 full_X_inner_init)
        job_name = common_job_name+"/X"
        X_init = Job(command=X_init_cmd, name=job_name)
        jobs.append(X_init)
        inner_fold_jobs.append(X_init)

        full_Y_inner_init = os.path.join(inner_fold_dir, 'Y_inner_init.npy')
        Y_init_cmd = svd_command(full_Y_inner_train,
                                 full_Y_inner_init)
        job_name = common_job_name+"/Y"
        Y_init = Job(command=Y_init_cmd, name=job_name)
        jobs.append(Y_init)
        inner_fold_jobs.append(Y_init)
        
        full_Z_inner_init = os.path.join(inner_fold_dir, 'Z_inner_init.npy')
        Z_init_cmd = svd_command(full_Z_inner_train,
                                 full_Z_inner_init)
        job_name = common_job_name+"/Z"
        Z_init = Job(command=Z_init_cmd, name=job_name)
        jobs.append(Z_init)
        inner_fold_jobs.append(Z_init)

        inner_fold_init = [full_X_inner_init,
                           full_Y_inner_init,
                           full_Z_inner_init]

        # Fit, transform and predict tasks
        for C_name, l1_params in PARAM_SET:
            C =  C_DICT[C_name]
            param_dir = os.path.join(inner_fold_dir,
                                     "{name}_{c[0]}_{c[1]}".format(name=C_name,
                                                                   c=l1_params))
            common_job_name = "{out}/{inn}/{name}/{c}".format(
                               out=outer_fold_index,
                               inn=inner_fold_index,
                               name=C_name,
                               c=l1_params)
            if not os.path.exists(param_dir):
                os.makedirs(param_dir)
            model_filename = os.path.join(param_dir, "model.pkl")
            param_transform_files=[os.path.join(param_dir, f)
                                   for f in ["X.transformed.npy", "Y.transformed.npy", "Z.transformed.npy"]]
            param_prediction_file = os.path.join(param_dir, "prediction.npz")

            # Fitting task
            fit_cmd = fit_command(inner_fold_training_files,
                                  3, C, l1_params, "centroid", inner_fold_init,
                                  model_filename)
            job_name = common_job_name+"/fit"
            fit = Job(command=fit_cmd, name=job_name)
            jobs.append(fit)
            inner_fold_jobs.append(fit)  # Just for grouping
            dependencies.append((X_init, fit))
            dependencies.append((Y_init, fit))
            dependencies.append((Z_init, fit))
            # TODO: ajouter dépendance à transferts

            # Transform task
            transform_cmd = transform_command(model_filename,
                                              inner_fold_testing_files,
                                              param_transform_files)
            job_name = common_job_name+"/transform"
            transform = Job(command=transform_cmd, name=job_name)
            jobs.append(transform)
            dependencies.append((fit, transform))
            inner_fold_jobs.append(transform)  # Just for grouping
            # Predict task
            predict_cmd = predict_command(param_transform_files,
                                          param_prediction_file)
            job_name = common_job_name+"/predict"
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
                    root_group=group_elements,
                    name=WF_NAME)

# save the workflow into a file
Helper.serialize(os.path.join(OUT_DIR, WF_NAME), workflow)
