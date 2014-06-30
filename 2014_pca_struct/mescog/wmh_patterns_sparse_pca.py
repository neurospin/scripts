# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:35:45 2014

@author: md238665

Read the data generated in 2013_mescog/proj_wmh_patterns.

We use the centered data.

Warning: some subjects are not in the clinic data (used for stratification)
so we remove them.

"""

import os
import json

import numpy as np

import sklearn
import sklearn.decomposition
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"

INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "datasets")

#
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "CAD-WMH-MNI.npy")
INPUT_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                             "CAD-WMH-MNI-subjects.txt")

INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

INPUT_CLINIC_DIR = os.path.join(INPUT_BASE_DIR,
                                "mescog", "proj_predict_cog_decline", "data")
INPUT_CSV = os.path.join(INPUT_CLINIC_DIR,
                         "dataset_clinic_niglob_20140121.csv")

OUTPUT_BASE_DIR = "/neurospin/brainomics"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "2014_pca_struct", "mescog")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_DATASET = os.path.join(OUTPUT_DIR, "dataset.npy")

# Output for scikit-learn sparse PCA: alpha will be replaced by actual value
OUTPUT_DIR  = os.path.join(OUTPUT_DIR, "SparsePCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
#OUTPUT_SPARSE_PCA_PRED = os.path.join(OUTPUT_SPARSE_PCA_DIR, "X_pred.npy")
#OUTPUT_SPARSE_PCA_COMP = os.path.join(OUTPUT_SPARSE_PCA_DIR, "components.npy")

##############
# Parameters #
##############

N_COMP = 5
SPARSE_PCA_ALPHA = np.arange(0, 10, 1)

#############
# Functions #
#############

def load_globals(config):
    pass

def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    print [toto.shape for toto in GLOBAL.DATA.values()]
    print [idx for idx in resample]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}

def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
    # GLOBAL.DATA
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    alpha, = key
    sparse_pca = sklearn.decomposition.SparsePCA(n_components=N_COMP,
                                                 alpha=alpha)
    sparse_pca.fit(GLOBAL.DATA_RESAMPLED["X"][0])
    X_transform = sparse_pca.transform(GLOBAL.DATA_RESAMPLED["X"][1])
    ret = dict(X_transform=X_transform,
               V=sparse_pca.V,
               U=sparse_pca.U)
    output_collector.collect(key, ret)

def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    return None


#################
# Actual script #
#################

if __name__ == "__main__":

    # Read clinic status (used to split groups)
    clinic_data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
    clinic_subjects_id = [int(subject_id[4:]) for subject_id in clinic_data.index]
    clinic_data.index = clinic_subjects_id
    print "Found", len(clinic_subjects_id), "clinic records"

    # Read ID of subjects
    with open(INPUT_SUBJECTS) as f:
        wmh_subjects_id = np.array([int(l) for l in f.readlines()])
    print "Found", len(wmh_subjects_id), "WMH maps"

    # Intersection of subjects list
    subjects_id = np.intersect1d(wmh_subjects_id, clinic_subjects_id)
    lines_to_keep = np.where(np.in1d(wmh_subjects_id, subjects_id))[0]
    print "Found", len(subjects_id), "correct subjects"
    subjects_to_remove = np.setdiff1d(wmh_subjects_id, clinic_subjects_id)
    lines_to_remove = np.where(np.in1d(wmh_subjects_id, subjects_to_remove))[0]
    print "Should remove subjects", subjects_to_remove, \
          "corresponding to lines", lines_to_remove

    # Subsample images
    all_images = np.load(INPUT_DATASET)
    dataset = all_images[lines_to_keep]
    np.save(OUTPUT_DATASET, dataset)

    # Subsample clinic data
    clinic_data_sub = clinic_data.loc[subjects_id]

    # Stratification of subjects
    y = clinic_data_sub['SITE'].map({'FR': 0, 'GE':1})
    skf = StratifiedKFold(y=y, n_folds=2)
    resample = [[tr.tolist(), te.tolist()] for tr, te in skf]

    # parameters grid
    params = SPARSE_PCA_ALPHA.reshape((-1,1)).tolist()

    # User map/reduce function file:
    user_func_filename = os.path.abspath(__file__)

    # Create config file
    config = dict(data=dict(X=OUTPUT_DATASET),
                  params=params,
                  resample=resample,
                  map_output=os.path.join(OUTPUT_DIR, "results"),
                  user_func=user_func_filename,
                  ncore=4,
                  reduce_input=os.path.join(OUTPUT_DIR, "results/*/*"),
                  reduce_group_by=os.path.join(OUTPUT_DIR, "results/.*/(.*)"),
                  reduce_output=os.path.join(OUTPUT_DIR, "results.csv"))
    json.dump(config, open(os.path.join(OUTPUT_DIR, "config.json"), "w"))
    print "mapreduce.py -m --ncore 2 %s/config.json" % OUTPUT_DIR
