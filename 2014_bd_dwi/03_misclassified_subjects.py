# -*- coding: utf-8 -*-
"""
Created on Mon Dec  15 08:36:00 2014

@author: md238665

Determine which subjects are misclassified and try to find links with
clinical data.
"""

import os
import json

import numpy as np
import pandas as pd

import mapreduce

#########
# Input #
#########

INPUT_BASE_DIR = "/neurospin/brainomics/2014_bd_dwi/"
INPUT_POPULATION = os.path.join(INPUT_BASE_DIR, "population.csv")

INPUT_DATASET_DIR_FORMAT = "enettv_bd_dwi_{dataset}"

INPUT_DATASETS = ["site",
                  "skel",
                  "trunc"]

INPUT_CONFIG_FILENAME = "config.json"
INPUT_PRED_FILENAME = "y_pred.npz"
INPUT_TRUE_FILENAME = "y_true.npz"

##########
# Output #
##########

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_MISSCLASSIF_FILE = os.path.join(OUTPUT_DIR,
                                       "missclassified.csv")

##########
# Params #
##########

#############
# Functions #
#############

##########
# Script #
##########

#, Load clinic
population = pd.io.parsers.read_csv(INPUT_POPULATION)
subjects_id = population.ID

missclassif = None
for dataset in INPUT_DATASETS:
    dataset_dir = INPUT_DATASET_DIR_FORMAT.format(dataset=dataset)
    dataset_full_dir = os.path.join(INPUT_BASE_DIR,
                                    dataset_dir)

    # Load config and create jobs
    config_full_filename = os.path.join(dataset_full_dir,
                                        INPUT_CONFIG_FILENAME)
    config = json.load(open(config_full_filename))
    jobs = mapreduce._build_job_table(config)
    params_list = [tuple(p) for p in config["params"]]
    resamples_list = config["resample"]

    # Create dataframe
    if missclassif is None:
        # Create index (dataset + params)
        # Assume that all datasets have the same parameters
        index_levels = [(dataset_, ) + params
                        for dataset_ in INPUT_DATASETS
                        for params in params_list]
        index = pd.MultiIndex.from_tuples(index_levels,
                                          names=['dataset',
                                                 'a', 'tv', 'l1', 'l2',
                                                 'k'])
        missclassif = pd.DataFrame(columns=subjects_id,
                                   index=index,
                                   dtype='bool')

    # Group by params
    groups = jobs.groupby("params")

    # Reload data and count misclassifications per subject
    # We skip resample[0] (whole population)
    for params, param_jobs in groups:
        for i, job in param_jobs.iterrows():
            param_res = pd.DataFrame(index=subjects_id,
                                     columns=['classif'],
                                     dtype='bool')
            if job["resample_index"] == 0:
                continue
                params = job.params
            loc = (dataset, ) + params
            resample_index = job.resample_index
            subjects = subjects_id[resamples_list[resample_index][1]]

            full_results_dir = os.path.join(dataset_full_dir,
                                            job["output dir"])

            y_pred_filename = os.path.join(full_results_dir,
                                           INPUT_PRED_FILENAME)
            print "Loading", y_pred_filename
            y_pred_file = np.load(y_pred_filename)
            y_pred = y_pred_file['arr_0']
            y_pred_file.close()

            y_true_filename = os.path.join(full_results_dir,
                                           INPUT_TRUE_FILENAME)
            print "Loading", y_true_filename
            y_true_file = np.load(y_true_filename)
            y_true = y_true_file['arr_0']
            y_true_file.close()

            param_res.loc[subjects] = (y_pred != y_true)
        missclassif.loc[loc] = param_res['classif']

# Store results
missclassif.to_csv(OUTPUT_MISSCLASSIF_FILE)
