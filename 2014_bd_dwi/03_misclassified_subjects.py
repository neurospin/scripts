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
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "enettv_bd_dwi_trunc")
INPUT_CONFIG_PATH = os.path.join(INPUT_DIR, "config.json")

INPUT_PRED_FILE_NAME = "y_pred.npz"
INPUT_TRUE_FILE_NAME = "y_true.npz"

##########
# Output #
##########

OUTPUT_DIR = INPUT_DIR
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

# Load jobs
config = json.load(open(INPUT_CONFIG_PATH))
jobs = mapreduce._build_job_table(config)
#results_dir = os.path.join(INPUT_DIR, config["map_output"])
params_list = [tuple(p) for p in config["params"]]
resamples = config["resample"]

# Group by params
groups = jobs.groupby("params")

# Reload data and count misclassifications per subject
# We skip resample[0] (whole population)
missclassif = pd.DataFrame(index=subjects_id,
                           columns=params_list,
                           dtype='int')
for params, param_jobs in groups:
    for i, job in param_jobs.iterrows():
        if job["resample_index"] != 0:
            params = job.params
            resample_index = job.resample_index
            full_results_dir = os.path.join(INPUT_DIR,
                                            job["output dir"])
            y_pred_file = os.path.join(full_results_dir,
                                       INPUT_PRED_FILE_NAME)
            #print "Loading", y_pred_file
            y_pred = np.load(y_pred_file)['arr_0']
            y_true_file = os.path.join(full_results_dir,
                                       INPUT_TRUE_FILE_NAME)
            #print "Loading", y_true_file
            y_true = np.load(y_true_file)['arr_0']
            subjects = subjects_id[resamples[resample_index][1]]
            missclassif[params].loc[subjects] = (y_pred != y_true)

# We store it in the reverse order for easier visualization with localc
missclassif.T.to_csv(OUTPUT_MISSCLASSIF_FILE)
