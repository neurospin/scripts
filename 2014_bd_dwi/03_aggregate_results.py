# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 01:18:36 2014

@author: md238665

Aggregates results of previous scripts.

"""

import os

import pandas as pd
import numpy as np

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_bd_dwi"
INPUT_DATASET_DIR_FORMAT = "enettv_bd_dwi_{dataset}"

INPUT_DATASETS = ["site",
                  "skel",
                  "trunc"]
INPUT_RESULT_FILE = "results.csv"

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

##########
# Script #
##########

# Read input files in a large df
input_files = []
total_df = None
N = 0
for dataset in INPUT_DATASETS:
    subdir = INPUT_DATASET_DIR_FORMAT.format(dataset=dataset)
    full_filename = os.path.join(INPUT_BASE_DIR, subdir, INPUT_RESULT_FILE)
    input_files.append(full_filename)
    if os.path.exists(full_filename):
        # Read it without index
        df = pd.io.parsers.read_csv(full_filename)
        n, p = df.shape
        print "Reading", full_filename, "(", n, "lines)"
        N += n
        # Add a column for SNR
        df['dataset'] = pd.Series.from_array(np.asarray([dataset]*n),
                                         name='Dataset',
                                         index=pd.Index(np.arange(n)))
        # Append to large df
        if total_df is None:
            total_df = df
        else:
            total_df = total_df.append(df)
# Create multiindex (dataset, params, k, a, tv, l1, l2)
total_df.set_index(['dataset', 'params', 'k', 'a', 'tv', 'l1', 'l2'],
                   inplace=True, drop=True)
total_df.to_csv(OUTPUT_RESULTS_FILE)
