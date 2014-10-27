# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:16:54 2014

@author: md238665

Add some columns to the results of the previous script.

This file was copied from scripts/2014_pca_struct/Olivetti_faces/02_aggregate_results.py.

"""

import os

import pandas as pd
import numpy as np

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/mescog/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "results")

INPUT_MODELS = ["pca", "sparse_pca", "struct_pca"]
INPUT_RESULT_FILE = os.path.join(INPUT_BASE_DIR,
                                 "results.csv")

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

##########
# Script #
##########

total_df = None

# Read results without index
df = pd.io.parsers.read_csv(INPUT_RESULT_FILE)
n, p = df.shape
print "Reading", INPUT_RESULT_FILE, "(", n, "lines)"

# Add a column for model, global penalization, l1 ratio and tv ratio
# from the key
# Model may contain '_' so I have to use this cryptic form
df['model'] = df['key'].map(lambda key: '_'.join(key.split('_')[:-3]))
df['global_pen'] = df['key'].map(lambda key: float(key.split('_')[-3]))
df['tv_ratio'] = df['key'].map(lambda key: float(key.split('_')[-2]))
df['l1_ratio'] = df['key'].map(lambda key: float(key.split('_')[-1]))

# Create multiindex (model, SNR, global_pen, tv_ratio, l1_ratio)
df.set_index(['model', 'global_pen', 'tv_ratio', 'l1_ratio'],
           inplace=True, drop=True)

# Save file
df.to_csv(OUTPUT_RESULTS_FILE)