# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:16:54 2014

@author: md238665

Aggregates results of previous scripts.

"""

import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "data")
INPUT_DATASET_DIR_FORMAT = "data_{s[0]}_{s[1]}_{snr}"

INPUT_SHAPE = (100, 100, 1)
INPUT_SNRS = np.array([0.1, 0.5, 1.0])

INPUT_MODELS = ["pca", "sparse_pca", "struct_pca"]
INPUT_RESULT_FILE = "results.csv"

OUTPUT_DIR = INPUT_BASE_DIR
OUTPUT_RESULTS = os.path.join(OUTPUT_DIR, "consolidated_results.csv")

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
for model in INPUT_MODELS:
    model_dir = os.path.join(INPUT_BASE_DIR, model)
    for snr in INPUT_SNRS:
        subdir = INPUT_DATASET_DIR_FORMAT.format(s=INPUT_SHAPE,
                                                 snr=snr)
        full_filename = os.path.join(model_dir, subdir, INPUT_RESULT_FILE)
        input_files.append(full_filename)
        if os.path.exists(full_filename):
            # Read it without index
            df = pd.io.parsers.read_csv(full_filename)
            n, p = df.shape
            print "Reading", full_filename, "(", n, "lines)"
            N += n
            # Add a column for global penalization and tv ratio
            if model == 'pca':
                global_pen = pd.Series.from_array(np.zeros(n),
                                                  name='Global penalization',
                                                  index=pd.Index(np.arange(n)))
                tv_ratio = pd.Series.from_array(np.zeros(n),
                                                name='TV ratio',
                                                index=pd.Index(np.arange(n)))


            if model == 'sparse_pca':
                param = df['key'].map(float)
                global_pen = pd.Series.from_array(param,
                                                  name='Global penalization',
                                                  index=pd.Index(np.arange(n)))
                tv_ratio = pd.Series.from_array(np.zeros(n),
                                                name='TV ratio',
                                                index=pd.Index(np.arange(n)))


            if model == 'struct_pca':
                params = df['key'].apply(
                    lambda key: [float(s) for s in key.split('_')])
                global_pen = pd.Series.from_array(params.map(sum),
                        name='Global penalization',
                        index=pd.Index(np.arange(n)))
                tv_ratio = pd.Series.from_array(params.map(lambda p: p[2]/sum(p)),
                        name='TV ratio',
                        index=pd.Index(np.arange(n)))

            df['Global penalization'] = global_pen
            df['TV ratio'] = tv_ratio
            # Create multiindex (model, alpha, key) for this file
            snrs = pd.Series.from_array(np.asarray([snr]*n),
                                        name='SNR',
                                        index=pd.Index(np.arange(n)))
            models = pd.Series.from_array(np.asarray([model]*n),
                                          name='model',
                                          index=pd.Index(np.arange(n)))
            df.index = pd.MultiIndex.from_arrays([models, snrs, df['key']])
            # Append to large df
            if total_df is None:
                total_df = df
            else:
                total_df = total_df.append(df)
total_df.to_csv(OUTPUT_RESULTS)
