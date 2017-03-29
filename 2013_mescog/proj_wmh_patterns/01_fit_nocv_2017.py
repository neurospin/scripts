#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:23:25 2017

@author: ed203246
"""
import sys
import os
import time

import numpy as np
import nibabel
import pandas as pd


sys.path.append('/home/ed203246/git/brainomics-team/2014_pca_tv/draft_code')
import pca_tv
import parsimony.functions.nesterov.tv


#from brainomics import plot_utilities
#import parsimony.utils.check_arrays as check_arrays

################
# Input/Output #
################

#INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/mescog/mescog_5folds"
#INPUT_DIR = os.path.join(INPUT_BASE_DIR,
#                         "results")
#INPUT_RESULTS_FILE = os.path.join(INPUT_BASE_DIR, "results.csv")



INPUT_MESCOG_DIR = "/neurospin/mescog/proj_wmh_patterns"

INPUT_POPULATION_FILE = os.path.join(INPUT_MESCOG_DIR,
                                     "population.csv")
INPUT_DATASET = os.path.join(INPUT_MESCOG_DIR,
                             "X_center.npy")

INPUT_MASK = os.path.join(INPUT_MESCOG_DIR,
                          "mask_bin.nii.gz")

OUTPUT = INPUT_MESCOG_DIR


# Load data & mask
mask_ima = nib.load(INPUT_MASK)
mask_arr = mask_ima.get_data() != 0
mask_arr.sum()
mask_indices = np.where(mask_arr)
X = np.load(INPUT_DATASET)

assert X.shape == (301, 1064455)
assert mask_arr.sum() == X.shape[1]

#
pop = pd.read_csv(INPUT_POPULATION_FILE)

# Fit model
N_COMP = 5

A = parsimony.functions.nesterov.tv.linear_operator_from_mask(mask_arr)

# Parameters settings

# 'struct_pca_0.03_0.64_0.33'
global_pen, tv_ratio = 1.0, 0.33,
l1max = pca_tv.PCA_L1_L2_TV.l1_max(X) * .9
# l1max = 0.025937425654559931
l1_ratio  = l1max / (global_pen * (1 - tv_ratio))
ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))

#  1/3, 1/3 1/3 such that ll1 < l1max
alpha, l1_ratio, l2_ratio, tv_ratio = 0.01, 1/3, 1/3, 1/3
ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio

key = "struct_pca_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)
OUTPUT_DIR = os.path.join(OUTPUT, key)
if not(os.path.exists(OUTPUT_DIR)):
    os.makedirs(OUTPUT_DIR)

model = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                            l1=ll1, l2=ll2, ltv=ltv,
                            Atv=A,
                            criterion="frobenius",
                            eps=1e-6,
                            max_iter=100,
                            inner_max_iter=int(1e4),
                            output=False)

t0 = time.clock()
model.fit(X)
model.l1_max(X)
t1 = time.clock()
_time = t1 - t0
# 4688

# Save results
np.savez_compressed(os.path.join(OUTPUT_DIR, "pca_enettv.npz"),
                    U=model.U, d=model.d, V=model.V)

m = np.load(os.path.join(OUTPUT_DIR, "pca_enettv.npz"))
U, d, V = m["U"], m["d"], m["V"]

fh = open(os.path.join(OUTPUT_DIR, "pca_enettv_info.txt"), "w")
fh.write("Time:" + str(_time) + "\n")
fh.write("max(|V|):" + str(np.abs(V).max(axis=0)) + "\n")
fh.write("mean(|V|):" + str(np.abs(V).mean(axis=0)) + "\n")
fh.write("sd(|V|):" + str(np.abs(V).std(axis=0)) + "\n")
fh.write("max(|U|):" + str(np.abs(U).max(axis=0)) + "\n")
fh.write("mean(|U|):" + str(np.abs(U).mean(axis=0)) + "\n")
fh.write("sd(|U|):" + str(np.abs(U).std(axis=0)) + "\n")
fh.close()

assert U.shape == (301, 5)
assert V.shape == (1064455, 5)
assert d.shape == (5,)

for pc in range(V.shape[1]):
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = V[:, pc].ravel()
    out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
    filename = os.path.join(OUTPUT_DIR, "pca_enettv_V%i.nii.gz" % pc)
    out_im.to_filename(filename)

import nilearn
from nilearn import plotting

filename = '/neurospin/mescog/proj_wmh_patterns/struct_pca_0.03_0.64_0.33/pca_enettv_V4.nii.gz'
nilearn.plotting.plot_glass_brain(filename,
                                      colorbar=True, plot_abs=False)#,
#                                      #threshold = t,
#                                      vmax=abs(vmax), vmin =-abs(vmax))