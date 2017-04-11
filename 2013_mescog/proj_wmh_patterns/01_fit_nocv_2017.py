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
import matplotlib.pylab as plt
import nilearn
from nilearn import plotting

from matplotlib.backends.backend_pdf import PdfPages


sys.path.append('/home/ed203246/git/scripts/2013_mescog/proj_wmh_patterns')
import pca_tv
import parsimony.functions.nesterov.tv


#from brainomics import plot_utilities
#import parsimony.utils.check_arrays as check_arrays

################
INPUT_MESCOG_DIR = "/neurospin/mescog/"

INPUT_POPULATION = os.path.join(INPUT_MESCOG_DIR, "proj_wmh_patterns",
                                     "population.csv")
INPUT_DATASET = os.path.join(INPUT_MESCOG_DIR, "proj_wmh_patterns",
                             "X_center.npy")

INPUT_MASK = os.path.join(INPUT_MESCOG_DIR, "proj_wmh_patterns",
                          "mask_bin.nii.gz")

INPUT_CLINIC = os.path.join(INPUT_MESCOG_DIR, "proj_predict_cog_decline/data/dataset_clinic_niglob_20140728.csv")


OUTPUT_DIR = os.path.join(INPUT_MESCOG_DIR, "proj_wmh_patterns", '{key}')

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

OUTPUT_BASE = INPUT_MESCOG_DIR



# Load data & mask
mask_ima = nibabel.load(INPUT_MASK)
mask_arr = mask_ima.get_data() != 0
mask_arr.sum()
mask_indices = np.where(mask_arr)
X = np.load(INPUT_DATASET)

assert X.shape == (301, 1064455)
assert mask_arr.sum() == X.shape[1]
assert np.allclose(X.mean(axis=0), 0)

#
pop = pd.read_csv(INPUT_POPULATION_FILE)

# Fit model
N_COMP = 5


# #############################################################################
# Fit PCAEnetTV
A = parsimony.functions.nesterov.tv.linear_operator_from_mask(mask_arr)

# Parameters settings 1

if False:
    # 'struct_pca_0.03_0.64_0.33'
    global_pen, tv_ratio = 1.0, 0.33,
    l1max = pca_tv.PCA_L1_L2_TV.l1_max(X) * .9
    # l1max = 0.025937425654559931
    l1_ratio  = l1max / (global_pen * (1 - tv_ratio))
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
    assert(np.allclose(ll1 + ll2 + ltv, global_pen))

# Parameters settings 2
#  1/3, 1/3 1/3 such that ll1 < l1max
alpha, l1_ratio, l2_ratio, tv_ratio = 0.01, 1/3, 1/3, 1/3
ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio

key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)
key = pca_enettv
OUTPUT_DIR.format(key=key)

if not(os.path.exists(OUTPUT_DIR.format(key=key))):
    os.makedirs(OUTPUT_DIR.format(key=key))

model = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                            l1=ll1, l2=ll2, ltv=ltv,
                            Atv=A,
                            criterion="frobenius",
                            eps=1e-6,
                            max_iter=100,
                            inner_max_iter=int(1e3),
                            verbose=True)

t0 = time.clock()
model.fit(X)
model.l1_max(X)
t1 = time.clock()
_time = t1 - t0
# 4688
"""
Outer iteration
Iteration 33
Iteration 64
Iteration 31
Iteration 16
Iteration 39

36.6 / PC

74633.05885 / (183 * 1000)
0.40783092267759563 s / iteration FISTA

20.73138888888889 h

Pour: (301, 1064455)

Time requested = n_components * 36.6 * inner_max_iter * 0.5

"""
# Save results
#model.U, model.d, model.V = m["U"], m["d"], m["V"]
PC, d = model.transform(X)

np.savez_compressed(os.path.join(OUTPUT_DIR.format(key=key), "model.npz"),
                    U=model.U, d=model.d, V=model.V, PC=PC)


m = np.load(os.path.join(OUTPUT_DIR.format(key=key), "model.npz"))
U, d, V, PC = m["U"], m["d"], m["V"], m["PC"]

fh = open(os.path.join(OUTPUT_DIR.format(key=key), "pca_enettv_info.txt"), "w")
fh.write("Time:" + str(_time) + "\n")
fh.write("max(|V|):" + str(np.abs(V).max(axis=0)) + "\n")
fh.write("mean(|V|):" + str(np.abs(V).mean(axis=0)) + "\n")
fh.write("sd(|V|):" + str(np.abs(V).std(axis=0)) + "\n")
fh.write("max(|U|):" + str(np.abs(U).max(axis=0)) + "\n")
fh.write("mean(|U|):" + str(np.abs(U).mean(axis=0)) + "\n")
fh.write("sd(|U|):" + str(np.abs(U).std(axis=0)) + "\n")
fh.close()

assert U.shape == (301, 5)
assert PC.shape == (301, 5)
assert V.shape == (1064455, 5)
assert d.shape == (5,)

# #############################################################################
# Fit Regular PCA
key_pca = "pca"
key = key_pca

OUTPUT_DIR.format(key=key)

if not(os.path.exists(OUTPUT_DIR.format(key=key))):
    os.makedirs(OUTPUT_DIR.format(key=key))

from sklearn.decomposition import PCA
pca = PCA(n_components=N_COMP)
pca.fit(X)
print(pca.explained_variance_ratio_)
# [ 0.20200558  0.03278055  0.0199719   0.01520858  0.01312169]
assert pca.components_.shape == (5, 1064455)
PC = pca.transform(X)
np.savez_compressed(os.path.join(OUTPUT_DIR.format(key=key), "model.npz"),
                    U=pca.transform(X), V=pca.components_.T, PC=PC)

# #############################################################################
# Plot map

keys = [key_pca_enettv, key_pca]

for key in keys:
    #key = key_pca_enettv
    #key = key_pca

    m = np.load(os.path.join(OUTPUT_DIR.format(key=key), "model.npz"))
    U, V, PC = m["U"], m["V"], m["PC"]
    assert U.shape == (301, 5)
    assert V.shape == (1064455, 5)
    assert PC.shape == (301, 5)

    # multiply second PC by -1 to stick with original article
    V[:, 1] *= -1
    PC[:, 1] *= -1

    np.abs(V).mean(axis=0)

    PCs = pd.DataFrame(U, columns=['PC%i' % (i+1) for i in range(U.shape[1])])
    clinic_pc = pd.concat([pop, PCs], axis=1)
    clinic_pc.to_csv(os.path.join(OUTPUT_DIR.format(key=key), "clinic_pc.csv"))

    pdf = PdfPages(os.path.join(OUTPUT_DIR.format(key=key), "maps.pdf"))

    for pc in range(V.shape[1]):
        #pc = 1
        arr = np.zeros(mask_arr.shape)
        arr[mask_arr] = V[:, pc].ravel()
        out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
        filename = os.path.join(OUTPUT_DIR.format(key=key), "V%i.nii.gz" % (pc+1))
        out_im.to_filename(filename)
        nilearn.plotting.plot_glass_brain(filename, colorbar=True, plot_abs=False, title="PC%i"% (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='y', cut_coords=7, title="PC%i" % (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='x', cut_coords=6, title="PC%i"% (pc+1))
        pdf.savefig(); plt.close()

    pdf.close()

# #############################################################################
# Plot map
#                                      #threshold = t,
#                                      vmax=abs(vmax), vmin =-abs(vmax))
"""
cd /neurospin/mescog/proj_wmh_patterns/struct_pca_0.003_0.003_0.003/
image_clusters_analysis_nilearn.py pca_enettv_V1.nii.gz --thresh_norm_ratio 0.99
image_clusters_analysis_nilearn.py pca_enettv_V2.nii.gz --thresh_norm_ratio 0.99
image_clusters_analysis_nilearn.py pca_enettv_V3.nii.gz --thresh_norm_ratio 0.99
image_clusters_analysis_nilearn.py pca_enettv_V4.nii.gz --thresh_norm_ratio 0.99
"""