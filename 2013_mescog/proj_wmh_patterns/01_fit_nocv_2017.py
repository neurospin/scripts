#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:23:25 2017

@author: ed203246


rsync -avuhn --delete /neurospin/mescog/proj_wmh_patterns/PCs /media/ed203246/usbed/neurospin/mescog/proj_wmh_patterns/
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
from brainomics import array_utils


#from brainomics import plot_utilities
#import parsimony.utils.check_arrays as check_arrays

################
INPUT_DIR = "/neurospin/mescog/proj_wmh_patterns"

INPUT_POP = os.path.join(INPUT_DIR, "population.csv")
#INPUT_IMPLICIT_MASK = os.path.join(INPUT_DIR, "mask_implicit.nii.gz")
#INPUT_ATLAS_MASK = os.path.join(INPUT_DIR, "mask_atlas.nii.gz")
INPUT_BIN_MASK = os.path.join(INPUT_DIR, "mask_bin.nii.gz")
INPUT_A_BIN_MASK = os.path.join(INPUT_DIR, "mask_bin_A.npz")
#INPUT_X = os.path.join(INPUT_DIR, "X.npy")
INPUT_CENTERED_X = os.path.join(INPUT_DIR, "X_center.npy")
#INPUT_MEANS = os.path.join(INPUT_DIR, "means.npy")


OUTPUT_DIR = os.path.join(INPUT_DIR, "PCs", '{key}')



# Load data & mask
mask_ima = nibabel.load(INPUT_BIN_MASK)
mask_arr = mask_ima.get_data() != 0
mask_arr.sum()
mask_indices = np.where(mask_arr)
X = np.load(INPUT_CENTERED_X)

assert X.shape == (301, 1064455)
assert mask_arr.sum() == X.shape[1]
assert np.allclose(X.mean(axis=0), 0)

#
pop = pd.read_csv(INPUT_POP)

# Fit model
N_COMP = 10


# #############################################################################
# Fit PCAEnetTV
from parsimony.utils.linalgs import LinearOperatorNesterov
A = LinearOperatorNesterov(filename=INPUT_A_BIN_MASK)


inner_max_iter = int(1e3)

if False:  # Parameters settings 1:  too much penalization
    # 'struct_pca_0.03_0.64_0.33'
    global_pen, tv_ratio = 1.0, 0.33
    l1max = pca_tv.PCA_L1_L2_TV.l1_max(X) * .9
    # l1max = 0.025937425654559931
    l1_ratio  = l1max / (global_pen * (1 - tv_ratio))
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
    assert(np.allclose(ll1 + ll2 + ltv, global_pen))


if False:  # Parameters settings 2: trop de l1 trop de TV pas assez de l2
    #  1/3, 1/3 1/3 such that ll1 < l1max
    alpha, l1_ratio, l2_ratio, tv_ratio = 0.01, 1/3, 1/3, 1/3
    ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio
    key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)

if False:  # Parameters settings 3:  too much penalization
    #  1/3, 1/3 1/3 such that ll1 < l1max
    alpha, l1_ratio, l2_ratio, tv_ratio = 1., 0.025937425654559931, 1/3, 1/3
    ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio
    key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)

if False:  # Parameters settings 3: trop de TV
    #  1/3, 1/3 1/3 such that ll1 < l1max
    alpha, l1_ratio, l2_ratio, tv_ratio = 1., 0.1 * 0.025937425654559931, 1/3, 1/3
    ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio
    key_pca_enettv_01l1max = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)
    key_pca_enettv = key_pca_enettv_01l1max

if False:  # Parameters settings 4:
    # ll1 < 0.1 * l1max,  tv = 0.01 * 1/3
    ll1, ll2, ltv = 0.1 * 0.025937425654559931, 1/3, 0.01 * 1/3
    key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)
    # Corr with old PC[-0.99920208144096823, 0.88560562252109543, -0.65506757554132422]

#
if False:  # Parameters settings 4 (max_inner_ite=100):
    # ll1 < 0.1 * l1max,  tv = 0.01 * 1/3
    ll1, ll2, ltv =  0.1 * 0.025937425654559931, 1/3, 0.01 * 1/3
    key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f_inner_max_iter100" % (ll1, ll2, ltv)
    inner_max_iter = int(1e2)

if False:  # Parameters settings 5: GOOD sufficient TV
    # ll1 < 0.01 * l1max,  tv = 0.01 * 1/3
    ll1, ll2, ltv = 0.01 * 0.025937425654559931, 1/3, 0.01 * 1/3
    key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)
    # pca_enettv_0.0003_0.333_0.003
    #Corr with old PC[-0.99984945338417142, 0.99586133438324298, 0.90527476131971074]
    # Explained variance:[ 0.19908058  0.22867141  0.24468919  0.25686902  0.26628877]

if False:  # Parameters settings 6: NO: too much TV too much l1
    # ll1 < 0.01 * l1max,  tv = 0.01 * 1/3
    ll1, ll2, ltv = 0.1 * 0.025937425654559931, 1, 0.1
    key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)
    # Corr with old PC[-0.9796910479344958, -0.1058878547057002, 0.018105886571454705]

if False:  # Parameters settings 7: Almost but too much TV
    # ll1 < 0.01 * l1max,  tv = 0.01 * 1/3
    ll1, ll2, ltv = 0.01 * 0.025937425654559931, 1, 0.01
    key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)
    # Corr with old PC[-0.99966211718252285, -0.99004655401439967, -0.74332811780676245]

# Parameters settings 8: take 5 reduce a little bit tv, GOOD but no less TV
ll1, ll2, ltv = 0.01 * 0.025937425654559931, 1, 0.001
key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)
# pending: pca_enettv_0.0003_1.000_0.001
# Corr with old PC[0.99982329115577584, -0.99440441127687851, 0.93785013498951075]
# Explained variance:[ 0.20122213  0.23290909  0.25154813  0.26544507  0.27724102]

# Parameters settings 9: take 5 reduce a little bit tv, increase a little l1: GOOD BINGO !!!
ll1, ll2, ltv = 0.05 * 0.025937425654559931, 1, 0.001
key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)
CHOICE = key_pca_enettv
# pending: for 10 PCs
#Corr with old PC[0.99999222883809447, 0.9994293857297728, -0.99247826586372279]
#Explained variance:[ 0.19876024  0.22844359  0.24310107  0.25474415  0.26392774]

# Parameters settings 10: take 9 increase a little tv (~5 with more l1)
# pending
ll1, ll2, ltv = 0.05 * 0.025937425654559931, 1, 0.003
key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)

## key_pca_enettv = CHOICE
key = key_pca_enettv
print(OUTPUT_DIR.format(key=key))

if not(os.path.exists(OUTPUT_DIR.format(key=key))):
    os.makedirs(OUTPUT_DIR.format(key=key))

model = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                            l1=ll1, l2=ll2, ltv=ltv,
                            Atv=A,
                            criterion="frobenius",
                            eps=1e-6,
                            max_iter=100,
                            inner_max_iter=inner_max_iter,
                            verbose=True)

t0 = time.clock()
model.fit(X)
model.l1_max(X)
t1 = time.clock()
_time = t1 - t0
print("Time TOT(s)",_time)


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

assert U.shape == (301, N_COMP)
assert PC.shape == (301, N_COMP)
assert V.shape == (1064455, N_COMP)
assert d.shape == (N_COMP,)

# #############################################################################
# Fit Regular PCA
if False:
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

# for pca
# array([ 0.20200558,  0.23478613,  0.25475803,  0.26996664,  0.2830884 ,  0.        ])

# #############################################################################
# Plot map + QC with old results
olddata_filename = '/neurospin/mescog/proj_wmh_patterns/proj_wmh_patterns_laptop-data-mescog/2017/pc_clinic_associations_20170124.xlsx'
olddata = pd.read_excel(olddata_filename, sheetname="data")
cols = [x for x in olddata.columns if # Of Interest
    (x.count('tvl1l2') and not x.count('tvl1l2_smalll1'))] + ['ID']
assert olddata.shape == (372, 69)
olddata = olddata[cols]
olddata['Subject ID'] =  [int(x.split("_")[1]) for x in olddata.ID]


keys = [key_pca_enettv]#, key_pca]

for key in keys:
    # key = key_pca_enettv
    # key = key_pca
    print(key)
    m = np.load(os.path.join(OUTPUT_DIR.format(key=key), "model.npz"))
    U, V, PC = m["U"], m["V"], m["PC"]
    assert U.shape == (301, N_COMP)
    assert V.shape == (1064455, N_COMP)
    assert PC.shape == (301, N_COMP)

    # multiply second PC by -1 to stick with original article
    #V[:, 1] *= -1
    #PC[:, 1] *= -1

    #sgn = np.sign(np.mean(V, axis=0))
    #V *= sgn
    #U *= sgn
    print(np.abs(V).mean(axis=0))

    print("# Merge with clinic ##############################################")

    PCs = pd.DataFrame(PC, columns=['PC%i' % (i+1) for i in range(U.shape[1])])
    clinic_pc = pd.concat([pop, PCs], axis=1)
    #clinic_pc.to_csv(os.path.join(OUTPUT_DIR.format(key=key), "clinic_pc.csv"))

    print("# Correlation with old coponents #################################")

    mrg = pd.merge(clinic_pc, olddata, left_on = 'Subject ID', right_on='Subject ID')
    assert mrg.shape[0] == 301
    cors = [np.corrcoef(mrg['PC%i'%i], mrg['pc%i__tvl1l2' % i])[0, 1] for i in range(1, 4)]

    # Change sign to align with old results
    cors = cors + [1] * (U.shape[1] - 3)
    U *= np.sign(cors)
    V *= np.sign(cors)
    for i in range(U.shape[1]):
        clinic_pc['PC%i' % (i+1)] *= np.sign(cors)[i]
    clinic_pc.to_csv(os.path.join(OUTPUT_DIR.format(key=key), "clinic_pc.csv"))


    fh = open(os.path.join(OUTPUT_DIR.format(key=key), "pca_enettv_info.txt"), "a")
    mrg = pd.merge(clinic_pc, olddata, left_on = 'Subject ID', right_on='Subject ID')
    cors = [np.corrcoef(mrg['PC%i'%i], mrg['pc%i__tvl1l2' % i])[0, 1] for i in range(1, 4)]
    fh.write("Corr with old PC" + str(cors) + "\n")
    fh.close()
    print("Corr with old PC" + str(cors))


    print("# Plots & build nii files #########################################")

    pdf = PdfPages(os.path.join(OUTPUT_DIR.format(key=key), "maps.pdf"))

    for pc in range(V.shape[1]):
        #pc = 1
        #pc = 0
        arr = np.zeros(mask_arr.shape)
        arr[mask_arr] = V[:, pc].ravel()
        out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
        filename = os.path.join(OUTPUT_DIR.format(key=key), "V%i.nii.gz" % (pc+1))
        out_im.to_filename(filename)
        nilearn.plotting.plot_glass_brain(filename, colorbar=True, plot_abs=False, title="PC%i"% (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1))
        #plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1), vmax=0.0001)
        #plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1), black_bg=True)
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='y', cut_coords=7, title="PC%i" % (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='x', cut_coords=6, title="PC%i"% (pc+1))
        pdf.savefig(); plt.close()

    pdf.close()

    print("# Plots & build nii files with thresholding at 99% of l2 norm ####")

    pdf = PdfPages(os.path.join(OUTPUT_DIR.format(key=key), "maps_thresh.pdf"))

    for pc in range(V.shape[1]):
        #pc = 1
        #pc = 0
        arr = np.zeros(mask_arr.shape)
        map_arr, thres = array_utils.arr_threshold_from_norm2_ratio(V[:, pc].ravel(), ratio=.95)
        arr[mask_arr] = map_arr.ravel()
        out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
        filename = os.path.join(OUTPUT_DIR.format(key=key), "V%i_thresh.nii.gz" % (pc+1))
        out_im.to_filename(filename)
        nilearn.plotting.plot_glass_brain(filename, colorbar=True, plot_abs=False, title="PC%i"% (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1))
        #plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1), vmax=0.0001)
        #plotting.plot_stat_map(filename, display_mode='z', cut_coords=7, title="PC%i" % (pc+1), black_bg=True)
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='y', cut_coords=7, title="PC%i" % (pc+1))
        pdf.savefig(); plt.close()
        plotting.plot_stat_map(filename, display_mode='x', cut_coords=6, title="PC%i"% (pc+1))
        pdf.savefig(); plt.close()

    pdf.close()



    print("# explained variance #############################################")
    #fh = open(os.path.join(OUTPUT_DIR.format(key=key), "pca_enettv_info.txt"), "a")
    mod = pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                                l1=ll1, l2=ll2, ltv=ltv,
                                Atv=A,
                                criterion="frobenius",
                                eps=1e-6,
                                max_iter=100,
                                inner_max_iter=1000,
                                verbose=True)

    #m = np.load(os.path.join(OUTPUT_DIR.format(key=key_pca_enettv), "model.npz"))
    mod.U, mod.V, mod.d =  m["U"], m["V"], m["d"]

    rsquared = np.zeros((N_COMP))
    for j in range(N_COMP):
        mod.n_components = j + 1
        X_predict = mod.predict(X)
        sse = np.sum((X - X_predict) ** 2)
        ssX = np.sum(X ** 2)
        rsquared[j] = 1 - sse / ssX

    fh = open(os.path.join(OUTPUT_DIR.format(key=key), "pca_enettv_info.txt"), "a")
    fh.write("Explained variance:"+str(rsquared) + "\n")
    fh.close()
    print("Explained variance:"+str(rsquared))

"""
run_triscotte_nohup.sh "python /home/ed203246/git/scripts/2013_mescog/proj_wmh_patterns/01_fit_nocv_2017.py"

cd /neurospin/mescog/proj_wmh_patterns/struct_pca_0.003_0.003_0.003/
image_clusters_analysis_nilearn.py pca_enettv_V1.nii.gz --thresh_norm_ratio 0.99
image_clusters_analysis_nilearn.py pca_enettv_V2.nii.gz --thresh_norm_ratio 0.99
image_clusters_analysis_nilearn.py pca_enettv_V3.nii.gz --thresh_norm_ratio 0.99
image_clusters_analysis_nilearn.py pca_enettv_V4.nii.gz --thresh_norm_ratio 0.99


pca_enettv_0.0003_0.333_0.003
r2 = np.array([ 0.19908058,  0.22867141,  0.24468919,  0.25686902,  0.26628877])
r2_bycomp = np.array([r2[0]] + [r2[i] - r2[i-1] for i in range(1 ,len(r2))])
# array([ 0.19908058,  0.02959083,  0.01601778,  0.01217983,  0.00941975])
r2_bycomp / r2
# array([ 1.        ,  0.12940328,  0.06546174,  0.0474165 ,  0.03537419])


r2 = np.array([ 0.19878513,  0.22846844,  0.24232428,  0.25432934,  0.2625482,   0.27135235,  0.2763483,  0.28049931,  0.28540894,  0.28823914])
r2_bycomp = np.array([r2[0]] + [r2[i] - r2[i-1] for i in range(1 ,len(r2))])
array([ 0.19878513,  0.02968331,  0.01385584,  0.01200506,  0.00821886, 0.00880415,  0.00499595,  0.00415101,  0.00490963,  0.0028302 ])
"""