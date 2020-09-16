#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:06:56 2020

@author: edouard.duchesnay@cea.fr

TODO: explore coef map

Fusar-Poli, P., S. Borgwardt, A. Crescini, G. Deste, Matthew J. Kempton, S. Lawrie, P. Mc Guire, and E. Sacchetti. “Neuroanatomy of Vulnerability to Psychosis: A Voxel-Based Meta-Analysis.” Neuroscience and Biobehavioral Reviews 35, no. 5 (April 2011): 1175–85. https://doi.org/10.1016/j.neubiorev.2010.12.005.
GM reductions in the frontal and temporal cortex associated with transition to psychosis (HR-NT > HR-T)


# Copy data

cd /home/ed203246/data/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition/*participants*.csv ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition*t1mri_mwp1_mask.nii.gz ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition/*mwp1_gs-raw_data64.npy ./

# NS => Laptop
rsync -azvun triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition/* /home/ed203246/data/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition/

"""
# %load_ext autoreload
# %autoreload 2

import os
import time
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_preprocessing as preproc
#from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
import matplotlib
# matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
import re
# import glob
import seaborn as sns
import copy
import pickle

from nitk.utils import arr_threshold_from_norm2_ratio
from nitk.image import img_to_array, global_scaling, compute_brain_mask
from nitk.stats import Residualizer
from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif

###############################################################################
#
# %% 1) Config
#
###############################################################################

INPUT_PATH = '/neurospin/psy/start-icaar-eugei/'
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/202009_start-icaar_cat12vbm_predict-transition'
NJOBS = 8
os.chdir(OUTPUT_PATH)

# On laptop
if not os.path.exists(INPUT_PATH):
    INPUT_PATH = INPUT_PATH.replace('/neurospin', '/home/ed203246/data')
    OUTPUT_PATH = OUTPUT_PATH.replace('/neurospin', '/home/ed203246/data' )
    NJOBS = 2


def PATH(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None,
    type=None, ext=None, basepath=""):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32

    return os.path.join(basepath, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) +
                 ("" if ext is None else "." + ext))

def INPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=INPUT_PATH)

def OUTPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=OUTPUT_PATH)

dataset, TARGET, TARGET_NUM = 'icaar-start', "transition", "transition_num" #, "diagnosis", "diagnosis_num"
scaling, harmo = 'gs', 'raw'
DATASET_TRAIN = dataset
VAR_CLINIC  = []
VAR_DEMO = ['age', 'sex']
NSPLITS = 5


###############################################################################
#
# %% 2) Utils
#
###############################################################################

###############################################################################
# %% 2.1) Mapper

import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

#from nitk.utils import dict_product, parallel, aggregate_cv
#from nitk.utils import dict_product, parallel, reduce_cv_classif
from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif
from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio

def fit_predict(key, estimator_img, residualize, split):
    estimator_img = copy.deepcopy(estimator_img)
    train, test = split
    Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Z_train, Z_test, y_train =\
        Xim[train, :], Xim[test, :], Xdemoclin[train, :], Xdemoclin[test, :], Z[train, :], Z[test, :], y[train]

    # Images based predictor

    # Residualization
    if residualize == 'yes':
        residualizer.fit(Xim_train, Z_train)
        Xim_train = residualizer.transform(Xim_train, Z_train)
        Xim_test = residualizer.transform(Xim_test, Z_test)

    elif residualize == 'biased':
        residualizer.fit(Xim, Z)
        Xim_train = residualizer.transform(Xim_train, Z_train)
        Xim_test = residualizer.transform(Xim_test, Z_test)

    scaler = StandardScaler()
    Xim_train = scaler.fit_transform(Xim_train)
    Xim_test = scaler.transform(Xim_test)

    try: # if coeficient can be retrieved given the key
        estimator_img.coef_ = KEY_VALS[key]['coef_img']
    except: # if not fit
        estimator_img.fit(Xim_train, y_train)

    y_test_img = estimator_img.predict(Xim_test)
    score_test_img = estimator_img.decision_function(Xim_test)
    score_train_img = estimator_img.decision_function(Xim_train)

    # Demographic/clinic based predictor
    estimator_democlin = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
    scaler = StandardScaler()
    Xdemoclin_train = scaler.fit_transform(Xdemoclin_train)
    Xdemoclin_test = scaler.transform(Xdemoclin_test)
    estimator_democlin.fit(Xdemoclin_train, y_train)
    y_test_democlin = estimator_democlin.predict(Xdemoclin_test)
    score_test_democlin = estimator_democlin.decision_function(Xdemoclin_test)
    score_train_democlin = estimator_democlin.decision_function(Xdemoclin_train)

    # STACK DEMO + IMG
    estimator_stck = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=True)
    # SVC
    # from sklearn.svm import SVC
    # estimator_stck = SVC(kernel='rbf', probability=True, gamma=1 / 100)
    # GB
    # from sklearn.ensemble import GradientBoostingClassifier
    # estimator_stck = GradientBoostingClassifier()

    Xstck_train = np.c_[score_train_democlin, score_train_img]
    Xstck_test = np.c_[score_test_democlin, score_test_img]
    scaler = StandardScaler()
    Xstck_train = scaler.fit_transform(Xstck_train)
    Xstck_test = scaler.transform(Xstck_test)
    estimator_stck.fit(Xstck_train, y_train)

    y_test_stck = estimator_stck.predict(Xstck_test)
    score_test_stck = estimator_stck.predict_log_proba(Xstck_test)[:, 1]
    score_train_stck = estimator_stck.predict_log_proba(Xstck_train)[:, 1]

    return dict(y_test_img=y_test_img, score_test_img=score_test_img,
                y_test_democlin=y_test_democlin, score_test_democlin=score_test_democlin,
                y_test_stck=y_test_stck, score_test_stck=score_test_stck,
                coef_img=estimator_img.coef_)

###############################################################################
# %% 2.2) Plot

def plot_coefmap_stats(coef_map, coef_maps, ref_img, vmax, prefix):

    arr_threshold_from_norm2_ratio(coef_map, .99)[0]
    coef_maps_t = np.vstack([arr_threshold_from_norm2_ratio(coef_maps[i, :], .99)[0] for i in range(coef_maps.shape[0])])

    w_selectrate = np.sum(coef_maps_t != 0, axis=0) / coef_maps_t.shape[0]
    w_zscore = np.nan_to_num(np.mean(coef_maps, axis=0) / np.std(coef_maps, axis=0))
    w_mean = np.mean(coef_maps, axis=0)
    w_std = np.std(coef_maps, axis=0)

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = coef_map
    coefmap_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_img.to_filename(prefix + "coefmap.nii.gz")

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = w_mean
    coefmap_cvmean_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_cvmean_img.to_filename(prefix + "coefmap_mean.nii.gz")

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = w_std
    coefmap_cvstd_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_cvstd_img.to_filename(prefix + "coefmap_std.nii.gz")

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = w_zscore
    coefmap_cvzscore_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_cvzscore_img.to_filename(prefix + "coefmap_zscore.nii.gz")

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = w_selectrate
    coefmap_cvselectrate_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_cvselectrate_img.to_filename(prefix + "coefmap_selectrate.nii.gz")

    # Plot
    pdf = PdfPages(prefix + "coefmap.pdf")

    fig = plt.figure(figsize=(11.69, 3 * 8.27))

    ax = fig.add_subplot(511)
    plotting.plot_glass_brain(coefmap_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, vmax=vmax, figure=fig, axes=ax, title="Coef")

    ax = fig.add_subplot(512)
    plotting.plot_glass_brain(coefmap_cvmean_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, vmax=vmax, figure=fig, axes=ax, title="Mean")

    ax = fig.add_subplot(513)
    plotting.plot_glass_brain(coefmap_cvstd_img, threshold=1e-6, colorbar=True, figure=fig, axes=ax, title="Std")

    ax = fig.add_subplot(514)
    plotting.plot_glass_brain(coefmap_cvzscore_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, figure=fig, axes=ax, title="Zscore")

    ax = fig.add_subplot(515)
    plotting.plot_glass_brain(coefmap_cvselectrate_img, threshold=1e-6, colorbar=True, figure=fig, axes=ax, title="Select. Rate")

    pdf.savefig(); plt.close(fig); pdf.close()

    maps = {"coefmap": coefmap_img, "coefmap_mean": coefmap_cvmean_img,
            "coefmap_cvstd": coefmap_cvstd_img, "coefmap_cvzscore": coefmap_cvzscore_img,
            "coefmap_cvselectrate": coefmap_cvselectrate_img}

    return maps

###############################################################################
#
# %% 3.1) Build dataset from images
#
###############################################################################

# Create dataset if needed
if not os.path.exists(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy")):

    # Clinic filename (Update 2020/06)
    clinic_filename = os.path.join(INPUT_PATH, "phenotype/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_202006.tsv")
    pop = pd.read_csv(clinic_filename, sep="\t")
    pop = pop[pop.study == 'ICAAR_EUGEI_START']

    ################################################################################
    # Images
    ni_icaar_filenames = glob.glob(os.path.join(INPUT_PATH, "derivatives/cat12/vbm/sub-*/ses-V1/mri/mwp1*.nii"))
    tivo_icaar = pd.read_csv(os.path.join(INPUT_PATH, 'derivatives/cat12/stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    assert tivo_icaar.shape == (171, 6)
    assert len(ni_icaar_filenames) == 171

    imgs_arr, pop_ni, target_img = img_to_array(ni_icaar_filenames)

    # Merge image with clinic and global volumes
    keep = pop_ni["participant_id"].isin(pop["participant_id"])
    assert np.sum(keep) == 170
    imgs_arr =  imgs_arr[keep]
    pop = pd.merge(pop_ni[keep], pop, on="participant_id", how= 'inner') # preserve the order of the left keys.
    pop = pd.merge(pop, tivo_icaar, on="participant_id", how= 'inner')

    # Global scaling
    imgs_arr = global_scaling(imgs_arr, axis0_values=np.array(pop.tiv), target=1500)

    # Compute mask
    #mask_img = preproc.compute_brain_mask(imgs_arr, target_img, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10,
    #                           verbose=1)
    mask_img = compute_brain_mask(imgs_arr, target_img, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10,
                               verbose=1)
    mask_arr = mask_img.get_fdata() > 0
    mask_img.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    pop.to_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"), index=False)
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), imgs_arr)

    del imgs_arr, target_img, mask_img

###############################################################################
#
# %% 3.2) Load dataset, select subjects
#
###############################################################################

pop = pd.read_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
imgs_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"))
mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_fdata() != 0
assert mask_arr.sum() == 369547

# Update 2020/06: include Non-UHR-NC as non transitors
pop[TARGET] = pop["diagnosis"].map({'UHR-C': "1", 'UHR-NC': "0", 'Non-UHR-NC':"0"}).values
pop[TARGET_NUM] = pop["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0, 'Non-UHR-NC':0}).values
pop["GM_frac"] = pop.gm / pop.tiv
pop["sex_c"] = pop["sex"].map({0: "M", 1: "F"})


###############################################################################
# %% 3.2.1) Select participants

# Update 2020/06: include Non-UHR-NC as non transitors
msk = pop["diagnosis"].isin(['UHR-C', 'UHR-NC', 'Non-UHR-NC']) &  pop["irm"].isin(['M0']) # UPDATE 2020/06
assert msk.sum() == 85

Xim = imgs_arr.squeeze()[:, mask_arr][msk]
del imgs_arr
y = pop[TARGET_NUM][msk].values

Xclin = pop.loc[msk, VAR_CLINIC].values
Xdemo = pop.loc[msk, VAR_DEMO].values
Xsite = pd.get_dummies(pop.site[msk]).values
Xdemoclin = np.concatenate([Xdemo, Xclin], axis=1)
formula_res, formula_full = "site + age + sex", "site + age + sex + " + TARGET_NUM
residualizer = Residualizer(data=pop[msk], formula_res=formula_res, formula_full=formula_full)
Z = residualizer.get_design_mat()

###############################################################################
#
# %% Explore data
#
###############################################################################

if False:
    # Explore data
    tab = pd.crosstab(pop["sex_c"][msk], pop[TARGET][msk], rownames=["sex_c"],
        colnames=[TARGET])
    print(pop[TARGET][msk].describe())
    print(pop[TARGET][msk].isin(['UHR-C']).sum())
    print(tab)
    """
    count     85
    unique     2
    top        0
    freq      58
    Name: transition, dtype: object
    0
    transition   0   1
    sex_c
    F           24   8
    M           34  19
    """

    # Some plot
    df_ = pop[msk]
    fig = plt.figure()
    sns.lmplot("age", "GM_frac", hue=TARGET, data=df_)
    fig.suptitle("Aging is faster in Convertors")

    fig = plt.figure()
    sns.lmplot("age", "GM_frac", hue=TARGET, data=df_[df_.age <= 24])
    fig.suptitle("Aging is faster in Convertors in <= 24 years")
    del df_

    ###########################################################################
    # PCA

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    pca_im = PCA(n_components=None)
    PC_im_ = pca_im.fit_transform(scaler.fit_transform(Xim))
    print("EV", pca_im.explained_variance_ratio_[:10])
    """
    EV [0.03318841 0.02634463 0.02491262 0.01924089 0.01895185 0.01720236
     0.01656014 0.01618993 0.01528729 0.01517617]
    """

    fig = plt.figure()
    sns.scatterplot(pop["GM_frac"][msk], PC_im_[:, 0], hue=pop[TARGET][msk])
    fig.suptitle("PC1 capture global GM atrophy")

    # sns.scatterplot(pop["GM_frac"][msk_tgt], PC_tgt_[:, 1], hue=pop[TARGET][msk_tgt])

    fig = plt.figure()
    sns.scatterplot(PC_im_[:, 0], PC_im_[:, 1], hue=pop[TARGET][msk])
    fig.suptitle("PC1-PC2 no specific pattern")

    del PC_im_

###############################################################################
#
# %% 4) Machine learning
#
###############################################################################

###############################################################################
# %% 4.1) Fit models over parameters grid: L2LR

mod_str = "l2lr-range"

xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")

if not os.path.exists(xls_filename):
    Cs = np.logspace(-2, 2, 5)
    #Cs = [1]

    estimators_dict = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=1)
    cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(Xim, y))}
    cv_dict["ALL"] = [np.arange(Xim.shape[0]), np.arange(Xim.shape[0])]

    key_values_input = dict_product(estimators_dict, dict(resdualizeNo="No", resdualizeYes="yes", resdualizeBiased="biased"), cv_dict)
    print("Nb Tasks=%i" % len(key_values_input))

    start_time = time.time()
    key_vals_output = MapReduce(n_jobs=16, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
    cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=y)
    cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
    cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
    cv_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
    print(cv_scores_mean)

    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals_output, fd)

    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)



###############################################################################
# %% 4.2) Fit models over parameters grid: ENETTV

print("# %% 2.2) Model + parameters grid: ENETTV")
#mod_str = "enettv-%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)
mod_str = "enettv-range"

xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")

mapreduce_sharedir = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext=None)

if not os.path.exists(xls_filename):

    # estimators
    import parsimony.algorithms as algorithms
    import parsimony.estimators as estimators
    import parsimony.functions.nesterov.tv as nesterov_tv
    from parsimony.utils.linalgs import LinearOperatorNesterov

    if not os.path.exists(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz")):
        Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
        Atv.save(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))

    Atv = LinearOperatorNesterov(filename=OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))
    assert np.allclose(Atv.get_singular_values(0), 11.942158510142576)

    def ratios_to_param(alpha, l1l2ratio, tvl2ratio):
        tv = alpha * tvl2ratio
        l1 = alpha * l1l2ratio
        l2 = alpha * 1
        return l1, l2, tv

    # Large range
    alphas = [.01, .1, 1.]
    l1l2ratios = [0, 0.0001, 0.001, 0.01, 0.1]
    tvl2ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1]

    # Smaller range
    # tv, l1, l2 from 202004 => tv, l1, l2 = 0.1, 0.001, 0.1
    # <=> alpha = 0.1, l1l2ratios=0.01, tvl2ratios = 1
    # alphas = [0.1]
    # l1l2ratios = [0.01]
    # tvl2ratios = [1]

    import itertools
    estimators_dict = dict()
    for alpha, l1l2ratio, tvl2ratio in itertools.product(alphas, l1l2ratios, tvl2ratios):
        print(alpha, l1l2ratio, tvl2ratio)
        l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvl2ratio)
        key = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)

        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
        estimators_dict[key] = estimator

    cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=1)
    cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(Xim, y))}
    cv_dict["ALL"] = [np.arange(Xim.shape[0]), np.arange(Xim.shape[0])]

    key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict)
    print("Nb Tasks=%i" % len(key_values_input))


    ###########################################################################
    # 3) Distributed Mapper

    if os.path.exists(mapreduce_sharedir):
        print("# Existing shared dir, delete for fresh restart: ")
        print("rm -rf %s" % mapreduce_sharedir)

    os.makedirs(mapreduce_sharedir, exist_ok=True)

    start_time = time.time()
    mp = MapReduce(n_jobs=NJOBS, shared_dir=mapreduce_sharedir, pass_key=True, verbose=20)
    mp.map(fit_predict, key_values_input)
    key_vals_output = mp.reduce_collect_outputs()


    ###########################################################################
    # 3) Centralized Mapper
    # start_time = time.time()
    # key_vals_output = MapReduce(n_jobs=NJOBS, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    # print("#  Centralized mapper completed in %.2f sec" % (time.time() - start_time))

    ###############################################################################
    # 4) Reducer: output key/value pairs => CV scores""")

    if key_vals_output is not None:
        print("# Distributed mapper completed in %.2f sec" % (time.time() - start_time))
        cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=y)
        cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
        cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
        cv_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        print(cv_scores_mean)

        with open(models_filename, 'wb') as fd:
            pickle.dump(key_vals_dist_output, fd)

        with pd.ExcelWriter(xls_filename) as writer:
            cv_scores.to_excel(writer, sheet_name='folds', index=False)
            cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)


###############################################################################
#
# %% 5) Plot parameters grid: ENETTV
#
###############################################################################

mod_str = "enettv-range"

xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
pdf_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pdf")

if not os.path.exists(pdf_filename):

    print("""
    #------------------------------------------------------------------------------
    # maps' similarity measures accross CV-folds
    """)
    #models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
    with open(models_filename, 'rb') as fd:
        key_vals = pickle.load(fd)

    # filter out "ALL" folds
    key_vals = {k:v for k, v in key_vals.items() if k[2] != "ALL"}

    # Agregate maps all key except the last whici is the fold
    nkey_beforefold = len(list(key_vals.keys())[0]) - 1

    by_param = {k:[] for k in set([k[:nkey_beforefold] for k in key_vals])}
    for k, v in key_vals.items():
        by_param[k[:nkey_beforefold]].append(v['coef_img'].ravel())

    maps_similarity_l = list()
    # Compute similarity measures
    for k, v in by_param.items():
        maps = np.array(v)
        maps_t = np.vstack([arr_threshold_from_norm2_ratio(maps[i, :], .99)[0] for i in range(maps.shape[0])])
        r_bar, dice_bar, fleiss_kappa_stat = maps_similarity(maps_t)
        prop_non_zeros_mean = np.count_nonzero(maps_t) / np.prod(maps_t.shape)
        maps_similarity_l.append(list(k) + [prop_non_zeros_mean, r_bar, dice_bar, fleiss_kappa_stat])

    map_sim = pd.DataFrame(maps_similarity_l,
                           columns=['param_%i' % i for i in range(nkey_beforefold)] +
                           ['prop_non_zeros_mean', 'r_bar', 'dice_bar', 'fleiss_kappa_stat'])

    # Write new sheet 'mean_img' in excel file
    sheets_ = pd.read_excel(xls_filename, sheet_name=None)
    cv_score_img_mean_ = sheets_['mean'][sheets_['mean']['pred'] == 'test_img']
    cv_score_img_mean_ = pd.merge(cv_score_img_mean_, map_sim)
    assert cv_score_img_mean_.shape[0] == map_sim.shape[0]
    sheets_['mean_img'] = cv_score_img_mean_

    with pd.ExcelWriter(xls_filename) as writer:
        for sheet_name, df_ in sheets_.items():
            print(sheet_name, df_.shape)
            df_.to_excel(writer, sheet_name=sheet_name, index=False)

    del sheets_, cv_score_img_mean_, sheets_, df_


    print("""
    #------------------------------------------------------------------------------
    # plot
    """)

    # xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    cv_scores = pd.read_excel(xls_filename, sheet_name='folds')
    cv_scores = cv_scores[(cv_scores['pred'] == 'test_img') & (cv_scores['fold'] != 'ALL')]
    cv_scores = cv_scores.reset_index(drop=True)
    cv_scores_mean = pd.read_excel(xls_filename, sheet_name='mean_img')
    cv_scores_mean = cv_scores_mean.reset_index(drop=True)

    keys_ = pd.DataFrame([[s.split("_")[0]] + [float(v) for v in s.split("_")[1].split(":")] for s in cv_scores["param_0"]],
                 columns=["model", "alpha", "l1l2ratio", "tvl2ratio"])
    cv_scores = pd.concat([keys_, cv_scores], axis=1)
    keys_ = pd.DataFrame([[s.split("_")[0]] + [float(v) for v in s.split("_")[1].split(":")] for s in cv_scores_mean["param_0"]],
                 columns=["model", "alpha", "l1l2ratio", "tvl2ratio"])
    cv_scores_mean = pd.concat([keys_, cv_scores_mean], axis=1)
    del keys_

    from matplotlib.backends.backend_pdf import PdfPages
    sns.set_style("whitegrid")
    import matplotlib.pylab as pl
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('text.latex', preamble=r'\usepackage{lmodern}')
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    # cv_scores["alpha"].unique(): array([0.01, 0.1 , 1.  ])
    # cv_scores["l1l2"].unique(): array([0.  , 0.01, 0.1 ])
    # cv_scores["tv"].unique() array([0.001, 0.01 , 0.1  , 1.   ])

    # pdf_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pdf")
    with PdfPages(pdf_filename) as pdf:
        for l1l2 in np.sort(cv_scores["l1l2ratio"].unique()):
            print("%.4f" % l1l2, l1l2)
            df_ = cv_scores[cv_scores["l1l2ratio"].isin([l1l2])]
            dfm_ = cv_scores_mean[cv_scores_mean["l1l2ratio"].isin([l1l2])]
            df_["alpha"] = df_["alpha"].map({0.01:"0.01", 0.1:"0.1" , 1.:"1'"})
            dfm_["alpha"] = dfm_["alpha"].map({0.01:"0.01", 0.1:"0.1" , 1.:"1'"})
            hue_order_ = np.sort(df_["alpha"].unique())
            df_.rename(columns={"alpha":"alpha", 'auc':'AUC', 'bacc':'bAcc'}, inplace=True)
            dfm_.rename(columns={'r_bar':'$r_w$', 'prop_non_zeros_mean':'non-null', 'dice_bar':'dice', 'fleiss_kappa_stat':'Fleiss-Kappa'}, inplace=True)

            fig, axs = plt.subplots(3, 2, figsize=(2 * 7.25, 3 * 5), dpi=300)
            g = sns.lineplot(x="tvl2ratio", y='AUC', hue="alpha", hue_order=hue_order_, data=df_, ax=axs[0, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tvl2ratio", y='bAcc', hue="alpha", hue_order=hue_order_, data=df_, ax=axs[0, 1], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tvl2ratio", y='$r_w$', hue="alpha", hue_order=hue_order_, data=dfm_, ax=axs[1, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tvl2ratio", y='non-null', hue="alpha", hue_order=hue_order_, data=dfm_, ax=axs[1, 1], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tvl2ratio", y='dice', hue="alpha", hue_order=hue_order_, data=dfm_, ax=axs[2, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tvl2ratio", y='Fleiss-Kappa', hue="alpha", hue_order=hue_order_, data=dfm_, ax=axs[2, 1], palette="Blues"); g.set(xscale="log")
            #plt.tight_layout()
            fig.suptitle('$\ell_1/\ell_2=%.5f$' % l1l2)
            #plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_auc", ext="pdf"))
            pdf.savefig()  # saves the current figure into a pdf page
            fig.clf()
            plt.close()


###############################################################################
#
# %% 6) Plot weight maps
#
###############################################################################
import nilearn.image
from nilearn import plotting
from matplotlib.backends.backend_pdf import PdfPages

mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_fdata() != 0
assert mask_arr.sum() == 369547


###############################################################################
# %% 6.1) ENETTV

mod_str = "enettv_0.100:0.010000:1.000000"
models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % "enettv-range", ext="pkl")
prefix = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type=mod_str+"_", ext=None)

if not os.path.exists(prefix + "coefmap.nii.gz"):

    with open(models_filename, 'rb') as fd:
        key_vals_output = pickle.load(fd)

    # Refit all coef map
    coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
    arr_threshold_from_norm2_ratio(coef_map, .99)
    # threshold= 0.0004182175137636338
    coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

    maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, vmax=0.001, prefix=prefix)

    # Cluster analysis
    import subprocess
    cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.99 --thresh_size 10" % (prefix + "coefmap.nii.gz")
    p = subprocess.run(cmd.split())



###############################################################################
# %% 6.2) L2LR

mod_str = "l2lr_C:10.000000"
models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % "l2lr-range", ext="pkl")
prefix = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type=mod_str+"_", ext=None)

if not os.path.exists(prefix + "coefmap.nii.gz"):

    with open(models_filename, 'rb') as fd:
        key_vals_output = pickle.load(fd)

    # Refit all coef map
    coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
    arr_threshold_from_norm2_ratio(coef_map, .99)
    # threshold= 0.0004182175137636338
    coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
         [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

    maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, vmax=0.001, prefix=prefix)

    # Cluster analysis
    import subprocess
    cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.99 --thresh_size 10" % (prefix + "coefmap.nii.gz")
    p = subprocess.run(cmd.split())

