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


# Build lobes atlas from HO
mkdir /neurospin/psy_sbox/analyses/202008_start-icaar_cat12vbm_predict-transition/ressources
cd /neurospin/psy_sbox/analyses/202008_start-icaar_cat12vbm_predict-transition/ressources
python ~/git/nitk/nitk/atlas/atlas_builder.py
"""
# %load_ext autoreload
# %autoreload 2

import os
import sys
import time
import glob
import re
import copy
import pickle
import shutil
import json
import subprocess

import numpy as np
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
# matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import nibabel
import nilearn.image
from nilearn.image import resample_to_img
import nilearn.image
from nilearn import plotting

from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio, arr_clusters
from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters, img_plot_glass_brain
from nitk.stats import Residualizer
from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif

import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sklearn.metrics as metrics

###############################################################################
#
# %% 1) Config
#
###############################################################################

INPUT_PATH = '/neurospin/psy/start-icaar-eugei/'
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/202008_start-icaar_cat12vbm_predict-transition'
# OUTPUT_PATH = '/neurospin/psy_sbox/analyses/202008_start-icaar_cat12vbm_predict-transition_wholebrain'

NJOBS = 4
#MSK_SIZE = 369542 # whole brain
MSK_SIZE = 330850 # without cerrebelum and BrainStem

os.chdir(OUTPUT_PATH)

# On laptop
if not os.path.exists(INPUT_PATH):
    INPUT_PATH = INPUT_PATH.replace('/neurospin', '/home/ed203246/data')
    OUTPUT_PATH = OUTPUT_PATH.replace('/neurospin', '/home/ed203246/data' )
    NJOBS = 2


def PATH(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None,
    masking=None,
    type=None, ext=None, basepath=""):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32

    return os.path.join(basepath, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if masking is None else "-" + masking) +
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
NBOOTS = 500


###############################################################################
#
# %% 2) Utils
#
###############################################################################

###############################################################################
# %% 2.1) Mapper

def fit_predict(key, estimator_img, residualize, split, Xim, y, Z, Xdemoclin, residualizer):
    estimator_img = copy.deepcopy(estimator_img)
    train, test = split
    print("fit_predict", Xim.shape, Xdemoclin.shape, Z.shape, y.shape, np.max(train), np.max(test))
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

    elif residualize == 'no':
        pass

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


def ratios_to_param(alpha, l1l2ratio, tvl2ratio):
    tv = alpha * tvl2ratio
    l1 = alpha * l1l2ratio
    l2 = alpha * 1
    return l1, l2, tv

###############################################################################
# %% 2.2) Plot

def plot_coefmap_stats(coef_map, coef_maps, ref_img, thresh_norm_ratio, vmax, prefix):

    # arr_threshold_from_norm2_ratio(coef_map, thresh_norm_ratio)[0]
    coef_maps_t = np.vstack([arr_threshold_from_norm2_ratio(coef_maps[i, :], thresh_norm_ratio)[0] for i in range(coef_maps.shape[0])])

    w_selectrate = np.sum(coef_maps_t != 0, axis=0) / coef_maps_t.shape[0]
    w_zscore = np.nan_to_num(np.mean(coef_maps, axis=0) / np.std(coef_maps, axis=0))
    w_mean = np.mean(coef_maps, axis=0)
    w_std = np.std(coef_maps, axis=0)
    # 95% CI compute sign product of lower and hhigher 95%CI
    coef_maps_ci = np.quantile(coef_maps, [0.025, 0.975], axis=0)
    #coef_maps_ci = np.quantile(coef_maps, [0.05, 0.95], axis=0)
    coef_maps_ci_sign = np.sign(coef_maps_ci.prod(axis=0))
    coef_maps_ci_sign[coef_maps_ci_sign == -1] = 0

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = coef_map
    coefmap_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_img.to_filename(prefix + "coefmap.nii.gz")

    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = w_mean
    coefmap_cvmean_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_cvmean_img.to_filename(prefix + "coefmap_mean.nii.gz")

    # Mean weight map within 95%CI
    w_mean[coef_maps_ci_sign != 1] = 0
    val_arr = np.zeros(ref_img.get_fdata().shape)
    val_arr[mask_arr] = w_mean
    coefmap_cvmean95ci_img = nibabel.Nifti1Image(val_arr, affine=ref_img.affine)
    coefmap_cvmean95ci_img.to_filename(prefix + "coefmap_mean95ci.nii.gz")

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

    #fig = plt.figure(figsize=(11.69, 6 * 11.69 * .32))
    fig = plt.figure(figsize=(11.69, 6 * 11.69 * .4))

    ax = fig.add_subplot(611)
    plotting.plot_glass_brain(coefmap_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, vmax=vmax, figure=fig, axes=ax, title="Coef")

    ax = fig.add_subplot(612)
    plotting.plot_glass_brain(coefmap_cvmean_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, vmax=vmax, figure=fig, axes=ax, title="Mean")

    ax = fig.add_subplot(613)
    plotting.plot_glass_brain(coefmap_cvmean95ci_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, vmax=vmax, figure=fig, axes=ax, title="Mean 95%CI")

    ax = fig.add_subplot(614)
    plotting.plot_glass_brain(coefmap_cvzscore_img, threshold=1e-6, plot_abs=False, colorbar=True, cmap=plt.cm.bwr, figure=fig, axes=ax, title="Zscore")

    ax = fig.add_subplot(615)
    plotting.plot_glass_brain(coefmap_cvselectrate_img, threshold=1e-6, plot_abs=True, colorbar=True, figure=fig, axes=ax, title="Select. Rate")

    ax = fig.add_subplot(616)
    plotting.plot_glass_brain(coefmap_cvstd_img, threshold=1e-8, plot_abs=True, colorbar=True, figure=fig, axes=ax, title="Std")

    pdf.savefig(); plt.close(fig); pdf.close()

    maps = {"coefmap": coefmap_img, "coefmap_mean": coefmap_cvmean_img,
            "coefmap_cvstd": coefmap_cvstd_img, "coefmap_cvzscore": coefmap_cvzscore_img,
            "coefmap_cvselectrate": coefmap_cvselectrate_img}

    return maps

###############################################################################
#
# %% 3.1) Build uhr from images
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
    mask_img = compute_brain_mask(imgs_arr, target_img, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10,
                               verbose=1)

    # Exclude 'Brain-Stem':7, and 'Cerebellum':8
    lobe_atlas_img = resample_to_img(os.path.join(OUTPUT_PATH, "ressources", "HarvardOxford_lobes.nii.gz"), target_img, interpolation= 'nearest')
    lobe_atlas_img.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="atlas-lobes", ext="nii.gz"))

    lobe_atlas_tab = pd.read_csv(os.path.join(OUTPUT_PATH, "ressources", "HarvardOxford_lobes.csv"))
    #lobe_atlas_dict = {row.label: row.lobe for row in lobe_atlas_tab.itertuples()}
    lobe_atlas_dict = {row.lobe:row.label for row in lobe_atlas_tab.itertuples()}
    lobe_atlas_arr = lobe_atlas_img.get_fdata()

    mask_arr = mask_img.get_fdata() > 0
    mask_arr[(lobe_atlas_arr == lobe_atlas_dict['Brain-Stem']) | (lobe_atlas_arr == lobe_atlas_dict['Cerebellum'])] = False
    mask_arr = rm_small_clusters(mask_arr, clust_size_thres=100)
    mask_img = nilearn.image.new_img_like(target_img, mask_arr)

    mask_img.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    pop.to_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"), index=False)
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), imgs_arr)

    # template
    template = nibabel.load("/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz")
    template_resamp = resample_to_img(template, mask_img)
    template_resamp.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="template", ext="nii.gz"))

    del imgs_arr, target_img, mask_img


###############################################################################
#
# %% 4) Datasets loader
#

###############################################################################
# %% 4.1) uhr loader

def load_uhr_dataset():
    dataset = 'icaar-start'

    pop = pd.read_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
    imgs_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"))
    mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_arr = mask_img.get_fdata() != 0
    assert mask_arr.sum() == MSK_SIZE

    # Update 2020/06: include Non-UHR-NC as non transitors
    # pop[TARGET] = pop["diagnosis"].map({'UHR-C': "1", 'UHR-NC': "0", 'Non-UHR-NC':"0"}).values
    # pop[TARGET_NUM] = pop["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0, 'Non-UHR-NC':0}).values
    pop[TARGET] = pop["diagnosis"].map({'UHR-C': "1", 'UHR-NC': "0"}).values
    pop[TARGET_NUM] = pop["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0}).values

    pop["GM_frac"] = pop.gm / pop.tiv
    pop["sex_c"] = pop["sex"].map({0: "M", 1: "F"})


    ###############################################################################
    # Select participants

    # Update 2020/06: include Non-UHR-NC as non transitors
    # msk = pop["diagnosis"].isin(['UHR-C', 'UHR-NC', 'Non-UHR-NC']) &  pop["irm"].isin(['M0']) # UPDATE 2020/06
    msk = pop["diagnosis"].isin(['UHR-C', 'UHR-NC']) &  pop["irm"].isin(['M0']) # UPDATE 2020/06

    assert msk.sum() == 82

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

    dataset = dict(Xim=Xim, y=y, Xdemoclin=Xdemoclin,
                  msk=msk, residualizer=residualizer, Z=Z,
                  mask_img=mask_img, mask_arr=mask_arr, pop=pop)

    return dataset

###############################################################################
# %% 4.2) prague_schizconnectvip loade

def load_prague_schizconnectvip_datasets():
    TARGET_NUM = "diagnosis_num"
    schizconnectvip_arr_filename = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/schizconnect-vip_t1mri_mwp1_gs-raw_data64.npy"
    schizconnectvip_csv_filename = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/schizconnect-vip_t1mri_mwp1_participants.csv"

    schizconnectvipprag_csv = pd.read_csv(schizconnectvip_csv_filename)
    schizconnectvipprag_imgs_arr = np.load(schizconnectvip_arr_filename)
    schizconnectvipprag_csv[TARGET_NUM] = schizconnectvipprag_csv["diagnosis"].map({'schizophrenia': 1, 'FEP':1, 'control': 0}).values

    # Select PRAGUE subjects
    msk_prag = schizconnectvipprag_csv.study == "PRAGUE"
    Xim_prag = schizconnectvipprag_imgs_arr.squeeze()[:, mask_arr][msk_prag]
    y_prag = schizconnectvipprag_csv.loc[msk_prag, TARGET_NUM].values
    Xclin_prag = schizconnectvipprag_csv.loc[msk_prag, VAR_CLINIC].values
    Xdemo_prag = schizconnectvipprag_csv.loc[msk_prag, VAR_DEMO].values
    Xsite_prag = pd.get_dummies(schizconnectvipprag_csv.site[msk_prag]).values
    Xdemoclin_prag = np.concatenate([Xdemo_prag, Xclin_prag], axis=1)
    formula_res, formula_full = "site + age + sex", "site + age + sex + " + TARGET_NUM
    residualizer_prag = Residualizer(data=schizconnectvipprag_csv[msk_prag], formula_res=formula_res, formula_full=formula_full)
    #Z_prag = residualizer_prag.get_design_mat()
    # residualizer_prag.fit(Xim_prag, Z_prag)
    # Xim_prag_res = residualizer_prag.transform(Xim_prag, Z_prag)

    # Select schizconnect-vip subjects
    msk_schizconnectvip = schizconnectvipprag_csv.study == "SCHIZCONNECT-VIP"
    Xim_schizconnectvip = schizconnectvipprag_imgs_arr.squeeze()[:, mask_arr][msk_schizconnectvip]
    y_schizconnectvip = schizconnectvipprag_csv.loc[msk_schizconnectvip, 'diagnosis'].map({'control':0, 'schizophrenia':1}).values
    Xclin_schizconnectvip = schizconnectvipprag_csv.loc[msk_schizconnectvip, VAR_CLINIC].values
    Xdemo_schizconnectvip = schizconnectvipprag_csv.loc[msk_schizconnectvip, VAR_DEMO].values
    Xsite_schizconnectvip = pd.get_dummies(schizconnectvipprag_csv.site[msk_schizconnectvip]).values
    Xdemoclin_schizconnectvip = np.concatenate([Xdemo_schizconnectvip, Xclin_schizconnectvip], axis=1)
    formula_res, formula_full = "site + age + sex", "site + age + sex + " + TARGET_NUM
    residualizer_schizconnectvip = Residualizer(data=schizconnectvipprag_csv[msk_schizconnectvip], formula_res=formula_res, formula_full=formula_full)
    #Z_schizconnectvip = residualizer_schizconnectvip.get_design_mat()

    del schizconnectvipprag_imgs_arr

    dataset_dict = dict(
    prague = dict(Xim=Xim_prag, y=y_prag, Xdemoclin=Xdemoclin_prag,
                  msk=msk_prag, residualizer=residualizer_prag),
    schizconnectvip = dict(Xim=Xim_schizconnectvip, y=y_schizconnectvip, Xdemoclin=Xdemo_schizconnectvip,
                  msk=msk_schizconnectvip, residualizer=residualizer_schizconnectvip))

    return dataset_dict



###############################################################################
#
# %% 5) Machine learning
#
###############################################################################

###############################################################################
# %% 5.1) L2 and L1 LR 5CV grid search

dataset = 'icaar-start'
mod_str = "l2l1lr_range"

xls_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")

if not os.path.exists(xls_filename):
    df = load_uhr_dataset()
    # Xim = df["Xim"]
    # y = df["y"]
    # residualizer = df["residualizer"]
    # Z = residualizer.get_design_mat()
    # Xdemoclin = df["Xdemoclin"]

    Cs = np.logspace(-2, 2, 5)
    # Cs = [10]

    estimators_dict = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    estimators_dict.update({"l1lr_C:%.6f" % C: lm.LogisticRegression(penalty='l1', solver= 'liblinear', C=C, class_weight='balanced', fit_intercept=False) for C in Cs})

    cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=1)
    cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    # key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
    key_values_input = dict_product(estimators_dict, dict(resdualizeNo="No", resdualizeYes="yes", resdualizeBiased="biased"), cv_dict,
        {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :df['y']},
        {'Z_%s' % dataset :df['Z']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})
    print("Nb Tasks=%i" % len(key_values_input))

    start_time = time.time()
    key_vals_output = MapReduce(n_jobs=16, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
    cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)

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
# %% 5.3) L2LR smoothed 5CV grid search

dataset = 'icaar-start'
mod_str = "s8_l2lr_range"

xls_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")

if not os.path.exists(xls_filename):

    df = load_uhr_dataset()
    mask_arr = df['mask_arr']
    mask_img = df['mask_img']
    msk = df['msk']

    # Smooth all image at fwhm=8mm
    imgs_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"))
    from nilearn.image import new_img_like, smooth_img
    imgs_arr = np.stack([np.expand_dims(smooth_img(new_img_like(mask_img, imgs_arr[i, 0, :]), fwhm=8).get_fdata(), axis=0)
        for i in range(imgs_arr.shape[0])])
    Xim = imgs_arr.squeeze()[:, mask_arr][msk]
    del imgs_arr

    df['Xim'] = Xim
    del Xim
    Cs = np.logspace(-2, 2, 5)
    #Cs = [1]

    estimators_dict = {"s8_l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=1)
    cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    key_values_input = dict_product(estimators_dict, dict(resdualizeNo="No", resdualizeYes="yes", resdualizeBiased="biased"), cv_dict,
        {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :df['y']},
        {'Z_%s' % dataset :df['Z']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})
    print("Nb Tasks=%i" % len(key_values_input))

    start_time = time.time()
    key_vals_output = MapReduce(n_jobs=16, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
    # cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
    cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
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
# %% 5.4) ENETTV 5CV grid search

dataset = 'icaar-start'
#mod_str = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)
mod_str = "enettv_range"

xls_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
# models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")
mapreduce_sharedir = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext=None)

if not os.path.exists(xls_filename):
    print(" %% 4.4) ENETTV 5CV grid search")

    df = load_uhr_dataset()
    mask_arr = df['mask_arr']
    mask_img = df['mask_img']
    msk = df['msk']

    # estimators
    import parsimony.algorithms as algorithms
    import parsimony.estimators as estimators
    import parsimony.functions.nesterov.tv as nesterov_tv
    from parsimony.utils.linalgs import LinearOperatorNesterov

    if not os.path.exists(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz")):
        Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
        Atv.save(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))

    Atv = LinearOperatorNesterov(filename=OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))
    #assert np.allclose(Atv.get_singular_values(0), 11.940682881834617) # whole brain
    assert np.allclose(Atv.get_singular_values(0), 11.925947387128584) # rm brainStem+cerrebelum

    # Large range
    alphas = [.01, 0.1, 1.]
    l1l2ratios = [0, 0.0001, 0.001, 0.01, 0.1]
    tvl2ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]

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
    cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
    cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

    #key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict)
    key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
        {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :df['y']},
        {'Z_%s' % dataset :df['Z']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})

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
    # key_vals_output = mp.reduce_collect_outputs(force=True)

    ###########################################################################
    # 3) Centralized Mapper
    # start_time = time.time()
    # key_vals_output = MapReduce(n_jobs=NJOBS, pass_key=True, verbose=20).map(fit_predict, key_values_input)
    # print("#  Centralized mapper completed in %.2f sec" % (time.time() - start_time))

    ###############################################################################
    # 4) Reducer: output key/value pairs => CV scores""")

    if key_vals_output is not None:
        print("# Distributed mapper completed in %.2f sec" % (time.time() - start_time))
        cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
        cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
        cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
        cv_scores_std = cv_scores.groupby(["param_1", "param_0", "pred"]).std().reset_index()
        cv_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        cv_scores_std.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        print(cv_scores_mean)

        # with open(models_filename, 'wb') as fd:
        #   pickle.dump(key_vals_output, fd)

        with pd.ExcelWriter(xls_filename) as writer:
            cv_scores.to_excel(writer, sheet_name='folds', index=False)
            cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)
            cv_scores_std.to_excel(writer, sheet_name='std', index=False)

# sys.exit("End")

###############################################################################
# %% 5.5) ENETTV Bootstrap "enettv_0.100:0.010000:1.000000"

# 0.100:0.010000:1.000000 has been found to provide good trade-off prediction/statibility/sparsity
# on other studies: deptms/ schyzconnect/ bsnip
dataset = 'icaar-start'
# alpha, l1l2ratio, tvl2ratio = 0.1, 0.01, 1
#mod_str = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)
mod_str = "enettv_range"

xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_boot" % mod_str, ext="xlsx")
#models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_boot" % mod_str, ext="pkl")
bootjson_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_boot" % mod_str, ext="json")
mapreduce_sharedir = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_boot" % mod_str, ext=None)

if not os.path.exists(xls_filename):
    print(" %% 6.2) ENETTV Bootstrap enettv")

    df = load_uhr_dataset()
    mask_arr = df['mask_arr']
    mask_img = df['mask_img']
    msk = df['msk']
    y = df["y"]

    # estimators
    import parsimony.algorithms as algorithms
    import parsimony.estimators as estimators
    import parsimony.functions.nesterov.tv as nesterov_tv
    from parsimony.utils.linalgs import LinearOperatorNesterov

    if not os.path.exists(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz")):
        Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
        Atv.save(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))

    Atv = LinearOperatorNesterov(filename=OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))
    assert np.allclose(Atv.get_singular_values(0), 11.925947387128584) # rm brainStem+cerrebelum

    # Large range
    # alphas = [.01, 0.1, 1.]
    # l1l2ratios = [0, 0.0001, 0.001, 0.01, 0.1]
    # tvl2ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]

    # Smaller range
    # tv, l1, l2 from 202004 => tv, l1, l2 = 0.1, 0.001, 0.1
    # <=> alpha = 0.1, l1l2ratios=0.01, tvl2ratios = 1
    alphas = [0.1]
    l1l2ratios = [0.001, 0.01]
    tvl2ratios = [0.1, 1]

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

    ###########################################################################
    # Boostraped sampling
    if not os.path.exists(bootjson_filename):
        orig_all = np.arange(y.shape[0])
        boot_dict, boot_i= dict(), 0
        while len(boot_dict) < NBOOTS:
            boot_tr = resample(orig_all, n_samples=len(y), replace=True, stratify=y, random_state=boot_i)
            boot_te = np.setdiff1d(orig_all, boot_tr, assume_unique=False)
            if len(np.unique(y[boot_te])) == len(np.unique(y)): # all classes are present in test
                boot_dict["BOOT%03d" % boot_i] = [boot_tr.tolist(), boot_te.tolist()]
            boot_i += 1

        with open(bootjson_filename, 'w') as fp:
            json.dump(boot_dict, fp)
    else:
        with open(bootjson_filename, 'r') as fp:
            boot_dict = json.load(fp)
            assert len(boot_dict) == NBOOTS

    # key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), boot_dict)
    key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), boot_dict,
        {'Xim_%s' % dataset :df['Xim']}, {'y_%s' % dataset :df['y']},
        {'Z_%s' % dataset :df['Z']}, {'Xdemoclin_%s' % dataset :df['Xdemoclin']},
        {'residualizer_%s' % dataset :df['residualizer']})

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
        # cv_scores_all = reduce_cv_classif(key_vals_output, boot_dict, y_true=y)
        cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
        cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
        cv_scores_mean = cv_scores.groupby(["param_1", "param_0", "pred"]).mean().reset_index()
        cv_scores_std = cv_scores.groupby(["param_1", "param_0", "pred"]).std().reset_index()
        cv_scores_mean.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        cv_scores_std.sort_values(["param_1", "param_0", "pred"], inplace=True, ignore_index=True)
        print(cv_scores_mean)

        # with open(models_filename, 'wb') as fd:
        #   pickle.dump(key_vals_output, fd)

        with pd.ExcelWriter(xls_filename) as writer:
            cv_scores.to_excel(writer, sheet_name='folds', index=False)
            cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)
            cv_scores_std.to_excel(writer, sheet_name='std', index=False)

sys.exit("End")

###############################################################################
#
# %% 6) Plot parameters grid: ENETTV
#
###############################################################################

dataset = 'icaar-start'
mod_str = "enettv_range"
mapreduce_sharedir = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext=None)
xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
pdf_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pdf")

if not os.path.exists(pdf_filename):

    print("""
    #------------------------------------------------------------------------------
    # maps' similarity measures accross CV-folds
    """)
    #models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
    #with open(models_filename, 'rb') as fd:
    #    key_vals = pickle.load(fd)

    mp = MapReduce(shared_dir=mapreduce_sharedir)
    key_vals_all = mp.reduce_collect_outputs(force=True)

    # filter out "ALL" folds
    key_vals = {k:v for k, v in key_vals_all.items() if k[2] != "ALL"}

    # Agregate maps all key except the last whici is the fold
    idx_fold = 2
    #nkey_beforefold = len(list(key_vals.keys())[0]) - 1

    by_param = {k:[] for k in set([k[:idx_fold] for k in key_vals])}
    for k, v in key_vals.items():
        by_param[k[:idx_fold]].append(v['coef_img'].ravel())

    maps_similarity_l = list()
    # Compute similarity measures and average number of cluster
    for k, v in by_param.items():
        maps = np.array(v)
        maps_t = np.vstack([arr_threshold_from_norm2_ratio(maps[i, :], .999)[0] for i in range(maps.shape[0])])
        r_bar, dice_bar, fleiss_kappa_stat = maps_similarity(maps_t)
        prop_non_zeros_mean = np.count_nonzero(maps_t) / np.prod(maps_t.shape)
        n_clusts = np.mean([arr_clusters(maps_t[i,: ], mask_arr)[1] for i in range(maps_t.shape[0])])
        maps_similarity_l.append(list(k) + [n_clusts, prop_non_zeros_mean, r_bar, dice_bar, fleiss_kappa_stat])

    map_sim = pd.DataFrame(maps_similarity_l,
                           columns=['param_%i' % i for i in range(idx_fold)] +
                           ['n_clusts', 'prop_non_zeros_mean', 'r_bar', 'dice_bar', 'fleiss_kappa_stat'])

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

    del sheets_, cv_score_img_mean_, df_


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
            g = sns.lineplot(x="tvl2ratio", y='n_clusts', hue="alpha", hue_order=hue_order_, data=dfm_, ax=axs[2, 1], palette="Blues"); g.set(xscale="log")
            #plt.tight_layout()
            fig.suptitle('$\ell_1/\ell_2=%.5f$' % l1l2)
            #plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_auc", ext="pdf"))
            pdf.savefig()  # saves the current figure into a pdf page
            fig.clf()
            plt.close()



###############################################################################
# %% 7) Discrimative power of ROIs

if False:

    mod_str = "enettv_0.100:0.010000:1.000000"
    #models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_5cv_" % "enettv_range", ext="pkl")
    prefix = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="%s_boot_" % mod_str, ext=None)

    coefmap_mean95ci_arr = nibabel.load(prefix + "coefmap_mean95ci.nii.gz").get_fdata()
    coefmap_mean95ci_clust_labels_label_arr = nibabel.load(prefix + "coefmap_mean95ci_clust_labels.nii.gz").get_fdata()


    ###############################################################################
    # Decision function global and for each sub roi
    # Age vs decision function
    pop_uhr = pop.loc[msk, ['participant_id', "diagnosis"] + VAR_DEMO]


    ###############################################################################
    # Rezidualize

    residualizer.fit(Xim, Z)
    Xim_res = residualizer.transform(Xim, Z)

    ###############################################################################
    # Decision function global and by ROIs

    pop_uhr["decfunc_glob"] = np.dot(Xim_res, coefmap_mean95ci_arr[mask_arr])

    for lab in np.unique(coefmap_mean95ci_clust_labels_label_arr):
        if lab == 0:
            continue
        print(lab)

        mask_roi = coefmap_mean95ci_clust_labels_label_arr == lab
        coefmap_roi = coefmap_mean95ci_arr.copy()
        coefmap_roi[mask_roi != 0] = 0
        pop_uhr["decfunc_roi%i" % lab] = np.dot(Xim_res, coefmap_roi[mask_arr])

        regex = re.compile("^(decfunc_roi[0-9]+)")
        roi_names = [col for col in pop_uhr.columns if regex.match(col)]

    from sklearn.linear_model import lars_path
    from sklearn import metrics

    X_ = pop_uhr[roi_names].values
    y_ = pop_uhr['diagnosis'].map({'UHR-NC':0, 'UHR-C':1}).values
    _, _, coefs_ = lars_path(X_, y_, return_path=True) # coefs= (n_features, n_alphas + 1)

    dec_func_path = np.dot(X_, coefs_)
    auc_path = [metrics.roc_auc_score(y_, dec_func_path[:, j]) for j in range(dec_func_path.shape[1])]
    feat_path = [np.array(roi_names)[coefs_[:, j] != 0].tolist() for j in range(dec_func_path.shape[1])]
    feat_path = [[feat.replace('decfunc_', '') for feat in  feat_set] for feat_set in feat_path]

    auc_path_df = pd.DataFrame(dict(auc=auc_path, features=[' '.join(feat_set) for feat_set in feat_path]))
    auc_path_df.to_csv("/tmp/path.csv", index=False)

    sns.pairplot(pop_uhr, hue="diagnosis")

    pop_uhr.corr()


    ###############################################################################
    # Apply to PRAGUE

    schizconnectvip_arr_filename = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/schizconnect-vip_t1mri_mwp1_gs-raw_data64.npy"
    schizconnectvip_csv_filename = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/schizconnect-vip_t1mri_mwp1_participants.csv"


    sczcnnct_csv = pd.read_csv(schizconnectvip_csv_filename)
    sczcnnct_imgs_arr = np.load(schizconnectvip_arr_filename)
    sczcnnct_csv[TARGET_NUM] = sczcnnct_csv["diagnosis"].map({'schizophrenia': 1, 'FEP':1, 'control': 0}).values

    # Select PRAGUE subjects
    msk_prag = sczcnnct_csv.study == "PRAGUE"
    Xim_prag = sczcnnct_imgs_arr.squeeze()[:, mask_arr][msk_prag]
    del sczcnnct_imgs_arr


    ###############################################################################
    # Decision function without residualization

    pop_prag = sczcnnct_csv.loc[msk_prag, ['participant_id', "diagnosis"] + VAR_DEMO]
    # pop_prag["dec_func"] = np.dot(Xim_prag, coefmap_arr[mask_arr])
    # sns.scatterplot(x='age', y='dec_func', hue=pop_prag.diagnosis.tolist(), data=pop_prag)


    ###############################################################################
    # Residualize PRAGUE DATA using icaar-starts rezidualizer

    # [col for col in residualizer.t_contrasts]
    # ['Intercept', 'site[T.Sainte-Anne]', 'age', 'sex', 'transition_num']
    N = Xim_prag.shape[0]
    Z_prg = np.column_stack([np.ones(N), np.zeros(N), pop_prag['age'].values,
                             pop_prag['sex'].values,
                             pop_prag['diagnosis'].map({'control':0, 'FEP':1})])
    Xim_prag_res = residualizer.transform(Xim_prag, Z_prg)

    pop_prag["decfunc_glob_uhrresidualizer"] = np.dot(Xim_prag_res, coefmap_mean95ci_arr[mask_arr])
    #pop_prag["decfunc_glob"] = np.dot(Xim_prag, coefmap_mean95ci_arr[mask_arr])
    metrics.roc_auc_score(pop_prag['diagnosis'].map({'control':0, 'FEP':1}).values, pop_prag["decfunc_glob_uhrresidualizer"])
    # 0.41886304909560723

    ###############################################################################
    # Residualize using PRAGUE DATA

    #Xclin_prg = sczcnnct_csv.loc[msk_prag, VAR_CLINIC].values
    #Xdemo_prg = sczcnnct_csv.loc[msk_prag, VAR_DEMO].values
    #Xsite_prg = pd.get_dummies(sczcnnct_csv.site[msk_prag]).values
    #Xdemoclin_prg = np.concatenate([Xdemo_prg, Xclin_prg], axis=1)
    formula_res, formula_full = "site + age + sex", "site + age + sex + " + TARGET_NUM
    residualizer_prg = Residualizer(data=sczcnnct_csv[msk_prag], formula_res=formula_res, formula_full=formula_full)
    Z_prg = residualizer_prg.get_design_mat()
    residualizer_prg.fit(Xim_prag, Z_prg)
    Xim_prag_res = residualizer_prg.transform(Xim_prag, Z_prg)

    pop_prag["decfunc_glob"] = np.dot(Xim_prag_res, coefmap_mean95ci_arr[mask_arr])
    sns.pairplot(pop_prag, hue="diagnosis")
    np.corrcoef(pop_prag.decfunc_glob, pop_prag.decfunc_glob_uhrresidualizer)[0, 1]
    # 0.9694918690446181

    #pop_prag["decfunc_glob"] = np.dot(Xim_prag, coefmap_mean95ci_arr[mask_arr])
    metrics.roc_auc_score(pop_prag['diagnosis'].map({'control':0, 'FEP':1}).values, pop_prag["decfunc_glob"])


    ###############################################################################
    # Decision function global and by ROIs

    for lab in np.unique(coefmap_mean95ci_clust_labels_label_arr):
        if lab == 0:
            continue
        print(lab)

        mask_roi = coefmap_mean95ci_clust_labels_label_arr == lab
        coefmap_roi = coefmap_mean95ci_arr.copy()
        coefmap_roi[mask_roi != 0] = 0
        pop_prag["decfunc_roi%i" % lab] = np.dot(Xim_prag_res, coefmap_roi[mask_arr])

        regex = re.compile("^(decfunc_roi[0-9]+)")
        roi_names = [col for col in pop_uhr.columns if regex.match(col)]

    from sklearn.linear_model import lars_path
    from sklearn import metrics

    X_ = pop_prag[roi_names].values
    y_ = pop_prag['diagnosis'].map({'control':0, 'FEP':1}).values
    _, _, coefs_ = lars_path(X_, y_, return_path=True) # coefs= (n_features, n_alphas + 1)

    dec_func_path = np.dot(X_, coefs_)
    auc_path = [metrics.roc_auc_score(y_, dec_func_path[:, j]) for j in range(dec_func_path.shape[1])]
    feat_path = [np.array(roi_names)[coefs_[:, j] != 0].tolist() for j in range(dec_func_path.shape[1])]
    feat_path = [[feat.replace('decfunc_', '') for feat in  feat_set] for feat_set in feat_path]

    auc_path_df = pd.DataFrame(dict(auc=auc_path, features=[' '.join(feat_set) for feat_set in feat_path]))
    auc_path_df.to_csv("/tmp/path.csv", index=False)

    sns.pairplot(pop_prag, hue="diagnosis")

    ###############################################################################
    # Visualize all

    pop_all = pop_uhr.append(pop_prag)
    sns.scatterplot(x='age', y='decfunc_glob', hue=pop_all.diagnosis.tolist(), data=pop_all)
    #sns.scatterplot(x='age', y='dec_func_res_nosite', hue=pop_all.diagnosis.tolist(), data=pop_all)


###############################################################################
# %% 9) PRAGUE & schizconnect

mod_str  = "enettv_range"

if np.any([not os.path.exists(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx"))
           for dataset in ["schizconnectvip", "prague"]]):

    ###########################################################################
    # Load datasets prague & schizconnectvip dataset
    dataset_dict = load_prague_schizconnectvip_datasets()

    ###########################################################################
    # Process datasets

    for dataset in ["schizconnectvip", "prague"]:
        print("PROCESS:", dataset, mod_str)

        #######################################################################
        # Prepare data
        xls_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="xlsx")
        models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")
        mapreduce_sharedir = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext=None)
        Xim = dataset_dict[dataset]["Xim"]
        y = dataset_dict[dataset]["y"]
        residualizer = dataset_dict[dataset]["residualizer"]
        Z = residualizer.get_design_mat()
        Xdemoclin = dataset_dict[dataset]["Xdemoclin"]
        assert Xim.shape[0] == y.shape[0] == Z.shape[0] and not np.any(np.isnan(Z))
        print("Data shapes:", Xim.shape, y.shape, Z.shape, Xdemoclin.shape)
        # print(xls_filename)

        #######################################################################
        # 5 CV
        cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=1)
        cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(df['Xim'], df['y']))}
        cv_dict["ALL"] = [np.arange(df['Xim'].shape[0]), np.arange(df['Xim'].shape[0])]

        #######################################################################
        # l2lr + l1lr with and without residualization

        Cs = [10.]
        estimators_l2lr_dict = {"l2lr_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
        estimators_l1lr_dict = {"l1lr_C:%.6f" % C: lm.LogisticRegression(penalty='l1', solver= 'liblinear', C=C, class_weight='balanced', fit_intercept=False) for C in Cs}

        estimators_dict = dict()
        estimators_dict.update(estimators_l2lr_dict)
        estimators_dict.update(estimators_l1lr_dict)

        key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes", resdualizeNo="no"), cv_dict)

        #######################################################################
        # Enettv with residualization

        import parsimony.algorithms as algorithms
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as nesterov_tv
        from parsimony.utils.linalgs import LinearOperatorNesterov

        if not os.path.exists(OUTPUT('icaar-start', scaling=None, harmo=None, type="Atv", ext="npz")):
            Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
            Atv.save(OUTPUT('icaar-start', scaling=None, harmo=None, type="Atv", ext="npz"))

        Atv = LinearOperatorNesterov(filename=OUTPUT('icaar-start', scaling=None, harmo=None, type="Atv", ext="npz"))
        #assert np.allclose(Atv.get_singular_values(0), 11.940682881834617) # whole brain
        assert np.allclose(Atv.get_singular_values(0), 11.925947387128584) # rm brainStem+cerrebelum

        # Large range
        # alphas = [.01, 0.1, 1.]
        # l1l2ratios = [0, 0.0001, 0.001, 0.01, 0.1]
        tvl2ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]

        # Smaller range
        # tv, l1, l2 from 202004 => tv, l1, l2 = 0.1, 0.001, 0.1
        # <=> alpha = 0.1, l1l2ratios=0.01, tvl2ratios = 1
        alphas = [0.1]
        l1l2ratios = [0.01]
        #tvl2ratios = [1]

        import itertools
        estimators_enettv_dict = dict()
        for alpha, l1l2ratio, tvl2ratio in itertools.product(alphas, l1l2ratios, tvl2ratios):
            print(alpha, l1l2ratio, tvl2ratio)
            l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvl2ratio)
            key = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)

            conesta = algorithms.proximal.CONESTA(max_iter=10000)
            estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
            estimators_enettv_dict[key] = estimator


        key_values_enettv_input = dict_product(estimators_enettv_dict, dict(resdualizeYes="yes"), cv_dict)

        #######################################################################
        # Pack all cases
        key_values_input.update(key_values_enettv_input)
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
            cv_scores_all = reduce_cv_classif(key_vals_output, cv_dict, y_true=df['y'], index_fold=2)
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
# %% Map organization as spatial clusters

if False:

    mask_img = nibabel.load(OUTPUT(dataset='icaar-start', scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_arr = mask_img.get_fdata() != 0
    assert mask_arr.sum() == MSK_SIZE


    # mod_enettv_str = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)
    mod_enettv_str = "enettv_range"
    #mod_l2_str = "l2lr_C:10.000000"
    #mod_l1_str = "l1lr_C:10.000000"

    for dataset in ["icaar-start", "prague", "schizconnectvip"]:
        # dataset = "icaar-start"
        # dataset = "prague"
        # dataset = "schizconnectvip"
        mapreduce_sharedir = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_enettv_str, ext=None)
        mp = MapReduce(shared_dir=mapreduce_sharedir)
        key_vals_output = mp.reduce_collect_outputs(force=True)

        # Filter ou useless keys
        key_vals = {k[0]: v for k, v in key_vals_output.items() if k[2] == 'ALL'}

        #mod_str_ ="l2lr_C:10.000000"
        tvl2ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
        mods_str = ["enettv_%.3f:%.6f:%.6f" % (0.1, 0.01, tvl2ratio) for tvl2ratio in tvl2ratios]
        mods_str = mods_str + ["enettv_%.3f:%.6f:%.6f" % (0.1, 0.001, tvl2ratio) for tvl2ratio in tvl2ratios]
        #mods_str.append("l2lr_C:10.000000")

        prefix = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_coefmap" % mod_enettv_str, ext=None)

        clust_stats = list()
        titles = list()
        imgs =  list()
        coefs = list()
        for k, mod_str_ in enumerate(mods_str):
            try:
                coef = key_vals[mod_str_]['coef_img'].ravel()
                #arr = np.zeros(mask_arr.shape)
                #arr[mask_arr] = coef
                coefs.append(coef)
                #coefs_img = nilearn.image.new_img_like(mask_img, arr)
                # coefs_img.to_filename(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv_coefmap" % mod_str_, ext="nii.gz"))
                coef_t, t = arr_threshold_from_norm2_ratio(coef, .999)
                _, n_clusts, clust_sizes = arr_clusters(coef_t, mask_arr)

                coef_arr = np.zeros(mask_arr.shape)
                coef_arr[mask_arr] = coef
                coef_img = nilearn.image.new_img_like(mask_img, coef_arr)
                coef_img.to_filename(prefix.replace("enettv_range", mod_str_) + ".nii.gz")

                vmax = np.abs(coef).max()
                clust_stats.append(
                    [mod_str_, n_clusts, np.median(clust_sizes), vmax, np.sum(coef == 0), np.sum(coef_t == 0)])
                #imgs.append(coefs_img)
                titles.append(mod_str_ + ", nclusts:%i" % n_clusts)
            except:
                pass


        clust_stats = pd.DataFrame(clust_stats, columns=["mod", 'nclust', 'med_clust_sizes', "max", 'nnuls', 'nnuls_t'])
        clust_stats.to_csv(prefix+".csv", index=False)
        print(clust_stats)
        vmax = [np.abs(coefs).max() / 10 for coef in coefs]
        pdf = PdfPages(prefix+".pdf")
        imgs, fig, axes = img_plot_glass_brain(imgs=None, coefs=coefs, mask_img=mask_img, vmax=vmax, titles=titles, threshold=1e-06, cmap=plt.cm.bwr, plot_abs=False, colorbar=True)
        pdf.savefig(); plt.close(fig); pdf.close()

        print(prefix)

# cd /neurospin/psy_sbox/analyses/202008_start-icaar_cat12vbm_predict-transition
# fsleyes icaar-start_t1mri_mwp1_template.nii.gz icaar-start_t1mri_mwp1_gs-raw_enettv_*_coefmap.nii.gz



###############################################################################
#
# %% Plot weight maps
#
###############################################################################

if False:
    dataset = 'icaar-start'

    mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_arr = mask_img.get_fdata() != 0
    assert mask_arr.sum() == MSK_SIZE

    ###############################################################################
    # 6.1) ENETTV 5CV

    mod_str = "enettv_0.100:0.010000:1.000000"
    models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % "enettv_range", ext="pkl")
    # models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")

    prefix = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv_" % mod_str, ext=None)

    if not os.path.exists(prefix + "coefmap.nii.gz"):

        with open(models_filename, 'rb') as fd:
            key_vals_output = pickle.load(fd)

        # Refit all coef map
        coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
        arr_threshold_from_norm2_ratio(coef_map, .999)
        # threshold= 7.173373337622994e-05
        coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
             [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

        maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, thresh_norm_ratio=0.999, vmax=0.0005, prefix=prefix)

        # Cluster analysis
        cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.999 --thresh_size 10  --save_atlas" % (prefix + "coefmap.nii.gz")
        p = subprocess.run(cmd.split())


    ###############################################################################
    # 6.2) ENETTV Boot
    mod_str = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)

    #mod_str = "enettv_0.100:0.010000:1.000000"
    models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_boot" % mod_str, ext="pkl")
    models_filename_refit = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_str, ext="pkl")
    prefix = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_boot_" % mod_str, ext=None)

    if not os.path.exists(prefix + "coefmap.nii.gz"):

        # Refit all coef map
        with open(models_filename_refit, 'rb') as fd:
            key_vals_output = pickle.load(fd)
            coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()

        with open(models_filename, 'rb') as fd:
            key_vals_output = pickle.load(fd)

        #coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
        #arr_threshold_from_norm2_ratio(coef_map, .999)
        # threshold= 7.173373337622994e-05
        coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
             [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

        maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, thresh_norm_ratio=0.999, vmax=0.0005, prefix=prefix)

        # Cluster analysis
        cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.999 --thresh_size 10  --save_atlas" % (prefix + "coefmap.nii.gz")
        p = subprocess.run(cmd.split())

        # Cluster analysis on coefmap_mean95ci
        cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.999 --thresh_size 10  --save_atlas" % (prefix + "coefmap_mean95ci.nii.gz")
        p = subprocess.run(cmd.split())

    ###############################################################################
    # 6.2) L2LR

    mod_str = "l2lr_C:10.000000"

    models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % "l2lr_range", ext="pkl")
    prefix = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv_" % mod_str, ext=None)

    if not os.path.exists(prefix + "coefmap.nii.gz"):

        with open(models_filename, 'rb') as fd:
            key_vals_output = pickle.load(fd)

        # Refit all coef map
        coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
        arr_threshold_from_norm2_ratio(coef_map, .999)
        # threshold= 7.274653751750078e-05
        coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
             [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

        maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, thresh_norm_ratio=0.999, vmax=0.0005, prefix=prefix)

        # Cluster analysis
        cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.999 --thresh_size 10" % (prefix + "coefmap.nii.gz")
        p = subprocess.run(cmd.split())

    ###############################################################################
    # 6.2) L1LR

    mod_str = "l1lr_C:10.000000"
    models_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % "l1lr_range", ext="pkl")
    prefix = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv_" % mod_str, ext=None)

    if not os.path.exists(prefix + "coefmap.nii.gz"):

        with open(models_filename, 'rb') as fd:
            key_vals_output = pickle.load(fd)

        # Refit all coef map
        coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
        arr_threshold_from_norm2_ratio(coef_map, .999)
        # threshold= 7.274653751750078e-05
        coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
             [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

        maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, thresh_norm_ratio=0.999, vmax=0.0005, prefix=prefix)

        # Cluster analysis
        cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.999 --thresh_size 10" % (prefix + "coefmap.nii.gz")
        p = subprocess.run(cmd.split())


    ###############################################################################
    # 6.2) L2LR smoothed data (8 mm)

    mod_str = "l2lr_C:10.000000"
    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="s8_%s_5cv" % "l2lr_range", ext="pkl")
    prefix = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="s8_%s_5cv_" % mod_str, ext=None)

    if not os.path.exists(prefix + "coefmap.nii.gz"):

        mapreduce_sharedir = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="%s_5cv" % mod_enettv_str, ext=None)
        mp = MapReduce(shared_dir=mapreduce_sharedir)
        key_vals_output = mp.reduce_collect_outputs(force=True)

        with open(models_filename, 'rb') as fd:
            key_vals_output = pickle.load(fd)

        # Refit all coef map
        coef_map = key_vals_output[(mod_str, 'resdualizeYes', 'ALL')]['coef_img'].ravel()
        arr_threshold_from_norm2_ratio(coef_map, .999)
        # threshold= 7.274653751750078e-05
        coef_maps = np.vstack([key_vals_output[k]['coef_img'].ravel() for k in
             [k for k in key_vals_output.keys() if (k[0] == mod_str and  k[1] == "resdualizeYes" and k[2] != "ALL")]])

        maps = plot_coefmap_stats(coef_map, coef_maps, ref_img=mask_img, thresh_norm_ratio=0.999, vmax=0.0005, prefix=prefix)

        # Cluster analysis
        cmd = "/home/ed203246/git/nitk/nitk/image/image_clusters_analysis.py %s -t 0.99 --thresh_size 10" % (prefix + "coefmap.nii.gz")
        p = subprocess.run(cmd.split())

    # sys.exit("End")

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