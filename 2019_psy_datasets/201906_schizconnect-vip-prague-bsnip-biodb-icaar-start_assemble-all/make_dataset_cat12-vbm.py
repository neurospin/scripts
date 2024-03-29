#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:08:41 CET 2019

@author: edouard.duchesnay


nohup python ~/git/scripts/2019_psy_datasets/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/make_dataset_cat12-vbm.py &

float 64 vs float 32
https://en.wikipedia.org/wiki/IEEE_754

%load_ext autoreload
%autoreload 2



###############################################################################
# storage double 64 vs simple 32

for all datasets for all targets similar results (diff < 1%)
=> use simple precision

    'icaar-start'
    'schizconnect-vip'
    'bsnip'
    'biobd'

###############################################################################
# COMPARISION OF PRE-PROCESSING STRATEGIES


## schizconnect-vip inter-study

target	dataset	tag	model	mae_test
age	schizconnect-vip	raw-raw	RidgeCV_inter	4.92782552726035
age	schizconnect-vip	gs-raw		RidgeCV_inter	**5.01495242618661**
age	schizconnect-vip	gs-ctrsite	RidgeCV_inter	15.4597558193428

target	dataset		tag		model		bacc_test		auc_test
sex	schizconnect-vip	raw-raw	LRCVinter	0.909564297582952	0.966949583454671
sex	schizconnect-vip	raw-raw	LRCVnointer	0.892740120252839	0.961594213759985
sex	schizconnect-vip	gs-raw		LRCVinter	0.880714279596416	**0.953279480788667**
sex	schizconnect-vip	gs-raw		LRCVnointer	0.857585350394847	0.944485185375519
sex	schizconnect-vip	gs-ctrsite	LRCVinter	0.803415214896277	0.890699469254444
sex	schizconnect-vip	gs-ctrsite	LRCVnointer	0.792076930983093	0.885717048951927

target	dataset		tag		model		bacc_test		auc_test
dx	schizconnect-vip	raw-raw	LRCVinter	0.742121212121212	0.829972451790634
dx	schizconnect-vip	raw-raw	LRCVnointer	0.753636363636364	0.833994490358127
dx	schizconnect-vip	gs-raw		LRCVinter	0.747575757575757	**0.826005509641873**
dx	schizconnect-vip	gs-raw		LRCVnointer	0.759393939393939	0.830082644628099
dx	schizconnect-vip	gs-ctrsite	LRCVinter	0.752727272727273	0.815426997245179
dx	schizconnect-vip	gs-ctrsite	LRCVnointer	0.753636363636363	0.819504132231405


## biodb inter-study

target	dataset tag		model		mae_test
age	biobd	raw-raw	RidgeCV_inter	4.96452954053455
age	biobd	gs-raw		RidgeCV_inter	**5.04393558083351**
age	biobd	gs-ctrsite	RidgeCV_inter	28.6553804648908

target	dataset tag		model		bacc_test		auc_test
sex	biobd	raw-raw	LRCVinter	0.732239873878244	0.786298774796228
sex	biobd	raw-raw	LRCVnointer	0.72603484517746	0.784560995772492
sex	biobd	gs-raw		LRCVinter	0.70314085556021	0.764522052582931
sex	biobd	gs-raw		LRCVnointer	0.709167897750241	**0.767147191746998**
sex	biobd	gs-ctrsite	LRCVinter	0.638340719834778	0.647879885091867
sex	biobd	gs-ctrsite	LRCVnointer	0.630803670956472	0.660747412099583

target	dataset tag		model		bacc_test		auc_test
dx	biobd	raw-raw	LRCVinter	0.670077456926765	0.751997994784619
dx	biobd	raw-raw	LRCVnointer	0.691678138931716	0.756543546194022
dx	biobd	gs-raw		LRCVinter	0.664073872697375	0.73970722655458
dx	biobd	gs-raw		LRCVnointer	0.676360037538824	**0.743236846308475**
dx	biobd	gs-ctrsite	LRCVinter	0.497257870035892	0.51834903450907
dx	biobd	gs-ctrsite	LRCVnointer	0.496390157148307	0.520185176219784


### bnsip inter-study

target	dataset tag		model		mae_test
age	bsnip	raw-raw	RidgeCV_inter	5.08211339671643
age	bsnip	gs-raw		RidgeCV_inter	**5.04579017671471**
age	bsnip	gs-ctrsite	RidgeCV_inter	9.16294421271349

target	dataset tag		model		bacc_test		auc_test
sex	bsnip	raw-raw	LRCVinter	0.913394990701162	0.972804536275248
sex	bsnip	raw-raw	LRCVnointer	0.910831568074446	0.968940483703162
sex	bsnip	gs-raw		LRCVinter	0.915353399405849	0.972788387112606
sex	bsnip	gs-raw		LRCVnointer	0.900731847617483	**0.968696247622391**
sex	bsnip	gs-ctrsite	LRCVinter	0.901883299628491	0.964367624787489
sex	bsnip	gs-ctrsite	LRCVnointer	0.89734738001085	0.965216524494645

target	dataset tag		model		bacc_test		auc_test
dx	bsnip	raw-raw	LRCVinter	0.737908232118758	0.807020917678813
dx	bsnip	raw-raw	LRCVnointer	0.735344129554656	0.808306342780027
dx	bsnip	gs-raw		LRCVinter	0.732908232118758	0.800570175438596
dx	bsnip	gs-raw		LRCVnointer	0.725472334682861	**0.798890013495277**
dx	bsnip	gs-ctrsite	LRCVinter	0.672078272604588	0.768694331983805
dx	bsnip	gs-ctrsite	LRCVnointer	0.677334682860999	0.766622807017544


### icaar-start inter-study

target	dataset	tag		model		mae_test
age	icaar-start	raw-raw	RidgeCV_inter	1.96619730303429
age	icaar-start	gs-raw		RidgeCV_inter	**1.93729897446262**
age	icaar-start	gs-ctrsite	RidgeCV_inter	1.86460950477642

target	dataset	tag		model		bacc_test		auc_test
sex	icaar-start	raw-raw	LRCVinter	0.816391941391941	0.901709401709402
sex	icaar-start	raw-raw	LRCVnointer	0.817307692307692	0.90030525030525
sex	icaar-start	gs-raw		LRCVinter	0.731593406593406	0.875457875457876
sex	icaar-start	gs-raw		LRCVnointer	0.801373626373626	**0.880830280830281**
sex	icaar-start	gs-ctrsite	LRCVinter	0.768956043956044	0.890659340659341
sex	icaar-start	gs-ctrsite	LRCVnointer	0.794139194139194	0.885225885225885

target	dataset	tag		model		bacc_test		auc_test
dx	icaar-start	raw-raw	LRCVinter	0.644688644688645	0.621367521367521
dx	icaar-start	raw-raw	LRCVnointer	0.50018315018315	0.598534798534798
dx	icaar-start	gs-raw		LRCVinter	0.631043956043956	0.696550671550672
dx	icaar-start	gs-raw		LRCVnointer	0.580769230769231	**0.696916971916972**
dx	icaar-start	gs-ctrsite	LRCVinter	0.640018315018315	0.685286935286935
dx	icaar-start	gs-ctrsite	LRCVnointer	0.618131868131868	0.689896214896215


###############################################################################
## raw vs global scaling
## ---------------------

### Intra

                   age          diag(AUC)
'icaar-start'      1.96 1.93    62 69(+7)
'schizconnect-vip' 4.9  5       74 75
'biobd'            4.9  5       67 66
'bsnip'            5    5       73 73

No big diff

### Inter

AGE-SEX: BIOBD+BSNIP+PRAGUE+SCHIZCONNECT-VIP_t1mri_mwp1_ml-scores_predict-AGE-SEX
SCZvsCTL: BSNIP+PRAGUE+SCHIZCONNECT-VIP_t1mri_mwp1_ml-scores_predict-SCZvsCTL
BDvsCTL: BIOBD+BSNIP_t1mri_mwp1_ml-scores_predict-BDvsCTL

AGE(MAE)             5.3 5.5
SEX(AUC)             89  89
SCZvsCTL(AUC)        77  77
BDvsCTL(AUC)         64  64

Similar

**CONCLUSION:** No big diff, by default do glob scale. Because we want to normalize for TIV


###############################################################################
## raw vs ressite
## --------------

### Intra

                   age            sex(AUC)       diag (AUC)
'icaar-start'      1.9 1.8        87 89(+2)      69 68
'schizconnect-vip' 5   15 (+10)   95 89(-6)      82 81
'biobd'            5   28(+24)    76 64(-8)      74 51(-23)
'bsnip'            5 9 (+4)       91 90          73 67(+6)

ressite is BAD !!! DO NOT USE !!!

**CONCLUSION:** ctrsite is BAD !!! DO NOT USE !!!

## ctrsite vs ressite
## ------------------

**CONCLUSION:** for all datasets for all targets same results

###############################################################################
### Inter

AGE(MAE)             5.3 8.3 (-3 bad)
SEX(AUC)             89  88
SCZvsCTL(AUC)        77  78
BDvsCTL(AUC)         64  62 (-2 bad)

###############################################################################
## ressite vs site(age+sex+diag)
## -----------------------------
# BIASED Anyway DO NOT USE

perf of adj

### Intra
                   age            sex(AUC)   diag (AUC)
'icaar-start'      1.86 1.87      75 73      69 75(+6)
'schizconnect-vip' 15   4.8 (-10) 80 89(+9)  81 84(+3)
'biobd'            28   4.5(-24)  65 79(+14) 51 80(+30)
'bsnip'            9    5 (-4)    96 97(+1)  76 81(+5)

=> always used adjusted residualization is better

### Inter

AGE(MAE)             8.3 6.9 (-1.5 good)
SEX(AUC)             89  88
SCZvsCTL(AUC)        78  72 (-6 bad)
BDvsCTL(AUC)         62  62

=> Not sot good


**CONCLUSION:** adjusted residualization is better on intra-study but not for inter-study

## raw-raw vs gs-site(age+sex+diag)
## --------------------------------

### Intra

                   age            sex(AUC)   diag (AUC)
'icaar-start'      1.9 1.8        89 89      62 75 (+13)
'schizconnect-vip' 4.9 4.8        90 90      83 84
'biobd'            5   4.5        78 79      75 80(+5)
'bsnip'            5   4.9        97 97      80 81


=> Adjusted residualization always better in half cases much better results for diag

### Inter

AGE(MAE)             5.3 6.9 (+1.5 bad)
SEX(AUC)             89  89
SCZvsCTL(AUC)        77  72 (-5 bad)
BDvsCTL(AUC)         64  62 (-2 bad)

=> bad

**CONCLUSION:**

# Intercept vs no intercept

gs-res:site(age+sex+diag)
Regression RidgeCV alpha large => large regularization
Classif: [C] large => low regularization

### Intra

                   age                    sex(AUC)             diag (AUC)
'icaar-start'      1.8[0.1] 21[10]        88[21]  88[1e-4]     75[21]   73[1e-4] (+2)
'schizconnect-vip' 4.9[10]  32[10]        96[166] 95[1e-4]     84[1200] 84[7e-5]
'biobd'            4.5[0.1] 39[10]        79[21]  79[1e-4]     80[166]  79[7e-5]
'bsnip'            5  [0.1] 38[10]        97[166]  97[1e-4]    81[166]  81[7e-5]

Regression: interpect
Classif: No big diff but models with no interpect are more regularized


"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import scipy, scipy.ndimage
#import xml.etree.ElementTree as ET
import re
# import glob
import seaborn as sns

# for the ROIs
BASE_PATH_icaar = '/neurospin/psy/start-icaar-eugei/derivatives/cat12'
BASE_PATH_schizconnect = '/neurospin/psy/schizconnect-vip-prague/derivatives/cat12'
BASE_PATH_bsnip = '/neurospin/psy/bsnip1/derivatives/cat12'
BASE_PATH_biobd = '/neurospin/psy/bipolar/biobd/derivatives/cat12'

# 1) Inputs: phenotype
PHENOTYPE_CSV = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv"
# for the phenotypes
# INPUT_CSV_icaar_bsnip_biobd = '/neurospin/psy_sbox/start-icaar-eugei/phenotype'
#INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
#INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'

# 3) Output
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/'

def OUTPUT(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None, type=None, ext=None):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(OUTPUT_PATH, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)


"""
OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv")
OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz")
OUTPUT(dataset, scaling='raw', harmo='raw', type="data64", ext="npy")
OUTPUT(dataset, scaling='gs', harmo='raw', type="data64", ext="npy")
OUTPUT(dataset, scaling='gs', harmo='ctrsite', type="data64", ext="npy")
OUTPUT(dataset, scaling='gs', harmo='ressite', type="data64", ext="npy")
OUTPUT(dataset, scaling='gs', harmo='adjsite', type="data64", ext="npy")
"""
########################################################################################################################
# CLINICAL DATA
########################################################################################################################

########################################################################################################################
# Read phenotypes

phenotypes = pd.read_csv(PHENOTYPE_CSV, sep='\t')
assert phenotypes.shape == (3871, 46)
# rm subjects with missing age or site
phenotypes = phenotypes[phenotypes.sex.notnull() & phenotypes.age.notnull()]
assert phenotypes.shape == (2711, 46)

########################################################################################################################
# Neuroimaging niftii and TIV
# mwp1 files
check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)) # excpected image dimensions

ni_icaar_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-V1/mri/mwp1*.nii")
ni_schizconnect_filenames = glob.glob("/neurospin/psy/schizconnect-vip-prague/derivatives/cat12/vbm/sub-*/mri/mwp1*.nii")
ni_bsnip_filenames = glob.glob("/neurospin/psy/bsnip1/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")
ni_biobd_filenames = glob.glob("/neurospin/psy/bipolar/biobd/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")

tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_schizconnect = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

assert tivo_icaar.shape == (171, 6)
assert len(ni_icaar_filenames) == 171

assert tivo_schizconnect.shape == (738, 6)
assert len(ni_schizconnect_filenames) == 738

assert tivo_bsnip.shape == (1042, 6)
assert len(ni_bsnip_filenames) == 1042

assert tivo_biobd.shape == (746, 6)
assert len(ni_biobd_filenames) == 746

########################################################################################################################
# FIX some issues: duplicated subjects

# 1) Remove subjects from biobd subject dublicated in schizconnect(vip)
# Duplicated between schizconnect and biobd
df = tivo_biobd.append(tivo_schizconnect)
duplicated_in_biobd =  df["participant_id"][df.iloc[:, 1:].duplicated(keep='last')]
assert len(duplicated_in_biobd) == 14
tivo_biobd = tivo_biobd[np.logical_not(tivo_biobd.participant_id.isin(duplicated_in_biobd))]
assert tivo_biobd.shape == (732, 6)

# 2) Remove dublicated subject from bsnip with inconsistant sex and age
# cd /neurospin/psy/bsnip1/sourcedata
# fslview sub-INVVV2WYKK6/ses-V1/anat/sub-INVVV2WYKK6_ses-V1_acq-1.2_T1w.nii.gz sub-INVXR8L3WRZ/ses-V1/anat/sub-INVXR8L3WRZ_ses-V1_acq-1.2_T1w.nii.gz  &
# Same image

df = tivo_bsnip
df.iloc[:, 1:].duplicated().sum() == 1
duplicated_in_bsnip = df[df.iloc[:, 1:].duplicated(keep=False)]["participant_id"]
print(phenotypes[phenotypes.participant_id.isin(duplicated_in_bsnip)][["participant_id",  "sex",   "age"]])
tivo_bsnip = tivo_bsnip[np.logical_not(tivo_bsnip.participant_id.isin(duplicated_in_bsnip))]
assert tivo_bsnip.shape == (1040, 6)

tivo = pd.concat([tivo_icaar, tivo_schizconnect, tivo_bsnip, tivo_biobd], ignore_index=True)
assert tivo.shape == (2681, 6)

########################################################################################################################
# Merge phenotypes with TIV

participants_df = pd.merge(phenotypes, tivo, on="participant_id")
assert participants_df.shape == (2642, 51)

# Check missing in phenotypes
assert len(set(tivo_icaar.participant_id).difference(set(phenotypes.participant_id))) == 4
# set(tivo_icaar.participant_id).difference(set(phenotypes.participant_id))
# Out[8]: {'5EU31000', 'ICAAR004', 'ICAAR047', 'SLBG3TPILOTICAAR'}
assert len(set(tivo_schizconnect.participant_id).difference(set(phenotypes.participant_id))) == 0
assert len(set(tivo_bsnip.participant_id).difference(set(phenotypes.participant_id))) == 0
assert len(set(tivo_biobd.participant_id).difference(set(phenotypes.participant_id))) == 35
"""
set(tivo_biobd.participant_id).difference(set(phenotypes.participant_id))
"""

########################################################################################################################
# VBM DATASETS
########################################################################################################################

########################################################################################################################
# Some global params

def do_ml(NI_arr, NI_participants_df, mask_arr, tag, dataset):
    """
    Machine learning for sex, age and DX
    """
    import sklearn.metrics as metrics
    import sklearn.ensemble
    import sklearn.linear_model as lm

    def balanced_acc(estimator, X, y, **kwargs):
        return metrics.recall_score(y, estimator.predict(X), average=None).mean()

    # estimators_clf = dict(LogisticRegressionCV_balanced_inter=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5),
    #                      gbc=sklearn.ensemble.GradientBoostingClassifier())
    # or
    estimators_clf = dict(LogisticRegressionCV_balanced_inter=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5),
                          LogisticRegressionCV_balanced_nointer=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5,
                                                                                        fit_intercept=False))
    estimators_reg = dict(RidgeCV_inter=lm.RidgeCV(), RidgeCV_nointer=lm.RidgeCV(fit_intercept=False))

    ml_age_, _, _ = ml_predictions(X=NI_arr, y=NI_participants_df["age"].values,
                                   estimators=estimators_reg, cv=None, mask_arr=mask_arr)
    ml_age_.insert(0, "tag", tag);
    ml_age_.insert(0, "dataset", dataset);
    ml_age_.insert(0, "target", "age");

    ml_sex_, _, _ = ml_predictions(X=NI_arr, y=NI_participants_df["sex"].astype(int).values,
                                   estimators=estimators_clf, cv=None, mask_arr=mask_arr)
    ml_sex_.insert(0, "tag", tag);
    ml_sex_.insert(0, "dataset", dataset);
    ml_sex_.insert(0, "target", "sex");

    if dataset == 'icaar-start':
        msk = NI_participants_df["diagnosis"].isin(['UHR-C', 'UHR-NC'])
        dx = NI_participants_df[msk]["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0}).values
        NI_arr_ = NI_arr[msk]
    elif dataset == 'schizconnect-vip':
        msk = NI_participants_df["diagnosis"].isin(['control', 'schizophrenia']) & NI_participants_df["study"].isin(['SCHIZCONNECT-VIP'])
        dx = NI_participants_df[msk]["diagnosis"].map({'schizophrenia': 1, 'control': 0}).values
        NI_arr_ = NI_arr[msk]
    elif dataset == 'bsnip':
        msk = NI_participants_df["diagnosis"].isin(['control', 'schizophrenia'])
        dx = NI_participants_df[msk]["diagnosis"].map({'schizophrenia': 1, 'control': 0}).values
        NI_arr_ = NI_arr[msk]
    elif dataset == 'biobd':
        msk = NI_participants_df["diagnosis"].isin(['control', 'bipolar disorder'])
        dx = NI_participants_df[msk]["diagnosis"].map({'bipolar disorder': 1, 'control': 0}).values
        NI_arr_ = NI_arr[msk]

    print(NI_participants_df[msk]["diagnosis"].describe())
    ml_dx_, _, _ = ml_predictions(X=NI_arr_, y=dx, estimators=estimators_clf, cv=None, mask_arr=mask_arr)
    ml_dx_.insert(0, "tag", tag);
    ml_dx_.insert(0, "dataset", dataset);
    ml_dx_.insert(0, "target", "dx");
    return ml_age_, ml_sex_, ml_dx_

########################################################################################################################
# Load images, intersect with pop and do preprocessing qnd dump 5d npy

datasets = {
    'icaar-start': dict(ni_filenames = ni_icaar_filenames),
    'schizconnect-vip': dict(ni_filenames = ni_schizconnect_filenames),
    'bsnip': dict(ni_filenames = ni_bsnip_filenames),
    'biobd': dict(ni_filenames = ni_biobd_filenames)}


for dataset in datasets:
    print("###########################################################################################################")
    print("#", dataset)
    # dataset = 'icaar-start'
    # dataset = 'schizconnect-vip'
    # dataset = 'bsnip'
    # dataset = 'biobd'
    NI_filenames = datasets[dataset]['ni_filenames']

    ml_age_l = list()
    ml_sex_l = list()
    ml_dx_l = list()

    ########################################################################################################################
    print("# 1) Read images")
    scaling, harmo = 'raw', 'raw'

    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df)
    NI_participants_df.to_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"), index=False)
    # Save (reload data in memory mapping)
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    """
    #NI_arr = np.load(OUTPUT(dataset, scaling='raw', harmo='raw', type="data64", ext="npy"))
    NI_arr = np.load(OUTPUT(dataset, scaling='raw', harmo='raw', type="data64", ext="npy"), mmap_mode='r')

    NI_participants_df = pd.read_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
    ref_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_img = ref_img
    """

    # Compute mask
    mask_img = preproc.compute_brain_mask(NI_arr, ref_img, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10,
                               verbose=1)
    mask_arr = mask_img.get_data() > 0
    mask_img.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))

    ########################################################################################################################
    print("# 2) Raw data")
    # Univariate stats

    # design matrix: Set missing diagnosis to 'unknown' to avoid missing data(do it once)
    dmat_df = NI_participants_df[['age', 'sex', 'diagnosis', 'tiv', 'site']]
    dmat_df.loc[:, "sex"] = dmat_df.sex.astype(int).astype('object')
    dmat_df.loc[dmat_df.diagnosis.isnull(), 'diagnosis'] = 'unknown'
    assert np.all(dmat_df.isnull().sum() == 0)

    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)

    # %time univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)

    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=scaling + '-' + harmo, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)
    # ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr.astype('float32'), NI_participants_df, mask_arr, tag=scaling + '-' + harmo + '-' + "x32", dataset=dataset)
    # ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    print("# 3) Global scaling")
    scaling, harmo = 'gs', 'raw'

    NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)
    # Save
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    # Univariate stats
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=scaling + '-' + harmo, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)
    # ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr.astype('float32'), NI_participants_df, mask_arr, tag=scaling + '-' + harmo + '-' + "x32", dataset=dataset)
    # ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    # Keep reference on this mmmaped data
    NI_arr_gs = NI_arr

    ########################################################################################################################
    print("# 4) Site-harmonization Center by site")
    scaling, harmo = 'gs', 'ctrsite'

    NI_arr = preproc.center_by_site(NI_arr_gs, site=NI_participants_df.site)

    # Save
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), NI_arr.astype('float32'))
    NI_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), mmap_mode='r')

    # Univariate stats
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=scaling + '-' + harmo, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)
    # ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr.astype('float32'), NI_participants_df, mask_arr, tag=scaling + '-' + harmo + '-' + "x32", dataset=dataset)
    # ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    print("# 5) Harmonization residualize on site")
    scaling, harmo = 'gs', 'res:site'

    Yres = residualize(Y=NI_arr_gs.squeeze()[:, mask_arr], formula_res="site", data=dmat_df)
    #Yadj = residualize(Y=NI_arr_gs.squeeze()[:, mask_arr], formula_res="site", data=dmat_df, formula_full="age + sex + diagnosis + site")
    NI_arr = np.zeros(NI_arr_gs.shape)
    NI_arr[:, 0, mask_arr] = Yres
    del Yres

    # Save
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), NI_arr.astype('float32'))
    NI_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), mmap_mode='r')

    # Univariate stats
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=scaling + '-' + harmo, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)
    # ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr.astype('float32'), NI_participants_df, mask_arr, tag=scaling + '-' + harmo + '-' + "x32", dataset=dataset)
    # ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    print("# 6) Harmonization res:site adjusted for (age+sex+diag)")
    scaling, harmo = 'gs', 'res:site(age+sex+diag)'

    Yadj = residualize(Y=NI_arr_gs.squeeze()[:, mask_arr], formula_res="site", data=dmat_df, formula_full="site + age + sex + diagnosis")
    NI_arr = np.zeros(NI_arr_gs.shape)
    NI_arr[:, 0, mask_arr] = Yadj
    del Yadj

    # Save
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), NI_arr.astype('float32'))
    NI_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), mmap_mode='r')

    # Univariate stats
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=scaling + '-' + harmo, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)
    # ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr.astype('float32'), NI_participants_df, mask_arr, tag=scaling + '-' + harmo + '-' + "x32", dataset=dataset)
    # ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    print("# 7) Harmonization res:age+sex+site adjusted for diag")
    scaling, harmo = 'gs', 'res:site+age+sex(diag)'

    Yadj = residualize(Y=NI_arr_gs.squeeze()[:, mask_arr], formula_res="site + age + sex", data=dmat_df, formula_full="site + age + sex + diagnosis")
    NI_arr = np.zeros(NI_arr_gs.shape)
    NI_arr[:, 0, mask_arr] = Yadj
    del Yadj

    # Save
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), NI_arr.astype('float32'))
    NI_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), mmap_mode='r')

    # Univariate stats
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=scaling + '-' + harmo, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)
    # ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr.astype('float32'), NI_participants_df, mask_arr, tag=scaling + '-' + harmo + '-' + "x32", dataset=dataset)
    # ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    #np.max(np.abs(NI_arr.astype('float32') - NI_arr))
    ########################################################################################################################
    # Save ML
    ml_age_df = pd.concat(ml_age_l)
    ml_sex_df = pd.concat(ml_sex_l)
    ml_dx_df = pd.concat(ml_dx_l)

    with pd.ExcelWriter(OUTPUT(dataset, scaling=None, harmo=None, type="ml-scores", ext="xlsx")) as writer:
        ml_age_df.to_excel(writer, sheet_name='age', index=False)
        ml_sex_df.to_excel(writer, sheet_name='sex', index=False)
        ml_dx_df.to_excel(writer, sheet_name='dx', index=False)

    del NI_arr

########################################################################################################################
# Make some public datasets
########################################################################################################################

# schizconnect ONLY

# Read clinical data
dataset = 'schizconnect-vip'

df = pd.read_csv(OUTPUT(dataset=dataset, scaling=None, harmo=None, type="participants", ext="csv"))
mask = np.logical_not(df['site'].isin(['PRAGUE', 'vip']))
df = df[mask]
df = df[['participant_id', 'sex', 'age', 'diagnosis', 'study', 'site', 'tiv', 'gm', 'wm', 'csf', 'wmh']]

scaling, harmo = 'gs', 'raw'
NI_arr = np.load(OUTPUT(dataset=dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')[mask]
mask_img = nibabel.load(OUTPUT(dataset='schizconnect-vip', scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_data() != 0

# Save at 1.5mm
mask_img.to_filename(OUTPUT(dataset="public/" + 'schizconnect_1.5mm', scaling=None, harmo=None, type="mask", ext="nii.gz"))
df.to_csv(OUTPUT(dataset="public/" + 'schizconnect_1.5mm', scaling=None, harmo=None, type="participants", ext="csv"), index=False)
np.save(OUTPUT(dataset="public/" + 'schizconnect_1.5mm', scaling=scaling, harmo=harmo, type="data32", ext="npy"), NI_arr.astype('float32'))

# Univariate stats
df[["age", "sex", "diagnosis", "tiv", "site"]].isnull().sum()
univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=df)
pdf_filename = OUTPUT("public/" + 'schizconnect_1.5mm', scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
plot_univ_stats(univstats, mask_img, data=df[["age", "sex", "diagnosis", "tiv", "site"]],
                grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                skip_intercept=True)

# Downsample at 3mm
import brainomics.image_resample
mask_3mm_img = brainomics.image_resample.down_sample(src_img=mask_img, factor=2)
# Binarrize mask
mask_3mm_arr = mask_3mm_img.get_data() > 1e-3
mask_3mm_img = nilearn.image.new_img_like(mask_3mm_img, data=mask_3mm_arr)

NI_arr_3mm = np.concatenate([
    brainomics.image_resample.down_sample(src_img=nilearn.image.new_img_like(mask_img, data=NI_arr[i, 0, :]),
                                          factor=2).get_data()[np.newaxis, np.newaxis, :] for i in range(NI_arr.shape[0])])
mask_3mm_img.to_filename(OUTPUT(dataset="public/" + 'schizconnect_3mm', scaling=None, harmo=None, type="mask", ext="nii.gz"))
df.to_csv(OUTPUT(dataset="public/" + 'schizconnect_3mm', scaling=None, harmo=None, type="participants", ext="csv"), index=False)
np.save(OUTPUT(dataset="public/" + 'schizconnect_3mm', scaling=scaling, harmo=harmo, type="data32", ext="npy"), NI_arr_3mm.astype('float32'))

# Univariate stats
univmods, univstats = univ_stats(NI_arr_3mm.squeeze()[:, mask_3mm_arr], formula="age + sex + diagnosis + tiv + site", data=df)
pdf_filename = OUTPUT("public/" + 'schizconnect_3mm', scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
plot_univ_stats(univstats, mask_3mm_img, data=df[["age", "sex", "diagnosis", "tiv", "site"]],
                grand_mean=NI_arr_3mm.squeeze()[:, mask_3mm_arr].mean(axis=1), pdf_filename=pdf_filename, thres_nlpval=3,
                skip_intercept=True)

########################################################################################################################
# INTER-STUDIES
########################################################################################################################

def ml_predictions_warpper(X, df, targets_reg, targets_clf, cv=None, mask_arr=None, tag_name=None, dataset_name=None):
    """
    Machine learning for sex, age and DX
    """
    import sklearn.metrics as metrics
    import sklearn.ensemble
    import sklearn.linear_model as lm

    def balanced_acc(estimator, X, y, **kwargs):
        return metrics.recall_score(y, estimator.predict(X), average=None).mean()

    scores_ml = dict()

    # Regression targets
    estimators_reg = dict(RidgeCV_inter=lm.RidgeCV(), RidgeCV_nointer=lm.RidgeCV(fit_intercept=False))
    #estimators_reg = dict(RidgeCV_inter=lm.RidgeCV())

    for target in targets_reg:
        ml_, ml_folds_,  _ = ml_predictions(X=X, y=df[target].values,
                                       estimators=estimators_reg, cv=cv, mask_arr=mask_arr)
        ml_folds_.insert(0, "tag", tag_name);
        ml_folds_.insert(0, "dataset", dataset_name);
        ml_folds_.insert(0, "target", target);
        ml_.insert(0, "tag", tag_name);
        ml_.insert(0, "dataset", dataset_name);
        ml_.insert(0, "target", target);

        scores_ml[target + "_folds"] = ml_folds_
        scores_ml[target] = ml_

    # Classification targets
    estimators_clf = dict(
        LogisticRegressionCV_balanced_inter=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1,
                                                                    cv=5),
        LogisticRegressionCV_balanced_nointer=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc,
                                                                      n_jobs=1, cv=5,
                                                                      fit_intercept=False))
    for target in targets_clf:
        ml_, ml_folds_,  _ = ml_predictions(X=X, y=df[target].values,
                                       estimators=estimators_clf, cv=cv, mask_arr=mask_arr)
        ml_folds_.insert(0, "tag", tag_name);
        ml_folds_.insert(0, "dataset", dataset_name);
        ml_folds_.insert(0, "target", target);
        ml_.insert(0, "tag", tag_name);
        ml_.insert(0, "dataset", dataset_name);
        ml_.insert(0, "target", target);

        scores_ml[target + "_folds"] = ml_folds_
        scores_ml[target] = ml_

    return scores_ml

# Leave out study CV
from sklearn.model_selection import BaseCrossValidator
class CVIterableWrapper(BaseCrossValidator):
    """Wrapper class for old style cv objects and iterables."""
    def __init__(self, cv):
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        for train, test in self.cv:
            yield train, test


########################################################################################################################
# Age, Sex : leave study out on CTL of on schizconnect-vip bsnip biobd
# 'icaar-start', 'schizconnect-vip', 'bsnip', 'biobd'

datasets = ['schizconnect-vip', 'bsnip', 'biobd']

# Read clinical data
df = pd.concat([pd.read_csv(OUTPUT(dataset=dataset, scaling=None, harmo=None, type="participants", ext="csv")) for dataset in datasets], axis=0)
mask = df['diagnosis'].isin(['control'])
df = df[mask]

# Leave study out CV
studies = np.sort(df["study"].unique())
# array(['BIOBD', 'BSNIP', 'PRAGUE', 'SCHIZCONNECT-VIP'], dtype=object)
folds = [[np.where(df["study"] != s)[0], np.where(df["study"] == s)[0]] for s in studies]
cv = CVIterableWrapper(folds)

# Merge masks
mask_filenames = glob.glob(OUTPUT("*", scaling=None, harmo=None, type="mask", ext="nii.gz"))
print([np.sum(nibabel.load(mask_filename).get_data()) for mask_filename in mask_filenames])
# [364610, 368680, 365280, 362619]
#
mask_arr = np.sum(np.concatenate([np.expand_dims(nibabel.load(mask_filename).get_data() > 0, axis=0) for mask_filename
                                  in mask_filenames]), axis=0) > (len(mask_filenames) - 1)
# 360348
mask_img = nilearn.image.new_img_like(mask_filenames[0], data=mask_arr)

print(mask_arr.sum())
mask_img.to_filename(OUTPUT("inter_studies/" + "+".join(studies), scaling=None, harmo=None, type="mask", ext="nii.gz"))


scores_ml = dict()

settings = [
    ['raw', 'raw', "data64"],
    ['gs', 'raw', "data64"],
    ['gs', 'ctrsite', "data32"],
    ['gs', 'res:site', "data32"],
    ['gs', 'res:site(age+sex+diag)', "data32"]]

for scaling, harmo, datatype in settings:
    print(scaling, harmo, datatype)
    # scaling, harmo, datatype = 'raw', 'raw', "data64"
    # scaling, harmo, datatype = 'gs', 'res:site(age+sex+diag)', "data32"

    NI_arr = np.concatenate([np.load(OUTPUT(dataset=dataset, scaling=scaling, harmo=harmo, type=datatype, ext="npy"), mmap_mode='r') for dataset in datasets])[mask]
    scores_ml_ = ml_predictions_warpper(X=NI_arr, df=df, targets_reg=["age"], targets_clf=["sex"], cv=cv, mask_arr=mask_arr,
                                       tag_name=scaling + "-" + harmo, dataset_name="+".join(studies))

    for key, dat in scores_ml_.items():
        if "_folds" in key:
            dat["fold"] = dat["fold"].map({"CV%i" % i: studies[i] for i in range(len(studies))})
        scores_ml[(scaling, harmo, key)] = dat


scalings, harmos, targets = zip(*[[scaling, harmo, target] for scaling, harmo, target in scores_ml])
scalings, harmos, targets = set(scalings), set(harmos), set(targets)
scores_ml_bytarget = {name:[] for name in targets}

for key, dat in scores_ml.items():
    scores_ml_bytarget[key[2]].append(dat)

scores_ml_bytarget = {key:pd.concat(dats) for key, dats in scores_ml_bytarget.items()}

with pd.ExcelWriter(OUTPUT("inter_studies/" + "+".join(studies), scaling=None, harmo=None, type="ml-scores", ext="xlsx")) as writer:
    for key, dat in scores_ml_bytarget.items():
        dat.to_excel(writer, sheet_name=key, index=False)

########################################################################################################################
# SCZ (schizconnect-vip <=> bsnip)

datasets = ['schizconnect-vip', 'bsnip']

# Read clinical data
df = pd.concat([pd.read_csv(OUTPUT(dataset=dataset, scaling=None, harmo=None, type="participants", ext="csv")) for dataset in datasets], axis=0)
mask = df['diagnosis'].isin(['schizophrenia', 'FEP', 'control'])
df = df[mask]
# FEP of PRAGUE becomes 1
df["diagnosis"] = df["diagnosis"].map({'schizophrenia': 1, 'FEP':1, 'control': 0}).values

# Leave study out CV
studies = np.sort(df["study"].unique())
# array(['BIOBD', 'BSNIP', 'PRAGUE', 'SCHIZCONNECT-VIP'], dtype=object)
folds = [[np.where(df["study"] != s)[0], np.where(df["study"] == s)[0]] for s in studies]
cv = CVIterableWrapper(folds)

# Check all sites have both labels
print([[studies[i], np.unique(df["diagnosis"].values[te])] for i, (tr, te) in enumerate(cv.split(None, df["diagnosis"].values))])

# Merge masks
mask_filenames = glob.glob(OUTPUT("*", scaling=None, harmo=None, type="mask", ext="nii.gz"))
print([np.sum(nibabel.load(mask_filename).get_data()) for mask_filename in mask_filenames])
# [364610, 368680, 365280, 362619]
#
mask_arr = np.sum(np.concatenate([np.expand_dims(nibabel.load(mask_filename).get_data() > 0, axis=0) for mask_filename
                                  in mask_filenames]), axis=0) > (len(mask_filenames) - 1)
# 360348
mask_img = nilearn.image.new_img_like(mask_filenames[0], data=mask_arr)

print(mask_arr.sum())
mask_img.to_filename(OUTPUT("inter_studies/" + "+".join(studies), scaling=None, harmo=None, type="mask", ext="nii.gz"))


scores_ml = dict()

settings = [
    ['raw', 'raw', "data64"],
    ['gs', 'raw', "data64"],
    ['gs', 'ctrsite', "data32"],
    ['gs', 'res:site', "data32"],
    ['gs', 'res:site(age+sex+diag)', "data32"],
    ['gs', 'res:site+age+sex(diag)', "data32"]]

for scaling, harmo, datatype in settings:
    print(scaling, harmo, datatype)
    # scaling, harmo, datatype = 'raw', 'raw', "data64"
    # 'raw', "data64"

    # scaling, harmo, datatype = 'gs', 'res:site(age+sex+diag)', "data32"

    NI_arr = np.concatenate([np.load(OUTPUT(dataset=dataset, scaling=scaling, harmo=harmo, type=datatype, ext="npy"), mmap_mode='r') for dataset in datasets])[mask]
    scores_ml_ = ml_predictions_warpper(X=NI_arr, df=df, targets_reg=[], targets_clf=["diagnosis"], cv=cv, mask_arr=mask_arr,
                                       tag_name=scaling + "-" + harmo, dataset_name="+".join(studies))

    for key, dat in scores_ml_.items():
        if "_folds" in key:
            dat["fold"] = dat["fold"].map({"CV%i" % i: studies[i] for i in range(len(studies))})
        scores_ml[(scaling, harmo, key)] = dat


scalings, harmos, targets = zip(*[[scaling, harmo, target] for scaling, harmo, target in scores_ml])
scalings, harmos, targets = set(scalings), set(harmos), set(targets)
scores_ml_bytarget = {name:[] for name in targets}

for key, dat in scores_ml.items():
    scores_ml_bytarget[key[2]].append(dat)

scores_ml_bytarget = {key:pd.concat(dats) for key, dats in scores_ml_bytarget.items()}

with pd.ExcelWriter(OUTPUT("inter_studies/" + "+".join(studies), scaling=None, harmo=None, type="ml-scores", ext="xlsx")) as writer:
    for key, dat in scores_ml_bytarget.items():
        dat.to_excel(writer, sheet_name=key, index=False)

########################################################################################################################
# BD (biobd <=> bsnip)

datasets = ['biobd', 'bsnip']

bsnip = pd.read_csv(OUTPUT(dataset='bsnip', scaling=None, harmo=None, type="participants", ext="csv"))
pd.DataFrame([[l, np.sum(bsnip["diagnosis"] == l)] for l in bsnip["diagnosis"].unique()])
"""
                                                   0    1
0  relative of proband with schizoaffective disorder  123
1                                            control  200
2  relative of proband with psychotic bipolar dis...  119
3                           schizoaffective disorder  112
4                                      schizophrenia  194
5             relative of proband with schizophrenia  175
6                         psychotic bipolar disorder  117

"""

########################################################################################################################
# BD (biobd <=> bsnip (schizoaffective disorder))
# => in bsnip consider "psychotic bipolar disorder"

suffix = ""

df = pd.concat([pd.read_csv(OUTPUT(dataset=dataset, scaling=None, harmo=None, type="participants", ext="csv")) for dataset in datasets], axis=0)
pd.DataFrame([[l, np.sum(df["diagnosis"] == l)] for l in df["diagnosis"].unique()])


mask = df['diagnosis'].isin(['bipolar disorder', 'psychotic bipolar disorder', 'control'])
df = df[mask]
print(pd.DataFrame([[l, np.sum(df["diagnosis"] == l)] for l in df["diagnosis"].unique()]))
"""
                          0    1
0                   control  556
1          bipolar disorder  306
2  schizoaffective disorder  112
"""
df["diagnosis"] = df["diagnosis"].map({'bipolar disorder': 1, 'psychotic bipolar disorder':1, 'control': 0}).values

# Leave study out CV
studies = np.sort(df["study"].unique())
# array(['BIOBD', 'BSNIP', 'PRAGUE', 'SCHIZCONNECT-VIP'], dtype=object)
folds = [[np.where(df["study"] != s)[0], np.where(df["study"] == s)[0]] for s in studies]
cv = CVIterableWrapper(folds)

# Check all sites have both labels
print([[studies[i]] + [np.sum(df["diagnosis"].values[te] == lab) for lab in np.unique(df["diagnosis"].values[te])] for i, (tr, te) in enumerate(cv.split(None, df["diagnosis"].values))])
# [['BIOBD', 356, 306], ['BSNIP', 200, 117]]

# Merge masks
mask_filenames = glob.glob(OUTPUT("*", scaling=None, harmo=None, type="mask", ext="nii.gz"))
print([np.sum(nibabel.load(mask_filename).get_data()) for mask_filename in mask_filenames])
# [364610, 368680, 365280, 362619]
#
mask_arr = np.sum(np.concatenate([np.expand_dims(nibabel.load(mask_filename).get_data() > 0, axis=0) for mask_filename
                                  in mask_filenames]), axis=0) > (len(mask_filenames) - 1)
# 360348
mask_img = nilearn.image.new_img_like(mask_filenames[0], data=mask_arr)

print(mask_arr.sum())
mask_img.to_filename(OUTPUT("inter_studies/" + "+".join(studies) + suffix, scaling=None, harmo=None, type="mask", ext="nii.gz"))


scores_ml = dict()

settings = [
    ['raw', 'raw', "data64"],
    ['gs', 'raw', "data64"],
    ['gs', 'ctrsite', "data32"],
    ['gs', 'res:site', "data32"],
    ['gs', 'res:site(age+sex+diag)', "data32"],
    ['gs', 'res:site+age+sex(diag)', "data32"]]

for scaling, harmo, datatype in settings:
    print(scaling, harmo, datatype)
    # scaling, harmo, datatype = 'raw', 'raw', "data64"
    # scaling, harmo, datatype = 'gs', 'res:site(age+sex+diag)', "data32"

    NI_arr = np.concatenate([np.load(OUTPUT(dataset=dataset, scaling=scaling, harmo=harmo, type=datatype, ext="npy"), mmap_mode='r') for dataset in datasets])[mask]
    scores_ml_ = ml_predictions_warpper(X=NI_arr, df=df, targets_reg=[], targets_clf=["diagnosis"], cv=cv, mask_arr=mask_arr,
                                       tag_name=scaling + "-" + harmo, dataset_name="+".join(studies) + suffix)

    for key, dat in scores_ml_.items():
        if "_folds" in key:
            dat["fold"] = dat["fold"].map({"CV%i" % i: studies[i] for i in range(len(studies))})
        scores_ml[(scaling, harmo, key)] = dat


scalings, harmos, targets = zip(*[[scaling, harmo, target] for scaling, harmo, target in scores_ml])
scalings, harmos, targets = set(scalings), set(harmos), set(targets)
scores_ml_bytarget = {name:[] for name in targets}

for key, dat in scores_ml.items():
    scores_ml_bytarget[key[2]].append(dat)

scores_ml_bytarget = {key:pd.concat(dats) for key, dats in scores_ml_bytarget.items()}

with pd.ExcelWriter(OUTPUT("inter_studies/" + "+".join(studies)  + suffix, scaling=None, harmo=None, type="ml-scores", ext="xlsx")) as writer:
    for key, dat in scores_ml_bytarget.items():
        dat.to_excel(writer, sheet_name=key, index=False)

################################################################################
"""
Première chose, j'ai effcivement oublié ici dans la stratification 'diagnosis':
'control' pour BIOBD (je l'ai bien mis dans le benchmark en revanche).

Age:
Train = HCP+IXI
validation = controls de BIOBD
test = controls de BSNIP

Sex:
same as age

Diagnostic :
Train/val: IXI+HCP(ctl)+Schizconnect+PRAGUE with CV
Test: BSNIP en


La stratification est bien pour les batchs puisque les set train/val/test sont fixés pour la partie (âge/sexe).

Encore une fois, pour le diagnostic je ne suis même pas sûr de ce que l'on veut (par exemple,
inclut on HCP alors que c'est une base différente de nos bases de controls dans SCHIZCONNECT?).
Le problème comparé au linéaire dans ce cas est la taille de l'échantillons, vraiment très petite,
qui occasionne des problèmes de convergence.
"""

# ## Sex, Age Benchmark used in PyNet (with PyTorch back-end)

# inputs_path = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/" \
#               "cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy"
# metadata_path = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/" \
#                 "cat12vbm/all_t1mri_mwp1_participants.tsv"
# df = pd.read_csv(metadata_path, sep='\t')
# batch_size = 16
# nb_folds = 1
# pin_memory = True
# drop_last = False
#
# stratif = {
#     'train': {'study': ['HCP', 'IXI']},
#     'validation': {'study': 'BIOBD'},
#     'test': {'study': 'BSNIP', 'diagnosis': 'control'}
# }
#
# add_to_input = None
# add_input = False
# labels=["age", "sex"]
# sampler="random"
# stratify_label='site'
# input_transforms=[Crop((1, 121, 128, 121)), Padding((1, 128, 128, 128)), Normalize(mean=0, std=1)]
# strat_label_transforms=[LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))})]
# labels_transforms=None
# data_augmentation=None
# output_transforms=None
# patch_size=None
# input_size=None
#
# manager1 = DataManager(inputs_path, metadata_path,
#                        batch_size=batch_size,
#                        number_of_folds=nb_folds,
#                        add_to_input=add_to_input,
#                        add_input=add_input,
#                        labels=labels,
#                        sampler=sampler,
#                        projection_labels=projection_labels,
#                        custom_stratification=stratif,
#                        stratify_label=stratify_label,
#                        input_transforms=input_transforms,
#                        stratify_label_transforms=strat_label_transforms,
#                        labels_transforms=labels_transforms,
#                        data_augmentation=data_augmentation,
#                        output_transforms=output_transforms,
#                        patch_size=patch_size,
#                        input_size=input_size,
#                        pin_memory=pin_memory,
#                        drop_last=drop_last)


################################################################################
# Global mask
import numpy as np
import nibabel
#from nilearn.image import resample_to_img
from nitk.image import compute_brain_mask

# ABIDE2_t1mri_mwp1_gs-raw_data64.npy
# biobd_t1mri_mwp1_gs-raw_data64.npy
# bsnip_t1mri_mwp1_gs-raw_data64.npy
# HCP_t1mri_mwp1_gs-raw_data64.npy
# icaar-start_t1mri_mwp1_gs-raw_data64.npy
# IXI_t1mri_mwp1_gs-raw_data64.npy
# schizconnect-vip_t1mri_mwp1_gs-raw_data64.npy

datasets = [
    'icaar-start',
    'schizconnect-vip',
    'bsnip',
    'biobd',
    'ABIDE2',
    'HCP',
    'IXI']

# IXI_t1mri_mwp1_mask.nii.gz

# Read mask images and check same affine => ref_img
mask_imgs = {dataset: nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz")) for dataset in datasets}
affines = [mask_img.affine for k, mask_img in mask_imgs.items()]
assert np.all([np.all(affines[0] == affine) for affine in affines])

target_img = mask_imgs["IXI"]
shape = ref_img.get_fdata().shape

# grand sum and ss
stats = dict()
for dataset in datasets:
    print(dataset)
    # dataset = datasets[0]
    #OUTPUT(dataset, scaling="gs", harmo="raw", type="data64", ext="npy")
    imgs_arr = np.load(OUTPUT(dataset, scaling="gs", harmo="raw", type="data64", ext="npy"), mmap_mode='r').squeeze()
    n = imgs_arr.shape[0]
    sum_ = np.mean(imgs_arr, axis=0) * n
    ss = np.var(imgs_arr, axis=0) * n
    stats[dataset] = [n, sum_, ss]

n = np.sum([v[0] for k, v in stats.items()])
assert n == 5571

grand_sum =  np.zeros(shape)
grand_ss =  np.zeros(shape)

for k, v in stats.items():
    grand_sum += v[1]
    grand_ss += v[2]

# grand mean and std

grand_mean = grand_sum / n
grand_std = np.sqrt(grand_ss / n)

np.save(os.path.join(OUTPUT_PATH, "grand_mean.npy"), grand_mean)
np.save(os.path.join(OUTPUT_PATH, "grand_std.npy"), grand_std)
# grand_mean_ = np.load(os.path.join(OUTPUT_PATH, "grand_mean.npy"))
# np.all(grand_mean_ == grand_mean)

# global mask

mask_thres_mean = 0.1
mask_thres_std = 1e-6

implicitmask_arr = np.ones(shape, dtype=bool).squeeze()
implicitmask_arr = implicitmask_arr & (np.abs(grand_mean) >= mask_thres_mean).squeeze()
implicitmask_arr = implicitmask_arr & (grand_std >= mask_thres_std).squeeze()

from  nitk.image import compute_brain_mask

mask_brain_img = compute_brain_mask(target_img=target_img, implicitmask_arr=implicitmask_arr, verbose=1)
# Clusters of connected voxels #1, sizes= [371414]
mask_cerebrum_img = compute_brain_mask(target_img=target_img, implicitmask_arr=implicitmask_arr, rm_brainstem=True, rm_cerebellum=True, verbose=1)
# Clusters of connected voxels #1, sizes= [331695]

mask_brain_img.to_filename(OUTPUT("ALL", scaling=None, harmo=None, type="brain-mask", ext="nii.gz"))
mask_cerebrum_img.to_filename(OUTPUT("ALL", scaling=None, harmo=None, type="cerebrum-mask", ext="nii.gz"))
