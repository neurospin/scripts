"""
# Copy data

cd /home/ed203246/data/psy_sbox/analyses/201906_biobd-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*participants*.csv ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*t1mri_mwp1_mask.nii.gz ./
rsync -azvu triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*mwp1_gs-raw_data64.npy ./

# NS => Laptop
rsync -azvun triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202004_biobd-bsnip_cat12vbm_predict-dx/* /home/ed203246/data/psy_sbox/analyses/202004_biobd-bsnip_cat12vbm_predict-dx/

# Laptop => NS
rsync -azvun /home/ed203246/data/psy_sbox/analyses/202004_biobd-bsnip_cat12vbm_predict-dx/ triscotte.intra.cea.fr:/neurospin/psy_sbox/analyses/202004_biobd-bsnip_cat12vbm_predict-dx/
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
if not hasattr(sys, 'ps1'): # if not interactive use pdf backend
    matplotlib.use('pdf')
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

#
# import os, sys
# import numpy as np
# import glob
# import pandas as pd
# import nibabel
# import brainomics.image_preprocessing as preproc
# from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
# import shutil
# # import mulm
# # import sklearn
# # import re
# # from nilearn import plotting
# import nilearn.image
# import matplotlib
# if not hasattr(sys, 'ps1'): # if not interactive use pdf backend
#     matplotlib.use('pdf')
# import matplotlib.pyplot as plt
# import re
# # import glob
# import seaborn as sns
# import copy
# import pickle
# import time

# # ML
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import sklearn.linear_model as lm
# import sklearn.ensemble as ensemble
# from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
# from nitk.stats import Residualizer
# from nitk.utils import dict_product, parallel, reduce_cv_classif
# from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio

INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm'
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/202010_biobd-bsnip_cat12vbm_predict-dx'
NJOBS = 8

# On laptop
if not os.path.exists(INPUT_PATH):
    INPUT_PATH = INPUT_PATH.replace('/neurospin', '/home/ed203246/data')
    OUTPUT_PATH = OUTPUT_PATH.replace('/neurospin', '/home/ed203246/data')
    NJOBS = 2

os.makedirs(OUTPUT_PATH, exist_ok=True)

scaling, harmo = 'gs', 'raw'
DATASET_FULL = 'biobd-bsnip'
DATASET_TRAIN = 'biobd'
target, target_num = "diagnosis", "diagnosis_num"
NSPLITS = 5

################################################################################
#
# Utils
#
################################################################################

def PATH(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None,
    type=None, ext=None, basepath=""):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32

    return os.path.join(basepath, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)

def INPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=INPUT_PATH)

def OUTPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=OUTPUT_PATH)

def fit_predict(key, estimator_img, residualize, split):
    print(key)
    start_time = time.time()
    train, test = split
    Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Z_train, Z_test, y_train =\
    Xim[train, :], Xim[test, :], Xdemoclin[train, :], Xdemoclin[test, :], Z[train, :], Z[test, :], y[train]

    # Images based predictor

    # Residualization
    if residualize:
        # residualizer.fit(Xim_, Z_) biased residualization
        residualizer.fit(Xim_train, Z_train)
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
    try:
        coef_img = estimator_img.coef_
    except:
        coef_img = None
    time_elapsed = round(time.time() - start_time, 2)

    return dict(y_test_img=y_test_img, score_test_img=score_test_img, time=time_elapsed,
                coef_img=coef_img)

"""
# Wrap user define CV to new sklearn CV (Leave out study CV)
from sklearn.model_selection import BaseCrossValidator
class CVIterableWrapper(BaseCrossValidator):
    "Wrapper class for old style cv objects and iterables."
    def __init__(self, cv):
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        for train, test in self.cv:
            yield train, test

# [(fold, train, test) for fold, (train, test) in enumerate(cv.split(X, y))]
# for fold, (train, test) in enumerate(cv.split(X, y)): print(fold, (train, test))
def scores_train_test(estimator, X_tr, X_te, y_tr, y_te):
    from sklearn import metrics
    y_pred_tr, y_pred_te = estimator.predict(X_tr), estimator.predict(X_te)
    return [metrics.accuracy_score(y_tr, y_pred_tr), metrics.accuracy_score(y_te, y_pred_te)]
"""


if not os.path.exists(OUTPUT(dataset=DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy")):

    print("""
    ################################################################################
    #
    # Dataset: concatenate [biobd  bsnip]
    #
    ################################################################################
    """)
    # BD (biobd <=> bsnip)

    datasets = ['biobd', 'bsnip']

    # Read clinical data
    pop = pd.concat([pd.read_csv(INPUT(dataset=dataset, scaling=None, harmo=None, type="participants", ext="csv")) for dataset in datasets], axis=0)
    pop = pop.reset_index(drop=True)
    # force participants id to be strings
    pop.participant_id = [str(id) for id in pop.participant_id]


    print(pd.DataFrame([[l, np.sum(pop["diagnosis"] == l)] for l in pop["diagnosis"].unique()]))
    """
                                                        0    1
    0                                             control  556
    1                                    bipolar disorder  306
    2                                            ADHD, SU    1
    3                                                 NaN    0
    4                                                 EDM    1
    5                                    MDE, ADHD, panic    1
    6                                           SU, panic    1
    7                                           MDE, PTSD    1
    8                                                ADHD    1
    9   relative of proband with schizoaffective disorder  123
    10  relative of proband with psychotic bipolar dis...  119
    11                           schizoaffective disorder  112
    12                                      schizophrenia  194
    13             relative of proband with schizophrenia  175
    14                         psychotic bipolar disorder  117
    """
    # keep only
    msk = pop.diagnosis.isin(['control', 'bipolar disorder', 'psychotic bipolar disorder'])
    assert pop[msk].shape == (979, 52)

    laurie = pd.read_csv(os.path.join(OUTPUT_PATH, 'norm_dataset_cat12_bsnip_biobd.tsv'), sep="\t")
    assert laurie.shape == (993, 183)

    merge = pd.merge(laurie, pop[msk], on='participant_id' )
    assert merge.shape == (976, 234)

    rm_from_laurie = laurie[~laurie.participant_id.isin(merge.participant_id)]
    rm_from_anton = pop[msk][~pop[msk].participant_id.isin(merge.participant_id)]
    anton = pop[msk]

    msk_merge = msk & pop.participant_id.isin(merge.participant_id)
    assert np.sum(msk_merge) == merge.shape[0]
    assert pd.merge(laurie, pop[msk_merge], on='participant_id' ).shape[0] == np.sum(msk_merge)

    # split mask in  since we will have to load dataset maskin and concatenating
    last_biobd = np.where(pop.study == "BIOBD")[0][-1]
    assert last_biobd + 1 == np.where(pop.study == "BSNIP")[0][0]
    mask_merge_biobd = msk_merge[:(last_biobd + 1)]
    mask_merge_bsnip = msk_merge[(last_biobd + 1):]
    assert np.all(np.concatenate([mask_merge_biobd, mask_merge_bsnip]) == msk_merge)
    del last_biobd

    # QC on merge
    df_ = pd.merge(
            pop[msk_merge][['participant_id',  'site', 'sex', 'age', 'diagnosis', 'study', 'Age of Onset']],
            laurie[['participant_id', 'siteID', 'Female', 'Age', 'DX', 'Age of Onset']],
            on='participant_id', suffixes=('_anton', '_laurie'))

    assert np.all(df_.site == df_.siteID)
    assert np.all(df_.sex == df_.Female)
    assert np.all((df_.age - df_.Age) < 1e-3)
    assert np.all(df_.diagnosis.map({'control':0, 'bipolar disorder':1, 'psychotic bipolar disorder':1}) == df_.DX)
    # OK so far

    correct_age_of_onset = df_[(df_["Age of Onset_anton"].notnull() | df_['Age of Onset_laurie'].notnull()) & (df_["Age of Onset_anton"] != df_['Age of Onset_laurie'])][['participant_id',  "site", "Age of Onset_anton", 'Age of Onset_laurie']]
    print(correct_age_of_onset.shape)
    # 36 patiens from Udine where Laurie has the Agae of Onset
    # Update:
    # PHENOTYPE_CSV = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv"
    # Correct
    for subject in correct_age_of_onset.participant_id:
        pop.loc[pop.participant_id == subject, 'Age of Onset'] = laurie.loc[laurie.participant_id == subject, 'Age of Onset'].values[0]


    xls_filename = os.path.join(OUTPUT_PATH + "/QC_MERGE_LAURIE-ANTON.xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        pop[msk_merge].to_excel(writer, sheet_name='merged', index=False)
        rm_from_laurie.to_excel(writer, sheet_name='removed_from_laurie', index=False)
        rm_from_anton.to_excel(writer, sheet_name='removed_from_ours', index=False)

    del df_, anton, correct_age_of_onset, laurie, merge, msk, rm_from_anton, rm_from_laurie, subject, xls_filename

    pop = pop[msk_merge]
    pop = pop.reset_index(drop=True)
    pop[target_num] = pop[target].map({'control':0, 'bipolar disorder':1, 'psychotic bipolar disorder':1}).values

    pop["GM_frac"] = pop.gm / pop.tiv
    pop["sex_c"] = pop["sex"].map({0: "M", 1: "F"})

    # Load arrays load separatly apply mask
    imgs_arr = np.zeros((pop.shape[0], 1, 121, 145, 121))
    imgs_arr[:mask_merge_biobd.sum()] = np.load(INPUT(dataset='biobd', scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')[mask_merge_biobd]
    imgs_arr[mask_merge_biobd.sum():] = np.load(INPUT(dataset='bsnip', scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')[mask_merge_bsnip]
    np.save(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), imgs_arr)
    del imgs_arr
    import gc
    gc.collect()

    # reload and do QC
    imgs_arr = np.load(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    assert np.all(imgs_arr[:mask_merge_biobd.sum()] == np.load(INPUT(dataset='biobd', scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')[mask_merge_biobd])
    assert np.all(imgs_arr[mask_merge_biobd.sum():] == np.load(INPUT(dataset='bsnip', scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')[mask_merge_bsnip])

    del msk_merge, mask_merge_biobd, mask_merge_bsnip

    # Use cerebrum mask: ALL_t1mri_mwp1_cerebrum-mask.nii.gz
    mask_img = nibabel.load(INPUT("ALL", scaling=None, harmo=None, type="cerebrum-mask", ext="nii.gz"))
    # mask_img = compute_brain_mask(imgs_arr, target_img=ref_img)
    mask_img.to_filename(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="cerebrum-mask", ext="nii.gz"))
    pop.to_csv(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv"), index=False)

    mask_arr = mask_img.get_data() != 0
    assert np.sum(mask_arr != 0) == 331695
    print(mask_arr.shape, imgs_arr.squeeze().shape)
    print("Sizes. mask_arr:%.2fGb" % (imgs_arr.nbytes / 1e9))


if not os.path.exists(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="residualization-l2", ext="xlsx")):
    print("""
    ###############################################################################
    #
    # Residualization study on biobd-bsnip
    #
    ###############################################################################
    """)

    pop = pd.read_csv(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv"))

    # Working population df with no NAs
    pop_w = pop.copy()
    assert np.all(pop_w[["sex", "age", "site"]].isnull().sum()  == 0)

    imgs_arr = np.load(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    mask_img = nibabel.load(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="cerebrum-mask", ext="nii.gz"))
    mask_arr = mask_img.get_data() != 0
    assert mask_arr.sum() == 331695

    print("""
    #==============================================================================
    # Select biobd+BSNIP : 5CV(BIOBD) + LSO(BIOBD+BSNIP)
    """)

    msk = pop.study.isin(['BIOBD', 'BSNIP'])
    assert msk.sum() == 976 # msk is all True here keep it for compatibility
    Xim = imgs_arr.squeeze()[:, mask_arr][msk]
    del imgs_arr
    y = pop[target + "_num"][msk].values
    print("Sizes. mask_arr:%.2fGb" % (Xim.nbytes / 1e9))

    vars_clinic = []
    vars_demo = ['age', 'sex']
    Xdemoclin = pop.loc[msk, vars_demo + vars_clinic].values

    # -----------------------------------------------------------------------------
    # Residualization bloc: Sex + Sites + age with some descriptives stats

    pop_ = pop_w[msk]
    print([[s, np.sum(pop_.sex == s)] for s in pop_.sex.unique()])
    # [[0.0, 592], [1.0, 407]]
    desc_stats = pd.DataFrame([pop_[pop_.site == s].study.unique().tolist() +
                        [s, np.sum(pop_.site == s),
                         round(pop_[pop_.site == s][target_num].mean(), 2),
                         round(pop_[pop_.site == s]["age"].mean(), 2),
                         round(pop_[pop_.site == s]["sex"].mean(), 2)] for s in pop_.site.unique()],
    columns=['study', "site", 'count', 'DX%', 'age_mean', "sex%F"])
    print(desc_stats)
    """
        study        site  count   DX%  age_mean  sex%F
    0   BIOBD    sandiego    117  0.37     50.65   0.62
    1   BIOBD    mannheim     79  0.52     41.06   0.56
    2   BIOBD     creteil     73  0.47     35.27   0.49
    3   BIOBD       udine    126  0.29     38.72   0.43
    4   BIOBD      galway     69  0.41     41.33   0.49
    5   BIOBD  pittsburgh    114  0.68     33.68   0.73
    6   BIOBD    grenoble     32  0.72     43.22   0.53
    7   BIOBD      geneve     52  0.46     31.25   0.46
    8   BSNIP      Boston     54  0.52     34.69   0.61
    9   BSNIP      Dallas     67  0.36     40.24   0.63
    10  BSNIP    Hartford     79  0.34     35.44   0.58
    11  BSNIP   Baltimore     88  0.35     41.70   0.64
    12  BSNIP     Detroit     26  0.19     32.88   0.54
    """

    formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
    residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
    Z = residualizer.get_design_mat()

    assert Xim.shape[0] == Z.shape[0] == y.shape[0]

    # -----------------------------------------------------------------------------
    # CV: 5CV(BIOBD) + LSO(biobd+BSNIP)

    pop_ = pop_w[msk]
    pop_ = pop_.reset_index(drop=True)

    # ~~~~~~~
    # CV LSO

    cv_lso_dict = {s:[np.where(pop_.site != s)[0], np.where(pop_.site == s)[0]] for s in pop_.site.unique()}

    # QC all test stem from single left-out site
    for k, fold in cv_lso_dict.items():
        sites_ =  pop_.site[fold[1]].unique()
        assert len(sites_) == 1 and sites_[0] == k

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5cv on BIOBD that will be applied on biobd+bsnip

    # store idx of the large dataset, cv in the smaller, map back using stored idx
    df_ = pop_[["participant_id", target_num]]
    df_["idx"] = np.arange(len(df_)) # store idx of large dataset
    df_ = df_[pop_.study.isin(["BIOBD"])] # select smaller
    cv_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    # do CV on the smaller but use index of the larger
    cv5_biobd = {"CV%i" % fold : [df_.idx[train].values, df_.idx[test].values] for fold, (train, test) in enumerate(cv_.split(df_[target_num].values, df_[target_num].values))}

    # Check all split cover all biobd sample
    assert np.all(np.array([len(np.unique(train.tolist() + test.tolist()))  for fold, (train, test) in cv5_biobd.items()]) == pop_.study.isin(["BIOBD"]).sum())
    # Check all test cover all biobd sample
    assert np.all(len(np.unique(np.concatenate([test.tolist()  for fold, (train, test) in cv5_biobd.items()]))) == pop_.study.isin(["BIOBD"]).sum())
    del df_, cv_

    print("""
    #==============================================================================
    # Run l2 5CV(BIOBD) + LSO(BIOBD+BSNIP)
    """)

    estimators_dict = {"l2_C:%f" % 1: lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)}

    # LSO
    args_collection = dict_product(estimators_dict, dict(noresidualize=False, residualize=True), cv_lso_dict)
    key_vals_lso = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)
    cv_scores_lso = reduce_cv_classif(key_vals_lso, cv_lso_dict, y_true=y)
    cv_scores_lso["CV"] = 'LSO(BIOBD+BSNIP)'

    # 5CV

    args_collection = dict_product(estimators_dict, dict(noresidualize=False, residualize=True), cv5_biobd)
    key_vals_cv5_biobd = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)
    cv_scores_cv5_biobd = reduce_cv_classif(key_vals_cv5_biobd, cv5_biobd, y_true=y)
    cv_scores_cv5_biobd["CV"] = '5CV(BIOBD)'

    cv_scores = cv_scores_lso.append(cv_scores_cv5_biobd)

    # =>
    #
    mean = cv_scores.groupby(["CV", "param_1"]).mean()
    sd = cv_scores.groupby(["CV", "param_1"]).std()
    sd = sd[["auc", "bacc"]].rename(columns={'auc':'auc_std', 'bacc':'bacc_std'})
    stat = pd.concat([mean, sd], axis=1)

    xls_filename = OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="residualization-l2", ext="xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        stat.to_excel(writer, sheet_name='mean')
        desc_stats.to_excel(writer, sheet_name='desc_stats', index=False)

    del cv_scores, cv_scores_lso, cv_scores_cv5_biobd

if not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")) or\
   not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")):

    print("""
    ###############################################################################
    #
    # Comparison analysis and Sensitivity study on biobd 5CV
    #
    ###############################################################################
    """)

    pop = pd.read_csv(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="participants", ext="csv"))
    imgs_arr = np.load(OUTPUT(DATASET_FULL, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
    mask_img = nibabel.load(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="cerebrum-mask", ext="nii.gz"))
    mask_arr = mask_img.get_data() != 0
    assert mask_arr.sum() == 367120

    print("""
    #==============================================================================
    # Select dataset 5CV on biobd
    """)

    msk = pop.study.isin(['BIOBD'])
    assert msk.sum() == 662
    Xim = imgs_arr.squeeze()[:, mask_arr][msk]
    del imgs_arr
    y = pop[target + "_num"][msk].values
    print("Sizes. mask_arr:%.2fGb" % (Xim.nbytes / 1e9))
    # Sizes. mask_arr:1.94Gb

    Xdemoclin = Z = np.zeros((Xim.shape[0], 1))

    cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)
    cv_dict = {"CV%i" % fold:split for fold, split in enumerate(cv.split(Xim, y))}
    cv_dict["ALL"] = [np.arange(Xim.shape[0]), np.arange(Xim.shape[0])]

    print([[lab, np.sum(y == lab)] for lab in np.unique(y)])
    #[[0, 356], [1, 306]]


if not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")):
    print("""
    #==============================================================================
    # l1, l2, enet, filter, rfe
    """)

    # parameters range:
    # from sklearn.svm import l1_min_c
    # Cmin = l1_min_c(StandardScaler().fit_transform(Xim), y, loss='log')
    # Cs = Cmin * np.logspace(0, -5, 10)

    Cs = np.logspace(14, -14, 20)
    l2 = {"l2_C:%.16f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}
    l1 = {"l1_C:%.16f" % C: lm.LogisticRegression(C=C, penalty='l1', class_weight='balanced', fit_intercept=False) for C in Cs}
    enet = {"enet_C:%.16f" % C: lm.LogisticRegression(C=C, penalty='elasticnet', class_weight='balanced', l1_ratio=.1, fit_intercept=False, solver='saga') for C in Cs}
    assert len(Cs) == len(l2) == len(l1) == len(enet)

    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    Ks = np.logspace(1, np.log10(Xim.shape[1]), 20).astype(int)
    fl2 = {"fl2_k:%i" % k:make_pipeline(SelectKBest(k=k), lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False))
            for k in Ks}

    rfe = {"rfel2_k:%i" % k:RFE(estimator=lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False), n_features_to_select=k, step=.05)
            for k in Ks}

    estimators_dict = dict()
    estimators_dict.update(l1)
    estimators_dict.update(l2)
    estimators_dict.update(enet)
    estimators_dict.update(fl2)
    estimators_dict.update(rfe)

    args_collection = dict_product(estimators_dict, dict(noresdualize=False), cv_dict)
    print("Nb Tasks=%i" % len(args_collection))

    key_vals = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)

    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="pkl")
    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals, fd)

    cv_scores = reduce_cv_classif(key_vals, cv_dict, y_true=y)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        cv_scores.groupby(["param_0"]).mean().to_excel(writer, sheet_name='mean')

    print("""
    #------------------------------------------------------------------------------
    # plot
    """)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-l1-l2-enet-filter-rfe", ext="xlsx")
    cv_scores = pd.read_excel(xls_filename)
    cv_scores["param"] = [float(s.split(":")[1]) for s in cv_scores["param_0"]]
    cv_scores["algo"] = [s.split("_")[0] for s in cv_scores["param_0"]]

    sns.set_style("whitegrid")
    import matplotlib.pylab as pl
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('text.latex', preamble=r'\usepackage{lmodern}')
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    # -------------

    # seach l2 C closre to 1
    x_ = cv_scores[cv_scores.algo == 'l2']["param"].unique()
    C1_almost = x_[np.argmin(np.abs(x_ - 1))]
    baseline_l2C1_auc = cv_scores[(cv_scores.algo == 'l2') & (cv_scores.param == C1_almost)]["auc"]
    baseline_l2C1_bacc = cv_scores[(cv_scores.algo == 'l2') & (cv_scores.param == C1_almost)]["bacc"]

    cv_scores["Model"] = cv_scores.algo.map({'l1':r'$\ell_1$', 'l2':r'$\ell_2$', "enet":r'$\ell_1\ell_2$', 'rfel2':'RFE+$\ell_2$',  'fl2':'Filter+$\ell_2$'})

    # filter and RFE + l2

    # AUC
    df_ = cv_scores[cv_scores["algo"].isin(["fl2", "rfel2"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="auc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'k', fontsize=20)
    plt.ylabel(ylabel=r'AUC', fontsize=16)
    g.axes.axhline(baseline_l2C1_auc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .8))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-filter-rfe_auc", ext="pdf"))

    # bACC
    df_ = cv_scores[cv_scores["algo"].isin(["fl2", "rfel2"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="bacc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'k', fontsize=20)
    plt.ylabel(ylabel=r'bACC', fontsize=16)
    g.axes.axhline(baseline_l2C1_bacc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .8))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-filter-rfe_bacc", ext="pdf"))

    # l1, enet

    # AUC
    df_ = cv_scores[cv_scores["algo"].isin(["l2", "l1", "enet"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="auc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'C', fontsize=20)
    plt.ylabel(ylabel=r'AUC', fontsize=16)
    g.axes.axhline(baseline_l2C1_auc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .8))
    g.set(xlim=(1e-4, 1e4))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_auc", ext="pdf"))

    # bACC
    df_ = cv_scores[cv_scores["algo"].isin(["l2", "l1", "enet"])]
    fig = plt.figure(figsize=(7.25, 5), dpi=300)
    g = sns.lineplot(x="param", y="bacc", hue="Model", data=df_)#, palette=palette)
    g.set(xscale="log")
    plt.xlabel(xlabel=r'C', fontsize=20)
    plt.ylabel(ylabel=r'bACC', fontsize=16)
    g.axes.axhline(baseline_l2C1_bacc.mean(), ls='--', color='black', linewidth=1, alpha=0.5)  # no feature selection performance
    g.set(ylim=(.45, .8))
    g.set(xlim=(1e-4, 1e4))
    plt.tight_layout()
    plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_bacc", ext="pdf"))


if not os.path.exists(OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")):

    print("""
    #==============================================================================
    # Enet-TV
    """)

    print("# Enet-TV")
    # estimators
    import parsimony.algorithms as algorithms
    import parsimony.estimators as estimators
    import parsimony.functions.nesterov.tv as nesterov_tv
    from parsimony.utils.linalgs import LinearOperatorNesterov

    if not os.path.exists(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="Atv", ext="npz")):
        Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
        Atv.save(OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="Atv", ext="npz"))

    Atv = LinearOperatorNesterov(filename=OUTPUT(DATASET_FULL, scaling=None, harmo=None, type="Atv", ext="npz"))
    assert np.allclose(Atv.get_singular_values(0), 11.941115174310951)

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
    # alphas = [0.010]
    # l1l2ratios = [0]
    # tvl2ratios = [0]

    import itertools
    estimators_dict = dict()
    for alpha, l1l2ratio, tvl2ratio in itertools.product(alphas, l1l2ratios, tvl2ratios):
        # print(alpha, l1l2ratio, tvl2ratio)
        l1, l2, tv = ratios_to_param(alpha, l1l2ratio, tvl2ratio)
        key = "enettv_%.3f:%.6f:%.6f" % (alpha, l1l2ratio, tvl2ratio)

        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
        estimators_dict[key] = estimator

    args_collection = dict_product(estimators_dict, dict(noresidualize=False), cv_dict)

    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
    if os.path.exists(models_filename):
        with open(models_filename, 'rb') as fd:
            KEY_VALS = pickle.load(fd)

    key_vals = parallel(fit_predict, args_collection, n_jobs=NJOBS, pass_key=True, verbose=20)

    with open(models_filename, 'wb') as fd:
        pickle.dump(key_vals, fd)

    cv_scores = reduce_cv_classif(key_vals, cv_dict, y_true=y)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)

    # [Parallel(n_jobs=8)]: Done 180 out of 180 | elapsed: 4900.6min finished

    print("""
    #------------------------------------------------------------------------------
    # maps' similarity measures accross CV-folds
    """)
    models_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pkl")
    with open(models_filename, 'rb') as fd:
        key_vals = pickle.load(fd)

    # filter out "ALL" folds
    key_vals = {k:v for k, v in key_vals.items() if k[2] != "ALL"}

    # Agregate maps by key[0]
    by_param = {k:[] for k in set([k[0] for k in key_vals])}
    for k, v in key_vals.items():
        by_param[k[0]].append(v['coef_img'].ravel())

    maps_similarity_l = list()
    # Compute similarity measures
    for k, v in by_param.items():
        maps = np.array(v)
        maps_t = np.vstack([arr_threshold_from_norm2_ratio(maps[i, :], .99)[0] for i in range(maps.shape[0])])
        r_bar, dice_bar, fleiss_kappa_stat = maps_similarity(maps_t)
        prop_non_zeros_mean = np.count_nonzero(maps_t) / np.prod(maps_t.shape)
        maps_similarity_l.append([k, prop_non_zeros_mean, r_bar, dice_bar, fleiss_kappa_stat])

    map_sim = pd.DataFrame(maps_similarity_l, columns=['param_0', 'prop_non_zeros_mean', 'r_bar', 'dice_bar', 'fleiss_kappa_stat'])

    # Update excel file with similariy measures
    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    cv_scores_all = pd.read_excel(xls_filename, sheet_name='folds')
    cv_scores = cv_scores_all[cv_scores_all.fold != "ALL"]
    pred_score_mean_ = cv_scores.groupby(["param_0"]).mean()
    pred_score_mean_ = pred_score_mean_.reset_index()
    pred_score_mean = pd.merge(pred_score_mean_, map_sim)
    assert pred_score_mean_.shape[0] == map_sim.shape[0] == pred_score_mean.shape[0]

    with pd.ExcelWriter(xls_filename) as writer:
        cv_scores.to_excel(writer, sheet_name='folds', index=False)
        pred_score_mean.to_excel(writer, sheet_name='mean')
        cv_scores_all.to_excel(writer, sheet_name='folds_with_all_in_train', index=False)
    del pred_score_mean_

    print("""
    #------------------------------------------------------------------------------
    # plot
    """)

    xls_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="xlsx")
    cv_scores = pd.read_excel(xls_filename, sheet_name='folds')
    cv_scores = cv_scores.reset_index(drop=True)
    cv_scores_mean = pd.read_excel(xls_filename, sheet_name='mean')
    cv_scores_mean = cv_scores_mean.reset_index(drop=True)

    keys_ = pd.DataFrame([[s.split("_")[0]] + [float(v) for v in s.split("_")[1].split(":")] for s in cv_scores["param_0"]],
                 columns=["model", "alpha", "l1l2", "tv"])
    cv_scores = pd.concat([keys_, cv_scores], axis=1)
    keys_ = pd.DataFrame([[s.split("_")[0]] + [float(v) for v in s.split("_")[1].split(":")] for s in cv_scores_mean["param_0"]],
                 columns=["model", "alpha", "l1l2", "tv"])
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

    pdf_filename = OUTPUT(DATASET_TRAIN, scaling=scaling, harmo=harmo, type="models-5cv-enettv", ext="pdf")
    with PdfPages(pdf_filename) as pdf:
        for l1l2 in cv_scores["l1l2"].unique():
            print("%.4f" % l1l2, l1l2)
            df_ = cv_scores[cv_scores["l1l2"].isin([l1l2])]
            dfm_ = cv_scores_mean[cv_scores_mean["l1l2"].isin([l1l2])]
            df_["alpha"] = df_["alpha"].map({0.01:"1e-2'", 0.1:"1e-1'" , 1.:"1'"})
            dfm_["alpha"] = dfm_["alpha"].map({0.01:"1e-2'", 0.1:"1e-1'" , 1.:"1'"})
            df_.rename(columns={"alpha":"alpha", 'auc':'AUC', 'bacc':'bAcc'}, inplace=True)
            dfm_.rename(columns={'r_bar':'$r_w$', 'prop_non_zeros_mean':'non-null', 'dice_bar':'dice', 'fleiss_kappa_stat':'Fleiss-Kappa'}, inplace=True)

            fig, axs = plt.subplots(3, 2, figsize=(2 * 7.25, 3 * 5), dpi=300)
            g = sns.lineplot(x="tv", y='AUC', hue="alpha", data=df_, ax=axs[0, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='bAcc', hue="alpha", data=df_, ax=axs[0, 1], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='$r_w$', hue="alpha", data=dfm_, ax=axs[1, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='non-null', hue="alpha", data=dfm_, ax=axs[1, 1], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='dice', hue="alpha", data=dfm_, ax=axs[2, 0], palette="Blues"); g.set(xscale="log")
            g = sns.lineplot(x="tv", y='Fleiss-Kappa', hue="alpha", data=dfm_, ax=axs[2, 1], palette="Blues"); g.set(xscale="log")
            #plt.tight_layout()
            fig.suptitle('$\ell_1/\ell_2=%.5f$' % l1l2)
            #plt.savefig(OUTPUT(DATASET_TRAIN, scaling=None, harmo=None, type="sensibility-l2-l1-enet_auc", ext="pdf"))
            pdf.savefig()  # saves the current figure into a pdf page
            fig.clf()
            plt.close()
