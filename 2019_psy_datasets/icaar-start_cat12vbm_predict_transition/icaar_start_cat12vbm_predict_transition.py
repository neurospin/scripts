#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:06:56 2020

@author: ed203246
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


###############################################################################
dataset = 'icaar-start'

scaling, harmo = 'gs', 'res:site+age+sex(diag)'


ni_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data32", ext="npy"), mmap_mode='r')
ni_participants_df = pd.read_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_data() != 0

msk = ni_participants_df["diagnosis"].isin(['UHR-C', 'UHR-NC']) &  ni_participants_df["irm"].isin(['M0'])
msk.sum()
ni_participants_df = ni_participants_df[msk]
ni_arr = ni_arr[msk]

ni_participants_df["transition"] = ni_participants_df["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0}).values

X = ni_arr[:, 0, mask_arr].astype('float64')
ni_arr.shape
X.shape


import sklearn.metrics as metrics
import sklearn.ensemble
import sklearn.linear_model as lm

def balanced_acc(estimator, X, y, **kwargs):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()

estimators_clf = dict(
    LogisticRegressionCV_balanced_inter=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1,
                                                                cv=5),
    LogisticRegressionCV_balanced_nointer=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc,
                                                                  n_jobs=1, cv=5,
                                                                  fit_intercept=False))

%time ml_, ml_folds_,  _ = ml_predictions(X=X, y=ni_participants_df["transition"].values, estimators=estimators_clf, cv=None)

with pd.ExcelWriter(OUTPUT(dataset, scaling=None, harmo=None, type="ml-scores-transition", ext="xlsx")) as writer:
    ml_.to_excel(writer, sheet_name='transition', index=False)
    ml_folds_.to_excel(writer, sheet_name='transition_folds', index=False)
