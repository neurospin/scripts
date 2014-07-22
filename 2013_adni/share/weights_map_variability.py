# -*- coding: utf-8 -*-
"""
@author: edouard.Duchesnay@cea.fr

ls -d */results|while read f ; do mv $f "$(dirname $f)/5cv"; done

Compute variability over weights maps

cd /neurospin/brainomics/2013_adni
find . -name beta.npy | while read f ; do gzip $f ; done

INPUT:
- mask.nii
- glob pattern directory/*/1.1

OUTPUT:

"""

import os
import numpy as np
import glob
import nibabel
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import mapreduce

BASE_PATH = "/neurospin/brainomics/2013_adni"
BASE_PATH = "/home/ed203246/tmp"

STUDY = "AD-CTL_cs"
#STUDY = "MCIc-CTL_cs"
#STUDY = "MCIc-MCInc_cs"

penalty_start = 2

PARAMS =\
[[0.10, 0.00, 1.00, 0.00, -1.0],
 [0.10, 0.00, 0.50, 0.50, -1.0],
 [0.10, 1.00, 0.00, 0.00, -1.0],
 [0.10, 0.50, 0.00, 0.50, -1.0],
 [0.10, 0.00, 0.00, 1.00, -1.0],
 [0.10, 0.50, 0.50, 0.00, -1.0],
 [0.10, 0.35, 0.35, 0.30, -1.0]]

param_sep = "_"
INPUT = '5cv'
OUTPUT = "results"

for param in PARAMS:
    #param = PARAMS[0]
    INPUT_MASK = os.path.join(BASE_PATH, STUDY, "mask.nii")
    param_str = param_sep.join([str(p) for p in param])
    INPUT_PATTERN = os.path.join(BASE_PATH, STUDY, INPUT, "*", param_str)
    OUTPUT_DIR = os.path.join(BASE_PATH, STUDY, OUTPUT)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    mask_image = nibabel.load(INPUT_MASK)
    mask = mask_image.get_data() != 0
    values = [mapreduce.OutputCollector(p) for p in glob.glob(INPUT_PATTERN)]
    values = [item.load() for item in values]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    fpr, tpr, testholds = roc_curve(y_true, prob_pred)
    auc_ = auc(fpr, tpr)
    n_ite = None
    betas = np.hstack([item["beta"] for item in values]).T
    R = np.corrcoef(betas)
    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    beta_sd = np.std(betas, axis=0)
    beta_mean = np.mean(betas, axis=0)
    #beta_sd.max()
    arr = np.zeros(mask.shape)
    arr[mask] = beta_sd[penalty_start:]
    out_im = nibabel.Nifti1Image(arr,affine=mask_image.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_DIR, param_str + "_sd.nii"))
    arr[mask] = beta_mean[penalty_start:]
    out_im = nibabel.Nifti1Image(arr,affine=mask_image.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_DIR, param_str + "_mean.nii"))

# run scripts/2013_adni/proj_classif_share/03_weights_map_variability.py

