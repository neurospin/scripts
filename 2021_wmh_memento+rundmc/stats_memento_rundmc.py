#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:28:43 2021

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
from nilearn.image import resample_to_img
from sklearn.decomposition import PCA
import argparse
import glob

from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('/home/ed203246/git/scripts/2021_wmh_memento+rundmc')
from file_utils import load_npy_nii

#import nilearn.datasets
#import brainomics.image_resample

FS = "/home/ed203246/data"

#%% MEMENTO_RUNDMC
MEMENTO_RUNDMC_PATH = "{FS}/2021_wmh_memento+rundmc".format(FS=FS)
MEMENTO_RUNDMC_DATA = os.path.join(MEMENTO_RUNDMC_PATH, "data")

#%% MEMENTO
MEMENTO_PATH = "{FS}/2017_memento/analysis/WMH".format(FS=FS)
MEMENTO_DATA = os.path.join(MEMENTO_PATH, "data")
MEMENTO_MODEL = os.path.join(MEMENTO_PATH, "models/pca_enettv_0.000010_1.000_0.001")
os.listdir(MEMENTO_MODEL)

#%% RUNDMC
RUNDMC_PATH = "{FS}/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca".format(FS=FS)
RUNDMC_DATA = os.path.join(RUNDMC_PATH, "data")
RUNDMC_MODEL = os.path.join(RUNDMC_PATH, "models/pca_enettv_0.000035_1.000_0.005")

os.listdir(RUNDMC_MODEL)

###############################################################################
#%% LOAD DATA
tissue_filename = os.path.join(MEMENTO_RUNDMC_DATA, "mask_cortex_ventricles_dill5mm_deepwm_mni_1mm_img.nii.gz")

# MEMENTO
model_filename = os.path.join(MEMENTO_MODEL, "model.npz")
loadings_img_filename = os.path.join(MEMENTO_MODEL, "components-brain-maps.nii.gz")
mask_img_filename = os.path.join(MEMENTO_DATA, "mask.nii.gz")
data_filename = os.path.join(MEMENTO_DATA, "WMH_arr_msk.npy")
data_shape = (1755, 116037)
# phenotypes_filename = os.path.join(MEMENTO_PATH , "population.csv")
# participants_with_phenotypes_filename = ??

# RUNDMC
model_filename = os.path.join(RUNDMC_MODEL, "model.npz")
loadings_img_filename = os.path.join(RUNDMC_MODEL, "components-brain-maps.nii.gz")
mask_img_filename = os.path.join(RUNDMC_DATA, "mask.nii.gz")
# See /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py
data_filename = os.path.join(RUNDMC_DATA, "WMH_2006.nii.gz")
#participants_filename = os.path.join(RUNDMC_DATA, "WMH_2006_participants.csv")
participants_with_phenotypes_filename = os.path.join(RUNDMC_DATA, "WMH_2006_participants_with_phenotypes.csv")

data_shape = (267, 371278)
# phenotypes_filename = os.path.join(MEMENTO_PATH , "population.csv")

def load_data(model_filename, loadings_img_filename, mask_img_filename,
              data_filename, data_shape, participants_with_phenotypes_filename):
    # Phenoypes
    phenotypes = pd.read_csv(participants_with_phenotypes_filename)

    # Model
    model = np.load(model_filename)
    U, d, V, PC, explained_variance = model['U'], model['d'], model['V'], model['PC'], \
                                      model['explained_variance']

    assert phenotypes.shape[0] == PC.shape[0]
    phenotypes["PC1"] = PC[:, 0]
    phenotypes["PC2"] = PC[:, 1]
    phenotypes["PC3"] = PC[:, 2]

    # Loading_img
    loadings_img = nibabel.load(loadings_img_filename)

    # mask_img
    mask_img = nibabel.load(mask_img_filename)
    assert loadings_img.get_fdata().shape[:-1] == mask_img.get_fdata().shape
    assert np.all(loadings_img.affine == mask_img.affine)

    mask_arr = mask_img.get_fdata() == 1
    assert mask_arr.sum() == data_shape[1]

    # Data
    data = load_npy_nii(data_filename)
    if isinstance(data, np.ndarray):
        pass
    elif hasattr(data, 'affine'):
        data_img = data
        data = data_img.get_fdata()[mask_arr].T

    phenotypes["wmh_tot"] = data.sum(axis=1)

    assert mask_arr.sum() == data.shape[1] == data_shape[1]
    assert data.shape[0] == data_shape[0]

    # wmh_by_tissue
    tissue_img = nibabel.load(tissue_filename)
    tissue_img = resample_to_img(source_img=tissue_img, target_img=mask_img, interpolation='nearest')
    tissue_arr = tissue_img.get_fdata().astype(int)
    tissue_labels = dict(cortex=1, ventricles=2, deepwm=3)

    assert np.all(tissue_arr == tissue_img.get_fdata())
    wmh_pc_by_tissue = pd.DataFrame({"wmh_pc_%s" % tissue:PCA(n_components=1).fit_transform(data[:, tissue_arr[mask_arr] == lab]).ravel()
        for tissue, lab in tissue_labels.items()})
    wmh_sum_by_tissue = pd.DataFrame({"wmh_sum_%s" % tissue:data[:, tissue_arr[mask_arr] == lab].sum(axis=1)
        for tissue, lab in tissue_labels.items()})
    phenotypes = pd.concat([phenotypes, wmh_pc_by_tissue, wmh_sum_by_tissue], axis=1)

    # center data
    data = data - data.mean(axis=0)

    return model, loadings_img, mask_img, data, phenotypes









