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
import argparse
import glob

from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('/home/ed203246/git/scripts/2021_wmh_memento+rundmc')
from file_utils import load_npy_nii

import nilearn.datasets
import brainomics.image_resample


FS = "/home/ed203246/data"

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

model_filename = os.path.join(MEMENTO_MODEL, "model.npz")
loadings_img_filename = os.path.join(MEMENTO_MODEL, "components-brain-maps.nii.gz")
mask_img_filename = os.path.join(MEMENTO_DATA, "mask.nii.gz")
data_filename = os.path.join(MEMENTO_DATA, "WMH_arr_msk.npy")
data_shape = (1755, 116037)
phenotypes_filename = os.path.join(MEMENTO_PATH , "population.csv")


model_filename = os.path.join(RUNDMC_MODEL, "model.npz")
loadings_img_filename = os.path.join(RUNDMC_MODEL, "components-brain-maps.nii.gz")
mask_img_filename = os.path.join(RUNDMC_DATA, "mask.nii.gz")
data_filename = os.path.join(RUNDMC_DATA, "WMH_2006.nii.gz")
data_shape = (267, 371278)

def load_data(model_filename, loadings_img_filename, mask_img_filename,
              data_filename, data_shape, phenotypes_filename):
    # Model
    model = np.load(model_filename)

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

    # center data
    data = data - data.mean(axis=0)

    assert mask_arr.sum() == data.shape[1] == data_shape[1]
    assert data.shape[0] == data_shape[0]

    # Phenoypes
    phenotypes = pd.read_csv(phenotypes_filename)

    return model, loadings_img, mask_img, data

# MEMENTO
mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
mask_arr = mask_img.get_data() == 1
assert mask_arr.sum() == 116037

# pop = pd.read_csv(os.path.join(ANALYSIS_DATA_PATH, "%s_participants.csv" % CONF["NI"]))

NI_arr_msk = np.load(os.path.join(ANALYSIS_DATA_PATH, "WMH_arr_msk.npy"))
assert NI_arr_msk.shape == (1755, 116037)

print("time elapsed={:.2f}s".format(time.time() - current_time))

# Check shape
#assert NI_arr.shape == tuple(list(CONF["shape"]) + [pop.shape[0]])
assert NI_arr_msk.shape == (pop.shape[0], mask_arr.sum())

X = NI_arr_msk - NI_arr_msk.mean(axis=0)
del NI_arr_msk



# assert X.shape == (503, 51637)
assert X.shape == (1755, 116037)
assert mask_arr.sum() == X.shape[1]
assert np.allclose(X.mean(axis=0), 0)


mod = "memento"
mod = "rundmc"

###############################################################################
#%% MEMENTO READ DATA

coefs_filename = os.path.join(MEMENTO_MODELS_PATH, "components-brain-maps.nii.gz")
coefs_img = nibabel.load(coefs_filename)
# mni152_t1_15mm_img = resample_to_img(mni152_t1_1mm_img, coefs_img)
#mni152_t1_15mm_img.to_filename(os.path.join(WD, "MNI152_T1_1.5mm.nii.gz"))

mod_memento = np.load(os.path.join(MEMENTO_MODELS_PATH, "model.npz"))
U, d, V, PC, explained_variance = mod_memento['U'], mod_memento['d'], mod_memento['V'], mod_memento['PC'], mod_memento['explained_variance']

NI_arr_msk = np.load(os.path.join(ANALYSIS_DATA_PATH, "WMH_arr_msk.npy"))
assert NI_arr_msk.shape == (1755, 116037)
wmhvol = NI_arr_msk.sum(axis=1)

phenotypes = pd.read_csv(PARTICIPANTS_CSV)

assert phenotypes.shape[0] == PC.shape[0]

pd.Series(wmhvol).describe()

thres = -np.inf
df_memento = phenotypes[wmhvol > thres]

df_memento["PC1"] = PC[wmhvol > thres, 0]
df_memento["PC2"] = PC[wmhvol > thres, 1]
df_memento["PC3"] = PC[wmhvol > thres, 2]
df_memento["wmh_tot"] = wmhvol[wmhvol > thres]
df_memento["sex"] = df_memento.sex.astype('object')


###############################################################################
#%% RUNDMC READ DATA












STUDY_PATH = '/neurospin/brainomics/2019_rundmc_wmh'
DATA_PATH = os.path.join(STUDY_PATH, 'sourcedata', 'wmhmask')

#ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analysis', '201905_rundmc_wmh_pca')
ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analyses', '201909_rundmc_wmh_pca')
ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
ANALYSIS_MODELS_PATH = os.path.join(ANALYSIS_PATH, "models")

OUTPUT_DIR = os.path.join(ANALYSIS_MODELS_PATH, '{key}')
