#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:54:23 2019

@author: ai258328
"""

import os
import numpy as np
import glob
import pandas as pd
#import nibabel
#import brainomics.image_atlas
import shutil
#import mulm
#import sklearn
import re
#from nilearn import plotting
#import matplotlib.pyplot as plt
#import scipy, scipy.ndimage
import xml.etree.ElementTree as ET
import re

# for the ROIs
BASE_PATH_icaar = '/neurospin/psy/start-icaar-eugei/derivatives/cat12/stats'
BASE_PATH_schizconnect = '/neurospin/psy/schizconnect-vip-prague/derivatives/cat12/stats'
BASE_PATH_bsnip = '/neurospin/psy/bsnip1/derivatives/cat12/stats'
BASE_PATH_biobd = '/neurospin/psy/bipolar/biobd/derivatives/cat12/stats'
# for the phenotypes
INPUT_CSV_icaar_bsnip_biobd = '/neurospin/psy_sbox/start-icaar-eugei/phenotype'
INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'

OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data'


"""
CREATION OF ONE SINGLE DATASET WITH ALL ROIs
"""

# STEP 1: concatenation of all cat12_rois_Vgm.tsv files

ROIs_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'cat12_rois_Vgm.tsv'),sep='\t')
ROIs_schizconnect = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'cat12_rois_Vgm.tsv'),sep='\t')
ROIs_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'cat12_rois_Vgm.tsv'),sep='\t')
ROIs_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd, 'cat12_rois_Vgm.tsv'),sep='\t')
assert ROIs_icaar.shape == (171, 143)
assert ROIs_schizconnect.shape == (738, 143)
assert ROIs_bsnip.shape == (1042, 143)
assert ROIs_biobd.shape == (746, 143)
ROIs = pd.concat([ROIs_icaar, ROIs_schizconnect, ROIs_bsnip, ROIs_biobd], ignore_index=True)
assert ROIs.shape == (2697, 143)

# STEP 2: concatenation of all cat12_tissues_volumes.tsv files

tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'cat12_tissues_volumes.tsv'),sep='\t')
tivo_schizconnect = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'cat12_tissues_volumes.tsv'),sep='\t')
tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'cat12_tissues_volumes.tsv'),sep='\t')
tivo_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd, 'cat12_tissues_volumes.tsv'),sep='\t')
assert tivo_icaar.shape == (171, 6)
assert tivo_schizconnect.shape == (738, 6)
assert tivo_bsnip.shape == (1042, 6)
assert tivo_biobd.shape == (746, 6)
tivo = pd.concat([tivo_icaar, tivo_schizconnect, tivo_bsnip, tivo_biobd], ignore_index=True)
assert tivo.shape == (2697, 6)

# STEP 3: normalization of the ROIs dataset

tivo['norm_ratio'] = 1500 / tivo.tiv
for i in range(ROIs.shape[0]):
    ROIs.iloc[i,1:] *= tivo.norm_ratio[i]
    
ROIs.to_csv(os.path.join(OUTPUT_PATH,'cat12_roi_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t', index=False)























