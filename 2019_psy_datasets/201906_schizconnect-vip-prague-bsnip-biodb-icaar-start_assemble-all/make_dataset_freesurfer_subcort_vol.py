#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:45:12 2019

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
BASE_PATH_icaar = '/neurospin/psy/start-icaar-eugei/derivatives/fsstats/ses-V1/stats'
BASE_PATH_schizconnect = '/neurospin/psy/schizconnect-vip-prague/derivatives/fsstats/ses-v1'
BASE_PATH_bsnip = '/neurospin/psy/bsnip1/derivatives/fsstats/ses-V1'
BASE_PATH_biobd = '/neurospin/psy/bipolar/biobd/derivatives/fsstats'

OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data'

# for ICAAR-EUGEI-START

vol_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aseg_stats_volume.csv'), sep=',')
assert vol_icaar.shape == (171, 67)
vol_icaar = vol_icaar.rename(columns = {'Measure:volume':'participant_id'})

# for SCHIZCONNECT-VIP-PRAGUE

vol_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aseg_stats_volume.csv'), sep=',')
assert vol_schiz.shape == (735, 67)
vol_schiz = vol_schiz.rename(columns = {'Measure:volume':'participant_id'})

# for BSNIP

vol_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aseg_stats_volume.csv'), sep=',')
assert vol_bsnip.shape == (1045, 67)
vol_bsnip = vol_bsnip.rename(columns = {'Measure:volume':'participant_id'})

# for BIOBD

vol_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aseg_stats_volume.csv'), sep=',')
assert vol_biobd.shape == (366, 67)
vol_biobd = vol_biobd.rename(columns = {'Measure:volume':'participant_id'})

# ATTENTION, dans fs_biobd, on a trois colonnes différentes des autres: 
#CorticalWhiteMatterVol
#lhCorticalWhiteMatterVol
#rhCorticalWhiteMatterVol
#Alors que dans les autres groupes, on a à la place:
#CerebralWhiteMatterVol
#lhCerebralWhiteMatterVol
#rhCerebralWhiteMatterVol
#!!! Je les renomme mais vérifier que c'est ok
vol_biobd = vol_biobd.rename(columns = {'lhCorticalWhiteMatterVol':'lhCerebralWhiteMatterVol',
    'rhCorticalWhiteMatterVol':'rhCerebralWhiteMatterVol',
    'CorticalWhiteMatterVol':'CerebralWhiteMatterVol'})

assert list(vol_icaar) == list(vol_schiz) == list(vol_bsnip) == list(vol_biobd)
vol_all = pd.concat([vol_icaar, vol_schiz, vol_bsnip, vol_biobd], axis=0)
assert vol_all.shape == (2317, 67)

regex = re.compile("sub-([^_]+)")
vol_all.iloc[:,0] = [regex.findall(s)[0] for s in vol_all.iloc[:,0]]

vol_all.to_csv(os.path.join(OUTPUT_PATH,'FS_subcort_vol_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t', index=False)
