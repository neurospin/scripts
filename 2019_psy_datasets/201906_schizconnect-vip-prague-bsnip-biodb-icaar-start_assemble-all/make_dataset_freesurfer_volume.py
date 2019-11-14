#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:17:30 2019

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

# CONCATENATION OF aparc_stats_lh(rh)_volume.csv files

# remarque: 'BrainSegVoNotVent' et 'eTIV' sont présents dans toutes les cohortes sauf BioBD
# pour homogénéiser les tableaux, je les supprime lors de la construction des données "volume",
# mais elles seront dans les données "subcort_vol" (où elles sont présentes pour toutes les cohortes, BioBD compris) 

# for ICAAR-EUGEI-START

l_vol_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aparc_stats_lh_volume.csv'), sep=',')
assert l_vol_icaar.shape == (171, 37)
list(l_vol_icaar)
for col_name in l_vol_icaar:
    if col_name not in {'lh.aparc.volume','BrainSegVolNotVent','eTIV'}:
        l_vol_icaar = l_vol_icaar.rename(columns = {col_name: 'lh_' + col_name})
l_vol_icaar = l_vol_icaar.rename(columns = {'lh.aparc.volume':'participant_id'})

r_vol_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aparc_stats_rh_volume.csv'), sep=',')
assert r_vol_icaar.shape == (171, 37)
for col_name in r_vol_icaar:
    if col_name not in {'rh.aparc.volume','BrainSegVolNotVent','eTIV'}:
        r_vol_icaar = r_vol_icaar.rename(columns = {col_name: 'rh_' + col_name})
r_vol_icaar = r_vol_icaar.rename(columns = {'rh.aparc.volume':'participant_id'})

assert l_vol_icaar.eTIV.equals(r_vol_icaar.eTIV)
assert l_vol_icaar.BrainSegVolNotVent.equals(r_vol_icaar.BrainSegVolNotVent)
assert l_vol_icaar.participant_id.equals(r_vol_icaar.participant_id)

vol_icaar = pd.merge(l_vol_icaar, r_vol_icaar, how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del vol_icaar['BrainSegVolNotVent']
del vol_icaar['eTIV']
assert vol_icaar.shape == (171, 69)

# for SCHIZCONNECT-VIP-PRAGUE

l_vol_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aparc_stats_lh_volume.csv'), sep=',')
assert l_vol_schiz.shape == (735, 37)

for col_name in l_vol_schiz:
    if col_name not in {'lh.aparc.volume','BrainSegVolNotVent','eTIV'}:
        l_vol_schiz = l_vol_schiz.rename(columns = {col_name: 'lh_' + col_name})
l_vol_schiz = l_vol_schiz.rename(columns = {'lh.aparc.volume':'participant_id'})

r_vol_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aparc_stats_rh_volume.csv'), sep=',')
assert r_vol_schiz.shape == (735, 37)
for col_name in r_vol_schiz:
    if col_name not in {'rh.aparc.volume','BrainSegVolNotVent','eTIV'}:
        r_vol_schiz = r_vol_schiz.rename(columns = {col_name: 'rh_' + col_name})
r_vol_schiz = r_vol_schiz.rename(columns = {'rh.aparc.volume':'participant_id'})

assert l_vol_schiz.eTIV.equals(r_vol_schiz.eTIV)
assert l_vol_schiz.BrainSegVolNotVent.equals(r_vol_schiz.BrainSegVolNotVent)
assert l_vol_schiz.participant_id.equals(r_vol_schiz.participant_id)

vol_schiz = pd.merge(l_vol_schiz, r_vol_schiz, how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV' ])
del vol_schiz['eTIV']
del vol_schiz['BrainSegVolNotVent']
assert vol_schiz.shape == (735, 69)

# for BSNIP

l_vol_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aparc_stats_lh_volume.csv'), sep=',')
assert l_vol_bsnip.shape == (1045, 37)

for col_name in l_vol_bsnip:
    if col_name not in {'lh.aparc.volume','BrainSegVolNotVent','eTIV'}:
        l_vol_bsnip = l_vol_bsnip.rename(columns = {col_name: 'lh_' + col_name})
l_vol_bsnip = l_vol_bsnip.rename(columns = {'lh.aparc.volume':'participant_id'})

r_vol_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aparc_stats_rh_volume.csv'), sep=',')
assert r_vol_bsnip.shape == (1045, 37)
for col_name in r_vol_bsnip:
    if col_name not in {'rh.aparc.volume','BrainSegVolNotVent','eTIV'}:
        r_vol_bsnip = r_vol_bsnip.rename(columns = {col_name: 'rh_' + col_name})
r_vol_bsnip = r_vol_bsnip.rename(columns = {'rh.aparc.volume':'participant_id'})

assert l_vol_bsnip.eTIV.equals(r_vol_bsnip.eTIV)
assert l_vol_bsnip.BrainSegVolNotVent.equals(r_vol_bsnip.BrainSegVolNotVent)
assert l_vol_bsnip.participant_id.equals(r_vol_bsnip.participant_id)

vol_bsnip = pd.merge(l_vol_bsnip, r_vol_bsnip, how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV' ])
del vol_bsnip['eTIV']
del vol_bsnip['BrainSegVolNotVent']
assert vol_bsnip.shape == (1045, 69)


# for BIOBD

l_vol_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aparc_stats_lh_volume.csv'), sep=',')
assert l_vol_biobd.shape == (366, 35)

for col_name in l_vol_biobd:
    if col_name not in {'lh.aparc.volume'}:
        l_vol_biobd = l_vol_biobd.rename(columns = {col_name: 'lh_' + col_name})
l_vol_biobd = l_vol_biobd.rename(columns = {'lh.aparc.volume':'participant_id'})

r_vol_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aparc_stats_rh_volume.csv'), sep=',')
assert r_vol_biobd.shape == (366, 35)

for col_name in r_vol_biobd:
    if col_name not in {'rh.aparc.volume'}:
        r_vol_biobd = r_vol_biobd.rename(columns = {col_name: 'rh_' + col_name})
r_vol_biobd = r_vol_biobd.rename(columns = {'rh.aparc.volume':'participant_id'})

assert l_vol_biobd.participant_id.equals(r_vol_biobd.participant_id)

vol_biobd = pd.merge(l_vol_biobd, r_vol_biobd, how='outer', on=['participant_id'])
assert vol_biobd.shape == (366, 69)


    
# concatenate all

assert list(vol_icaar) == list(vol_schiz) == list(vol_bsnip) == list(vol_biobd)
vol_all = pd.concat([vol_icaar, vol_schiz, vol_bsnip, vol_biobd], axis=0)
assert vol_all.shape == (2317, 69)

regex = re.compile("sub-([^_]+)")
vol_all.iloc[:,0] = [regex.findall(s)[0] for s in vol_all.iloc[:,0]]

vol_all.to_csv(os.path.join(OUTPUT_PATH,'FS_volume_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t', index=False)
