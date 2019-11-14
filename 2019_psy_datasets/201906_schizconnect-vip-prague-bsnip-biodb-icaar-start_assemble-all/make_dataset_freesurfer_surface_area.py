#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:41:04 2019

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

# CONCATENATION OF aparc_stats_lh(and _rh)_area.csv

# remarque: 'BrainSegVoNotVent' et 'eTIV' sont présents dans toutes les cohortes sauf BioBD
# pour homogénéiser les tableaux, je les supprime lors de la construction des données "surface_area",
# mais elles seront dans les données "subcort_vol" (où elles sont présentes pour toutes les cohortes, BioBD compris) 

# for ICAAR-EUGEI-START

l_area_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aparc_stats_lh_area.csv'), sep=',')
assert l_area_icaar.shape == (171, 38)

for col_name in l_area_icaar:
    if col_name not in {'lh.aparc.area','BrainSegVolNotVent','eTIV'}:
        l_area_icaar = l_area_icaar.rename(columns = {col_name: 'lh_' + col_name})
l_area_icaar = l_area_icaar.rename(columns = {'lh.aparc.area':'participant_id'})

r_area_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aparc_stats_rh_area.csv'), sep=',')
assert r_area_icaar.shape == (171, 38)

for col_name in r_area_icaar:
    if col_name not in {'rh.aparc.area','BrainSegVolNotVent','eTIV'}:
        r_area_icaar = r_area_icaar.rename(columns = {col_name: 'rh_' + col_name})
r_area_icaar = r_area_icaar.rename(columns = {'rh.aparc.area':'participant_id'})

assert l_area_icaar.eTIV.equals(r_area_icaar.eTIV)
assert l_area_icaar.BrainSegVolNotVent.equals(r_area_icaar.BrainSegVolNotVent)
assert l_area_icaar.participant_id.equals(r_area_icaar.participant_id)

area_icaar = pd.merge(l_area_icaar, r_area_icaar,how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del area_icaar['BrainSegVolNotVent']
del area_icaar['eTIV']
assert area_icaar.shape == (171, 71)


# for SCHIZCONNECT-VIP-PRAGUE

l_area_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aparc_stats_lh_area.csv'), sep=',')
assert l_area_schiz.shape == (735, 38)

for col_name in l_area_schiz:
    if col_name not in {'lh.aparc.area','BrainSegVolNotVent','eTIV'}:
        l_area_schiz = l_area_schiz.rename(columns = {col_name: 'lh_' + col_name})
l_area_schiz = l_area_schiz.rename(columns = {'lh.aparc.area':'participant_id'})

r_area_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aparc_stats_rh_area.csv'), sep=',')
assert r_area_schiz.shape == (735, 38)

for col_name in r_area_schiz:
    if col_name not in {'rh.aparc.area','BrainSegVolNotVent','eTIV'}:
        r_area_schiz = r_area_schiz.rename(columns = {col_name: 'rh_' + col_name})
r_area_schiz = r_area_schiz.rename(columns = {'rh.aparc.area':'participant_id'})

assert l_area_schiz.eTIV.equals(r_area_schiz.eTIV)
assert l_area_schiz.BrainSegVolNotVent.equals(r_area_schiz.BrainSegVolNotVent)
assert l_area_schiz.participant_id.equals(r_area_schiz.participant_id)

area_schiz = pd.merge(l_area_schiz, r_area_schiz,how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del area_schiz['BrainSegVolNotVent']
del area_schiz['eTIV']
assert area_schiz.shape == (735, 71)


# for BSNIP

l_area_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aparc_stats_lh_area.csv'), sep=',')
assert l_area_bsnip.shape == (1045, 38)

for col_name in l_area_bsnip:
    if col_name not in {'lh.aparc.area','BrainSegVolNotVent','eTIV'}:
        l_area_bsnip = l_area_bsnip.rename(columns = {col_name: 'lh_' + col_name})
l_area_bsnip = l_area_bsnip.rename(columns = {'lh.aparc.area':'participant_id'})

r_area_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aparc_stats_rh_area.csv'), sep=',')
assert r_area_bsnip.shape == (1045, 38)

for col_name in r_area_bsnip:
    if col_name not in {'rh.aparc.area','BrainSegVolNotVent','eTIV'}:
        r_area_bsnip = r_area_bsnip.rename(columns = {col_name: 'rh_' + col_name})
r_area_bsnip = r_area_bsnip.rename(columns = {'rh.aparc.area':'participant_id'})

assert l_area_bsnip.eTIV.equals(r_area_bsnip.eTIV)
assert l_area_bsnip.BrainSegVolNotVent.equals(r_area_bsnip.BrainSegVolNotVent)
assert l_area_bsnip.participant_id.equals(r_area_bsnip.participant_id)

area_bsnip = pd.merge(l_area_bsnip, r_area_bsnip, how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del area_bsnip['BrainSegVolNotVent']
del area_bsnip['eTIV']
assert area_bsnip.shape == (1045, 71)


# for BIOBD

l_area_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aparc_stats_lh_area.csv'), sep=',')
assert l_area_biobd.shape == (366, 36)

for col_name in l_area_biobd:
    if col_name not in {'lh.aparc.area'}:
        l_area_biobd = l_area_biobd.rename(columns = {col_name: 'lh_' + col_name})
l_area_biobd = l_area_biobd.rename(columns = {'lh.aparc.area':'participant_id'})

r_area_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aparc_stats_rh_area.csv'), sep=',')
assert r_area_biobd.shape == (366, 36)

for col_name in r_area_biobd:
    if col_name not in {'rh.aparc.area'}:
        r_area_biobd = r_area_biobd.rename(columns = {col_name: 'rh_' + col_name})
r_area_biobd = r_area_biobd.rename(columns = {'rh.aparc.area':'participant_id'})

assert l_area_biobd.participant_id.equals(r_area_biobd.participant_id)

area_biobd = pd.merge(l_area_biobd, r_area_biobd, how='outer', on=['participant_id'])
assert area_biobd.shape == (366, 71)



# concatenate all

assert list(area_icaar) == list(area_schiz) == list(area_bsnip) == list(area_biobd)
area_all = pd.concat([area_icaar, area_schiz, area_bsnip, area_biobd], axis=0)
assert area_all.shape == (2317, 71)

regex = re.compile("sub-([^_]+)")
area_all.iloc[:,0] = [regex.findall(s)[0] for s in area_all.iloc[:,0]]

area_all.to_csv(os.path.join(OUTPUT_PATH,'FS_surface_area_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t', index=False)

