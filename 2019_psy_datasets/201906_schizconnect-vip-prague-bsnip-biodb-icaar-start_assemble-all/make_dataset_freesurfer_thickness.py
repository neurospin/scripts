#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:07:49 2019

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

# CONCATENATION OF aparc_stats_lh(rh)_thickness.csv and aseg_stats_volume.csv

# remarque: 'BrainSegVoNotVent' et 'eTIV' sont présents dans toutes les cohortes sauf BioBD
# pour homogénéiser les tableaux, je les supprime lors de la construction des données "thickness",
# mais elles seront dans les données "subcort_vol" (où elles sont présentes pour toutes les cohortes, BioBD compris) 

# for ICAAR-EUGEI-START

l_th_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aparc_stats_lh_thickness.csv'), sep=',')
assert l_th_icaar.shape == (171, 38)
for col_name in l_th_icaar:
    if col_name not in {'lh.aparc.thickness','BrainSegVolNotVent','eTIV'}:
        l_th_icaar = l_th_icaar.rename(columns = {col_name: 'lh_' + col_name})
l_th_icaar = l_th_icaar.rename(columns = {'lh.aparc.thickness':'participant_id'})

r_th_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar,'aparc_stats_rh_thickness.csv'), sep=',')
assert r_th_icaar.shape == (171, 38)
for col_name in r_th_icaar:
    if col_name not in {'rh.aparc.thickness','BrainSegVolNotVent','eTIV'}:
        r_th_icaar = r_th_icaar.rename(columns = {col_name: 'rh_' + col_name})
r_th_icaar = r_th_icaar.rename(columns = {'rh.aparc.thickness':'participant_id'})

assert l_th_icaar.eTIV.equals(r_th_icaar.eTIV)
assert l_th_icaar.BrainSegVolNotVent.equals(r_th_icaar.BrainSegVolNotVent)
assert l_th_icaar.participant_id.equals(r_th_icaar.participant_id)

th_icaar = pd.merge(l_th_icaar, r_th_icaar,how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del th_icaar['BrainSegVolNotVent']
del th_icaar['eTIV']
assert th_icaar.shape == (171, 71)

# for SCHIZCONNECT-VIP-PRAGUE

l_th_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aparc_stats_lh_thickness.csv'), sep=',')
assert l_th_schiz.shape == (735, 38)
for col_name in l_th_schiz:
    if col_name not in {'lh.aparc.thickness','BrainSegVolNotVent','eTIV'}:
        l_th_schiz = l_th_schiz.rename(columns = {col_name: 'lh_' + col_name})
l_th_schiz = l_th_schiz.rename(columns = {'lh.aparc.thickness':'participant_id'})

r_th_schiz = pd.read_csv(os.path.join(BASE_PATH_schizconnect,'aparc_stats_rh_thickness.csv'), sep=',')
assert r_th_schiz.shape == (735, 38)
for col_name in r_th_schiz:
    if col_name not in {'rh.aparc.thickness','BrainSegVolNotVent','eTIV'}:
        r_th_schiz = r_th_schiz.rename(columns = {col_name: 'rh_' + col_name})
r_th_schiz = r_th_schiz.rename(columns = {'rh.aparc.thickness':'participant_id'})

assert l_th_schiz.eTIV.equals(r_th_schiz.eTIV)
assert l_th_schiz.BrainSegVolNotVent.equals(r_th_schiz.BrainSegVolNotVent)
assert l_th_schiz.participant_id.equals(r_th_schiz.participant_id)

th_schiz = pd.merge(l_th_schiz, r_th_schiz,how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del th_schiz['BrainSegVolNotVent']
del th_schiz['eTIV']
assert th_schiz.shape == (735, 71)


# for BSNIP

l_th_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aparc_stats_lh_thickness.csv'), sep=',')
assert l_th_bsnip.shape == (1045, 38)
for col_name in l_th_bsnip:
    if col_name not in {'lh.aparc.thickness','BrainSegVolNotVent','eTIV'}:
        l_th_bsnip = l_th_bsnip.rename(columns = {col_name: 'lh_' + col_name})
l_th_bsnip = l_th_bsnip.rename(columns = {'lh.aparc.thickness':'participant_id'})

r_th_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip,'aparc_stats_rh_thickness.csv'), sep=',')
assert r_th_bsnip.shape == (1045, 38)
for col_name in r_th_bsnip:
    if col_name not in {'rh.aparc.thickness','BrainSegVolNotVent','eTIV'}:
        r_th_bsnip = r_th_bsnip.rename(columns = {col_name: 'rh_' + col_name})
r_th_bsnip = r_th_bsnip.rename(columns = {'rh.aparc.thickness':'participant_id'})

assert l_th_bsnip.eTIV.equals(r_th_bsnip.eTIV)
assert l_th_bsnip.BrainSegVolNotVent.equals(r_th_bsnip.BrainSegVolNotVent)
assert l_th_bsnip.participant_id.equals(r_th_bsnip.participant_id)

th_bsnip = pd.merge(l_th_bsnip, r_th_bsnip,how='outer', on=['participant_id','BrainSegVolNotVent', 'eTIV'])
del th_bsnip['BrainSegVolNotVent']
del th_bsnip['eTIV']
assert th_bsnip.shape == (1045, 71)


# for BIOBD

l_th_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aparc_stats_lh_thickness.csv'), sep=',')
assert l_th_biobd.shape == (366, 36)
for col_name in l_th_biobd:
    if col_name not in {'lh.aparc.thickness'}:
        l_th_biobd = l_th_biobd.rename(columns = {col_name: 'lh_' + col_name})
l_th_biobd = l_th_biobd.rename(columns = {'lh.aparc.thickness':'participant_id'})

r_th_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd,'aparc_stats_rh_thickness.csv'), sep=',')
assert r_th_biobd.shape == (366, 36)
for col_name in r_th_biobd:
    if col_name not in {'rh.aparc.thickness'}:
        r_th_biobd = r_th_biobd.rename(columns = {col_name: 'rh_' + col_name})
r_th_biobd = r_th_biobd.rename(columns = {'rh.aparc.thickness':'participant_id'})

assert l_th_biobd.participant_id.equals(r_th_biobd.participant_id)

th_biobd = pd.merge(l_th_biobd, r_th_biobd, how='outer', on=['participant_id'])
assert th_biobd.shape == (366, 71)

    
# concatenate all

assert list(th_icaar) == list(th_schiz) == list(th_bsnip) == list(th_biobd)
th_all = pd.concat([th_icaar, th_schiz, th_bsnip, th_biobd], axis=0)
assert th_all.shape == (2317, 71)

regex = re.compile("sub-([^_]+)")
th_all.iloc[:,0] = [regex.findall(s)[0] for s in th_all.iloc[:,0]]

th_all.to_csv(os.path.join(OUTPUT_PATH,'FS_thickness_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t', index=False)

