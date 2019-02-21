#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:08:34 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import glob
import nibabel as nib  # import generate a FutureWarning
import matplotlib.pyplot as plt
import os.path
import scipy.io
import scipy.linalg

#BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects'
BASE_PATH = '/neurospin/brainomics/2016_schizConnect/2019_analysis/all_studies+VIP/VBM/all_subjects/results/Leave_One_Site_Out/LOSO_enet_centered_by_site_all'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/completed_schizconnect_metaData_1829.csv'
OUTPUT_CSV = os.path.join(BASE_PATH,"participants.tsv")


INPUT_CSV_COBRE = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"
INPUT_CSV_NMorphCH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
INPUT_CSV_NUSDAST = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
INPUT_CSV_VIP = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population.csv"


clinic_COBRE = pd.read_csv(INPUT_CSV_COBRE)
clinic_NMorphCH = pd.read_csv(INPUT_CSV_NMorphCH)
clinic_NUSDAST = pd.read_csv(INPUT_CSV_NUSDAST)
clinic_VIP = pd.read_csv(INPUT_CSV_VIP)
pop = pd.concat([clinic_COBRE, clinic_NMorphCH, clinic_NUSDAST], axis=0)
pop = pop[['subjectid', 'path_VBM','sex_num','dx_num','site','age']]

#Since VIP is not a SchizConnect database, need to reorganise some infos
clinic_VIP = clinic_VIP[['path_VBM','sex_code','dx','age']]
clinic_VIP["site"] = "vip"
clinic_VIP["sex_num"] = clinic_VIP["sex_code"]
clinic_VIP["dx_num"] = clinic_VIP["dx"]
clinic_VIP['subjectid'] = [os.path.splitext(os.path.basename(p))[0].replace('mwc1', '') for p in clinic_VIP['path_VBM']]
clinic_VIP = clinic_VIP[['subjectid', 'path_VBM','sex_num','dx_num','site','age']]


pop = pd.concat([pop, clinic_VIP], axis=0)
pop = pop[['subjectid',"dx_num","path_VBM","sex_num","site","age"]]
assert pop.shape == (606, 6)

#pop.site.unique()
SITE_MAP = {'MRN': 1, 'NU': 2, "WUSTL" : 3,"vip" : 4}
pop['site_num'] = pop["site"].map(SITE_MAP)
assert sum(pop.site_num.values==1) == 164
assert sum(pop.site_num.values==2) == 80
assert sum(pop.site_num.values==3) == 270
assert sum(pop.site_num.values==4) == 92


assert sum(pop.dx_num.values==0) == 330
assert sum(pop.dx_num.values==1) == 276

assert sum(pop.sex_num.values==0) == 374
assert sum(pop.sex_num.values==1) == 232
pop.reset_index(drop=True, inplace=True)
pop.columns = ['participant_id', 'dx_num', 'path', 'sex_num', 'site', 'age', 'site_num']
#pop[pop.site_num==4].age.mean()
#Out[116]: 34.37954139368672
#
#pop[pop.site_num==3].age.mean()
#Out[117]: 30.581481481481482
#
#pop[pop.site_num==2].age.mean()
#Out[118]: 32.05
#
#pop[pop.site_num==1].age.mean()
#Out[119]: 37.84146341463415

##############################################################################
# Compute TIV

tissues_vol = list()

for row in pop.itertuples():
    print(row.path)
    # Load an threshold map, load t1
    gm_filename = row.path.replace("mwc1", "c1")
    wm_filename = row.path.replace("mwc1", "c2")
    csf_filename = row.path.replace("mwc1", "c3")

    gm_img = nib.load(gm_filename)
    voxsize = np.asarray(gm_img.header.get_zooms())
    voxvol = voxsize.prod()  # mm3
    gm_arr = gm_img.get_data().squeeze()
    gm_vol = gm_arr.sum() * voxvol / (10 ** 6) # l

    wm_img = nib.load(wm_filename)
    wm_arr = wm_img.get_data().squeeze()
    wm_vol = wm_arr.sum() * voxvol / (10 ** 6) # l

    csf_img = nib.load(csf_filename)
    csf_arr = csf_img.get_data().squeeze()
    csf_vol = csf_arr.sum() * voxvol / (10 ** 6) # l

    mgm_img = nib.load(row.path)
    voxsize = np.asarray(mgm_img.header.get_zooms())
    voxvol = voxsize.prod()  # mm3
    mgm_arr = mgm_img.get_data().squeeze()
    mgm_vol = mgm_arr.sum() * voxvol / (10 ** 6) # l

    # Get global scaling from linear transfo
    mat = scipy.io.loadmat(row.path.replace("mwc1", "").replace(".nii", "_seg8.mat"))
    det = scipy.linalg.det(mat['Affine'][:3, :3])

    tissues_vol.append([row.participant_id, gm_vol, wm_vol, csf_vol, mgm_vol, det])

tissues_vol = pd.DataFrame(tissues_vol, columns = ["participant_id", "GMvol_l", "WMvol_l", "CSFvol_l", "mwGMvol_l", "gscaling"])
tissues_vol["TIV_l"] = tissues_vol[["GMvol_l", "WMvol_l", "CSFvol_l"]].sum(axis=1)
tissues_vol["GMratio"] = tissues_vol["GMvol_l"] / tissues_vol["TIV_l"]
tissues_vol["WMratio"] = tissues_vol["WMvol_l"] / tissues_vol["TIV_l"]
tissues_vol["CSFratio"] = tissues_vol["CSFvol_l"] / tissues_vol["TIV_l"]
assert np.all(tissues_vol.participant_id == pop.participant_id)
tissues_vol.pop('participant_id')
pop = pd.concat([pop, tissues_vol], axis=1)

# Remove outliers

val = pop["GMratio"]
mad = 1.4826 * np.median(np.abs(val - val.median()))
pop_ = pop[((val - val.median()).abs() < 3 * mad)]
#pop_ = pop
plt.plot(pop_.GMvol_l, pop_.mwGMvol_l, "o")
plt.plot(pop_.TIV_l, pop_.gscaling, "o")

np.corrcoef(pop_.TIV_l, pop_.gscaling)

#plt.plot(pop_.GMvol_l + pop_.WMvol_l, pop_.gscaling, "o")

pop_.describe()
"""
            index      dx_num     sex_num         age    site_num     ...        gscaling       TIV_l     GMratio     WMratio    CSFratio
count  603.000000  603.000000  603.000000  603.000000  603.000000     ...      603.000000  603.000000  603.000000  603.000000  603.000000
mean    94.187396    0.452736    0.383085   33.294977    2.480929     ...        1.255652    1.463532    0.494375    0.307119    0.198506
std     70.888240    0.498174    0.486542   12.344030    1.044310     ...        0.129916    0.155243    0.048188    0.022876    0.054847
min      0.000000    0.000000    0.000000   14.000000    1.000000     ...        0.924040    1.004842    0.350563    0.230926    0.060704
25%     37.500000    0.000000    0.000000   23.000000    1.000000     ...        1.157707    1.348996    0.462102    0.292638    0.160356
50%     75.000000    0.000000    0.000000   30.000000    3.000000     ...        1.245652    1.462985    0.492955    0.306661    0.196806
75%    140.500000    1.000000    1.000000   43.000000    3.000000     ...        1.333024    1.573949    0.527826    0.321272    0.235753
max    269.000000    1.000000    1.000000   66.000000    4.000000     ...        1.631502    1.982385    0.634946    0.371960    0.408846
"""
pop = pop_
# Save population information
#pop.to_csv(OUTPUT_CSV, index=False)
pop.to_csv(OUTPUT_CSV, sep='\t', index=False)
