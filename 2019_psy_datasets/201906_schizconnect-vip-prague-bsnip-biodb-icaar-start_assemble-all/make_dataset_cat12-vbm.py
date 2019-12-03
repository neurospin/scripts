#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:08:41 CET 2019

@author: edouard.duchesnay

float 64 vs float 32
https://en.wikipedia.org/wiki/IEEE_754

%load_ext autoreload
%autoreload 2

HC age
train:

HC between 15-30
train: schizconnect-vip_mwp1_gs_data-64
val:

set(df.study)
Out[278]: {'BIOBD', 'BSNIP', 'ICAAR_EUGEI_START', 'PRAGUE', 'SCHIZCONNECT-VIP'}

df = participants_df.copy()

df = df[df.diagnosis.isin(['control']) & (df.age <= 30) & (df.age => 15)]
df_tr = df[df.study.isin(['SCHIZCONNECT-VIP', 'PRAGUE', 'BIOBD'])]
316 from 12 sites

df_val = df[df.study.isin(['BSNIP'])]
70 from 5 site

"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univariate_statistics, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
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

# 1) Inputs: phenotype
PHENOTYPE_CSV = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv"
# for the phenotypes
# INPUT_CSV_icaar_bsnip_biobd = '/neurospin/psy_sbox/start-icaar-eugei/phenotype'
#INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
#INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'



# 3) Output
OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/{dataset}_{modality}_{tags}_{type}.{ext}'
# OUTPUT_PATH.format(dataset='', modality='mwp1', tags='', type='', ext='')
# OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='mask', ext='nii.gz')

########################################################################################################################
# Read phenotypes

phenotypes = pd.read_csv(PHENOTYPE_CSV, sep='\t')
assert phenotypes.shape == (3871, 46)
# rm subjects with missing age or site
phenotypes = phenotypes[phenotypes.sex.notnull() & phenotypes.age.notnull()]
assert phenotypes.shape == (2711, 46)

########################################################################################################################
# Neuroimaging niftii and TIV
# mwp1 files
check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)) # excpected image dimensions

ni_icaar_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-V1/mri/mwp1*.nii")
ni_schizconnect_filenames = glob.glob("/neurospin/psy/schizconnect-vip-prague/derivatives/cat12/vbm/sub-*/mri/mwp1*.nii")
ni_bsnip_filenames = glob.glob("/neurospin/psy/bsnip1/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")
ni_biobd_filenames = glob.glob("/neurospin/psy/bipolar/biobd/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")

tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_schizconnect = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

assert tivo_icaar.shape == (171, 6)
assert len(ni_icaar_filenames) == 171

assert tivo_schizconnect.shape == (738, 6)
assert len(ni_schizconnect_filenames) == 738

assert tivo_bsnip.shape == (1042, 6)
assert len(ni_bsnip_filenames) == 1042

assert tivo_biobd.shape == (746, 6)
assert len(ni_biobd_filenames) == 746

########################################################################################################################
# FIX some issues: duplicated subjects

# 1) Remove subjects from biobd subject dublicated in schizconnect(vip)
# Duplicated between schizconnect and biobd
df = tivo_biobd.append(tivo_schizconnect)
duplicated_in_biobd =  df["participant_id"][df.iloc[:, 1:].duplicated(keep='last')]
assert len(duplicated_in_biobd) == 14
tivo_biobd = tivo_biobd[np.logical_not(tivo_biobd.participant_id.isin(duplicated_in_biobd))]
assert tivo_biobd.shape == (732, 6)

# 2) Remove dublicated subject from bsnip with inconsistant sex and age
# cd /neurospin/psy/bsnip1/sourcedata
# fslview sub-INVVV2WYKK6/ses-V1/anat/sub-INVVV2WYKK6_ses-V1_acq-1.2_T1w.nii.gz sub-INVXR8L3WRZ/ses-V1/anat/sub-INVXR8L3WRZ_ses-V1_acq-1.2_T1w.nii.gz  &
# Same image

df = tivo_bsnip
df.iloc[:, 1:].duplicated().sum() == 1
duplicated_in_bsnip = df[df.iloc[:, 1:].duplicated(keep=False)]["participant_id"]
print(phenotypes[phenotypes.participant_id.isin(duplicated_in_bsnip)][["participant_id",  "sex",   "age"]])
tivo_bsnip = tivo_bsnip[np.logical_not(tivo_bsnip.participant_id.isin(duplicated_in_bsnip))]
assert tivo_bsnip.shape == (1040, 6)

tivo = pd.concat([tivo_icaar, tivo_schizconnect, tivo_bsnip, tivo_biobd], ignore_index=True)
assert tivo.shape == (2681, 6)

########################################################################################################################
# Merge phenotypes with TIV

participants_df = pd.merge(phenotypes, tivo, on="participant_id")
assert participants_df.shape == (2642, 51)

# Check missing in phenotypes
assert len(set(tivo_icaar.participant_id).difference(set(phenotypes.participant_id))) == 4
# set(tivo_icaar.participant_id).difference(set(phenotypes.participant_id))
# Out[8]: {'5EU31000', 'ICAAR004', 'ICAAR047', 'SLBG3TPILOTICAAR'}
assert len(set(tivo_schizconnect.participant_id).difference(set(phenotypes.participant_id))) == 0
assert len(set(tivo_bsnip.participant_id).difference(set(phenotypes.participant_id))) == 0
assert len(set(tivo_biobd.participant_id).difference(set(phenotypes.participant_id))) == 35
"""
set(tivo_biobd.participant_id).difference(set(phenotypes.participant_id))
"""
########################################################################################################################
# Some global params


def do_ml(NI_arr, NI_participants_df, mask_arr, tag, dataset):
    """
    Machine learning for sex, age and DX
    """
    import sklearn.metrics as metrics
    import sklearn.ensemble
    import sklearn.linear_model as lm

    def balanced_acc(estimator, X, y, **kwargs):
        return metrics.recall_score(y, estimator.predict(X), average=None).mean()

    estimators_clf = dict(lrl2=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5),
                          gbc=sklearn.ensemble.GradientBoostingClassifier())
    # or
    estimators_clf = dict(lrl2=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5))
    estimators_reg = dict(lrl2=lm.RidgeCV())

    ml_age_, _, _ = ml_predictions(NI_arr=NI_arr, y=NI_participants_df["age"].values,
                                   estimators=estimators_reg, cv=None, mask_arr=mask_arr)
    ml_age_.insert(0, "tag", tag);
    ml_age_.insert(0, "dataset", dataset);
    ml_age_.insert(0, "target", "age");

    ml_sex_, _, _ = ml_predictions(NI_arr=NI_arr, y=NI_participants_df["sex"].astype(int).values,
                                   estimators=estimators_clf, cv=None, mask_arr=mask_arr)
    ml_sex_.insert(0, "tag", tag);
    ml_sex_.insert(0, "dataset", dataset);
    ml_sex_.insert(0, "target", "sex");

    if dataset == 'icaar-start':
        msk = NI_participants_df["diagnosis"].isin(['UHR-C', 'UHR-NC'])
        dx = NI_participants_df[msk]["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0}).values
        NI_arr_ = NI_arr[msk]
    elif dataset == 'schizconnect-vip':
        msk = NI_participants_df["diagnosis"].isin(['control', 'schizophrenia']) & NI_participants_df["study"].isin(['SCHIZCONNECT-VIP'])
        dx = NI_participants_df[msk]["diagnosis"].map({'schizophrenia': 1, 'control': 0}).values
        NI_arr_ = NI_arr[msk]
    elif dataset == 'bsnip':
        msk = NI_participants_df["diagnosis"].isin(['control', 'schizophrenia'])
        dx = NI_participants_df[msk]["diagnosis"].map({'schizophrenia': 1, 'control': 0}).values
        NI_arr_ = NI_arr[msk]
    elif dataset == 'biobd':
        msk = NI_participants_df["diagnosis"].isin(['control', 'bipolar disorder'])
        dx = NI_participants_df[msk]["diagnosis"].map({'bipolar disorder': 1, 'control': 0}).values
        NI_arr_ = NI_arr[msk]

    print(NI_participants_df[msk]["diagnosis"].describe())
    ml_dx_, _, _ = ml_predictions(NI_arr=NI_arr_, y=dx, estimators=estimators_clf, cv=None, mask_arr=mask_arr)
    ml_dx_.insert(0, "tag", tag);
    ml_dx_.insert(0, "dataset", dataset);
    ml_dx_.insert(0, "target", "dx");
    return ml_age_, ml_sex_, ml_dx_

########################################################################################################################
# Load images, intersect with pop and do preprocessing qnd dump 5d npy

datasets = {
    'icaar-start': dict(ni_filenames = ni_icaar_filenames),
    'schizconnect-vip': dict(ni_filenames = ni_schizconnect_filenames),
    'bsnip': dict(ni_filenames = ni_bsnip_filenames),
    'biobd': dict(ni_filenames = ni_biobd_filenames)}

# On triscotte
datasets = {
    #'icaar-start': dict(ni_filenames = ni_icaar_filenames),
    #'schizconnect-vip': dict(ni_filenames = ni_schizconnect_filenames),
    'bsnip': dict(ni_filenames = ni_bsnip_filenames),
    'biobd': dict(ni_filenames = ni_biobd_filenames)}

for dataset in datasets:
    print("###########################################################################################################")
    print("#", dataset)
    # dataset = 'icaar-start'
    # dataset = 'schizconnect-vip'
    # dataset = 'bsnip'
    # dataset = 'biobd'
    NI_filenames = datasets[dataset]['ni_filenames']

    ml_age_l = list()
    ml_sex_l = list()
    ml_dx_l = list()

    ########################################################################################################################
    print("# 1) Read images")
    tag = 'raw'
    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df)

    mask_arr = preproc.compute_brain_mask(NI_arr, ref_img, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10, verbose=1).get_data() > 0
    # mask_img = nilearn.image.new_img_like(ref_img, mask_arr)
    # mask_img.to_filename("/tmp/msk.nii")

    # Univariate stats
    pdf_filename = OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags=tag, type='stats', ext='pdf')
    design_mat = NI_participants_df[["age", "sex", "tiv", "site"]]  #  Design matrix for Univariate statistics
    vars, mask_arr = univariate_statistics(NI_arr=NI_arr, ref_img=ref_img, design_mat=design_mat, pdf_filename=pdf_filename, mask_arr=mask_arr, thres_nlpval=3)

    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=tag, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    print("# 2) Global scaling")
    tag = 'g'
    NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)

    # Univariate stats
    pdf_filename = OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags=tag, type='stats', ext='pdf')
    vars, mask_arr = univariate_statistics(NI_arr=NI_arr, ref_img=ref_img, design_mat=design_mat, pdf_filename=pdf_filename, mask_arr=mask_arr, thres_nlpval=3)

    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=tag, dataset=dataset)
    ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    print("# 3) Center by site")
    tag = 'gs'
    NI_arr = preproc.center_by_site(NI_arr, site=NI_participants_df.site)

    mask_arr_ = mask_arr & (np.std(NI_arr, axis=0) >= 1e-6).squeeze()
    print("Mask BEFORE Global scaling and Center by site", mask_arr.sum())
    print("Mask AFTER Global scaling and Center by site", mask_arr_.sum())
    mask_arr = mask_arr_

    # Univariate stats
    pdf_filename = OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags=tag, type='stats', ext='pdf')
    vars, mask_arr = univariate_statistics(NI_arr=NI_arr, ref_img=ref_img, design_mat=design_mat, pdf_filename=pdf_filename, mask_arr=mask_arr, thres_nlpval=3)

    # ML
    ml_age_, ml_sex_, ml_dx_ = do_ml(NI_arr, NI_participants_df, mask_arr, tag=tag, dataset=dataset)
    #ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    # DEBUG

    NI_arr = np.load(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='data-32', ext='npy'))
    ml_age_32, ml_sex_32, ml_dx_32 = do_ml(NI_arr, NI_participants_df, mask_arr, tag=tag, dataset=dataset)

    NI_arr = np.load(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='data-64', ext='npy'))
    ml_age_64, ml_sex_64, ml_dx_64 = do_ml(NI_arr, NI_participants_df, mask_arr, tag=tag, dataset=dataset)

    #ml_age_l.append(ml_age_); ml_sex_l.append(ml_sex_); ml_dx_l.append(ml_dx_)

    ########################################################################################################################
    # Save
    np.save(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags=tag, type='data-64', ext='npy'), NI_arr)
    np.save(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags=tag, type='data-32', ext='npy'), NI_arr.astype('float32'))
    NI_participants_df.to_csv(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags=tag, type='participants', ext='csv'), index=False)

    mask_img = nilearn.image.new_img_like(ref_img, mask_arr)
    mask_filename = OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='mask', ext='nii.gz')
    mask_img.to_filename(mask_filename)
    print("Check mask: fslview %s &" % mask_filename)

    # Save ML
    ml_age_df = pd.concat(ml_age_l)
    ml_sex_df = pd.concat(ml_sex_l)
    ml_dx_df = pd.concat(ml_dx_l)
    xls_filename = OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='ml', ext='xlsx')
    with pd.ExcelWriter(xls_filename) as writer:
        ml_age_df.to_excel(writer, sheet_name='age', index=False)
        ml_sex_df.to_excel(writer, sheet_name='sex', index=False)
        ml_dx_df.to_excel(writer, sheet_name='dx', index=False)

    ########################################################################################################################
    # Reload and check precision
    # x64 = np.load(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='data-64', ext='npy'))
    # assert np.max(np.abs(NI_arr - x64)) == 0
    # del x64

    x32 = np.load(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='data-32', ext='npy'))
    print("x32= %.2f GB; x64=%.2f GB" % (x32.nbytes / 1e9, NI_arr.nbytes / 1e9))
    print(np.max(np.abs(NI_arr[:, :, mask_arr] - x32[:, :, mask_arr])))
    #np.min(np.abs(NI_arr[:, :, mask_arr]))
    # schizconnect 1.160547182799121e-07
    del NI_arr, x32




"""
dataset='schizconnect-vip'
df = pd.read_csv(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='participants', ext='csv'))
np.all(NI_participants_df.age == df.age)
np.all(NI_participants_df.site == df.site)

pd.options.display.float_format = '{:,.2f}'.format
import statsmodels.api as sm
import statsmodels.formula.api as smfrmla
import seaborn as sns

# age
sns.violinplot("site", "age", data=df)
df[["site", 'age']].groupby("site").describe()
          age                                          
        count  mean   std   min   25%   50%   75%   max
site                                                   
MRN    164.00 37.84 12.63 18.00 26.00 36.50 49.00 65.00
NU      80.00 32.05  7.33 20.00 25.00 31.50 37.25 46.00
PRAGUE 133.00 28.21  6.57 19.00 23.00 27.00 33.00 49.00
WUSTL  269.00 30.61 13.04 14.00 21.00 25.00 41.00 66.00
vip     92.00 34.38 10.69 18.57 24.46 33.03 42.51 55.51

sm.stats.anova_lm(smfrmla.ols("age ~ site", data=df).fit(), typ=2)
           sum_sq     df     F  PR(>F)
site      8,419.19   4.00 16.78    0.00
Residual 91,946.87 733.00   nan     nan

# sex
df[["site", 'sex']].groupby("site").describe()
          sex                                   
        count mean  std  min  25%  50%  75%  max
site                                            
MRN    164.00 0.23 0.42 0.00 0.00 0.00 0.00 1.00
NU      80.00 0.42 0.50 0.00 0.00 0.00 1.00 1.00
PRAGUE 133.00 0.52 0.50 0.00 0.00 1.00 1.00 1.00
WUSTL  269.00 0.45 0.50 0.00 0.00 0.00 1.00 1.00
vip     92.00 0.45 0.50 0.00 0.00 0.00 1.00 1.00


###########################################################################################################
# icaar-start
# 1) Read images
Clusters of connected voxels #3, sizes= [368616, 45, 19]
# 2) Global scaling
# 3) Center by site
Mask BEFORE Global scaling and Center by site 368680
Mask AFTER Global scaling and Center by site 368680
Check mask: fslview /neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/icaar-start_mwp1_gs_mask.nii.gz &
x32= 1.42 GB; x64=2.84 GB
8.20750178931462e-08

###########################################################################################################
# schizconnect-vip
# 1) Read images
Clusters of connected voxels #4, sizes= [365159, 44, 36, 41]
# 2) Global scaling
# 3) Center by site
Mask BEFORE Global scaling and Center by site 365280
Mask AFTER Global scaling and Center by site 365280
Check mask: fslview /neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/schizconnect-vip_mwp1_gs_mask.nii.gz &
x32= 6.27 GB; x64=12.53 GB
1.160547182799121e-07

ML:
count         605
unique          2
top       control
freq          330
Name: diagnosis, dtype: object
###########################################################################################################
# bsnip
# 1) Read images
Clusters of connected voxels #4, sizes= [362509, 33, 24, 53]
# 2) Global scaling
# 3) Center by site
Mask BEFORE Global scaling and Center by site 362619
Mask AFTER Global scaling and Center by site 362619
Check mask: fslview /neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/bsnip_mwp1_gs_mask.nii.gz &
x32= 8.83 GB; x64=17.66 GB
2.2977780922417423e-07

###########################################################################################################
# biobd
# 1) Read images
Clusters of connected voxels #5, sizes= [364481, 33, 13, 27, 56]
# 2) Global scaling
# 3) Center by site
Mask BEFORE Global scaling and Center by site 364610
Mask AFTER Global scaling and Center by site 364610
Check mask: fslview /neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/biobd_mwp1_gs_mask.nii.gz &
x32= 5.92 GB; x64=11.84 GB
1.188412546149209e-07
"""
########################################################################################################################
# 



datasets = {
    'icaar-start': ni_icaar_filenames,
    'schizconnect-vip': ni_schizconnect_filenames,
    'bsnip': ni_bsnip_filenames,
    'biobd': ni_biobd_filenames}

tags = ['raw', 'g', 'gs']
for dataset in datasets:
    print("###########################################################################################################")

# dataset = 'icaar-start'
# dataset = 'schizconnect-vip'
# dataset = 'bsnip'
# dataset = 'biobd'    print("#", dataset)

# Same mask and participants for all tags
mask_img = nibabel.load(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='mask', ext='nii.gz'))
mask_arr = mask_img.get_data() == 1
pop = pd.read_csv(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='participants', ext='csv'))


for tag in tags:
    tag = "raw"

NI_arr = np.load(OUTPUT_PATH.format(dataset=dataset, modality='mwp1', tags='gs', type='data-64', ext='npy'), mmap_mode='r')

#X = NI_arr[:, :, mask_arr].squeeze()

# provide CV and score



