#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:08:41 CET 2019

@author: edouard.duchesnay

%load_ext autoreload
%autoreload 2
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univariate_statistics
import shutil
# import mulm
# import sklearn
import re
# from nilearn import plotting
import matplotlib.pyplot as plt
# import scipy, scipy.ndimage
#import xml.etree.ElementTree as ET
import re
import glob
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

mwgm_icaar_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-V1/mri/mwp1*.nii")
tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')

mwgm_schizconnect_filenames = glob.glob("/neurospin/psy/schizconnect-vip-prague/derivatives/cat12/vbm/sub-*/mri/mwp1*.nii")
mwgm_bsnip_filenames = glob.glob("/neurospin/psy/bsnip1/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")
mwgm_biobd_filenames = glob.glob("/neurospin/psy/bipolar/biobd/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")

tivo_schizconnect = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

assert tivo_icaar.shape == (171, 6)
assert len(mwgm_icaar_filenames) == 171

assert tivo_schizconnect.shape == (738, 6)
assert len(mwgm_schizconnect_filenames) == 738

assert tivo_bsnip.shape == (1042, 6)
assert len(mwgm_bsnip_filenames) == 1042

assert tivo_biobd.shape == (746, 6)
assert len(mwgm_biobd_filenames) == 746

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
# Read GM images, intersect with pop
from matplotlib.backends.backend_pdf import PdfPages

# ICAAR-START
NI_filenames = mwgm_icaar_filenames

########################################################################################################################
print("# 1) Read images")
NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df)

pdf_filename = OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='raw', type='qc', ext='pdf')
pdf = PdfPages(pdf_filename)
design_mat = NI_participants_df[["age", "sex", "tiv", "site"]] # Design matrix for Univariate statistics
fig_title = os.path.splitext(os.path.basename(pdf_filename))[0]

vars, mask_arr = univariate_statistics(NI_arr=NI_arr, ref_img=ref_img, design_mat=design_mat, fig_title=fig_title, pdf=pdf, mask_arr=None, thres_nlpval=3)
pdf.close()


########################################################################################################################
print("# 2) Global scaling")
NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)

pdf_filename = OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='g', type='qc', ext='pdf')
pdf = PdfPages(pdf_filename)
design_mat = NI_participants_df[["age", "sex", "tiv", "site"]] # Design matrix for Univariate statistics
fig_title = os.path.splitext(os.path.basename(pdf_filename))[0]
vars, mask_arr = univariate_statistics(NI_arr=NI_arr, ref_img=ref_img, design_mat=design_mat, fig_title=fig_title, pdf=pdf, mask_arr=None, thres_nlpval=3)
pdf.close()


########################################################################################################################
print("# 3) Center by site")
NI_arr = preproc.center_by_site(NI_arr, site=NI_participants_df.site)
pdf_filename = OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='qc', ext='pdf')
pdf = PdfPages(pdf_filename)
design_mat = NI_participants_df[["age", "sex", "tiv", "site"]] # Design matrix for Univariate statistics
fig_title = os.path.splitext(os.path.basename(pdf_filename))[0]
vars, mask_arr = univariate_statistics(NI_arr=NI_arr, ref_img=ref_img, design_mat=design_mat, fig_title=fig_title, pdf=pdf, mask_arr=None, thres_nlpval=3)
pdf.close()



########################################################################################################################
mask_arr = preproc.compute_brain_mask(NI_arr, ref_img).get_data() > 0
mask_arr = compute_brain_mask(NI_arr, ref_img, mask_thres_mean=0).get_data() > 0

np.save(OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='data-64', ext='npy'), NI_arr)
np.save(OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='data-32', ext='npy'), NI_arr.astype('float32'))

NI_participants_df.to_csv(OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='participants', ext='csv'), index=False)


a = np.load(OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='data-64', ext='npy'))
a = np.load(OUTPUT_PATH.format(dataset='icaar-start', modality='mwp1', tags='gs', type='data-32', ext='npy'))


np.max(np.abs(NI_arr[:, :, mask_arr] - a[:, :, mask_arr]))
# np.min(np.abs(NI_arr[:, :, mask_arr]))

NI_arr.save()

Y = NI_arr.reshape(NI_arr.shape[0], -1)
mod = mulm.MUOLS(Y, X)
mulm_tvals, mulm_pvals, mulm_df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

fig = plt.figure(1)

plt.subplot(311)
plt.hist(mulm_pvals[0, :], bins=100)
plt.title("Age, p-value histo (y~age+sex+tiv)")
plt.subplot(312)
plt.hist(mulm_pvals[1, :], bins=100)
plt.title("Sex, p-value histo (y~age+sex+tiv)")
plt.subplot(313)
sns.violinplot(x="site", y="gm", hue='sex', data=pd.DataFrame(dict(gm=[s.mean() for s in NI_arr], site=NI_participants_df.site, sex=NI_participants_df.sex)))
plt.title("Subject Mean per site")
fig.suptitle('3) Center by site')
pdf.savefig(); plt.close()

pdf.close()









NI_arr.shape


ref_img_schizconnect, NI_schizconnect, pop_ni_schizconnect = load_images(ni_filenames=mwgm_schizconnect_filenames, check=check)
ref_img_bsnip, NI_bsnip, pop_ni_bsnip = load_images(ni_filenames=mwgm_bsnip_filenames, check=check)
ref_img_biobd, NI_biobd, pop_ni_biobd = load_images(ni_filenames=mwgm_biobd_filenames, check=check)

#pop_icaar = pd.merge(pop_ni_icaar, pop, on="participant_id", how= 'inner') # preserve the order of the left keys.
#NI, participants, pop_ref = NI_icaar, pop_ni_icaar, pop
ref_img_icaar, NI_icaar, pop_ni_icaar = load_images(ni_filenames=mwgm_icaar_filenames, check=check)

NI_icaar, pop_ni_icaar = keep_those_in_popref(NI=NI_icaar, ni_participants=pop_ni_icaar, pop_ref=pop)
assert NI_icaar.shape[0] == 170
NI_icaar = global_scaling(X=NI_icaar, axis0_values=np.array(pop_ni_icaar.tiv), target=1500)
NI_icaar = center_by_site(X=NI_icaar, site=pop_ni_icaar.site)

#############################################################################
# Remove site effect



# STEP 3: normalization of the ROIs dataset

tivo['norm_ratio'] = 1500 / tivo.tiv
for i in range(ROIs.shape[0]):
    ROIs.iloc[i, 1:] *= tivo.norm_ratio[i]

# ROIs.to_csv(os.path.join(OUTPUT_PATH, 'cat12_roi_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t',
#            index=False)











