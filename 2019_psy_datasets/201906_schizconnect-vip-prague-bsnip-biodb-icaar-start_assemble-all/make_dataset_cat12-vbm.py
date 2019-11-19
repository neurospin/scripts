#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:08:41 CET 2019

@author: edouard.duchesnay
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
import brainomics.image_preprocessing as preproc
import shutil
# import mulm
# import sklearn
import re
# from nilearn import plotting
# import matplotlib.pyplot as plt
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

check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

mwgm_icaar_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-V1/mri/mwp1*.nii")
assert len(mwgm_icaar_filenames) == 171

mwgm_schizconnect_filenames = glob.glob("/neurospin/psy/schizconnect-vip-prague/derivatives/cat12/vbm/sub-*/mri/mwp1*.nii")
assert len(mwgm_schizconnect_filenames) == 738

mwgm_bsnip_filenames = glob.glob("/neurospin/psy/bsnip1/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")
assert len(mwgm_bsnip_filenames) == 1042

mwgm_biobd_filenames = glob.glob("/neurospin/psy/bipolar/biobd/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")
assert len(mwgm_biobd_filenames) == 746


PHENOTYPE_CSV = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv"
# for the phenotypes
# INPUT_CSV_icaar_bsnip_biobd = '/neurospin/psy_sbox/start-icaar-eugei/phenotype'
#INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
#INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'

OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data'

#CONF = dict(clust_size_thres=20, NI="mwp1", vs=1.5, shape=(121, 145, 121))

"""
CREATION OF ONE SINGLE DATASET WITH ALL ROIs
"""

########################################################################################################################
# Read TIV

tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_schizconnect = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd = pd.read_csv(os.path.join(BASE_PATH_biobd, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

assert tivo_icaar.shape == (171, 6)
assert tivo_schizconnect.shape == (738, 6)
assert tivo_bsnip.shape == (1042, 6)
assert tivo_biobd.shape == (746, 6)
tivo = pd.concat([tivo_icaar, tivo_schizconnect, tivo_bsnip, tivo_biobd], ignore_index=True)
assert tivo.shape == (2697, 6)

########################################################################################################################
# Read phenotypes

phenotypes = pd.read_csv(PHENOTYPE_CSV, sep='\t')
assert phenotypes.shape == (3871, 46)

########################################################################################################################
# Merge phenotypes with TIV

pop = pd.merge(phenotypes, tivo, on="participant_id")
assert pop.shape == (2661, 51)

# Check missing in phenotypes
assert len(set(tivo_icaar.participant_id).difference(set(phenotypes.participant_id))) == 1
# {'SLBG3TPILOTICAAR'}
assert len(set(tivo_schizconnect.participant_id).difference(set(phenotypes.participant_id))) == 0
assert len(set(tivo_bsnip.participant_id).difference(set(phenotypes.participant_id))) == 0
assert len(set(tivo_biobd.participant_id).difference(set(phenotypes.participant_id))) == 35
"""
set(tivo_biobd.participant_id).difference(set(phenotypes.participant_id))
"""
########################################################################################################################
# Read GM images, intersect with pop
other_df = pop

# ICAAR-START
NI_filenames=mwgm_icaar_filenames

NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, other_df)
sns.violinplot(x="site", y="gm", hue='sex', data=pd.DataFrame(dict(gm=[s.mean() for s in NI_arr], site=NI_participants_df.site, sex=NI_participants_df.sex)))


NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)
sns.violinplot(x="site", y="gm", hue='sex', data=pd.DataFrame(dict(gm=[s.mean() for s in NI_arr], site=NI_participants_df.site, sex=NI_participants_df.sex)))

NI_arr = preproc.center_by_site(NI_arr, site=NI_participants_df.site)
sns.violinplot(x="site", y="gm", hue='sex', data=pd.DataFrame(dict(gm=[s.mean() for s in NI_arr], site=NI_participants_df.site, sex=NI_participants_df.sex)))

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











