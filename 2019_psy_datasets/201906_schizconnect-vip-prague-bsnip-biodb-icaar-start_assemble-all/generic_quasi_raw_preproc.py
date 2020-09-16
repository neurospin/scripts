#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benoit.dufumier


"""
import sys; sys.path.append('../../')
import os
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
import brainomics.image_preprocessing as preproc
#from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy, scipy.ndimage
#import xml.etree.ElementTree as ET
import re
import glob
import seaborn as sns

def OUTPUT(dataset, output_path, modality='t1mri', mri_preproc='quasi_raw', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)


def nii2npy(nii_path, phenotype_path, dataset_name, output_path, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)), merge_ni_path=True):
    ########################################################################################################################
    # Read phenotypes

    phenotype = pd.read_csv(phenotype_path, sep=sep)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype_path,
                                                                                   set(keys_required)-set(phenotype.columns))

    assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"

    phenotype['participant_id'] = phenotype['participant_id'].astype(id_type)

    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    ########################################################################################################################
    #  Neuroimaging niftii and TIV
    #  mwp1 files
      #  excpected image dimensions
    NI_filenames = glob.glob(nii_path)

    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name)

    print("# 1) Read images")
    scaling, harmo = 'raw', 'raw'
    print("## Load images", flush=True)
    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames,check=check)
    NI_participants_df.participant_id = NI_participants_df.participant_id.astype(id_type)
    print('--> {} img loaded'.format(len(NI_arr)))
    print("## Merge nii's participant_id with participants.csv")
    NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df,
                                                     merge_ni_path=merge_ni_path)
    print('--> Total number of participants: %i'%len(participants_df))
    print('--> Total number of MRI scans annotated: %i'%len(NI_participants_df))
    print("## Save the new participants.csv")
    NI_participants_df.to_csv(OUTPUT(dataset_name, output_path, type="participants", ext="csv"),
                              index=False)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT(dataset_name, output_path, type="data64", ext="npy"), NI_arr)

    """
    #NI_arr = np.load(OUTPUT(dataset, output_path, type="data64", ext="npy"))
    NI_arr = np.load(OUTPUT(dataset, output_path, type="data64", ext="npy"), mmap_mode='r')

    NI_participants_df = pd.read_csv(OUTPUT(dataset, output_path, type="participants", ext="csv"))
    ref_img = nibabel.load(OUTPUT(dataset, output_path, type="mask", ext="nii.gz"))
    mask_img = ref_img
    """
    ######################################################################################################################
    # Deallocate the memory
    del NI_arr


if __name__=="__main__":
    OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/quasi_raw/'

    # Case specific
    nii_path = "/neurospin/psy_sbox/bsnip1/derivatives/quasi-raw/sub-*/ses-*/anat/sub-*_ses-*_*.nii.gz"
    dataset_name = "bsnip"
    phenotype_path = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/" \
                     "data/cat12vbm/bsnip_t1mri_mwp1_participants.csv"

    # # General case
    nii2npy(nii_path, phenotype_path, dataset_name, OUTPUT_PATH, sep=',', id_type=str,
            check=dict(shape=(182, 218, 182), zooms=(1, 1, 1)), merge_ni_path=True)

    ## Particular case of HCP
    if dataset_name == 'HCP':
        hcp_restricted = pd.read_csv(
            '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/HCP_restricted_data.csv')
        df = pd.read_csv('/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/quasi_raw/'
                         'HCP_t1mri_quasi_raw_participants.csv', sep=',')
        assert set(hcp_restricted.Subject.astype(str)) >= set(df[df.study.eq('HCP')].participant_id.astype(str))
        for id, age in hcp_restricted[['Subject', 'Age_in_Yrs']].values:
            df.loc[df.participant_id.astype(str).eq(str(id)), 'age'] = float(age)
        df.to_csv('/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/quasi_raw/'
                         'HCP_t1mri_quasi_raw_participants.csv', sep=',', index=False)
