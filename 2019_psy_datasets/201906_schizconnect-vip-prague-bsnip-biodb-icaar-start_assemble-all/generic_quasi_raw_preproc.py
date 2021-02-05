#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benoit.dufumier


"""
import sys; sys.path.append('../../')
import os, argparse
import numpy as np
import pandas as pd
import brainomics.image_preprocessing as preproc
import matplotlib
matplotlib.use('Agg')
import glob

def OUTPUT(dataset, output_path, modality='t1mri', mri_preproc='quasi_raw', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)


def nii2npy(nii_path, phenotype_path, dataset_name, output_path, qc=None, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)), merge_ni_path=True):
    ########################################################################################################################


    phenotype = pd.read_csv(phenotype_path, sep=sep)
    qc = pd.read_csv(qc, sep=sep) if qc is not None else None

    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype_path,
                                                                                   set(keys_required)-set(phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.csv
    #assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"


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
    print("## Load images")
    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames,check=check)
    print('--> {} img loaded'.format(len(NI_participants_df)))
    print("## Merge nii's participant_id with participants.csv")
    NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df,
                                                         qc=qc, id_type=id_type)
    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))

    print("## Save the new participants.csv")
    NI_participants_df.to_csv(OUTPUT(dataset_name, output_path, type="participants", ext="csv"),
                              index=False)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT(dataset_name, output_path, type="data64", ext="npy"), NI_arr)

    ######################################################################################################################
    # Deallocate the memory
    del NI_arr


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--nii_regex_path', type=str, required=True)
    parser.add_argument('--phenotype_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--qc_path', type=str, required=False)
    parser.add_argument('--id_type', type=str, choices=['str', 'int'], default='str',
                        help='Type of <participant_id> and <session> used for casting')
    parser.add_argument('--sep', type=str, choices=[',', '\t'], default='\t', help='Separator used in participants.csv')

    args = parser.parse_args()

    # # General case
    nii2npy(args.nii_regex_path,
            args.phenotype_path,
            args.dataset,
            args.output_path,
            qc=args.qc_path,
            sep=args.sep,
            id_type=eval(args.id_type),
            check=dict(shape=(182, 218, 182),
                       zooms=(1, 1, 1)))

    # !/bin/bash

    # MAIN = ~ / PycharmProjects / neurospin / scripts / 2019
    # _psy_datasets / 201906
    # _schizconnect - vip - prague - bsnip - biodb - icaar - start_assemble - all / generic_quasi_raw_preproc.py
    # MAIN_RESAMPLING = ~ / PycharmProjects / neurospin / scripts / 2019
    # _psy_datasets / 201906
    # _schizconnect - vip - prague - bsnip - biodb - icaar - start_assemble - all / quasi_raw_resample.py
    # OUTPUT_PATH = / neurospin / psy_sbox / analyses / 201906
    # _schizconnect - vip - prague - bsnip - biodb - icaar - start_assemble - all / data / quasi_raw /

    ## .nii to .npy for id-type == int and id-type == str (important to make the distinction, depends on the participant_id
    ## values)

    # for DATASET in  biobd hcp nar rbp  mpi-leipzig cnp abide1 abide2 ixi
    # do
    # PHENOTYPE=/neurospin/psy_sbox/$DATASET/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/$DATASET/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/'$DATASET'/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &
    # done

    # for DATASET in oasis3  #icbm localizer candi
    # do
    # PHENOTYPE=/neurospin/psy_sbox/$DATASET/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/$DATASET/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/'$DATASET'/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type str &> $DATASET.txt &
    # done

    ## Particular cases

    # DATASET=gsp
    # PHENOTYPE=/neurospin/psy_sbox/${DATASET^^}/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/${DATASET^^}/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/'${DATASET^^}'/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &
    #
    # DATASET=corr
    # PHENOTYPE=/neurospin/psy_sbox/CoRR/CoRR_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/CoRR/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/CoRR/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &

    # DATASET=bsnip
    # PHENOTYPE=/neurospin/psy_sbox/bsnip1/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/bsnip1/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/bsnip1/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type str &> $DATASET.txt &
    #
    # DATASET=schizconnect-vip
    # PHENOTYPE=/neurospin/psy_sbox/schizconnect-vip-prague/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/schizconnect-vip-prague/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/schizconnect-vip-prague/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type str &> $DATASET.txt &