#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benoit.dufumier


"""

import os, sys
sys.path.extend(['../../', '/home/bd261576/PycharmProjects/pylearn-mulm'])
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy, scipy.ndimage
#import xml.etree.ElementTree as ET
import re
import glob
import seaborn as sns

def OUTPUT(dataset, output_path, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None, type=None, ext=None):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)


def nii2npy(nii_path, phenotype_path, dataset_name, output_path, qc=None, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))):
    ########################################################################################################################
    # Read phenotypes

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
    NI_participants_df.to_csv(OUTPUT(dataset_name, output_path, scaling=None, harmo=None, type="participants", ext="csv"),
                              index=False)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    """
    #NI_arr = np.load(OUTPUT(dataset, output_path, scaling='raw', harmo='raw', type="data64", ext="npy"))
    NI_arr = np.load(OUTPUT(dataset, output_path, scaling='raw', harmo='raw', type="data64", ext="npy"), mmap_mode='r')

    NI_participants_df = pd.read_csv(OUTPUT(dataset, output_path, scaling=None, harmo=None, type="participants", ext="csv"))
    ref_img = nibabel.load(OUTPUT(dataset, output_path, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_img = ref_img
    """

    print("## Compute brain mask")
    mask_img = preproc.compute_brain_mask(NI_arr, ref_img, mask_thres_mean=0.1, mask_thres_std=1e-6,
                                          clust_size_thres=10,
                                          verbose=1)
    mask_arr = mask_img.get_data() > 0
    print("## Save the mask")
    mask_img.to_filename(OUTPUT(dataset_name, output_path, scaling=None, harmo=None, type="mask", ext="nii.gz"))

    ########################################################################################################################
    print("# 2) Raw data")
    # Univariate stats

    # design matrix: Set missing diagnosis to 'unknown' to avoid missing data(do it once)
    dmat_df = NI_participants_df[['age', 'sex', 'tiv']]
    assert np.all(dmat_df.isnull().sum() == 0)
    print("## Do univariate stats on age, sex and TIV")
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + tiv", data=dmat_df)

    # %time univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT(dataset_name, output_path, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1),
                    pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)

    ########################################################################################################################
    print("# 3) Global scaling")
    scaling, harmo = 'gs', 'raw'

    print("## Apply global scaling")
    NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)
    # Save
    print("## Save the new .npy array")
    np.save(OUTPUT(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    # Univariate stats
    print("## Recompute univariate stats on age, sex and TIV")
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + tiv", data=dmat_df)
    pdf_filename = OUTPUT(dataset_name, output_path, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1),
                    pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # Deallocate the memory
    del NI_arr


def do_ml(NI_arr, NI_participants_df, mask_arr, tag, dataset):
    """
    Machine learning for sex, age and DX
    """
    import sklearn.metrics as metrics
    import sklearn.ensemble
    import sklearn.linear_model as lm

    def balanced_acc(estimator, X, y, **kwargs):
        return metrics.recall_score(y, estimator.predict(X), average=None).mean()

    # estimators_clf = dict(LogisticRegressionCV_balanced_inter=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5),
    #                      gbc=sklearn.ensemble.GradientBoostingClassifier())
    # or
    estimators_clf = dict(LogisticRegressionCV_balanced_inter=lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=1, cv=5))
    estimators_reg = dict(RidgeCV_inter=lm.RidgeCV())

    age = NI_participants_df["age"].map({'26-30': 28, '31-35': 33, '22-25': 23.5, '36+': 38}).values
    ml_age_, _, _ = ml_predictions(X=NI_arr, y=age,
                                   estimators=estimators_reg, cv=None, mask_arr=mask_arr)
    ml_age_.insert(0, "tag", tag);
    ml_age_.insert(0, "dataset", dataset);
    ml_age_.insert(0, "target", "age");

    ml_sex_, _, _ = ml_predictions(X=NI_arr, y=NI_participants_df["sex"].astype(int).values,
                                   estimators=estimators_clf, cv=None, mask_arr=mask_arr)
    ml_sex_.insert(0, "tag", tag);
    ml_sex_.insert(0, "dataset", dataset);
    ml_sex_.insert(0, "target", "sex");

    return ml_age_, ml_sex_

def dx_predict(NI_arr, df, mask, train_filter, test_filter, filter=None):
    def get_mask(df, projection_labels=None, check_nan=None):
        mask = np.ones(len(df), dtype=np.bool)
        if projection_labels is not None:
            for (col, val) in projection_labels.items():
                if isinstance(val, list):
                    mask &= getattr(df, col).isin(val)
                elif val is not None:
                    mask &= getattr(df, col).eq(val)
        if check_nan is not None:
            for col in check_nan:
                mask &= ~getattr(df, col).isna()
        return mask

    from sklearn.linear_model import LogisticRegressionCV
    clf = LogisticRegressionCV(class_weight='balanced', cv=5, n_jobs=5, max_iter=200, solver='saga')
    train_mask = get_mask(df, train_filter)
    test_mask = get_mask(df, test_filter)
    age_train = df.age[train_mask].astype(float).values.reshape(-1, 1)
    age_test = df.age[test_mask].astype(float).values.reshape(-1, 1)
    Y = df.diagnosis[train_mask].eq('control').values

    # Train
    X = NI_arr[train_mask]
    if filter is not None:
        X = filter(X)
    X = X.squeeze()[:, mask].reshape(np.sum(train_mask), -1)
    clf.fit(X, Y)
    # Test
    X = NI_arr[test_mask]
    if filter is not None:
        X = filter(X)
    X = X.squeeze()[:, mask].reshape(np.sum(test_mask), -1)
    Y_pred = clf.predict(X)

    return Y_pred, df.diagnosis[test_mask].eq('control').values, clf


# Apply the 3D Sobel filter to an input pytorch tensor
class Sobel3D:
    def __init__(self, padding=0, norm=False, device='cpu', batch=None):
        import torch
        h = [1, 2, 1]
        h_d = [1, 0, -1]
        G_z = [[[h_d[k] * h[i] * h[j] for k in range(3)] for j in range(3)] for i in range(3)]
        G_y = [[[h_d[j] * h[i] * h[k] for k in range(3)] for j in range(3)] for i in range(3)]
        G_x = [[[h_d[i] * h[j] * h[k] for k in range(3)] for j in range(3)] for i in range(3)]
        self.G = torch.tensor([[G_x], [G_y], [G_z]], dtype=torch.float, device=device)
        self.padding = padding
        self.norm = norm
        self.device = device
        self.batch = batch

    def __call__(self, x):
        import torch.nn as nn
        import torch
        # x: 3d tensor (B, C, T, H, W)
        to_np = isinstance(x, np.ndarray)

        if self.batch is not None:
            x_filtered = []
            for i in range(len(x)//self.batch+1):
                x_b_f = nn.functional.conv3d(torch.tensor(x[i*self.batch:(i+1)*self.batch], device=self.device),
                                             self.G, padding=self.padding)
                x_filtered.append(x_b_f.cpu().numpy())
            x_filtered = np.concatenate(x_filtered, axis=0)
            if self.norm:
                x_filtered = np.expand_dims(np.sqrt(np.sum(x_filtered ** 2, axis=1)), 1)
        else:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float, device=self.device)
            x_filtered =  nn.functional.conv3d(x, self.G, padding=self.padding)
            if self.norm:
                x_filtered = torch.sqrt(torch.sum(x_filtered ** 2, dim=1)).unsqueeze(1)
            if to_np:
                x_filtered = x_filtered.cpu().numpy()

        return x_filtered

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
            check=dict(shape=(121, 145, 121),
                       zooms=(1.5, 1.5, 1.5)))

    # MAIN = ~/PycharmProjects/neurospin/scripts/2019
    # _psy_datasets / 201906
    # _schizconnect - vip - prague - bsnip - biodb - icaar - start_assemble - all / generic_cat12vbm_preproc.py
    # OUTPUT_PATH = / neurospin / psy_sbox / analyses / 201906
    # _schizconnect - vip - prague - bsnip - biodb - icaar - start_assemble - all / data / cat12vbm
    #
    # for DATASET in ncp
    # do
    # PHENOTYPE=/neurospin/psy_sbox/$DATASET/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/$DATASET/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/'$DATASET'/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &
    # done

    ## 4 Particular cases
    # DATASET=hcp
    # PHENOTYPE=/neurospin/psy_sbox/$DATASET/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/$DATASET/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/'$DATASET'/derivatives/cat12-12.6_vbm/sub-*/mri/mwp1*.nii' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &

    # DATASET=gsp
    # PHENOTYPE=/neurospin/psy_sbox/${DATASET^^}/${DATASET^^}_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/${DATASET^^}/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/'${DATASET^^}'/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &

    # DATASET=corr
    # PHENOTYPE=/neurospin/psy_sbox/CoRR/CoRR_t1mri_mwp1_participants.csv
    # QC=/neurospin/psy_sbox/CoRR/derivatives/cat12-12.6_vbm_qc/qc.tsv
    # python3 $MAIN --nii_regex_path '/neurospin/psy_sbox/CoRR/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii' --phenotype_path $PHENOTYPE --output_path $OUTPUT_PATH --qc_path $QC --dataset $DATASET --id_type int &> $DATASET.txt &

    # DATASET = localizer
    # PHENOTYPE = / neurospin / psy_sbox /$DATASET /${DATASET ^ ^}
    # _t1mri_mwp1_participants.csv
    # QC = / neurospin / psy_sbox /$DATASET / derivatives / cat12 - 12.6
    # _vbm_qc / qc.tsv
    # python3 $MAIN - -nii_regex_path
    # '/neurospin/psy/'$DATASET
    # '/derivatives/cat12/vbm/sub-*/ses-*/anat/mri/mwp1*.nii' - -phenotype_path $PHENOTYPE - -output_path $OUTPUT_PATH - -qc_path $QC - -dataset $DATASET - -id_type
    # str & > $DATASET.txt &

    # mask_arr=nibabel.load(OUTPUT("bsnip", OUTPUT_PATH, scaling=None, harmo=None, type="mask", ext="nii.gz")).get_data()>0
    # NI_arr = np.load(os.path.join(OUTPUT_PATH,'all_t1mri_mwp1_gs-raw_data32_tocheck.npy'), mmap_mode='r')
    # df = pd.read_csv(os.path.join(OUTPUT_PATH,'all_t1mri_mwp1_participants.tsv'), sep='\t')
    # train_filter = {'diagnosis': [ 'control', 'schizophrenia'], 'study': ['SCHIZCONNECT-VIP']}
    # test_filter = {'diagnosis': ['control', 'schizophrenia'], 'study': ['BSNIP']}
    # sobel = Sobel3D(1, True, 'cuda', 25)
    # Y_pred, Y_true, clf = dx_predict(NI_arr, df, mask_arr, train_filter, test_filter, filter=sobel)


