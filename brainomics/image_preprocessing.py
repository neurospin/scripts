#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/19/19

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import pandas as pd
import nibabel
import re


def load_images(NI_filenames, check=dict()):
    """
    Load images assuming paths contain a BIDS pattern to retrieve participant_id such /sub-<participant_id>/

    Parameters
    ----------
    NI_filenames : [str], filenames to NI_arri images?
    check : dict, optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
        NI_arr: ndarray, of shape (n_subjects, 1, image_shape). Shape should respect (n_subjects, n_channels, image_axis0, image_axis1, ...)
        participants: Dataframe, with 2 columns "participant_id", "ni_path"
        ref_img: first niftii image, to be use to map back ndarry to image.

    Example
    -------
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_filenames = ['/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR017/ses-V1/mri/mwp1sub-ICAAR017_ses-V1_acq-s03_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR033/ses-V1/mri/mwp1sub-ICAAR033_ses-V1_acq-s07_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-STARTRA160489/ses-V1/mri/mwp1sub-STARTRA160489_ses-v1_T1w.nii']
    >>> NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    >>> NI_arr.shape
    (3, 1, 121, 145, 121)
    >>> NI_participants_df
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    """

    match_filename_re = re.compile("/sub-([^/]+)/")
    pop_columns = ["participant_id", "ni_path"]


    NI_participants_df = pd.DataFrame([[match_filename_re.findall(NI_filename)[0]] + [NI_filename]
        for NI_filename in NI_filenames], columns=pop_columns)

    NI_imgs = [nibabel.load(NI_filename) for NI_filename in NI_participants_df.ni_path]

    ref_img = NI_imgs[0]

    # Check
    if 'shape' in check:
        assert ref_img.get_data().shape == check['shape']
    if 'zooms' in check:
        assert ref_img.header.get_zooms() == check['zooms']
    assert np.all([np.all(img.affine == ref_img.affine) for img in NI_imgs])
    assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in NI_imgs])

    # Load image subjects x chanels (1) x image
    NI_arr = np.stack([np.expand_dims(img.get_data(), axis=0) for img in NI_imgs])
    return NI_arr, NI_participants_df, ref_img

def merge_ni_df(NI_arr, NI_participants_df, participants_df, qc=None, participant_id="participant_id", id_type=str,
                merge_ni_path=True):
    """
    Select participants of NI_arr and NI_participants_df participants that are also in participants_df

    Parameters
    ----------
    NI_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    NI_participants_df: DataFrame, with at least 2 columns: participant_id, ni_path
    participants_df: DataFrame, with 2 at least 1 columns participant_id
    qc: DataFrame, with at least 1 column participant_id
    participant_id: column that identify participant_id
    id_type: the type of participant_id and session, eventually, that should be used for every DataFrame

    Returns
    -------
     NI_arr (ndarray) and NI_participants_df (DataFrame) participants that are also in participants_df


    >>> import numpy as np
    >>> import pandas as pd
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_filenames = ['/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR017/ses-V1/mri/mwp1sub-ICAAR017_ses-V1_acq-s03_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR033/ses-V1/mri/mwp1sub-ICAAR033_ses-V1_acq-s07_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-STARTRA160489/ses-V1/mri/mwp1sub-STARTRA160489_ses-v1_T1w.nii']
    >>> NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    >>> NI_arr.shape
    (3, 1, 121, 145, 121)
    >>> NI_participants_df
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    >>> other_df=pd.DataFrame(dict(participant_id=['ICAAR017', 'STARTRA160489']))
    >>> NI_arr2, NI_participants_df2 = preproc.merge_ni_df(NI_arr, NI_participants_df, other_df)
    >>> NI_arr2.shape
    (2, 1, 121, 145, 121)
    >>> NI_participants_df2
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    >>> np.all(NI_arr[[0, 2], ::] == NI_arr2)
    True
    """

    # 1) Extracts the session + run if available in participants_df/qc from <ni_path> in NI_participants_df
    unique_key_pheno = [participant_id]
    unique_key_qc = [participant_id]
    NI_participants_df.participant_id = NI_participants_df.participant_id.astype(id_type)
    participants_df.participant_id = participants_df.participant_id.astype(id_type)
    if 'session' in participants_df or (qc is not None and 'session' in qc):
        NI_participants_df['session'] = NI_participants_df.ni_path.str.extract('ses-([^_/]+)/')[0].astype(str) # .astype(id_type)?
        if 'session' in participants_df:
            participants_df.session = participants_df.session.astype(str)  # .astype(id_type)?
            unique_key_pheno.append('session')
        if qc is not None and 'session' in qc:
            qc.session = qc.session.astype(str)  # .astype(id_type)?
            unique_key_qc.append('session')
    if 'run' in participants_df or (qc is not None and 'run' in qc):
        NI_participants_df['run'] = NI_participants_df.ni_path.str.extract('run-([^_/]+)\_.*nii')[0].fillna(1).astype(str)
        if 'run' in participants_df:
            unique_key_pheno.append('run')
            participants_df.run = participants_df.run.astype(str)
        if qc is not None and 'run' in qc:
            unique_key_qc.append('run')
            qc.run = qc.run.astype(str)

    # 2) Keeps only the matching (participant_id, session, run) from both NI_participants_df and participants_df by
    #    preserving the order of NI_participants_df
    # !! Very import to have a clean index (to retrieve the order after the merge)
    NI_participants_df = NI_participants_df.reset_index(drop=True).reset_index() # stores a clean index from 0..len(df)
    NI_participants_merged = pd.merge(NI_participants_df, participants_df, on=unique_key_pheno,
                                      how='inner', validate='m:1')
    print('--> {} {} have missing phenotype'.format(len(NI_participants_df)-len(NI_participants_merged),
          unique_key_pheno))

    # 3) If QC is available, filters out the (participant_id, session, run) who did not pass the QC
    if qc is not None:
        assert np.all(qc.qc.eq(0) | qc.qc.eq(1)), 'Unexpected value in qc.csv'
        qc = qc.reset_index(drop=True) # removes an old index
        qc_val = qc.qc.values
        if np.all(qc_val==0):
            raise ValueError('No participant passed the QC !')
        elif np.all(qc_val==1):
            pass
        else:
            #idx_first_occurence = len(qc_val) - (qc_val[::-1] != 1).argmax()
            #assert np.all(qc.iloc[idx_first_occurence:].qc == 1)
            qc_passed = qc.qc.eq(1).to_numpy()
            assert np.all(qc[qc_passed].qc == 1)
            keep = qc[qc_passed][unique_key_qc]
            init_len = len(NI_participants_merged)
            # Very important to have 1:1 correspondance between the QC and the NI_participant_array
            NI_participants_merged = pd.merge(NI_participants_merged, keep, on=unique_key_qc,
                                              how='inner', validate='1:1')
            print('--> {} {} did not pass the QC'.format(init_len - len(NI_participants_merged), unique_key_qc))

    if merge_ni_path and 'ni_path' in participants_df:
        # Keep only the matching session and acquisition nb according to <participants_df>
        sub_sess_to_keep = NI_participants_merged['ni_path_y'].str.extract(r".*/.*sub-(\w+)_ses-(\w+)_.*")
        sub_sess = NI_participants_merged['ni_path_x'].str.extract(r".*/.*sub-(\w+)_ses-(\w+)_.*")
        # Some participants have only one acq, in which case it is not mentioned
        acq_to_keep = NI_participants_merged['ni_path_y'].str.extract(r"(acq-[a-zA-Z0-9\-\.]+)").fillna('')
        acq = NI_participants_merged['ni_path_x'].str.extract(r"(acq-[a-zA-Z0-9\-\.]+)").fillna('')

        assert not (sub_sess.isnull().values.any() or sub_sess_to_keep.isnull().values.any()), \
            "Extraction of session_id or participant_id failed"

        keep_unique_participant_ids = sub_sess_to_keep.eq(sub_sess).all(1).values.flatten() & \
                                      acq_to_keep.eq(acq).values.flatten()

        NI_participants_merged = NI_participants_merged[keep_unique_participant_ids]
        NI_participants_merged.drop(columns=['ni_path_y'], inplace=True)
        NI_participants_merged.rename(columns={'ni_path_x': 'ni_path'}, inplace=True)


    unique_key = unique_key_qc if set(unique_key_qc) >= set(unique_key_pheno) else unique_key_pheno
    assert len(NI_participants_merged.groupby(unique_key)) == len(NI_participants_merged), \
        '{} similar pairs {} found'.format(len(NI_participants_merged)-len(NI_participants_merged.groupby(unique_key)),
                                           unique_key)

    # Get back to NI_arr using the indexes kept in NI_participants through all merges
    idx_to_keep = NI_participants_merged['index'].values

    # NI_participants_merged.drop('index')
    return NI_arr[idx_to_keep], NI_participants_merged

def global_scaling(NI_arr, axis0_values=None, target=1500):
    """
    Apply a global proportional scaling, such that axis0_values * gscaling == target

    Parameters
    ----------
    NI_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    axis0_values: 1-d array, if None (default) use global average per subject: NI_arr.mean(axis=1)
    target: scalar, the desired target

    Returns
    -------
    The scaled array

    >>> import numpy as np
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_arr = np.array([[9., 11], [0, 2],  [4, 6]])
    >>> NI_arr
    array([[ 9., 11.],
           [ 0.,  2.],
           [ 4.,  6.]])
    >>> axis0_values = [10, 1, 5]
    >>> preproc.global_scaling(NI_arr, axis0_values, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    >>> preproc.global_scaling(NI_arr, axis0_values=None, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    """
    if axis0_values is None:
        axis0_values = NI_arr.mean(axis=1)
    gscaling = target / np.asarray(axis0_values)
    gscaling = gscaling.reshape([gscaling.shape[0]] + [1] * (NI_arr.ndim - 1))
    return gscaling * NI_arr


def center_by_site(NI_arr, site, in_place=False):
    """
    Center by site

    Parameters
    ----------
    NI_arr :  ndarray, of shape (n_subjects, 1, image_shape).
    site : 1-d array of site labels
    in_place: boolean perform inplace operation

    Returns
    -------
    >>> import numpy as np
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_arr = np.array([[8., 10], [9, 14],  [3, 5], [4, 7]])
    >>> NI_arr
    array([[ 8., 10.],
           [ 9., 14.],
           [ 3.,  5.],
           [ 4.,  7.]])
    >>> preproc.center_by_site(NI_arr, site=[1, 1, 0, 0])
    array([[-0.5, -2. ],
           [ 0.5,  2. ],
           [-0.5, -1. ],
           [ 0.5,  1. ]])
    """
    if not in_place:
        NI_arr = NI_arr.copy()
    site = np.asarray(site)
    for s in set(site):
        # s = 1
        m = site == s
        NI_arr[m] -= NI_arr[m, :].mean(axis=0)

    return NI_arr


def compute_brain_mask(NI_arr, target_img, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10, verbose=1):
    """
    Compute brain mask:
        (1) Implicit mask threshold `mean >= mask_thres_mean` and `std >= mask_thres_std`
        (2) Use brain mask from `nilearn.masking.compute_gray_matter_mask(target_img)`
        (3) mask = Implicit mask & brain mask
        (4) Remove small branches with `scipy.ndimage.binary_opening`
        (5) Avoid isolated clusters: remove clusters (of connected voxels) smaller that `clust_size_thres`

    Parameters
    ----------
    NI_arr :  ndarray, of shape (n_subjects, 1, image_shape).
    target_img : image.
    mask_thres_mean : Implicit mask threshold `mean >= mask_thres_mean`
    mask_thres_std : Implicit mask threshold `std >= mask_thres_std`
    clust_size_thres : remove clusters (of connected voxels) smaller that `clust_size_thres`
    verbose : int. verbosity level

    Returns
    -------
    image of mask

    """
    import scipy
    import nilearn
    import nilearn.masking

    # (1) Implicit mask
    mask_arr = np.ones(NI_arr.shape[1:], dtype=bool).squeeze()
    if mask_thres_mean is not None:
        mask_arr = mask_arr & (np.abs(np.mean(NI_arr, axis=0)) >= mask_thres_mean).squeeze()
    if mask_thres_std is not None:
        mask_arr = mask_arr & (np.std(NI_arr, axis=0) >= mask_thres_std).squeeze()

    # (2) Brain mask
    mask_img = nilearn.masking.compute_gray_matter_mask(target_img)

    # (3) mask = Implicit mask & brain mask
    mask_arr = (mask_img.get_data() == 1) & mask_arr

    # (4) Remove small branches
    mask_arr = scipy.ndimage.binary_opening(mask_arr)

    # (5) Avoid isolated clusters: remove all cluster smaller that clust_size_thres
    mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)

    labels = np.unique(mask_clustlabels_arr)[1:]
    for lab in labels:
        clust_size = np.sum(mask_clustlabels_arr == lab)
        if clust_size <= clust_size_thres:
            mask_arr[mask_clustlabels_arr == lab] = False

    if verbose >= 1:
        mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
        labels = np.unique(mask_clustlabels_arr)[1:]
        print("Clusters of connected voxels #%i, sizes=" % len(labels),
              [np.sum(mask_clustlabels_arr == lab) for lab in labels])

    return nilearn.image.new_img_like(target_img, mask_arr)
