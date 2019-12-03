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

    # Load image subjects x chanels (1) x image
    NI_arr = np.stack([np.expand_dims(img.get_data(), axis=0) for img in NI_imgs])
    return NI_arr, NI_participants_df, ref_img

def merge_ni_df(NI_arr, NI_participants_df, participants_df, participant_id="participant_id"):
    """
    Select participants of NI_arr and NI_participants_df participants that are also in participants_df

    Parameters
    ----------
    NI_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    NI_participants_df: DataFrame, with at leas 2 columns: participant_id, "ni_path"
    participants_df: DataFrame, with 2 at least 1 columns participant_id
    participant_id: column that identify participant_id

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
    keep = NI_participants_df[participant_id].isin(participants_df[participant_id])
    return NI_arr[keep], pd.merge(NI_participants_df[keep], participants_df, on=participant_id, how= 'inner') # preserve the order of the left keys.

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
