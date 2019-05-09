# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:31:01 2014

@author: edouard.duchesnay@cea.fr
"""
import os
import numpy as np
import scipy.ndimage
import nibabel as nib
import subprocess
import nilearn
import pandas as pd

def smooth_labels(arr, size=(3, 3, 3)):
    def func(buffer):
        return np.argmax(np.bincount(buffer.astype(int)))
    arr = scipy.ndimage.generic_filter(arr, func, size=size)
    return arr


def dilation_labels(arr, size=(3, 3, 3)):
    def func(buffer):
        buffer = buffer.astype(int)
        if np.any(buffer > 0):
            buffer = buffer[buffer != 0]
            return np.argmax(np.bincount(buffer))
        else:
            return 0
    arr = scipy.ndimage.generic_filter(arr, func, size=size)
    return arr

def resample_atlas_harvard_oxford(ref, output,
        atlas_base_dir="/usr/share/data/harvard-oxford-atlases/HarvardOxford",
        fsl_cmd = ['fsl5.0-applywarp', '-i', '%s', '-r', '%s', '-o', '%s', '--interp=nn'],
        smooth_size=(3, 3, 3), dilation_size=(3, 3, 3),
        fill_wm=True):
    """Resample HarvardOxford atlas (cortical and subcortical) into reference
    space. Add sub-cortical GM to cortical GM. Add 100 to sub label to avoid
    confusion. Smooth and dilate all those GM structures. At the end, voxel
    with no labels that correspond to WM (in sub atlas) are labeled WM
    (1+100, 12+100). Smoothing and dilation may take a while. This should prevent
    to remove too many voxels towoard ventriculus for instance.

    Example
    -------
    from brainomics.image_atlas import resample_atlas_harvard_oxford
    im = resample_atlas_harvard_oxford(ref="mwrc1Test_Sbj1.nii.gz", output="atlas.nii.gz")
    im = resample_atlas_harvard_oxford(ref="vol0000.nii.gz", output="atlas.nii.gz")
    im = resample_atlas_harvard_oxford(ref="swrLil1_DasDa_Presto_0655.nii", output="atlas.nii.gz", dilation_size=None, fill_wm=False)
    """
    cort_filename = os.path.join(atlas_base_dir, "HarvardOxford-cort-maxprob-thr0-1mm.nii.gz")
    sub_filename = os.path.join(atlas_base_dir, "HarvardOxford-sub-maxprob-thr0-1mm.nii.gz")
    # resamp subcortical
    #os.system(fsl_cmd % (sub_filename, ref, "/tmp/sub"))
    fsl_cmd[2], fsl_cmd[4], fsl_cmd[6] = sub_filename, ref, "/tmp/sub"
    #cmd = fsl_cmd[] % (sub_filename, ref, "/tmp/sub")
    print(fsl_cmd)
    subprocess.call(fsl_cmd)
    # rename WM, GM, ventriculus
    sub_image = nib.load("/tmp/sub.nii.gz")
    sub_arr = sub_image.get_data()
    sub_arr[(sub_arr == 1)  | (sub_arr == 2)  |     # Left WM & Cortex
            (sub_arr == 12) | (sub_arr == 13) |     # Right WM & Cortex
            (sub_arr == 3)  | (sub_arr == 14) |     # Ventriculus
            (sub_arr == 8)] = 0 # Brain-Stem

    #sub_image.to_filename("/tmp/sub.nii.gz")
    # resamp cortical
    #cmd = fsl_cmd % (cort_filename, ref, "/tmp/cort")
    fsl_cmd[2], fsl_cmd[4], fsl_cmd[6] = cort_filename, ref, "/tmp/cort"
    print("".join(fsl_cmd))
    subprocess.call(fsl_cmd)
    #os.system(fsl_cmd % (cort_filename, ref, "/tmp/cort"))
    cort_image = nib.load("/tmp/cort.nii.gz")
    atlas_arr = cort_image.get_data()
    atlas_arr[sub_arr != 0] = 100 + sub_arr[sub_arr != 0]
    cort_image.to_filename("/tmp/merge.nii.gz")

    if smooth_size is not None:  # smooth labels
        atlas_arr = smooth_labels(atlas_arr, size=smooth_size)
        atlas_im = nib.Nifti1Image(atlas_arr, affine=cort_image.affine)
        atlas_im.to_filename("/tmp/atlas_smoothed.nii.gz")

    if dilation_size is not None:  # dilate labels
        atlas_arr = dilation_labels(atlas_arr, size=dilation_size)

    if fill_wm:  # fill missing that are WM in sub
        sub_arr = nib.load("/tmp/sub.nii.gz").get_data()
        sub_wm_vent_mask = ((sub_arr == 1) | (sub_arr == 12)) | \
                           ((sub_arr == 3) | (sub_arr == 14))
        wm_vent_missing_mask = (atlas_arr == 0) & sub_wm_vent_mask
        atlas_arr[wm_vent_missing_mask] = 100 + sub_arr[wm_vent_missing_mask]

    atlas_arr_int = atlas_arr.astype('int16')
    assert np.all(atlas_arr_int == atlas_arr)
    atlas_im = nib.Nifti1Image(atlas_arr_int, affine=cort_image.affine)
    atlas_im.to_filename(output)
    print("Watch if everything is OK:")
    print("fslview %s %s" % (output, ref))
    return atlas_im

def resample_atlas_bangor_cerebellar(ref, output,
        atlas_base_dir="/usr/share/data/bangor-cerebellar-atlas/Cerebellum",
        fsl_cmd = ['fsl5.0-applywarp', '-i', '%s', '-r', '%s', '-o', '%s', '--interp=nn'],
        smooth_size=(3, 3, 3), dilation_size=(3, 3, 3),
        fill_wm=True):
    """Resample HarvardOxford atlas (cortical and subcortical) into reference
    space. Add sub-cortical GM to cortical GM. Add 100 to sub label to avoid
    confusion. Smooth and dilate all those GM structures. At the end, voxel
    with no labels that correspond to WM (in sub atlas) are labeled WM
    (1+100, 12+100). Smoothing and dilation may take a while. This should prevent
    to remove too many voxels towoard ventriculus for instance.

    Example
    -------
    from brainomics.image_atlas import resample_atlas_harvard_oxford
    im = resample_atlas_harvard_oxford(ref="mwrc1Test_Sbj1.nii.gz", output="atlas.nii.gz")
    im = resample_atlas_harvard_oxford(ref="vol0000.nii.gz", output="atlas.nii.gz")
    im = resample_atlas_harvard_oxford(ref="swrLil1_DasDa_Presto_0655.nii", output="atlas.nii.gz", dilation_size=None, fill_wm=False)
    """
    atlas_filename = os.path.join(atlas_base_dir, "Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz")

    #sub_image.to_filename("/tmp/sub.nii.gz")
    # resamp cortical
    #cmd = fsl_cmd % (atlas_filename, ref, "/tmp/cort")
    fsl_cmd[2], fsl_cmd[4], fsl_cmd[6] = atlas_filename, ref, "/tmp/cerebellar"
    print("".join(fsl_cmd))
    subprocess.call(fsl_cmd)
    #os.system(fsl_cmd % (atlas_filename, ref, "/tmp/cort"))
    atlas_image = nib.load("/tmp/cerebellar.nii.gz")
    atlas_arr = atlas_image.get_data()

    if smooth_size is not None:  # smooth labels
        atlas_arr = smooth_labels(atlas_arr, size=smooth_size)
        atlas_im = nib.Nifti1Image(atlas_arr, affine=atlas_image.affine)
        atlas_im.to_filename("/tmp/cerebellar_smoothed.nii.gz")

    if dilation_size is not None:  # dilate labels
        atlas_arr = dilation_labels(atlas_arr, size=dilation_size)

    atlas_arr_int = atlas_arr.astype('int16')
    assert np.all(atlas_arr_int == atlas_arr)
    atlas_im = nib.Nifti1Image(atlas_arr_int, affine=atlas_image.affine)
    atlas_im.to_filename(output)
    print("Watch if everything is OK:")
    print("fslview %s %s" % (output, ref))
    return atlas_im


def roi_average(maps_img, atlas="harvard_oxford", mask_img=None):
    """
    :param maps_img: List of 3D Nifti images
    :param atlas:
    :mask_img: Mask Nifti images
    :return: DataFrames average value over atlase ROI on Cortical ROI and Subcortical ROI
    """
    ref_img = maps_img[0]
    assert np.all([(ref_img.affine == maps_img[i].affine) for i in range(len(maps_img))])
    maps_arr = [maps_img[i].get_data() for i in range(len(maps_img))]
    if mask_img:
        mask_arr = mask_img.get_data() != 0
    else:
        mask_arr = np.ones(ref_img.get_data().shape) != 0
    # Fetch atlases
    if atlas == "harvard_oxford":
        atlascort = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm", data_dir=None, symmetric_split=False, resume=True, verbose=1)
        atlassub = nilearn.datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm", data_dir=None, symmetric_split=False, resume=True, verbose=1)
        # FIX bug nilearn.datasets.fetch_atlas_harvard_oxford: Errors in HarvardOxford.tgz / sub-maxprob-thr0-1mm
        atlassub.maps = os.path.join('/usr/share/data/harvard-oxford-atlases/HarvardOxford', os.path.basename(atlassub.maps))
    else:
        raise Exception('Undefined atlas')

    atlascort_img = nilearn.image.resample_to_img(source_img=atlascort.maps, target_img=ref_img, interpolation='nearest', copy=True, order='F')
    atlascort_arr, atlascort_labels = atlascort_img.get_data(), atlascort.labels
    assert len(np.unique(atlascort_arr)) == len(atlascort_labels), "Atlas %s : array labels must match labels table" %  atlas

    atlassub_img = nilearn.image.resample_to_img(source_img=atlassub.maps, target_img=ref_img, interpolation='nearest', copy=True, order='F')
    atlassub_arr, atlassub_labels = atlassub_img.get_data(), atlassub.labels
    atlassub_arr = atlassub_arr.astype(int)
    assert len(np.unique(atlassub_arr)) == len(atlassub_labels), "Atlas %s : array labels must match labels table" %  atlas

    assert np.all((ref_img.affine == atlassub_img.affine) & (ref_img.affine == atlascort_img.affine))
    # roi_cort_df = pd.DataFrame([[np.mean(maps_img[i].get_data()[(atlascort_arr == lab) & mask_arr]) for lab in np.unique(atlascort_arr)]
    #               for i in range(len(maps_img))], columns=atlascort_labels)
    #
    # roi_sub_df = pd.DataFrame([[np.mean(maps_img[i].get_data()[(atlassub_arr == lab) & mask_arr]) for lab in np.unique(atlassub_arr)]
    #               for i in range(len(maps_img))], columns=atlassub_labels)
    def stats_rois(maps_img, atlas_arr, atlas_labels):
        rois_stats = dict()
        rois_labels = dict()
        for lab in np.unique(atlas_arr):
            #lab = 9
            rois_labels[atlas_labels[lab]] = int(lab)
            roi_mask = (atlas_arr == lab)
            rois_val =[maps_img[i].get_data()[roi_mask] for i in range(len(maps_img))]
            rois_stats[atlas_labels[lab] + "_mean"] = [np.mean(v) for v in rois_val]
            rois_stats[atlas_labels[lab] + "_std"] = [np.std(v, ddof=1) for v in rois_val]
            rois_stats[atlas_labels[lab] + "_med"] = [np.median(v) for v in rois_val]
        return(pd.DataFrame(rois_stats), rois_labels)

    atlassub_arr[~mask_arr] = 0
    atlascort_arr[~mask_arr] = 0

    rois_sub_stats, rois_sub_labels = stats_rois(maps_img, atlas_arr=atlassub_arr, atlas_labels=atlassub_labels)
    rois_cort_stats, rois_cort_labels = stats_rois(maps_img, atlas_arr=atlascort_arr, atlas_labels=atlascort_labels)

    atlas_sub_img = nib.Nifti1Image(atlassub_arr, affine=ref_img.affine)
    atlas_cort_img = nib.Nifti1Image(atlascort_arr, affine=ref_img.affine)

    return(rois_sub_stats, rois_cort_stats, atlas_sub_img, atlas_cort_img, rois_sub_labels, rois_cort_labels)
