# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:31:01 2014

@author: edouard.duchesnay@cea.fr
"""
from __future__ import print_function
import os
import numpy as np
import scipy.ndimage
import nibabel as nib
import subprocess

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
        atlas_im = nib.Nifti1Image(atlas_arr, affine=cort_image.get_affine())
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
    atlas_im = nib.Nifti1Image(atlas_arr_int, affine=cort_image.get_affine())
    atlas_im.to_filename(output)
    print("Watch if everything is OK:")
    print("fslview %s %s" % (output, ref))
    return atlas_im

