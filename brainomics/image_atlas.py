# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:31:01 2014

@author: edouard.duchesnay@cea.fr
"""
import os
import numpy as np
import scipy.ndimage
import nibabel as nib


def smooth_labels(arr, size):
    def func(buffer):
        return np.argmax(np.bincount(buffer.astype(int)))
    arr = scipy.ndimage.generic_filter(arr, func, size=size)
    return arr


def dilation_labels(arr, size):
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
        fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s --interp=nn",
        smooth_size=(3, 3, 3), dilation_size=(3, 3, 3)):
    """Resample HarvardOxford atlas (cortical and subcortical) into reference
    space. Smooth labels and dilate. Smoothing and dilation may take a while.

    Example
    -------
    from brainomics.image_atlas import resample_atlas_harvard_oxford
    resample_atlas_harvard_oxford("mwrc1Test_Sbj1.nii.gz", "atlas.nii.gz")
    """
    cort_filename = os.path.join(atlas_base_dir, "HarvardOxford-cort-maxprob-thr0-1mm.nii.gz")
    sub_filename = os.path.join(atlas_base_dir, "HarvardOxford-sub-maxprob-thr0-1mm.nii.gz")
    # resamp subcortical
    os.system(fsl_cmd % (sub_filename, ref, "/tmp/sub"))
    # remone WM, GM, ventriculus (3, 14)
    sub_image = nib.load("/tmp/sub.nii.gz")
    sub_arr = sub_image.get_data()
    sub_arr[(sub_arr == 1)  | (sub_arr == 2)  |
            (sub_arr == 12) | (sub_arr == 13) |
            (sub_arr == 3)  | (sub_arr == 14)] = 0
    sub_image.to_filename("/tmp/sub.nii.gz")
    # resamp cortical
    os.system(fsl_cmd % (cort_filename, ref, "/tmp/cort"))
    cort_image = nib.load("/tmp/cort.nii.gz")
    atlas_arr = cort_image.get_data()
    atlas_arr[sub_arr != 0] = sub_arr[sub_arr != 0]
    cort_image.to_filename("/tmp/atlas.nii.gz")
    # smooth labels
    if smooth_size is not None:
        atlas_arr = smooth_labels(atlas_arr, size=smooth_size)
        atlas_im = nib.Nifti1Image(atlas_arr, affine=cort_image.get_affine())
        atlas_im.to_filename("/tmp/atlas_smoothed.nii.gz")
    if dilation_size is not None:
        atlas_arr = dilation_labels(atlas_arr, size=dilation_size)
    atlas_arr_int = atlas_arr.astype('int16')
    assert np.all(atlas_arr_int == atlas_arr)
    atlas_im = nib.Nifti1Image(atlas_arr_int, affine=cort_image.get_affine())
    atlas_im.to_filename(output)
    print "Watch everithing is OK"
    print "fslview /tmp/sub.nii.gz /tmp/cort.nii.gz /tmp/atlas.nii.gz \
        /tmp/atlas_smoothed.nii.gz %s %s" % (output, ref)


