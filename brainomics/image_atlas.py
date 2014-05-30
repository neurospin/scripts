# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:31:01 2014

@author: edouard.duchesnay@cea.fr
"""
import os
import numpy as np
import scipy.ndimage
import nibabel as nib

# adni
#ref = "/home/ed203246/data/data_sample/adni/mw002_S_0295_S13408_I45109_Nat_dartel_greyProba.nii"
ref = "/home/ed203246/data/data_sample/adni/Template_6.nii"
ref = "/home/ed203246/data/data_sample/mlc2014/mwrc1Test_Sbj1.nii.gz"

def resample_atlas_harvard_oxford(ref, output,
        atlas_base_dir="/usr/share/data/harvard-oxford-atlases/HarvardOxford",
        fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s --interp=nn",
        smooth_size=(3, 3, 3), dilation_size=(3, 3, 3)):
    """Resample HarvardOxford atlas (cortical and subcortical) into reference
    space. Smooth labels and dilate. Smoothing and dilation may take a while.

    Example
    -------
    resample_atlas_harvard_oxford("mwrc1Test_Sbj1.nii.gz", "atlas.nii")
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
    def smooth_labels(buffer):
        return np.argmax(np.bincount(buffer.astype(int)))
    if smooth_size is not None:
        atlas_arr = scipy.ndimage.generic_filter(atlas_arr, smooth_labels, size=smooth_size)
        atlas_im = nib.Nifti1Image(atlas_arr, affine=cort_image.get_affine())
        atlas_im.to_filename("/tmp/atlas_smoothed.nii.gz")
    # dilation
    def dilation_labels(buffer):
        buffer = buffer.astype(int)
        if np.any(buffer > 0):
            buffer = buffer[buffer != 0]
            return np.argmax(np.bincount(buffer))
        else:
            return 0
    if dilation_size is not None:
        atlas_arr = scipy.ndimage.generic_filter(atlas_arr, dilation_labels, size=dilation_size)
        atlas_im = nib.Nifti1Image(atlas_arr, affine=cort_image.get_affine())
    atlas_im.to_filename(output)
    print "Watch everithing is OK"
    print "fslview /tmp/sub.nii.gz /tmp/cort.nii.gz /tmp/atlas.nii.gz \
        /tmp/atlas_smoothed.nii.gz %s %s" % (output, ref)


