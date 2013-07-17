# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:54:31 2013

@author: ed203246
"""

# Test samples with missing: 0

import nibabel as nib
import numpy as np


def image_to_file(values, file_path, mask_image, background=0):
    res_arr = np.zeros(mask_image.get_data().shape, dtype=values.dtype)
    if background != 0:
        res_arr[::] = background
    res_arr[mask_image.get_data() != 0] = values
    res_im = nib.Nifti1Image(res_arr, affine=mask_image.get_affine())
    res_im.to_filename(file_path+'.nii.gz')
    return res_im

"""
import nibabel as nib
mask_im = nib.load(get_mask_path())
mask_arr = mask_im.get_data()
mask_bool = mask_arr == 1

#
tmp_im = np.zeros(mask_arr.shape, dtype=np.int16)
nans = np.isnan(X_tr)

tmp_im[mask_bool] = np.isnan(X_tr).sum(0)
#tmp_im[mask_bool] = 1 # OK
# 327

img = nib.Nifti1Image(tmp_im, affine=mask_im.get_affine())
img.to_filename(os.path.join('/tmp','nan.nii.gz'))
"""

#anatomist /neurospin/adhd200/python_analysis/data/mask_t0.1_sum0.8_closing.nii /tmp/nan.nii.gz
