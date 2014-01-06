# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:59:00 2013

@author: ed203246
"""

import nibabel as nib
import numpy as np
import os.path

WD = "/neurospin/brainomics/2012_imagen_subdepression"

input_filename = os.path.join(WD, "results/svm_weight_map/all_voxels_C=0.1_penalty=l2.nii")
output_filename = ""


image = nib.load(input_filename)
image_arr = image.get_data()


res_im = nib.Nifti1Image(res_arr, affine=mask_image.get_affine())
res_im.to_filename(output_filename)


def image_to_file(values, file_path, mask_image, background=0):
    _arr = np.zeros(image.get_data().shape, dtype=values.dtype)
    if background != 0:
        res_arr[::] = background
    res_arr[mask_image.get_data() != 0] = values
    res_im = nib.Nifti1Image(res_arr, affine=mask_image.get_affine())
    res_im.to_filename(file_path+'.nii.gz')
    return res_im
