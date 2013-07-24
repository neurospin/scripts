# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:59:00 2013

@author: ed203246
"""

import os
import os.path
import nibabel as nib
import numpy as np


WD = os.path.join(os.environ["HOME"], "data", "neuroimaging_ressources")
os.chdir(WD)
input_filename = "images/weights_map_l2.nii"
output_filename = "images/weights_map_l2_scaled.nii"

image = nib.load(input_filename)
image_arr = image.get_data()
image_arr[image_arr > 0] = image_arr[image_arr > 0] / image_arr.max()
image_arr[image_arr < 0] = image_arr[image_arr < 0] / np.abs(image_arr.min())
out_im = nib.Nifti1Image(image_arr, affine=image.get_affine())
out_im.to_filename(output_filename)