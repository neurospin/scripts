# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:59:00 2013

@author: ed203246
"""
# ############################################################################
# Read MNI template
# ############################################################################
import os

WD = "/neurospin/brainomics/neuroimaging_ressources/"
os.chdir(WD)
input_filename = "spm8/templates/T1.nii"
image = nib.load(input_filename)
h = image.get_header()
image.get_affine()

# ############################################################################
# Rescale image values to [-1, 1]
# ############################################################################
import os
import os.path
import nibabel as nib
import numpy as np


WD = "/neurospin/brainomics/neuroimaging_ressources/"
os.chdir(WD)
input_filename = "examples_images/weights_map_l2.nii"
output_filename = "examples_images/weights_map_l2_scaled.nii"

image = nib.load(input_filename)
image_arr = image.get_data()
image_arr[image_arr > 0] = image_arr[image_arr > 0] / image_arr.max()
image_arr[image_arr < 0] = image_arr[image_arr < 0] / np.abs(image_arr.min())
out_im = nib.Nifti1Image(image_arr, affine=image.get_affine())
out_im.to_filename(output_filename)

# ############################################################################
# Add large cluster in l1 weigths map: threshold l2 and take cluster 
# larger than 300
# ############################################################################
import os
import os.path
import nibabel as nib
import numpy as np
import scipy, scipy.ndimage

WD = "/neurospin/brainomics/neuroimaging_ressources/"
os.chdir(WD)
input_filename1 = "examples_images/weights_map_l1.nii"
input_filename2 = "examples_images/weights_map_l2.nii"
output_filename = "examples_images/weights_map_mixte.nii"

im1 = nib.load(input_filename1)
arr1 = im1.get_data()
print np.sum(arr1 !=0)
im2 = nib.load(input_filename2)
arr2 = im2.get_data()

l1l2_scale = \
np.abs(arr1[arr1 !=0 ]).mean() / np.abs(arr2[arr2 !=0 ]).mean()

tpos = np.mean(arr2[arr2>0]) +  np.std(arr2[arr2>0])
tneg = np.mean(arr2[arr2<0]) -  np.std(arr2[arr2<0])
clust_bool = np.zeros(arr2.shape, dtype=bool)
clust_bool[arr2 > tpos] = True
clust_bool[arr2 < tneg] = True

# remove small clusters <= 300
clust_labeled, n_clusts = scipy.ndimage.label(clust_bool)
clust_sizes = scipy.ndimage.measurements.histogram(clust_labeled, 1, n_clusts, n_clusts)
label_of_large_regions = np.arange(1, (n_clusts+1))[clust_sizes > 300]
size_of_large_regions = clust_sizes[clust_sizes > 300]
# 
for lab in label_of_large_regions:
    mask = clust_labeled == lab
    arr1[mask] = l1l2_scale * arr2[mask]

print np.sum(arr1 !=0)

out_im = nib.Nifti1Image(arr1, affine=im1.get_affine())
out_im.to_filename(output_filename)

# Check
im3 = nib.load(output_filename)
arr3 = im3.get_data()
print np.sum(arr3 !=0)

