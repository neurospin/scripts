# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:07:02 2013

@author: md238665

Remove cerebellum from mask using the resampled version of TD_lobe.

To do so we first extract a mask of cerebellum regions from the TD_lobe atlas.
Those voxels are removed from the mask and the largest connected component is
extracted.

It is necessary to dilate the cerebellum mask and reapply the same operations.
We take care of not overlapping with regions not in cerebellum when performing the dilation.

At each step we save:
 - the cerebellum mask (dilated or not)
 - the largest CC of the mask without the cerebellum mask
 - the diff with the original mask

"""
import os

import numpy
import scipy.ndimage

import nibabel

BASE_DIR='/neurospin/brainomics/2013_imagen_bmi/'

DATA_PATH = os.path.join(BASE_DIR, 'data')
OUT_PATH = os.path.join(DATA_PATH, 'mask_without_cerebellum/')
MASK_PATH = os.path.join(DATA_PATH, 'mask.nii')
ATLAS_FILE = os.path.join(OUT_PATH, 'TD_lobe_1.5mm.nii')
ATLAS_LABEL_FILE = '/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/TD_lobe.txt'
CEREBELLUM_NAMES = ['Cerebellum Anterior Lobe', 'Cerebellum Posterior Lobe']

# Load mask and atlas file
babel_mask  = nibabel.load(MASK_PATH)
mask        = babel_mask.get_data()
binary_mask = mask!=0
useful_voxels = numpy.ravel_multi_index(numpy.where(binary_mask), mask.shape)
n_useful_voxels = len(useful_voxels)

babel_atlas = nibabel.load(ATLAS_FILE)
numpy_atlas = babel_atlas.get_data()

label_to_name = {}
name_to_label = {}
for line_index, line in enumerate(open(ATLAS_LABEL_FILE)):
    if line_index != 0:
        tokens = line.strip().split('\t')
        label = int(tokens[0])
        region = tokens[1]
        label_to_name[label] = region
        name_to_label[region] = label
CEREBELLUM_LABELS = [name_to_label[name] for name in CEREBELLUM_NAMES]

CEREBELLUM_MASK = (numpy_atlas==CEREBELLUM_LABELS[0]) | (numpy_atlas==CEREBELLUM_LABELS[1])
OTHER_MASK = (numpy_atlas!= 0) & (~CEREBELLUM_MASK)

def smart_cerebellum_dilation(cerebellum_mask, other_mask, dilating_element=None):
    '''Return a dilated cerebellum which doesn't overlap with other structures.
       Inputs
       ------

       cerebellum mask:  mask of the cerebellum
       other_mask:       mask of other structures
       dilating_element: structuring element used for the dilation'''
    dilated_cerebellum_mask = scipy.ndimage.binary_dilation(cerebellum_mask)
    overlap_mask = dilated_cerebellum_mask & other_mask
    dilated_cerebellum_mask[overlap_mask] = False
    # Check: intersection between dilated_cerebellum_mask and other mask must be void
    if (dilated_cerebellum_mask & other_mask).any():
        print "Warning: dilated mask seems to overlap with mask of other brain structures"
    return dilated_cerebellum_mask

def dilate_cerebellum_and_save(cerebellum_mask, other_mask, cerebellum_affine, filename, size=3):
    # Dilate the cerebellum mask
    dilated_cerebellum_mask = smart_cerebellum_dilation(cerebellum_mask, other_mask, numpy.ones((size,size,size)))
    # Save
    outimg = nibabel.Nifti1Image(cerebellum_mask.astype(numpy.uint8), cerebellum_affine)
    nibabel.save(outimg, os.path.join(OUT_PATH, filename))
    return dilated_cerebellum_mask

def remove_cerebellum_mask_and_save(mask, cerebellum_mask, mask_affine, filename):
    # Remove the cerebellum
    mask_without_cerebellum = mask.copy()
    mask_without_cerebellum[cerebellum_mask] = False
    # Extract larger connected component
    label_im, nb_labels = scipy.ndimage.label(mask_without_cerebellum)
    labels =  range(nb_labels + 1)
    sizes = scipy.ndimage.sum(mask, label_im, labels)
    max_size_index = sizes.argmax()
    max_cc = labels[max_size_index]
    mask_without_cerebellum = label_im == max_cc
    # Save
    outimg = nibabel.Nifti1Image(mask_without_cerebellum.astype(numpy.uint8), mask_affine)
    nibabel.save(outimg, os.path.join(OUT_PATH, filename))
    return mask_without_cerebellum

def diff_and_save(original_mask, new_mask, diff_affine, filename):
    diff = numpy.zeros(original_mask.shape, dtype=numpy.int8)
    # Pixel set in original_mask but not in new_mask
    removed_mask = (original_mask) & (~new_mask)
    diff[removed_mask] = -1
    # Pixel set in new_mask but not original_mask
    added_mask = (new_mask) & (~original_mask)
    diff[added_mask] = -1
    # Save
    outimg = nibabel.Nifti1Image(diff, diff_affine)
    nibabel.save(outimg, os.path.join(OUT_PATH, filename))

# Save mask of other structures
outimg = nibabel.Nifti1Image(OTHER_MASK.astype(numpy.uint8), babel_atlas.get_affine())
nibabel.save(outimg, os.path.join(OUT_PATH, 'other_mask.nii'))

# Save cerebellum mask
outimg = nibabel.Nifti1Image(CEREBELLUM_MASK.astype(numpy.uint8), babel_atlas.get_affine())
nibabel.save(outimg, os.path.join(OUT_PATH, 'cerebellum_mask.nii'))

# Remove the cerebellum
mask_without_cerebellum = remove_cerebellum_mask_and_save(binary_mask, CEREBELLUM_MASK, babel_mask.get_affine(), 'mask_without_cerebellum.nii')
diff_and_save(binary_mask, mask_without_cerebellum, babel_mask.get_affine(), 'diff_mask_without_cerebellum.nii')

# Dilate the cerebellum mask
CEREBELLUM_MASK_3 = dilate_cerebellum_and_save(CEREBELLUM_MASK, OTHER_MASK, babel_atlas.get_affine(), 'cerebellum_mask_3.nii')

# Remove the dilated cerebellum
mask_without_cerebellum = remove_cerebellum_mask_and_save(binary_mask, CEREBELLUM_MASK_3, babel_mask.get_affine(), 'mask_without_cerebellum_3.nii')
diff_and_save(binary_mask, mask_without_cerebellum, babel_mask.get_affine(), 'diff_mask_without_cerebellum_3.nii')

# Larger dilation of the cerebellum mask
CEREBELLUM_MASK_7 = dilate_cerebellum_and_save(CEREBELLUM_MASK_3, OTHER_MASK, babel_atlas.get_affine(), 'cerebellum_mask_7.nii')

# Remove the cerebellum
mask_without_cerebellum = remove_cerebellum_mask_and_save(binary_mask, CEREBELLUM_MASK_7, babel_mask.get_affine(), 'mask_without_cerebellum_7.nii')
diff_and_save(binary_mask, mask_without_cerebellum, babel_mask.get_affine(), 'diff_mask_without_cerebellum_7.nii')

# Even larger dilation
CEREBELLUM_MASK_9 = dilate_cerebellum_and_save(CEREBELLUM_MASK_7, OTHER_MASK, babel_atlas.get_affine(), 'cerebellum_mask_9.nii')

# Remove the cerebellum
mask_without_cerebellum = remove_cerebellum_mask_and_save(binary_mask, CEREBELLUM_MASK_9, babel_mask.get_affine(), 'mask_without_cerebellum_9.nii')
diff_and_save(binary_mask, mask_without_cerebellum, babel_mask.get_affine(), 'diff_mask_without_cerebellum_9.nii')