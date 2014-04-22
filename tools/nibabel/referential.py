# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:40:18 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import nibabel as nib

input_filename = "/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz"
im_in = nib.load(input_filename)
arr = im_in.get_data()

im_out = nib.Nifti1Image(arr, affine=im_in.get_affine(), header=im_in.get_header().copy())
im_out.to_filename('/tmp/toto.nii')
