# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:50:14 2014

@author: md238665
"""


#############
# UTILS
#############

def save_model(out_dir, mod, coef, mask_im=None):
    import os, os.path, pickle, nibabel, numpy
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pickle.dump(mod, open(os.path.join(out_dir, "model.pkl"), "w"))
    if mask_im is not None:
        mask = mask_im.get_data() != 0
        arr = numpy.zeros(mask.shape)
        arr[mask] = coef.ravel()
        im_out = nibabel.Nifti1Image(arr, affine=mask_im.get_affine(), header=mask_im.get_header().copy())
        im_out.to_filename(os.path.join(out_dir,"beta.nii"))

