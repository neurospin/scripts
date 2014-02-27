# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:50:14 2014

@author: md238665
"""


#############
# UTILS
#############

def save_model(out_dir, mod, coef, mask_im=None, **kwargs):
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
    for k in kwargs:
        numpy.save(os.path.join(out_dir, k+".npy"), kwargs[k])

def load(input_dir):
    #input_dir = '/neurospin/brainomics/2013_adni/proj_classif/tv/10-0.1-0.4-0.5'
    import os, os.path, pickle, numpy, glob
    res = dict()
    for arr_filename in glob.glob(os.path.join(input_dir, "*.npy")):
        #print arr_filename
        name, ext = os.path.splitext(os.path.basename(arr_filename))
        res[name] = numpy.load(arr_filename)
    for pkl_filename in glob.glob(os.path.join(input_dir, "*.pkl")):
        #print pkl_filename
        name, ext = os.path.splitext(os.path.basename(pkl_filename))
        res[name] = pickle.load(open(pkl_filename, "r"))
    return res
