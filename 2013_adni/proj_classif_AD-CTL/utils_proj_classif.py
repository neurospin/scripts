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
    for k in kwargs:
        numpy.save(os.path.join(out_dir, k+".npy"), kwargs[k])
    if mask_im is not None:
        mask = mask_im.get_data() != 0
        arr = numpy.zeros(mask.shape)
        arr[mask] = coef.ravel()
        im_out = nibabel.Nifti1Image(arr, affine=mask_im.get_affine())#, header=mask_im.get_header().copy())
        im_out.to_filename(os.path.join(out_dir,"beta.nii"))

def load(input_dir):
    #input_dir = '/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/10-0.1-0.4-0.5'
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


def get_threshold_from_norm2_ratio(v, ratio=.99):
    """Threshold to apply an input_vector such
    norm2(output_vector) / norm2(input_vector) == ratio
    return the thresholded vector and the threshold"""
    #shape = v.shape
    import numpy as np
    v = v.copy().ravel()
    v2 = (v ** 2)
    v2.sort()
    v2 = v2[::-1]
    v_n2 = np.sqrt(np.sum(v2))
    #(v_n2 * ratio) ** 2
    cumsum2 = np.cumsum(v2)  #np.sqrt(np.cumsum(v2))
    select = cumsum2 <= (v_n2 * ratio) ** 2
    thres = np.sqrt(v2[select][-1])
    return thres

def mcnemar_test_prediction(y_pred1, y_pred2, y_true):
    """Compute the mcnemar_test, return """
    import scipy.stats
    import numpy as np
    r1 = y_pred1.ravel() == y_true.ravel()
    r2 = y_pred2.ravel() == y_true.ravel()
    #a = np.sum(r1 & r2)
    b = np.sum(r1 & np.logical_not(r2))
    c = np.sum(np.logical_not(r1) & r2)
    #d = np.sum(np.logical_not(r1) & np.logical_not(r2))
    #mcnemar_chi2 = float((b -c) ** 2) / float(b + c)
    #The statistic with Yates's correction for continuity
    mcnemar_chi2 = float((np.abs(b - c) - 1) ** 2) / float(b + c)
    pval_chi2 = 1 - scipy.stats.chi2.cdf(mcnemar_chi2, 1)
    # If either b or c is small (b + c < 25) then \chi^2 is not well-approximated
    # by the chi-squared distribution
    # b is compared to a binomial distribution with size parameter equal to b + c
    # and "probability of success" = ½
    #b + c < 25, the binomial calculation should be performed, and indeed, most
    #software packages simply perform the binomial calculation in all cases,
    #since the result then is an exact test in all cases.
    #  to achieve a two-sided p-value in the case of the exact binomial test,
    # the p-value of the extreme tail should be multiplied by 2.
    pval_binom = np.minimum(scipy.stats.binom_test(b, n=b + c, p=0.5) * 2, 1.)
    return pval_chi2, pval_binom