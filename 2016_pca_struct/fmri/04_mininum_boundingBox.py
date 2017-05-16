#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:45:23 2017

@author: ad247405
"""

mask = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz').get_data()
corner, size = smallest_bounding_box(X1)




def smallest_bounding_box(msk):
    """
    Extract the smallest bounding box from a mask

    Parameters
    ----------
    msk : ndarray
      Array of boolean

    Returns
    -------
    corner: ndarray
      3-dimensional coordinates of bounding box corner
    size: ndarray
      3-dimensional size of bounding box
    """
    x, y, z = np.where(msk > 10)
    corner = np.array([x.min(), y.min(), z.min()])
    size = np.array([x.max() + 1, y.max() + 1, z.max() + 1])
    return corner, size
    
    

def smallest_bounding_box(msk):
    """
    Extract the smallest bounding box from a mask

    Parameters
    ----------
    msk : ndarray
      Array of boolean

    Returns
    -------
    corner: ndarray
      3-dimensional coordinates of bounding box corner
    size: ndarray
      3-dimensional size of bounding box
    """
    x, y, z = np.where(msk > 0)
    corner = np.array([x.min(), y.min(), z.min()])
    size = np.array([x.max() + 1, y.max() + 1, z.max() + 1])
    return corner, size



