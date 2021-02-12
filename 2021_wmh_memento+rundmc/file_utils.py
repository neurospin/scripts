#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:34:14 2021

@author: ed203246
"""

def filename_type(filename):
    """Get filnename from type


    Parameters
    ----------
    filename : str
        filename.

    Returns
    -------
    str in [npy, nii] or None.

    Examples
    --------
    >>> filename_type('/tmp/image.nii.gz')
    'nii'
    >>> filename_type('/tmp/image.nii')
    'nii'
    >>> filename_type('/tmp/array.npy')
    'npy'
    >>> filename_type('/tmp/array.npz')
    'npy'
    """
    import re

    nii_re = re.compile(".+(nii.gz)$|.+(nii)$")
    npy_re = re.compile(".+(npy)$|.+(npz)$")


    if len(nii_re.findall(filename)):
        return 'nii'
    elif len(npy_re.findall(filename)):
        return 'npy'
    return None

def load_npy_nii(filename):
    """Load Nifti1Image or Array depending on file extension


    Parameters
    ----------
    filename : str
        filename.

    Returns
    -------
    array or Nifti1Image or None.
        data.

    """
    import numpy as np
    import nibabel

    if filename_type(filename) == 'nii':
        return nibabel.load(filename)

    elif filename_type(filename) == 'npy':
        return np.load(filename)

    return None
