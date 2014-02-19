# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:38:27 2014

@author: jl237561
"""

import os
import numpy
import tables
import numpy as np


def check_array_NaN(nparray):
    """
    Example
    -------
    array_data = [np.log(-1.),1.,np.log(0)]
    check_array_NaN(array_data)
    array_data = [1,2,3]
    check_array_NaN(array_data)
    """
    for any_element in nparray:
        if np.isnan(any_element).any():
            raise ValueError("np.array contain NaN")


def sub_check_array_zero(nparray):
    '''
    Example
    -------
    import numpy as np
    nparray = np.asarray([1.0, 2.0, 3.0, 0.0])
    check_array_zero(nparray)
    '''
    zero_mat = numpy.zeros(nparray.shape,
                           dtype=numpy.uint8)
    nsum = np.sum(nparray == zero_mat)
    if nsum > 0:
        raise ValueError("np.array contain zero")


def check_array_zero(nparray):
    '''
    Example
    -------
    import numpy as np
    nparray = np.asarray([1.0, 2.0, 3.0, 0.0])
    check_array_zero(nparray)
    '''
    for any_element in nparray:
        sub_check_array_zero(any_element)


def check_image(image_array):
    check_array_NaN(image_array)
    check_array_zero(image_array)


def check_snps_128(nparray):
    if np.sum(nparray == 128) > 1:
        raise ValueError("Snps contain 128")


def check_snps(nparray):
    check_snps_128(nparray)
    check_array_NaN(nparray)


if __name__ == "__main__":
    # Input
    BASE_DIR = '/neurospin/brainomics/2014_imagen_anat_vgwas_gpu_ridge'
    COR_BASE_DIR = os.path.join(BASE_DIR, "2013_imagen_anat_vgwas_gpu")
    OUT_DATA_BASE = os.path.join(BASE_DIR, "2013_imagen_anat_vgwas_gpu")
    # Input files from 2013_imagen_anat_vgwas_gpu
    IN_DIR = os.path.join(COR_BASE_DIR, 'data')
    IN_SNP_NPZ = os.path.join(IN_DIR, 'snp')
    IN_SNP_LIST_NPZ = os.path.join(IN_DIR, 'snp_list')
    IN_HDF5_FILE_FULRES_INTER = os.path.join(IN_DIR,
                                            'cache_full_res_inter.hdf5')
    # Output to 2014_imagen_anat_vgwas_gpu_ridge
    OUT_DATA_DIR = os.path.join(BASE_DIR, "data")
    OUT_IMAGE_DATA = os.path.join(OUT_DATA_DIR, "images.mem")
    OUT_IMAGE_HDF5_DATA = os.path.join(OUT_DATA_DIR, "images.hdf5")
    OUT_SNPS_PATH = os.path.join(OUT_DATA_DIR, "snps")
    OUT_SNPS_LIST_PATH = os.path.join(OUT_DATA_DIR, "snps_list")
    OUT_COV_PATH = os.path.join(OUT_DATA_DIR, "cov")

    # Load and save geno data
    geno_data = np.load(IN_SNP_NPZ + ".npz")
    geno_data = geno_data[geno_data.files[0]]
    check_snps(geno_data)
    np.savez(OUT_SNPS_PATH, data=geno_data)

    # Load and save snp list
    snp_list = np.load(IN_SNP_LIST_NPZ + ".npy")
    np.save(OUT_SNPS_LIST_PATH, snp_list)

    # Load and save image data
    h5file = tables.openFile(IN_HDF5_FILE_FULRES_INTER, mode="r")
    images = h5file.getNode(h5file.root, 'images')
    atom = tables.Atom.from_dtype(images.dtype)
    image_shape = images.shape
    check_image(images)

    # Save as hdf5 file
    h5file_out = tables.openFile(OUT_IMAGE_HDF5_DATA,
                                 mode="w",
                                 title='dataset_pa_prace')
    ds = h5file_out.createCArray(h5file_out.root,
                             'images',
                             atom,
                             image_shape)
    ds[:] = np.asarray(images)[:]
    h5file_out.close()

    # Save as memmap file
    images_mem = np.memmap(OUT_IMAGE_DATA,
                           dtype='float64',
                           mode='w+', shape=image_shape)
    images_mem[:] = images[:]
    del images_mem
    h5file.close()

    # To test mem file
    test_image_mem = np.memmap(OUT_IMAGE_DATA,
                               dtype='float64',
                               mode='r',
                               shape=image_shape)
    del test_image_mem

    # covariates
    cov_data = np.load(OUT_COV_PATH + ".npy")
    check_array_NaN(cov_data)
