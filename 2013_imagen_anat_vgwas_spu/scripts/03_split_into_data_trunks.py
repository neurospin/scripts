# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5th 2013

@author: jl237561

This script perform :
    - split data into trunks

"""
import sys
sys.path.append('/home/vf140245/gits/igutils')
import os
import igutils as ig
import numpy as np
import tables

def split_into_chunks(data,
                      filename_prefix,
                      num_chunks,
                      is_col=True,
                      is_npz=True,
                      fixed_size=None):
    """
    Split into sets of columns or rows

    Parameters
    ----------
    data: numpy array
        data is a numpy array to split into chunks

    filename_prefix: string
        where to save chunks

    num_chunks:
        the number of chunks you want split

    is_npz: boolean
        Save chunk in npz format. If false, save in npy format

    Example
    -------
    import numpy as np
    data = np.random.random((20, 500))
    filename_prefix = "/tmp/data"
    split_into_chunks(data, filename_prefix, 10)
    one_chunk_data_filename = filename_prefix + "_chunk_4.npz"
    one_chunk_data = np.load(one_chunk_data_filename)
    one_chunk_data = one_chunk_data[one_chunk_data.files[0]]
    print one_chunk_data.shape
    print np.all(one_chunk_data == data[:, 200:250])

    split_into_chunks(data, filename_prefix, 10, fixed_size=50)
    one_chunk_data_filename = filename_prefix + "_chunk_4.npz"
    one_chunk_data = np.load(one_chunk_data_filename)
    one_chunk_data = one_chunk_data[one_chunk_data.files[0]]
    print one_chunk_data.shape
    print np.all(one_chunk_data == data[:, 200:250])

    split_into_chunks(data, filename_prefix, 10, fixed_size=49)
    one_chunk_data_filename = filename_prefix + "_chunk_4.npz"
    one_chunk_data = np.load(one_chunk_data_filename)
    one_chunk_data = one_chunk_data[one_chunk_data.files[0]]
    print one_chunk_data.shape
    print np.all(one_chunk_data == data[:, 49*4:49*5])

    split_into_chunks(data, filename_prefix, 10, fixed_size=49)
    one_chunk_data_filename = filename_prefix + "_chunk_10.npz"
    one_chunk_data = np.load(one_chunk_data_filename)
    one_chunk_data = one_chunk_data[one_chunk_data.files[0]]
    print one_chunk_data.shape
    print np.all(one_chunk_data == data[:, 49*10:])

    split_into_chunks(data, filename_prefix, 10, is_npz=False)
    one_chunk_data_filename = filename_prefix + "_chunk_4.npy"
    one_chunk_data = np.load(one_chunk_data_filename)
    print one_chunk_data.shape
    print np.all(one_chunk_data == data[:, 200:250])

    split_into_chunks(data, filename_prefix, 10, is_npz=False, is_col=False)
    one_chunk_data_filename = filename_prefix + "_chunk_4.npy"
    one_chunk_data = np.load(one_chunk_data_filename)
    print one_chunk_data.shape
    print np.all(one_chunk_data == data[8:10, :])
    """
    if not fixed_size:
        data_chunk_offsets = np.linspace(0,
                                         data.shape[1],
                                         num_chunks + 1).astype(int)
    else:
        data_chunk_offsets = range(0, data.shape[1], fixed_size)
        if data_chunk_offsets[-1] < data.shape[1]:
            data_chunk_offsets.append(data.shape[1])
    if is_npz:
        np.savez(filename_prefix + "_chunk_offsets", data_chunk_offsets)
    else:
        np.save(filename_prefix + "_chunk_offsets", data_chunk_offsets)
    for i in xrange(len(data_chunk_offsets) - 1):
        data_chunk = data[:,
                         data_chunk_offsets[i]:
                         data_chunk_offsets[i + 1]]
        if not is_col:
            data_chunk = data_chunk.T
        if is_npz:
            np.savez(filename_prefix +
                     "_chunk_" +
                     repr(i) +
                     ".npz",
                     data=data_chunk, offset=data_chunk_offsets[i])
        else:
            np.save(filename_prefix +
                    "_chunk_" +
                    repr(i) +
                    ".npy",
                    data_chunk)

if __name__ == "__main__":
    # Input
    BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_spu'
    DATA_DIR = os.path.join(BASE_DIR, '2013_imagen_bmi', 'data')
    CLINIC_DIR = os.path.join(DATA_DIR, 'clinic')

    NUM_CHUNK_SNP = 40
    NUM_CHUNK_IMG = 876
    CHUNK_SIZE_IMG = 384

    # Output files
    OUT_DIR = os.path.join(BASE_DIR, 'data')
    OUT_SNP_NPZ = os.path.join(OUT_DIR, 'snp')
    OUT_SNP_LIST_NPZ = os.path.join(OUT_DIR, 'snp_list')
    OUT_TR_IMAGE_NPZ = os.path.join(OUT_DIR, 'tr_image')
    OUT_TR_IMAGE_VOX = os.path.join(OUT_DIR, 'vox')
    OUT_TR_IMAGE_NO_CERE_NPZ = os.path.join(OUT_DIR,
                                            'tr_images_without_cerebellum')
    OUT_HDF5_FILE_FULRES_INTER = os.path.join(OUT_DIR,
                                            'cache_full_res_inter.hdf5')
    # Load geno data
    geno_data = np.load(OUT_SNP_NPZ + ".npz")
    geno_data = geno_data[geno_data.files[0]]

    # Load image data
    h5file = tables.openFile(OUT_HDF5_FILE_FULRES_INTER, mode="r")
    images = h5file.getNode(h5file.root, 'images')
    images_without_cerebellum = h5file.getNode(h5file.root,
                                               "images_without_cerebellum")
    # Split data into data chunks
    split_into_chunks(data=geno_data,
                      filename_prefix=OUT_SNP_NPZ,
                      num_chunks=NUM_CHUNK_SNP)
    split_into_chunks(data=images,
                      filename_prefix=OUT_TR_IMAGE_VOX,
                      num_chunks=NUM_CHUNK_IMG,
                      is_col=False,
                      fixed_size=CHUNK_SIZE_IMG)
#    split_into_chunks(data=images_without_cerebellum,
#                      filename_prefix=OUT_TR_IMAGE_NO_CERE_NPZ,
#                      num_chunks=NUM_CHUNK_IMG,
#                      is_col=False)
    h5file.close()
    print "Finish"
