# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:01:25 2014

@author: jl237561
"""
import nibabel as ni
import numpy as np
import joblib
import os
import tables
import matplotlib.pyplot as plt
from numpy import array, ones, linalg


def get_linear_weights(x, y):
    '''
    Example
    -------
    x = arange(0,9)
    y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
    weights = get_linear_weights(x, y)
    predict_y = weights[0] * x + weights[1]
    '''
    A = array([x, ones(len(x))])
    # linearly generated sequence
    w = linalg.lstsq(A.T, y)[0]  # obtaining the parameters
    return w

BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
PATH_RES_RESULT = os.path.join(BASE_DIR,
                               'data',
                               'tmp_reduce',
                               'result.joblib')
DATA_DIR = os.path.join(BASE_DIR, '2013_imagen_bmi', 'data')
MASK_FILE = os.path.join(DATA_DIR, 'mask', 'mask.nii')
mask_file = MASK_FILE
OUT_IMAGES = os.path.join(BASE_DIR,
                          'interesting_snp_brain_img')

OUT_DIR = os.path.join(BASE_DIR, 'data')
OUT_HDF5_FILE_FULRES_INTER = os.path.join(OUT_DIR,
                                            'cache_full_res_inter.hdf5')
OUT_SNP_NPZ = os.path.join(OUT_DIR, 'snp')
OUT_SNP_LIST_NPY = os.path.join(OUT_DIR, 'snp_list')

# Load image data
h5file = tables.openFile(OUT_HDF5_FILE_FULRES_INTER, mode="r")
images = h5file.getNode(h5file.root, 'images')

# Load geno data
geno_data = np.load(OUT_SNP_NPZ + ".npz")
geno_data = geno_data[geno_data.files[0]]

# Load snp list
snp_list_name = np.load(OUT_SNP_LIST_NPY + ".npz")
snp_list_name = snp_list_name[snp_list_name.files[0]]

ar = joblib.load(PATH_RES_RESULT)
h1 = ar['h1']
h0 = ar['h0']
snp_of_interest = [122664, 379105]
n_voxels = 336188

for snp in snp_of_interest:
    sp_h1_snp = h1[h1['x_id'] == snp]
    # compute average grep matter value
    voxels_snp = images[:, sp_h1_snp['y_id']]
    subjects_snp = geno_data[:, snp]
    print "======================================"
    print "The snp name is ", snp_list_name[snp]
    for snp_val in np.unique(subjects_snp):
        print "The number of '", snp_val, "'=", np.sum(subjects_snp == snp_val)
    # Plot new figure
    plot_x = subjects_snp
    plot_y_avg = np.average(voxels_snp, axis=1)
    plot_y_max = np.max(voxels_snp, axis=1)
    # Linear regression using least square
    avg_weights = get_linear_weights(plot_x, plot_y_avg)
    max_weights = get_linear_weights(plot_x, plot_y_max)
    regression_line_x = np.sort(np.unique(plot_x))
    print "avg_weights =", avg_weights
    print "max_weights =", max_weights
    # Plot first subplot
    fig = plt.figure("%s (a point denotes a subject)" % snp_list_name[snp])
    ax = fig.add_subplot(211)
    ax.set_title('%s (a point denotes a subject)' % snp_list_name[snp])
    ax.set_xlabel('snp value')
    ax.set_ylabel('average voxel value')
    ax.scatter(plot_x, plot_y_avg, s=30, facecolors='none', edgecolors='r', alpha=0.3)
    ax.plot(regression_line_x, regression_line_x * avg_weights[0] + avg_weights[1])
    # Plot second subplot
    ax = fig.add_subplot(212)
    ax.set_title('%s (a point denotes a subject)' % snp_list_name[snp])
    ax.set_xlabel('snp value')
    ax.set_ylabel('max voxel value')
    ax.scatter(plot_x, plot_y_max, s=30, facecolors='none', edgecolors='r', alpha=0.3)
    ax.plot(regression_line_x, regression_line_x * max_weights[0] + max_weights[1])

plt.show()
